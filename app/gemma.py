"""The only module that talks to the Gemini API.

Everything else in the app calls ask_gemma(). Keeping the model call in one
place is deliberate: this file is the single spot that knows about the SDK,
retries, and which Gemma variant is in use, so moving to a different backend
(a local Ollama server, say) means rewriting this file and nothing else.

The retry policy exists because the API returned 500 and 503 errors on plain
prompts during development. See docs/adr/0002-retry-then-fall-back.md.
"""

import logging
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger(__name__)

# 26B A4B first: Google's docs suggest it as the general-purpose starting point.
# 31B is the fallback because it is the only other Gemma 4 model the Gemini API
# lists for this key.
DEFAULT_MODELS = ("gemma-4-26b-a4b-it", "gemma-4-31b-it")

# 429 is rate limiting. 500 and 503 are the codes actually seen during
# development; 502/504 are the usual gateway cousins.
TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})

# Network-level failures don't carry an HTTP code. Matched by class name so we
# don't have to import (and pin to) whichever HTTP library the SDK uses today.
TRANSIENT_ERROR_NAMES = frozenset(
    {
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutException",
        "ConnectError",
        "RemoteProtocolError",
    }
)

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.txt"


class ConfigError(RuntimeError):
    """The server is missing something it needs (e.g. the API key)."""


class GemmaUnavailable(RuntimeError):
    """Every configured model failed, even after retries."""

    def __init__(self, message: str, last_error: Optional[BaseException] = None):
        super().__init__(message)
        self.last_error = last_error


class _EmptyResponse(Exception):
    """The API answered 200 but gave us no text."""


@dataclass(frozen=True)
class Settings:
    models: Tuple[str, ...] = DEFAULT_MODELS
    retries_per_model: int = 2
    base_delay_s: float = 1.0
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        raw_models = os.environ.get("GEMMA_MODELS", "")
        models = tuple(m.strip() for m in raw_models.split(",") if m.strip())
        return cls(
            models=models or DEFAULT_MODELS,
            retries_per_model=int(os.environ.get("GEMMA_RETRIES_PER_MODEL", 2)),
            base_delay_s=float(os.environ.get("GEMMA_BASE_DELAY_S", 1.0)),
            timeout_s=float(os.environ.get("GEMMA_TIMEOUT_S", 30)),
        )


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    attempts: int


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def get_client(timeout_s: float = 30.0) -> Any:
    # Imported lazily so tests that inject a fake client never need the SDK
    # configured, and importing this module never needs a key.
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key "
            "from https://aistudio.google.com/apikey"
        )
    # Without an explicit timeout, one stuck request can hang for minutes
    # (a plain call took 1m45s to return a 500 while debugging). The SDK takes
    # the timeout in milliseconds.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )


def _error_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _EmptyResponse):
        return True
    if _error_code(exc) in TRANSIENT_CODES:
        return True
    return type(exc).__name__ in TRANSIENT_ERROR_NAMES


def ask_gemma(
    prompt: str,
    image_path: Optional[os.PathLike] = None,
    *,
    client: Any = None,
    settings: Optional[Settings] = None,
    system_prompt: Optional[str] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """Ask Gemma a question, optionally about an image.

    Retries transient failures with exponential backoff, then moves on to the
    next configured model. Errors that retrying cannot fix (bad key, malformed
    request) are raised immediately: falling back would just repeat the mistake
    against a second model.

    `client`, `settings` and `sleep` are injectable so the retry logic can be
    tested without network access or real waiting.
    """
    settings = settings or Settings.from_env()
    client = client or get_client(settings.timeout_s)
    if system_prompt is None:
        system_prompt = load_system_prompt()

    config = None
    if system_prompt:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system_prompt)

    uploaded = None  # uploaded lazily, then reused across every retry and model
    attempts = 0
    last_error: Optional[BaseException] = None

    for model in settings.models:
        for attempt in range(1, settings.retries_per_model + 1):
            attempts += 1
            try:
                if image_path is not None and uploaded is None:
                    uploaded = client.files.upload(file=str(image_path))
                contents = [uploaded, prompt] if uploaded is not None else [prompt]
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                text = (response.text or "").strip()
                if not text:
                    raise _EmptyResponse()
                return Answer(text=text, model=model, attempts=attempts)
            except Exception as exc:  # noqa: BLE001 - classified just below
                last_error = exc
                if _error_code(exc) == 404:
                    # This key can't see this model. Retrying won't change that,
                    # but the next model in the list might work.
                    log.warning("%s not found for this key, trying next model", model)
                    break
                if not _is_transient(exc):
                    raise
                log.warning(
                    "%s attempt %d/%d failed: %s",
                    model, attempt, settings.retries_per_model, exc,
                )
                if attempt < settings.retries_per_model:
                    # Jitter keeps a room full of people hitting the same
                    # overloaded endpoint from retrying in lockstep.
                    delay = settings.base_delay_s * 2 ** (attempt - 1)
                    sleep(delay + random.uniform(0, settings.base_delay_s / 2))

    raise GemmaUnavailable(
        f"No Gemma model answered after {attempts} attempts "
        f"(tried: {', '.join(settings.models)})",
        last_error=last_error,
    )
