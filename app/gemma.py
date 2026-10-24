"""The only module that talks to the Gemini API.

Everything else in the app calls ask_gemma(). Keeping the model call in one
place is deliberate: this file is the single spot that knows about the SDK,
retries, and which Gemma variant is in use, so moving to a different backend
(a local Ollama server, say) means rewriting this file and nothing else.

The retry policy exists because the API returned 500 and 503 errors on plain
prompts during development. See docs/adr/0002-retry-then-fall-back.md.
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

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
    data: Optional[dict] = None


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def parse_json_payload(text: str) -> Optional[dict]:
    """Extract and parse structured JSON from model output."""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()

    # 1. Direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. Extract from ```json { ... } ``` markdown fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 3. Find outermost braces { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def ensure_structured_data(text: str, data: Optional[dict] = None) -> dict:
    """Ensure a complete structured RxGuard response dictionary is available."""
    if data and isinstance(data, dict) and "safety_score" in data:
        return data

    text_lower = (text or "").lower()
    has_warning = any(
        w in text_lower
        for w in ("warning", "danger", "conflict", "avoid", "do not", "contraindication", "overdose")
    )
    score = "DANGER" if "danger" in text_lower or "overdose" in text_lower else ("CAUTION" if has_warning else "SAFE")

    return {
        "safety_score": score,
        "headline": "Medication Analysis Summary",
        "audio_summary": (text[:250] + "...") if len(text) > 250 else text,
        "medications": [],
        "conflicts_and_warnings": [
            {
                "severity": "HIGH" if score == "DANGER" else ("MEDIUM" if score == "CAUTION" else "LOW"),
                "title": "Clinical & Safety Guidance",
                "description": text or "No specific warnings identified.",
            }
        ],
        "daily_schedule": {
            "morning": [],
            "afternoon": [],
            "evening": [],
            "bedtime": [],
        },
        "emergency_card": {
            "patient_action": "Consult your prescribing doctor or pharmacist before making medication changes.",
            "active_medications": [],
            "critical_allergies_or_contraindications": [],
            "clinician_notes": text,
        },
    }


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
    image_path: Optional[Union[os.PathLike, Sequence[os.PathLike], str]] = None,
    *,
    client: Any = None,
    settings: Optional[Settings] = None,
    system_prompt: Optional[str] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """Ask Gemma a question, optionally about one or more medication images.

    Retries transient failures with exponential backoff, then moves on to the
    next configured model. Errors that retrying cannot fix (bad key, malformed
    request) are raised immediately.
    """
    settings = settings or Settings.from_env()
    client = client or get_client(settings.timeout_s)
    if system_prompt is None:
        system_prompt = load_system_prompt()

    config = None
    if system_prompt:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system_prompt)

    uploaded: Optional[list] = None
    attempts = 0
    last_error: Optional[BaseException] = None

    # Normalize image_path to a list of paths
    paths = []
    if image_path is not None:
        if isinstance(image_path, (list, tuple)):
            paths = [str(p) for p in image_path]
        else:
            paths = [str(image_path)]

    for model in settings.models:
        for attempt in range(1, settings.retries_per_model + 1):
            attempts += 1
            try:
                if paths and uploaded is None:
                    uploaded = [client.files.upload(file=p) for p in paths]

                contents = [*uploaded, prompt] if uploaded else [prompt]
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                text = (response.text or "").strip()
                if not text:
                    raise _EmptyResponse()
                parsed_data = parse_json_payload(text)
                return Answer(text=text, model=model, attempts=attempts, data=parsed_data)
            except Exception as exc:
                last_error = exc
                if _error_code(exc) == 404:
                    log.warning("%s not found for this key, trying next model", model)
                    break
                if not _is_transient(exc):
                    raise
                log.warning(
                    "%s attempt %d/%d failed: %s",
                    model,
                    attempt,
                    settings.retries_per_model,
                    exc,
                )
                if attempt < settings.retries_per_model:
                    delay = settings.base_delay_s * 2 ** (attempt - 1)
                    sleep(delay + random.uniform(0, settings.base_delay_s / 2))

    raise GemmaUnavailable(
        f"No Gemma model answered after {attempts} attempts "
        f"(tried: {', '.join(settings.models)})",
        last_error=last_error,
    )
