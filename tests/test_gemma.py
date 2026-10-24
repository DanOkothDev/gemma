"""Tests for the retry and fallback policy in app/gemma.py.

A scripted fake client stands in for the Gemini API, so these run offline and
without sleeping. They pin down the behaviour that matters on stage: transient
errors are retried, an unhealthy model is abandoned for the next one, and
errors that retrying can't fix surface immediately.
"""

import pytest

from app.gemma import Answer, GemmaUnavailable, Settings, ask_gemma, load_system_prompt


class FakeAPIError(Exception):
    """Mimics the SDK's errors, which expose the HTTP status as `.code`."""

    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Plays back one scripted outcome per generate_content call."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []  # model names, in call order
        self.uploads = []
        # The real client is client.models.generate_content / client.files.upload.
        self.models = self
        self.files = self

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    def upload(self, *, file):
        self.uploads.append(file)
        return "uploaded-file-handle"


@pytest.fixture
def settings():
    return Settings(models=("m1", "m2"), retries_per_model=2, base_delay_s=0.0, timeout_s=5)


@pytest.fixture
def sleeps():
    return []


def run(client, settings, sleeps, prompt="hi", image=None):
    # system_prompt="" skips building the SDK config object, keeping these pure.
    return ask_gemma(
        prompt,
        image,
        client=client,
        settings=settings,
        system_prompt="",
        sleep=sleeps.append,
    )


def test_first_try_success(settings, sleeps):
    client = FakeClient(["hello"])
    assert run(client, settings, sleeps) == Answer(text="hello", model="m1", attempts=1)
    assert sleeps == []


def test_transient_error_is_retried_on_same_model(settings, sleeps):
    client = FakeClient([FakeAPIError(503), "ok"])
    answer = run(client, settings, sleeps)
    assert (answer.model, answer.attempts) == ("m1", 2)
    assert client.calls == ["m1", "m1"]
    assert len(sleeps) == 1


def test_falls_back_to_next_model_after_retries_exhausted(settings, sleeps):
    client = FakeClient([FakeAPIError(500), FakeAPIError(503), "ok"])
    answer = run(client, settings, sleeps)
    assert answer.model == "m2"
    assert client.calls == ["m1", "m1", "m2"]


def test_raises_when_every_model_fails(settings, sleeps):
    client = FakeClient([FakeAPIError(503)] * 4)
    with pytest.raises(GemmaUnavailable) as info:
        run(client, settings, sleeps)
    assert "m1, m2" in str(info.value)
    assert info.value.last_error.code == 503


def test_non_transient_error_is_not_retried_or_fallen_back(settings, sleeps):
    client = FakeClient([FakeAPIError(400)])
    with pytest.raises(FakeAPIError):
        run(client, settings, sleeps)
    assert client.calls == ["m1"]


def test_unknown_model_skips_straight_to_next_without_waiting(settings, sleeps):
    client = FakeClient([FakeAPIError(404), "ok"])
    answer = run(client, settings, sleeps)
    assert answer.model == "m2"
    assert client.calls == ["m1", "m2"]
    assert sleeps == []


def test_empty_response_counts_as_a_failed_attempt(settings, sleeps):
    client = FakeClient(["", "ok"])
    answer = run(client, settings, sleeps)
    assert (answer.text, answer.attempts) == ("ok", 2)


def test_timeout_style_errors_are_retried(settings, sleeps):
    class ReadTimeout(Exception):  # name-matched, like httpx's
        pass

    client = FakeClient([ReadTimeout("slow"), "ok"])
    assert run(client, settings, sleeps).attempts == 2


def test_image_is_uploaded_once_across_retries_and_models(settings, sleeps, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"not really a jpeg")
    client = FakeClient([FakeAPIError(503), FakeAPIError(503), "ok"])
    answer = run(client, settings, sleeps, image=image)
    assert answer.model == "m2"
    assert client.uploads == [str(image)]


def test_backoff_doubles_between_retries(sleeps):
    settings = Settings(models=("m1",), retries_per_model=3, base_delay_s=1.0)
    client = FakeClient([FakeAPIError(503), FakeAPIError(503), "ok"])
    run(client, settings, sleeps)
    # delay = base * 2**(attempt-1) plus up to base/2 of jitter
    assert 1.0 <= sleeps[0] < 1.5
    assert 2.0 <= sleeps[1] < 2.5


def test_settings_read_model_list_from_env(monkeypatch):
    monkeypatch.setenv("GEMMA_MODELS", "a, b ,c")
    assert Settings.from_env().models == ("a", "b", "c")


def test_settings_default_models_when_env_unset(monkeypatch):
    monkeypatch.delenv("GEMMA_MODELS", raising=False)
    assert Settings.from_env().models == ("gemma-4-26b-a4b-it", "gemma-4-31b-it")


def test_system_prompt_file_is_present_and_nonempty():
    assert load_system_prompt()
