"""Tests for the retry and fallback policy in app/gemma.py.

A scripted fake client stands in for the Gemini API, so these run offline and
without sleeping. They pin down the behaviour that matters on stage: transient
errors are retried, an unhealthy model is abandoned for the next one, and
errors that retrying can't fix surface immediately.
"""

import pytest

from app.gemma import (
    Answer,
    GemmaUnavailable,
    Settings,
    ask_gemma,
    ensure_structured_data,
    load_system_prompt,
    parse_json_payload,
)


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
        return f"uploaded-{file}"


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
    class ReadTimeout(Exception):
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


def test_parse_json_payload_markdown_and_raw():
    raw_json = '{"safety_score": "SAFE", "headline": "All clear"}'
    assert parse_json_payload(raw_json) == {"safety_score": "SAFE", "headline": "All clear"}

    markdown_json = '```json\n{"safety_score": "DANGER", "headline": "Overdose risk"}\n```'
    assert parse_json_payload(markdown_json) == {"safety_score": "DANGER", "headline": "Overdose risk"}

    assert parse_json_payload("not json") is None


def test_structured_answer_contains_data(settings, sleeps):
    json_text = '{"safety_score": "CAUTION", "headline": "Take with food"}'
    client = FakeClient([json_text])
    answer = run(client, settings, sleeps)
    assert answer.data == {"safety_score": "CAUTION", "headline": "Take with food"}
    assert answer.model == "m1"


def test_multi_image_upload_to_client(settings, sleeps, tmp_path):
    img1 = tmp_path / "bottle1.jpg"
    img2 = tmp_path / "bottle2.png"
    img1.write_bytes(b"image 1")
    img2.write_bytes(b"image 2")
    client = FakeClient(["ok"])
    answer = run(client, settings, sleeps, image=[img1, img2])
    assert answer.attempts == 1
    assert client.uploads == [str(img1), str(img2)]


def test_ensure_structured_data_fallback():
    result = ensure_structured_data(
        "Warning: dangerous duplicate acetaminophen detected!"
    )

    assert result["safety_score"] == "UNVERIFIED"
    assert result["headline"] == "Medication safety could not be verified"
    assert result["daily_schedule"] == {
        "morning": [],
        "afternoon": [],
        "evening": [],
        "bedtime": [],
    }


def test_ensure_structured_data_does_not_infer_safety_from_text():
    result = ensure_structured_data(
        "No warnings were identified in this response."
    )

    assert result["safety_score"] == "UNVERIFIED"


def test_ensure_structured_data_downgrades_unverified_safe_result():
    result = ensure_structured_data(
        "The regimen appears safe.",
        data={
            "safety_score": "SAFE",
            "headline": "All clear",
            "medications": [],
            "conflicts_and_warnings": [],
        },
    )

    assert result["safety_score"] == "UNVERIFIED"
    assert result["headline"] == "Safety has not been independently verified"


def test_ensure_structured_data_preserves_caution():
    result = ensure_structured_data(
        "A potential interaction was reported.",
        data={
            "safety_score": "CAUTION",
            "headline": "Potential interaction",
        },
    )

    assert result["safety_score"] == "CAUTION"


def test_ensure_structured_data_preserves_danger():
    result = ensure_structured_data(
        "A high-risk interaction was reported.",
        data={
            "safety_score": "DANGER",
            "headline": "Potential high-risk interaction",
        },
    )

    assert result["safety_score"] == "DANGER"