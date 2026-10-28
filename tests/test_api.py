"""HTTP-layer tests. ask_gemma is stubbed, so these check validation and error
mapping only; the model call itself is covered in test_gemma.py."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.gemma import Answer, GemmaUnavailable
from app.models import User

client = TestClient(main.app)


csrf_token = None


@pytest.fixture(autouse=True)
def authenticated_test_user(monkeypatch):
    global csrf_token

    monkeypatch.setenv("RXGUARD_CSRF_SECRET", "test-only-csrf-secret")
    client.cookies.clear()

    test_user = User(
        id=1,
        email="tester@example.com",
        password_hash="test-only",
    )

    main.app.dependency_overrides[main.get_current_user] = (
        lambda: test_user
    )

    client.cookies.set(
        "rxguard_session",
        "test-session-token",
        domain="testserver.local",
        path="/",
    )

    csrf_response = client.get("/api/auth/csrf")
    assert csrf_response.status_code == 200
    csrf_token = csrf_response.json()["csrf_token"]

    yield

    main.app.dependency_overrides.pop(main.get_current_user, None)
    client.cookies.clear()


def post(question="what is this?", files=None, data=None):
    payload = {"question": question}
    if data:
        payload.update(data)

    return client.post(
        "/api/ask",
        data=payload,
        files=files,
        headers={"X-CSRF-Token": csrf_token},
    )


def test_health():
    assert client.get("/health").json() == {"status": "ok"}



def test_index_page_is_served():
    res = client.get("/")

    assert res.status_code == 200
    assert "RxGuard" in res.text
    assert "Make more informed decisions about your medication." in res.text
    assert "potential interactions" in res.text


def test_blank_question_is_rejected():
    assert post("   ").status_code == 400


def test_non_image_upload_is_rejected():
    res = post(files={"image": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_oversized_image_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 10)
    res = post(files={"image": ("big.png", b"x" * 11, "image/png")})
    assert res.status_code == 413


def test_text_only_question(monkeypatch):
    seen = {}

    def fake(prompt, image_path=None, **_):
        seen["args"] = (prompt, image_path)
        return Answer(text="an answer", model="m1", attempts=1)

    monkeypatch.setattr(main, "ask_gemma", fake)
    res = post("hello there")
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "an answer"
    assert body["model"] == "m1"
    assert body["attempts"] == 1
    assert body["data"]["safety_score"] == "UNVERIFIED"
    assert seen["args"] == ("hello there", None)


def test_image_is_written_with_the_right_extension(monkeypatch):
    seen = {}

    def fake(prompt, image_path=None, **_):
        seen["suffix"] = Path(image_path).suffix
        seen["bytes"] = Path(image_path).read_bytes()
        return Answer(text="ok", model="m1", attempts=1)

    monkeypatch.setattr(main, "ask_gemma", fake)
    res = post(files={"image": ("photo.png", b"png-bytes", "image/png")})
    assert res.status_code == 200
    assert seen == {"suffix": ".png", "bytes": b"png-bytes"}


def test_unavailable_maps_to_503(monkeypatch):
    def fake(*_, **__):
        raise GemmaUnavailable("nope")

    monkeypatch.setattr(main, "ask_gemma", fake)
    res = post()
    assert res.status_code == 503
    assert "busy" in res.json()["detail"]


def test_unexpected_error_maps_to_502_without_leaking_details(monkeypatch):
    def fake(*_, **__):
        raise ValueError("secret internal detail")

    monkeypatch.setattr(main, "ask_gemma", fake)
    res = post()
    assert res.status_code == 502
    assert "secret" not in res.text


def test_calendar_export_endpoint():
    schedule = {
        "morning": ["Lisinopril 10mg"],
        "afternoon": [],
        "evening": ["Metformin 500mg"],
        "bedtime": ["Atorvastatin 20mg"],
    }
    res = client.post(
        "/api/calendar/export",
        json=schedule,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert res.status_code == 200
    assert "text/calendar" in res.headers["content-type"]
    text = res.text
    assert "BEGIN:VCALENDAR" in text
    assert "Morning Medications" in text
    assert "Lisinopril 10mg" in text
    assert "Evening Medications" in text
    assert "RRULE:FREQ=DAILY" in text
    assert "END:VCALENDAR" in text


def test_structured_data_returned_when_available(monkeypatch):
    structured = {
        "safety_score": "DANGER",
        "headline": "Duplicate acetaminophen toxicity",
        "audio_summary": "Do not take both medications.",
    }

    def fake(prompt, image_path=None, **_):
        return Answer(text='{"safety_score": "DANGER"}', model="m1", attempts=1, data=structured)

    monkeypatch.setattr(main, "ask_gemma", fake)
    res = post("can I take these?")
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["data"] == structured
    assert json_data["model"] == "m1"


def test_integrations_status_endpoint():
    res = client.get("/api/status/integrations")
    assert res.status_code == 200
    data = res.json()
    assert data["hermes_agent"] is True
    assert "https://inference.do-ai.run/v1" in data["do_inference_endpoint"]
    assert data["droplet_1click_ready"] is True


def test_hermes_dispatch_endpoint():
    alert_payload = {
        "headline": "Severe duplicate acetaminophen alert",
        "safety_score": "DANGER",
        "medications": ["Percocet", "Tylenol"],
    }
    res = client.post(
        "/api/hermes/dispatch",
        json=alert_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "dispatched"
    assert "Hermes" in data["message"]


def test_api_downgrades_ai_safe_to_unverified(monkeypatch):
    structured = {
        "safety_score": "SAFE",
        "headline": "No risks found",
        "audio_summary": "The medicines appear safe.",
        "medications": [],
        "conflicts_and_warnings": [],
    }

    def fake(prompt, image_path=None, **_):
        return Answer(
            text='{"safety_score": "SAFE"}',
            model="m1",
            attempts=1,
            data=structured,
        )

    monkeypatch.setattr(main, "ask_gemma", fake)

    res = post("Can I take these medicines together?")

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["safety_score"] == "UNVERIFIED"
    assert data["headline"] == "Safety has not been independently verified"


def test_api_returns_unverified_when_structured_data_is_missing(monkeypatch):
    def fake(prompt, image_path=None, **_):
        return Answer(
            text="I could not assess these medicines.",
            model="m1",
            attempts=1,
        )

    monkeypatch.setattr(main, "ask_gemma", fake)

    res = post("Check these medicines")

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["safety_score"] == "UNVERIFIED"
    assert data["conflicts_and_warnings"]

def test_digitalocean_api_downgrades_safe_to_unverified(monkeypatch):
    monkeypatch.setattr(main, "is_do_configured", lambda: True)

    def fake_do(prompt, system_prompt=None):
        return {
            "text": '{"safety_score": "SAFE", "headline": "No risks found"}',
            "model": "digitalocean/test-model",
            "data": {
                "safety_score": "SAFE",
                "headline": "No risks found",
            },
        }

    monkeypatch.setattr(main, "ask_digitalocean_ai", fake_do)

    res = post(
        "Check these medicines",
        data={"engine": "digitalocean"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "digitalocean/test-model"
    assert body["data"]["safety_score"] == "UNVERIFIED"
    assert body["data"]["headline"] == (
        "Safety has not been independently verified"
    )