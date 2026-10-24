"""HTTP-layer tests. ask_gemma is stubbed, so these check validation and error
mapping only; the model call itself is covered in test_gemma.py."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.gemma import Answer, GemmaUnavailable

client = TestClient(main.app)


def post(question="what is this?", files=None):
    return client.post("/api/ask", data={"question": question}, files=files)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_page_is_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Ask Gemma" in res.text


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
    assert res.json() == {"answer": "an answer", "model": "m1", "attempts": 1}
    assert seen["args"] == ("hello there", None)


def test_image_is_written_with_the_right_extension(monkeypatch):
    seen = {}

    def fake(prompt, image_path=None, **_):
        # The temp file only exists during the call, so inspect it in here.
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
