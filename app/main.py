import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import (
    SESSION_DURATION_DAYS,
    create_session,
    get_user_from_session,
    hash_password,
    revoke_session,
    verify_password,
)

from app.database import get_db
from app.do_ai import (
    DigitalOceanAIError,
    ask_digitalocean_ai,
    dispatch_to_hermes,
    is_do_configured,
)
from app.gemma import (
    ConfigError,
    GemmaUnavailable,
    ask_gemma,
    ensure_structured_data,
    load_system_prompt,
    parse_json_payload,
)
from app.models import User
from app.schemas import LoginRequest, RegisterRequest


log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_QUESTION_CHARS = 2000

app = FastAPI(title="RxGuard - Visual Medication Safety Copilot & Hermes Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    token = request.cookies.get("rxguard_session")

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    user = get_user_from_session(db, token)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session.",
        )

    return user

@app.get("/api/auth/me")
def get_me(
    current_user: User = Depends(get_current_user),
) -> dict:
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_active": current_user.is_active,
    }

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/api/auth/register", status_code=201)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> dict:
    email = str(payload.email).strip().lower()

    existing_user = db.scalar(
        select(User).where(User.email == email)
    )

    if existing_user is not None:
        raise HTTPException(409, "An account with this email already exists.")

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            409,
            "An account with this email already exists.",
        )

    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
    }

@app.post("/api/auth/login")
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> dict:
    email = str(payload.email).strip().lower()

    user = db.scalar(
        select(User).where(User.email == email)
    )

    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(401, "Invalid email or password.")

    token, session = create_session(db, user)

    secure_cookie = os.getenv("RXGUARD_COOKIE_SECURE", "false").lower() == "true"

    response.set_cookie(
        key="rxguard_session",
        value=token,
        max_age=SESSION_DURATION_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )

    return {
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "email": user.email,
            "is_active": user.is_active,
        },
    }

@app.post("/api/auth/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> dict:
    token = request.cookies.get("rxguard_session")

    if token:
        revoke_session(db, token)

    secure_cookie = (
        os.getenv("RXGUARD_COOKIE_SECURE", "false").lower() == "true"
    )

    response.delete_cookie(
        key="rxguard_session",
        path="/",
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
    )

    return {"message": "Logout successful."}

@app.get("/api/status/integrations")
def integrations_status(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return status of AI providers and Hermes Agent integration."""
    return {
        "gemini_gemma": True,
        "digitalocean_ai": is_do_configured(),
        "hermes_agent": True,
        "do_inference_endpoint": "https://inference.do-ai.run/v1",
        "droplet_1click_ready": True,
    }


def generate_ics(schedule: dict) -> str:
    """Generate an iCalendar (.ics) string for recurring daily medication reminders."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    times = {
        "morning": ("080000", "081500", "Morning Medications (8:00 AM)"),
        "afternoon": ("130000", "131500", "Afternoon Medications (1:00 PM)"),
        "evening": ("180000", "181500", "Evening Medications (6:00 PM)"),
        "bedtime": ("220000", "221500", "Bedtime Medications (10:00 PM)"),
    }

    events = []
    for slot, (t_start, t_end, title) in times.items():
        meds = schedule.get(slot, [])
        if not meds:
            continue
        med_list = "\\n".join(f"- {m}" for m in meds)
        event = f"""BEGIN:VEVENT
UID:rxguard-{slot}-{now_str}@rxguard.app
DTSTAMP:{now_str}
DTSTART:{today_str}T{t_start}
DTEND:{today_str}T{t_end}
RRULE:FREQ=DAILY
SUMMARY:{title}
DESCRIPTION:{med_list}
BEGIN:VALARM
TRIGGER:-PT10M
ACTION:DISPLAY
DESCRIPTION:Time for your {slot} medications
END:VALARM
END:VEVENT"""
        events.append(event)

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//RxGuard//Medication Safety Copilot//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
{"".join(events)}
END:VCALENDAR""".replace("\n\n", "\n")
    return ics_content


@app.post("/api/calendar/export")
def export_calendar(
    schedule: dict = Body(...),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Export medication daily schedule as an iCalendar (.ics) download."""
    ics_text = generate_ics(schedule)
    return Response(
        content=ics_text,
        media_type="text/calendar",
        headers={"Content-Disposition": "attachment; filename=rxguard-schedule.ics"},
    )


@app.post("/api/hermes/dispatch")
def dispatch_hermes_alert(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Dispatch high-risk medication alert to 24/7 Hermes Caregiver Agent on DigitalOcean."""
    return dispatch_to_hermes(payload)


@app.post("/api/do/ask")
def ask_do(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Run medication safety reasoning on DigitalOcean Serverless Inference."""
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(400, "Please provide a question.")
    try:
        sys_prompt = load_system_prompt()
        res = ask_digitalocean_ai(question, sys_prompt)
        return res
    except DigitalOceanAIError as exc:
        raise HTTPException(502, str(exc))

# A plain `def` (not async) on purpose: FastAPI runs it in a worker thread, and
# ask_gemma blocks on network calls and sleeps between retries.
@app.post("/api/ask")
def ask(
    question: str = Form(...),
    engine: Optional[str] = Form("gemini"),
    image: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    current_user: User = Depends(get_current_user),
) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question is too long (max {MAX_QUESTION_CHARS} characters).")

    # Collect all uploaded images (single or multiple)
    file_list = []
    if image is not None and bool(image.filename):
        file_list.append(image)
    if images:
        for f in images:
            if f is not None and bool(f.filename):
                file_list.append(f)

    # Route to DigitalOcean Serverless Inference if explicitly requested and text-only
        
    if engine == "digitalocean" and not file_list and is_do_configured():
        try:
            do_res = ask_digitalocean_ai(question, load_system_prompt())

            do_text = do_res.get("text", "")
            do_data = do_res.get("data")

            if not isinstance(do_data, dict):
                do_data = parse_json_payload(do_text)

            safe_data = ensure_structured_data(
                do_text,
                do_data if isinstance(do_data, dict) else None,
            )

            return {
                "answer": do_text,
                "model": do_res["model"],
                "attempts": 1,
                "data": safe_data,
            }
        except Exception as exc:
            log.warning("DigitalOcean AI failed, falling back to Gemma: %s", exc)

    try:
        if not file_list:
            answer = ask_gemma(question)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                saved_paths = []
                for idx, upload in enumerate(file_list):
                    suffix = ALLOWED_IMAGE_TYPES.get(upload.content_type or "")
                    if suffix is None:
                        raise HTTPException(415, "Please upload a JPEG, PNG or WebP image.")
                    data = upload.file.read(MAX_IMAGE_BYTES + 1)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise HTTPException(413, f"Image is too large (max {MAX_IMAGE_BYTES // 1024 // 1024} MB).")
                    path = Path(tmp) / f"upload_{idx}{suffix}"
                    path.write_bytes(data)
                    saved_paths.append(path)

                target = saved_paths[0] if len(saved_paths) == 1 else saved_paths
                answer = ask_gemma(question, target)
    except HTTPException:
        raise
    except GemmaUnavailable:
        raise HTTPException(503, "Gemma is busy right now. Please try again in a minute.")
    except ConfigError:
        log.exception("server misconfigured")
        raise HTTPException(500, "The server isn't configured correctly.")
    except Exception:
        log.exception("Gemma request failed")
        raise HTTPException(502, "The model rejected this request.")

    res = {
    "answer": answer.text,
    "model": answer.model,
    "attempts": answer.attempts,
    }

    structured_data = answer.data

    if not isinstance(structured_data, dict):
        structured_data = parse_json_payload(answer.text)

    res["data"] = ensure_structured_data(
        answer.text,
        structured_data if isinstance(structured_data, dict) else None,
    )

    return res
