from pwdlib import PasswordHash
import hashlib
import secrets
import hmac
import os

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User, UserSession


SESSION_DURATION_DAYS = 7

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)

def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def generate_csrf_token(session_token: str) -> str:
    secret = os.getenv("RXGUARD_CSRF_SECRET")

    if not secret:
        raise RuntimeError("RXGUARD_CSRF_SECRET is not configured.")

    return hmac.new(
        secret.encode("utf-8"),
        session_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_csrf_token(session_token: str, csrf_token: str) -> bool:
    if not csrf_token:
        return False

    try:
        expected_token = generate_csrf_token(session_token)
    except RuntimeError:
        return False

    return hmac.compare_digest(expected_token, csrf_token)


def create_session(db: Session, user: User) -> tuple[str, UserSession]:
    token = generate_session_token()

    session = UserSession(
        user_id=user.id,
        token_hash=hash_session_token(token),
        expires_at=datetime.now(timezone.utc)
        + timedelta(days=SESSION_DURATION_DAYS),
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return token, session


def get_user_from_session(db: Session, token: str) -> User | None:
    token_hash = hash_session_token(token)

    session = db.scalar(
        select(UserSession).where(UserSession.token_hash == token_hash)
    )

    if session is None:
        return None

    now = datetime.now(timezone.utc)
    expires_at = session.expires_at

    # SQLite may return naive datetimes in tests.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if session.revoked_at is not None or expires_at <= now:
        return None

    user = db.get(User, session.user_id)

    if user is None or not user.is_active:
        return None

    return user


def revoke_session(db: Session, token: str) -> bool:
    token_hash = hash_session_token(token)

    session = db.scalar(
        select(UserSession).where(UserSession.token_hash == token_hash)
    )

    if session is None or session.revoked_at is not None:
        return False

    session.revoked_at = datetime.now(timezone.utc)
    db.commit()

    return True