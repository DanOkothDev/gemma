import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.auth import verify_password
from app.database import Base, get_db
from app.models import User


@pytest.fixture
def auth_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = override_get_db

    with TestClient(main.app) as client:
        client.testing_session_local = TestingSessionLocal
        yield client

    main.app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

def test_register_creates_user_with_hashed_password(auth_client):
    password = "SecureTestPassword123!"

    response = auth_client.post(
        "/api/auth/register",
        json={
            "email": "Dan@Example.com",
            "password": password,
        },
    )

    assert response.status_code == 201
    assert response.json()["email"] == "dan@example.com"
    assert response.json()["is_active"] is True

    with auth_client.testing_session_local() as db:
        user = db.query(User).filter_by(
            email="dan@example.com"
        ).first()

        assert user is not None
        assert user.password_hash != password
        assert verify_password(password, user.password_hash)
        assert not verify_password(
            "WrongPassword123!",
            user.password_hash,
        )

def test_register_rejects_duplicate_email(auth_client):
    payload = {
        "email": "dan@example.com",
        "password": "SecureTestPassword123!",
    }

    first_response = auth_client.post(
        "/api/auth/register",
        json=payload,
    )
    second_response = auth_client.post(
        "/api/auth/register",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409

def test_login_sets_session_cookie(auth_client):
    auth_client.post(
        "/api/auth/register",
        json={
            "email": "dan@example.com",
            "password": "SecureTestPassword123!",
        },
    )

    response = auth_client.post(
        "/api/auth/login",
        json={
            "email": "dan@example.com",
            "password": "SecureTestPassword123!",
        },
    )

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "dan@example.com"

    cookie = response.cookies.get("rxguard_session")
    assert cookie is not None
    assert len(cookie) > 20
    assert "httponly" in response.headers["set-cookie"].lower()


def test_login_rejects_wrong_password(auth_client):
    auth_client.post(
        "/api/auth/register",
        json={
            "email": "dan@example.com",
            "password": "SecureTestPassword123!",
        },
    )

    response = auth_client.post(
        "/api/auth/login",
        json={
            "email": "dan@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401


def test_authenticated_user_can_access_me(auth_client):
    auth_client.post(
        "/api/auth/register",
        json={
            "email": "dan@example.com",
            "password": "SecureTestPassword123!",
        },
    )

    login_response = auth_client.post(
        "/api/auth/login",
        json={
            "email": "dan@example.com",
            "password": "SecureTestPassword123!",
        },
    )

    assert login_response.status_code == 200

    response = auth_client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == "dan@example.com"
    assert response.json()["is_active"] is True

def test_unauthenticated_user_cannot_access_me(auth_client):
    response = auth_client.get("/api/auth/me")

    assert response.status_code == 401

def test_logout_revokes_session(auth_client):
    credentials = {
        "email": "dan@example.com",
        "password": "SecureTestPassword123!",
    }

    auth_client.post(
        "/api/auth/register",
        json=credentials,
    )

    login_response = auth_client.post(
        "/api/auth/login",
        json=credentials,
    )
    assert login_response.status_code == 200

    me_response = auth_client.get("/api/auth/me")
    assert me_response.status_code == 200

    logout_response = auth_client.post("/api/auth/logout")
    assert logout_response.status_code == 200

    me_after_logout = auth_client.get("/api/auth/me")
    assert me_after_logout.status_code == 401

def test_unauthenticated_user_cannot_access_medication_analysis(auth_client):
    response = auth_client.post(
        "/api/ask",
        json={"question": "Can I take these medicines together?"},
    )

    assert response.status_code == 401