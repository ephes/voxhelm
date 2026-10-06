from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import ImproperlyConfigured
from django.test import Client

from config.settings import DEV_ONLY_SECRET_KEY, resolve_secret_key

REPO_ROOT = Path(__file__).resolve().parent.parent
PRINT_SETTINGS = (
    "import config.settings as s; "
    "print(s.SECRET_KEY, s.SESSION_COOKIE_SECURE, s.CSRF_COOKIE_SECURE, sep='|')"
)


def import_settings(**overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DJANGO_SECRET_KEY", "DJANGO_DEBUG", "VOXHELM_SECURE_COOKIES"}
    }
    env.update(overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", PRINT_SETTINGS],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_settings_import_fails_without_secret_key_when_not_debug():
    for overrides in ({}, {"DJANGO_SECRET_KEY": "  "}, {"DJANGO_DEBUG": "false"}):
        completed = import_settings(**overrides)

        assert completed.returncode != 0, overrides
        assert "ImproperlyConfigured" in completed.stderr
        assert "DJANGO_SECRET_KEY must be set" in completed.stderr


def test_settings_import_uses_dev_key_and_plain_cookies_in_debug():
    completed = import_settings(DJANGO_DEBUG="true")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"{DEV_ONLY_SECRET_KEY}|False|False"


def test_settings_cookies_are_secure_by_default_outside_debug():
    completed = import_settings(DJANGO_SECRET_KEY="prod-key")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "prod-key|True|True"


def test_secure_cookie_override_allows_plain_http_outside_debug():
    completed = import_settings(DJANGO_SECRET_KEY="prod-key", VOXHELM_SECURE_COOKIES="false")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "prod-key|False|False"


def test_resolve_secret_key():
    assert resolve_secret_key(" real-key ", debug=False) == "real-key"
    assert resolve_secret_key(None, debug=True) == DEV_ONLY_SECRET_KEY
    with pytest.raises(ImproperlyConfigured):
        resolve_secret_key("", debug=False)


@pytest.mark.django_db
def test_root_page_denies_framing(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.django_db
def test_session_and_csrf_cookies_are_secure_with_debug_off():
    get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client = Client()

    response = client.get("/", HTTP_X_FORWARDED_PROTO="https")
    csrf_cookie = response.cookies["csrftoken"]
    assert csrf_cookie["secure"] is True

    response = client.post(
        "/",
        data={
            "username": "jochen",
            "password": "secret",
        },
        HTTP_X_FORWARDED_PROTO="https",
    )

    assert response.status_code == 302
    assert response.cookies["sessionid"]["secure"] is True
    assert response.cookies["sessionid"]["httponly"] is True


def post_login(client: Client, username: str, password: str, address: str = "10.0.0.5"):
    return client.post("/", data={"username": username, "password": password}, REMOTE_ADDR=address)


@pytest.mark.django_db
def test_login_is_refused_after_too_many_failures_until_window_expires(
    client, settings, monkeypatch
):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 3
    settings.VOXHELM_LOGIN_LOCKOUT_SECONDS = 600
    get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)

    for _ in range(3):
        response = post_login(client, "jochen", "wrong")
        assert response.status_code == 200
        assert "Invalid username or password." in response.content.decode()

    response = post_login(client, "jochen", "wrong")
    assert response.status_code == 429
    assert "Too many failed sign-in attempts" in response.content.decode()

    # The correct password is refused too while locked, without logging in.
    response = post_login(client, "jochen", "secret")
    assert response.status_code == 429
    assert "_auth_user_id" not in client.session

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 601)

    response = post_login(client, "jochen", "secret")
    assert response.status_code == 302
    assert client.session["_auth_user_id"]


@pytest.mark.django_db
def test_login_throttle_counts_per_username_across_addresses(client, settings):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 2
    get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)

    post_login(client, "jochen", "wrong", address="10.0.0.1")
    post_login(client, "Jochen ", "wrong", address="10.0.0.2")

    response = post_login(client, "jochen", "secret", address="10.0.0.3")
    assert response.status_code == 429


@pytest.mark.django_db
def test_login_throttle_counts_per_address_across_usernames(client, settings):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 2
    get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)

    post_login(client, "alice", "wrong")
    post_login(client, "bob", "wrong")

    assert post_login(client, "jochen", "secret").status_code == 429
    assert post_login(client, "jochen", "secret", address="10.0.0.9").status_code == 302


@pytest.mark.django_db
def test_successful_login_resets_username_failures(client, settings):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 2
    get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)

    post_login(client, "jochen", "wrong", address="10.0.0.1")
    assert post_login(client, "jochen", "secret", address="10.0.0.2").status_code == 302
    client.logout()

    post_login(client, "jochen", "wrong", address="10.0.0.3")
    assert post_login(client, "jochen", "secret", address="10.0.0.4").status_code == 302
