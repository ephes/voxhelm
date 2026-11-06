from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from django.test import Client, RequestFactory

from operators import login_throttle


def request(address="192.0.2.1"):
    return RequestFactory().post("/", REMOTE_ADDR=address)


@pytest.mark.parametrize("same_username", [True, False])
def test_login_admits_only_one_held_authenticator(settings, monkeypatch, same_username):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 1
    entered = Event()
    release = Event()
    calls = []

    def authenticate(*args, **kwargs):
        calls.append(kwargs["username"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return None

    monkeypatch.setattr("operators.views.authenticate", authenticate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            Client().post,
            "/",
            {"username": "synthetic", "password": "wrong"},
            REMOTE_ADDR="192.0.2.1",
        )
        try:
            assert entered.wait(5)
            second = Client().post(
                "/",
                {"username": "synthetic" if same_username else "other", "password": "wrong"},
                REMOTE_ADDR="192.0.2.2" if same_username else "192.0.2.1",
            )
            assert second.status_code == 429
            assert calls == ["synthetic"]
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200


def test_success_keeps_other_pending_guesses_and_address_failures(settings):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 2
    first = login_throttle.reserve_attempt(request(), "synthetic")
    second = login_throttle.reserve_attempt(request("192.0.2.2"), "synthetic")
    assert first and second
    login_throttle.settle_attempt(first, success=True)
    third = login_throttle.reserve_attempt(request("192.0.2.3"), "synthetic")
    assert third
    assert login_throttle.reserve_attempt(request("192.0.2.4"), "synthetic") is None
    login_throttle.settle_attempt(second, success=False)
    login_throttle.settle_attempt(third, success=False)
    assert login_throttle.reserve_attempt(request("192.0.2.5"), "synthetic") is None
    # A valid account cannot erase the shared address budget for other usernames.
    valid = login_throttle.reserve_attempt(request("192.0.2.2"), "valid")
    assert valid
    login_throttle.settle_attempt(valid, success=True)
    bad = login_throttle.reserve_attempt(request("192.0.2.2"), "other")
    assert bad
    login_throttle.settle_attempt(bad, success=False)
    assert login_throttle.reserve_attempt(request("192.0.2.2"), "fresh") is None


def test_window_expiry_does_not_drop_pending_or_let_old_settlement_clear_new_state(
    settings, monkeypatch
):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 1
    settings.VOXHELM_LOGIN_LOCKOUT_SECONDS = 10
    clock = [100.0]
    monkeypatch.setattr(login_throttle.time, "time", lambda: clock[0])
    held = login_throttle.reserve_attempt(request(), "synthetic")
    assert held
    clock[0] += 20
    assert login_throttle.reserve_attempt(request(), "synthetic") is None
    login_throttle.settle_attempt(held, success=False)
    clock[0] += 9
    assert login_throttle.reserve_attempt(request(), "synthetic") is None
    clock[0] += 2
    fresh = login_throttle.reserve_attempt(request(), "synthetic")
    assert fresh
    login_throttle.settle_attempt(held, success=True)
    assert login_throttle.reserve_attempt(request(), "synthetic") is None
    login_throttle.settle_attempt(fresh, success=True)
    assert login_throttle.reserve_attempt(request(), "synthetic")


def test_authenticator_exception_settles_reservation(settings, monkeypatch):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 1

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic authentication failure")

    monkeypatch.setattr("operators.views.authenticate", broken)
    with pytest.raises(RuntimeError):
        Client().post("/", {"username": "synthetic", "password": "wrong"})
    assert Client().post("/", {"username": "synthetic", "password": "wrong"}).status_code == 429


@pytest.mark.parametrize("active,staff", [(False, True), (True, False)])
def test_denied_accounts_consume_budget_even_with_correct_password(
    settings, monkeypatch, active, staff
):
    settings.VOXHELM_LOGIN_MAX_FAILURES = 1
    calls = []

    def authenticate(*args, **kwargs):
        calls.append(kwargs["password"])
        return SimpleNamespace(is_active=active, is_staff=staff)

    monkeypatch.setattr("operators.views.authenticate", authenticate)
    client = Client()
    response = client.post("/", {"username": "synthetic", "password": "correct"})
    assert response.status_code == 200
    assert b"inactive" in response.content or b"not allowed" in response.content
    assert client.post("/", {"username": "synthetic", "password": "correct"}).status_code == 429
    assert calls == ["correct"]
