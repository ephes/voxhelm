"""Bound failed and in-flight operator logins within the single server process.

Failures use fixed windows starting at the first completed failure. Pending
password checks reserve capacity until settlement, even across window expiry
and cache eviction. REMOTE_ADDR is trusted; forwarded addresses are deliberately
ignored. A process lock protects admission within the single uvicorn process.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from dataclasses import dataclass

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest

KEY_PREFIX = "voxhelm:operator-login-admission"
_LOCK = threading.Lock()
_PENDING: dict[str, set[str]] = {}


@dataclass
class _Bucket:
    failures: int = 0
    expires_at: float = 0


@dataclass(frozen=True)
class LoginReservation:
    keys: tuple[str, str]
    token: str


def _keys(request: HttpRequest, username: str) -> tuple[str, str]:
    address = request.META.get("REMOTE_ADDR", "") or "unknown"
    normalized = username.strip().casefold()
    username_digest = hashlib.sha256(normalized.encode()).hexdigest()
    address_digest = hashlib.sha256(address.encode()).hexdigest()
    return f"{KEY_PREFIX}:ip:{address_digest}", f"{KEY_PREFIX}:user:{username_digest}"


def _load(key: str, now: float) -> _Bucket:
    bucket = cache.get(key, _Bucket())
    if bucket.expires_at <= now:
        bucket.failures = 0
        bucket.expires_at = 0
    return bucket


def _save(key: str, bucket: _Bucket, now: float) -> None:
    if bucket.failures:
        cache.set(key, bucket, timeout=max(1, math.ceil(bucket.expires_at - now)))
    else:
        cache.delete(key)


def reserve_attempt(request: HttpRequest, username: str) -> LoginReservation | None:
    """Reserve both budgets before calling the password authenticator."""
    with _LOCK:
        now = time.time()
        keys = _keys(request, username)
        if any(
            _load(key, now).failures + len(_PENDING.get(key, ()))
            >= settings.VOXHELM_LOGIN_MAX_FAILURES
            for key in keys
        ):
            return None
        token = uuid.uuid4().hex
        for key in keys:
            _PENDING.setdefault(key, set()).add(token)
        return LoginReservation(keys=keys, token=token)


def settle_attempt(reservation: LoginReservation, *, success: bool) -> None:
    """Release only this reservation; success resets username failures only."""
    with _LOCK:
        if any(reservation.token not in _PENDING.get(key, ()) for key in reservation.keys):
            return
        for key in reservation.keys:
            pending = _PENDING[key]
            pending.remove(reservation.token)
            if not pending:
                del _PENDING[key]

        now = time.time()
        for index, key in enumerate(reservation.keys):
            bucket = _load(key, now)
            if success:
                if index == 1:
                    bucket.failures = 0
                    bucket.expires_at = 0
            else:
                if not bucket.failures:
                    bucket.expires_at = now + settings.VOXHELM_LOGIN_LOCKOUT_SECONDS
                bucket.failures += 1
            _save(key, bucket, now)
