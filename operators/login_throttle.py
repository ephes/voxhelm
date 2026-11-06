"""Bound failed and in-flight operator logins within the single server process.

Failures use fixed windows starting at the first completed failure. Pending
password checks reserve capacity until settlement, even across window expiry.
REMOTE_ADDR is trusted; forwarded addresses are deliberately ignored. Cache
state and a process lock match the supported single-uvicorn-process scope.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from dataclasses import dataclass, field

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest

KEY_PREFIX = "voxhelm:operator-login-admission"
_LOCK = threading.Lock()


@dataclass
class _Bucket:
    failures: int = 0
    expires_at: float = 0
    pending: set[str] = field(default_factory=set)


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
    if bucket.pending:
        # A slow authentication cannot escape its reservation by outliving a window.
        cache.set(key, bucket, timeout=None)
    elif bucket.failures:
        cache.set(key, bucket, timeout=max(1, math.ceil(bucket.expires_at - now)))
    else:
        cache.delete(key)


def reserve_attempt(request: HttpRequest, username: str) -> LoginReservation | None:
    """Reserve both budgets before calling the password authenticator."""
    with _LOCK:
        now = time.time()
        keys = _keys(request, username)
        buckets = [_load(key, now) for key in keys]
        if any(b.failures + len(b.pending) >= settings.VOXHELM_LOGIN_MAX_FAILURES for b in buckets):
            return None
        token = uuid.uuid4().hex
        for key, bucket in zip(keys, buckets, strict=True):
            bucket.pending.add(token)
            _save(key, bucket, now)
        return LoginReservation(keys=keys, token=token)


def settle_attempt(reservation: LoginReservation, *, success: bool) -> None:
    """Release only this reservation; success resets username failures only."""
    with _LOCK:
        now = time.time()
        for index, key in enumerate(reservation.keys):
            bucket = _load(key, now)
            if reservation.token not in bucket.pending:
                # An evicted/replaced cache entry must not be changed by an old attempt.
                continue
            bucket.pending.remove(reservation.token)
            if success:
                if index == 1:
                    bucket.failures = 0
                    bucket.expires_at = 0
            else:
                if not bucket.failures:
                    bucket.expires_at = now + settings.VOXHELM_LOGIN_LOCKOUT_SECONDS
                bucket.failures += 1
            _save(key, bucket, now)
