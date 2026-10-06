"""Cache-based throttle for failed operator UI logins.

Failures are counted per client address and per username in fixed windows of
``VOXHELM_LOGIN_LOCKOUT_SECONDS`` that start at the first failure. Once either
counter reaches ``VOXHELM_LOGIN_MAX_FAILURES``, further attempts for that address
or username are refused without checking the password until the window expires.

The client address is ``REMOTE_ADDR``. Behind the Traefik ingress that is the
proxy's address for every browser, so the address bucket is effectively shared;
``X-Forwarded-For`` is deliberately not trusted because the backend port is also
reachable directly on the LAN. Counters live in the default Django cache (the
per-process local-memory cache unless ``CACHES`` is configured), which matches
the single uvicorn process Voxhelm runs.
"""

from __future__ import annotations

import hashlib

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest

KEY_PREFIX = "voxhelm:operator-login-failures"


def _keys(request: HttpRequest, username: str) -> list[str]:
    address = request.META.get("REMOTE_ADDR", "") or "unknown"
    normalized = username.strip().casefold()
    username_digest = hashlib.sha256(normalized.encode()).hexdigest()
    address_digest = hashlib.sha256(address.encode()).hexdigest()
    return [f"{KEY_PREFIX}:ip:{address_digest}", f"{KEY_PREFIX}:user:{username_digest}"]


def is_locked_out(request: HttpRequest, username: str) -> bool:
    limit = settings.VOXHELM_LOGIN_MAX_FAILURES
    return any(int(cache.get(key, 0)) >= limit for key in _keys(request, username))


def record_failure(request: HttpRequest, username: str) -> None:
    timeout = settings.VOXHELM_LOGIN_LOCKOUT_SECONDS
    for key in _keys(request, username):
        # add() starts the window only if no counter exists; incr() keeps its expiry.
        cache.add(key, 0, timeout)
        try:
            cache.incr(key)
        except ValueError:
            # The counter expired between add() and incr(); start a new window.
            cache.set(key, 1, timeout)


def clear_failures(request: HttpRequest, username: str) -> None:
    """Reset the username counter after a successful login.

    The address counter is left to expire so a valid account cannot be used to
    reset the budget for guessing other accounts from the same address.
    """
    cache.delete(_keys(request, username)[1])
