"""Small API-key primitives shared by the service and tests."""

from __future__ import annotations

import hashlib
import secrets


API_KEY_PREFIX = "old_"


def generate_api_key() -> str:
    """Return a participant key; only its digest is stored by the service."""

    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def api_key_prefix(api_key: str) -> str:
    """Return a short non-secret identifier useful for operator support."""

    value = api_key.strip()
    return value[:12]


def bearer_api_key(authorization: str | None) -> str:
    if not authorization:
        raise ValueError("missing API key; use Authorization: Bearer <key>")
    scheme, separator, value = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not value.strip():
        raise ValueError("use Authorization: Bearer <key>")
    return value.strip()
