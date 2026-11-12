"""A small login for the operator dashboard.

The dashboard shows grading decisions, so it should not be open to anyone on the
network. This keeps it simple and safe: passwords are never stored in the clear,
only as a salted PBKDF2-HMAC-SHA256 hash, and signing in returns a short-lived
token signed with HMAC so it cannot be forged or reused after it expires.

The signing secret comes from the FG_SECRET environment variable in production.
A default is used only for local demos.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
import base64
import json

SECRET = os.environ.get("FG_SECRET", "freshgrade-demo-secret").encode()
_ITER = 200_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITER)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$")
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), _ITER)
    return hmac.compare_digest(dk.hex(), dk_hex)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(user: str, ttl: int = 3600) -> str:
    body = _b64(json.dumps({"user": user, "exp": int(time.time()) + ttl}).encode())
    sig = _b64(hmac.new(SECRET, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def check_token(token: str) -> str | None:
    try:
        body, sig = token.split(".")
    except (ValueError, AttributeError):
        return None
    good = _b64(hmac.new(SECRET, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, good):
        return None
    data = json.loads(_unb64(body))
    if data.get("exp", 0) < time.time():
        return None
    return data.get("user")


# demo operator account (hash only, never a plain password in the repo)
USERS = {"packer": hash_password("freshgrade")}


def login(user: str, password: str) -> str | None:
    stored = USERS.get(user)
    if stored and verify_password(password, stored):
        return make_token(user)
    return None
