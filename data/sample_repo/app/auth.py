"""Authentication: password hashing and signed, expiring session tokens."""
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

import config


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and return 'salt$hash' in hex."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + "$" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored 'salt$hash' value using constant-time comparison."""
    salt_hex, _, _ = stored.partition("$")
    candidate = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, stored)


def issue_token(member_id: int, is_admin: bool = False) -> str:
    """Create a signed session token containing the member id and an expiry timestamp."""
    payload = {"sub": member_id, "admin": is_admin, "exp": int(time.time()) + config.TOKEN_TTL_SECONDS}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(config.SECRET_KEY.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    """Return the token payload if the signature is valid and it has not expired, else None."""
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(config.SECRET_KEY.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body.encode()))
        return payload if payload["exp"] >= time.time() else None
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
