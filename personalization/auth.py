"""Minimal account + auth layer (email/password, optional 2FA).

Accounts and session tokens live in `user_data/_auth.json`. Passwords are stored as
PBKDF2-HMAC-SHA256 hashes with a per-user salt (stdlib only — no bcrypt dependency).
Each account links to a user_id in the existing user_store (which holds all their data).

2FA here is a DEV MOCK: we generate a 6-digit code and RETURN it in the API response
(and log it) instead of sending a real SMS/email — wiring a provider (Twilio / SMTP)
would replace `_send_code`. The demo account dummy@test.com has 2FA disabled.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time

from personalization import user_store

_AUTH_FILE = user_store.USER_DIR / "_auth.json"
_ITER = 200_000
_CODE_TTL = 300  # seconds


def _load() -> dict:
    if not _AUTH_FILE.exists():
        return {"accounts": {}, "tokens": {}, "pending": {}}
    return json.loads(_AUTH_FILE.read_text(encoding="utf-8"))


def _save(d: dict) -> None:
    user_store.USER_DIR.mkdir(parents=True, exist_ok=True)
    _AUTH_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITER).hex()


def _norm(email: str) -> str:
    return email.strip().lower()


class AuthError(Exception):
    pass


def account_exists(email: str) -> bool:
    return _norm(email) in _load()["accounts"]


def signup(email: str, password: str, phone: str | None = None,
           enable_2fa: bool = False, user_id: str | None = None) -> dict:
    email = _norm(email)
    if not email or "@" not in email:
        raise AuthError("Enter a valid email.")
    if len(password) < 4:
        raise AuthError("Password must be at least 4 characters.")
    d = _load()
    if email in d["accounts"]:
        raise AuthError("An account with that email already exists.")

    uid = user_id or ("u_" + secrets.token_hex(4))
    if not user_store.exists(uid):
        user_store.create(uid)
        u = user_store.load(uid)
        u["account"] = {"email": email, "phone": phone}
        user_store.save(u)

    salt = secrets.token_hex(16)
    d["accounts"][email] = {"user_id": uid, "salt": salt, "hash": _hash(password, salt),
                            "phone": phone, "twofa": bool(enable_2fa)}
    _save(d)
    return {"email": email, "user_id": uid, "twofa": bool(enable_2fa)}


def _check_password(acct: dict, password: str) -> bool:
    return secrets.compare_digest(acct["hash"], _hash(password, acct["salt"]))


def _send_code(email: str, phone: str | None, code: str) -> str:
    """DEV: pretend to send; return the channel. Real impl -> Twilio/SMTP."""
    print(f"[2FA] code for {email}: {code}")
    return "phone" if phone else "email"


def login(email: str, password: str) -> dict:
    email = _norm(email)
    d = _load()
    acct = d["accounts"].get(email)
    if not acct or not _check_password(acct, password):
        raise AuthError("Wrong email or password.")
    if acct.get("twofa"):
        code = f"{secrets.randbelow(1_000_000):06d}"
        d["pending"][email] = {"code": code, "exp": time.time() + _CODE_TTL}
        _save(d)
        channel = _send_code(email, acct.get("phone"), code)
        # dev_code is returned so the demo works without a real SMS/email provider.
        return {"status": "2fa_required", "channel": channel, "dev_code": code}
    token = _issue(email)
    return {"status": "ok", "token": token, "user_id": acct["user_id"], "email": email}


def verify_2fa(email: str, code: str) -> dict:
    email = _norm(email)
    d = _load()
    pend = d["pending"].get(email)
    if not pend or pend["exp"] < time.time():
        raise AuthError("Code expired — request a new one.")
    if not secrets.compare_digest(str(pend["code"]), str(code).strip()):
        raise AuthError("Incorrect code.")
    d["pending"].pop(email, None)
    _save(d)
    token = _issue(email)
    return {"status": "ok", "token": token, "user_id": d["accounts"][email]["user_id"],
            "email": email}


def _issue(email: str) -> str:
    d = _load()
    token = secrets.token_urlsafe(24)
    d["tokens"][token] = email
    _save(d)
    return token


def user_for_token(token: str) -> dict | None:
    d = _load()
    email = d["tokens"].get(token or "")
    if not email:
        return None
    acct = d["accounts"].get(email)
    return {"email": email, "user_id": acct["user_id"]} if acct else None


def logout(token: str) -> None:
    d = _load()
    d["tokens"].pop(token, None)
    _save(d)


def ensure_demo_account() -> None:
    """Guarantee the demo login exists: dummy@test.com / dummy -> the seeded u_demo."""
    if not user_store.exists("u_demo"):
        user_store.seed_demo("u_demo")
    d = _load()
    if "dummy@test.com" not in d["accounts"]:
        signup("dummy@test.com", "dummy", enable_2fa=False, user_id="u_demo")
