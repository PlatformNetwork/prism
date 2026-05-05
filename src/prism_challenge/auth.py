from __future__ import annotations

import hmac
import time
from hashlib import sha256
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from .config import PrismSettings


def canonical_submission_message(*, hotkey: str, nonce: str, timestamp: str, body: bytes) -> bytes:
    body_hash = sha256(body).hexdigest()
    return f"prism:{hotkey}:{nonce}:{timestamp}:{body_hash}".encode()


def _decode_signature(signature: str) -> bytes | str:
    value = signature.removeprefix("0x")
    try:
        return bytes.fromhex(value)
    except ValueError:
        return signature


def verify_hotkey_signature(hotkey: str, message: bytes, signature: str) -> bool:
    try:
        import bittensor as bt  # type: ignore

        keypair = bt.Keypair(ss58_address=hotkey)
        return bool(keypair.verify(message, _decode_signature(signature)))
    except Exception:
        return False


def verify_dev_signature(secret: str, message: bytes, signature: str) -> bool:
    expected = hmac.new(secret.encode(), message, sha256).hexdigest()
    return hmac.compare_digest(expected, signature.removeprefix("sha256="))


async def authenticate_miner(
    request: Request,
    x_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
    x_signature: Annotated[str, Header(min_length=1)],
    x_nonce: Annotated[str, Header(min_length=1, max_length=128)],
    x_timestamp: Annotated[str, Header(min_length=1)],
) -> str:
    app_settings: PrismSettings = request.app.state.settings
    try:
        timestamp = int(x_timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid timestamp") from exc
    if abs(int(time.time()) - timestamp) > app_settings.signature_ttl_seconds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "stale signature")
    body = await request.body()
    message = canonical_submission_message(
        hotkey=x_hotkey, nonce=x_nonce, timestamp=x_timestamp, body=body
    )
    valid = verify_hotkey_signature(x_hotkey, message, x_signature)
    if not valid and app_settings.allow_insecure_signatures:
        valid = verify_dev_signature(app_settings.internal_token(), message, x_signature)
    if not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid signature")
    async with request.app.state.database.connect() as conn:
        try:
            await conn.execute(
                "INSERT INTO nonces(hotkey, nonce, created_at) VALUES (?, ?, datetime('now'))",
                (x_hotkey, x_nonce),
            )
        except Exception as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "nonce already used") from exc
    return x_hotkey


def authenticate_internal(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> None:
    app_settings: PrismSettings = request.app.state.settings
    expected = f"Bearer {app_settings.internal_token()}"
    if authorization != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal token")
