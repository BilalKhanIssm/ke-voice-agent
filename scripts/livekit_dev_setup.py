#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign_hs256(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}.{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


@dataclass(frozen=True)
class LiveKitCreds:
    ws_url: str
    api_url: str
    api_key: str
    api_secret: str
    agent_name: str


def _http_api_url_from_livekit_url(url: str) -> str:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme == "wss":
        scheme = "https"
    elif scheme == "ws":
        scheme = "http"
    elif scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported LIVEKIT_URL scheme: {parts.scheme}")
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _load_creds() -> LiveKitCreds:
    ws_url = os.environ["LIVEKIT_URL"].rstrip("/")
    return LiveKitCreds(
        ws_url=ws_url,
        api_url=_http_api_url_from_livekit_url(ws_url),
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
        agent_name=os.environ.get("LIVEKIT_AGENT_NAME", "telephony-agent"),
    )


def _server_token(
    *,
    api_key: str,
    api_secret: str,
    grants: dict,
    identity: str,
    ttl_seconds: int = 3600,
) -> str:
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": identity,
        "nbf": now - 5,
        "iat": now,
        "exp": now + ttl_seconds,
        "video": grants,
    }
    return _sign_hs256(payload, api_secret)


def _twirp_post(url: str, token: str, service_method: str, body: dict) -> dict:
    endpoint = f"{url}/twirp/{service_method}"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_method} failed: HTTP {exc.code} {detail}") from exc


def create_room(creds: LiveKitCreds, room_name: str) -> dict:
    token = _server_token(
        api_key=creds.api_key,
        api_secret=creds.api_secret,
        identity="dev-room-bootstrap",
        grants={"roomCreate": True},
    )
    return _twirp_post(
        creds.api_url,
        token,
        "livekit.RoomService/CreateRoom",
        {"name": room_name},
    )


def create_dispatch(creds: LiveKitCreds, room_name: str, agent_name: str) -> dict:
    token = _server_token(
        api_key=creds.api_key,
        api_secret=creds.api_secret,
        identity="dev-agent-dispatch",
        grants={"roomAdmin": True, "room": room_name},
    )
    return _twirp_post(
        creds.api_url,
        token,
        "livekit.AgentDispatchService/CreateDispatch",
        {"room": room_name, "agent_name": agent_name},
    )


def create_participant_token(
    creds: LiveKitCreds,
    room_name: str,
    identity: str,
    name: str,
    ttl_seconds: int,
) -> str:
    now = int(time.time())
    payload = {
        "iss": creds.api_key,
        "sub": identity,
        "name": name,
        "nbf": now - 5,
        "iat": now,
        "exp": now + ttl_seconds,
        "video": {
            "roomJoin": True,
            "room": room_name,
            "canPublish": True,
            "canSubscribe": True,
        },
    }
    return _sign_hs256(payload, creds.api_secret)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LiveKit dev room + playground token, optionally dispatch agent.")
    p.add_argument("--room", default=f"ke-dev-{int(time.time())}", help="Room name to create")
    p.add_argument("--identity", default="playground-user", help="Playground participant identity")
    p.add_argument("--name", default="Playground User", help="Playground participant display name")
    p.add_argument("--ttl", type=int, default=3600, help="Participant token TTL seconds")
    p.add_argument("--dispatch-agent", action="store_true", help="Dispatch agent to room using LIVEKIT_AGENT_NAME")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        creds = _load_creds()
    except KeyError as exc:
        print(f"Missing env var: {exc.args[0]}")
        return 1

    room_info = create_room(creds, args.room)
    user_token = create_participant_token(creds, args.room, args.identity, args.name, args.ttl)

    dispatch_info: dict | None = None
    if args.dispatch_agent:
        try:
            dispatch_info = create_dispatch(creds, args.room, creds.agent_name)
        except Exception as exc:
            print(f"Agent dispatch failed: {exc}")

    print("\nLiveKit dev setup complete:\n")
    print(f"WS_URL={creds.ws_url}")
    print(f"ROOM_NAME={args.room}")
    print(f"PARTICIPANT_IDENTITY={args.identity}")
    print(f"PARTICIPANT_NAME={args.name}")
    print(f"PARTICIPANT_TOKEN={user_token}")
    if room_info:
        print(f"ROOM_CREATED={room_info.get('name', args.room)}")
    if dispatch_info is not None:
        print(f"AGENT_DISPATCHED={bool(dispatch_info)}")

    print("\nUse in LiveKit Playground:")
    print("1) Open playground and choose 'Connect with token'")
    print("2) Paste WS_URL")
    print("3) Paste PARTICIPANT_TOKEN")
    print("4) Keep your worker running (`python -m app.telephony.entrypoint start`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
