#!/usr/bin/env python3
"""capsuled: an isolated Unix-socket reference monitor for libcapsule.

The linked-library artifact is useful for mechanism measurements, but a client
that maps the store namespace can bypass it.  This service keeps the store,
HMAC secret, manifests, extents, and epoch table in a separate process.  A
small length-prefixed RPC protocol mediates every operation.  Connections are
authenticated with per-principal keys provisioned by the operator; create
requests are restricted to the principal's configured dedup domains.  Capsule
tokens remain delegatable bearer capabilities after creation.

The service intentionally does not promise remote unmapping or erasure of
bytes already returned to a client.  Revocation denies subsequent RPCs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import socket
import socketserver
import struct
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from agentsim.capsule_ffi import (  # noqa: E402
    CAP_R_ALL,
    CAPSULE_E_INVAL,
    CAPSULE_STORE_CREATE,
    CapsuleCap,
    CapsuleError,
    CapsuleStore,
)

MAX_FRAME = 64 * 1024 * 1024
AUTH_LABEL = b"capsule-session-v1\0"
E_AUTH = -100
E_DOMAIN = -101
E_PROTOCOL = -102


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    parts: list[bytes] = []
    left = size
    while left:
        chunk = sock.recv(left)
        if not chunk:
            raise EOFError
        parts.append(chunk)
        left -= len(chunk)
    return b"".join(parts)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if size == 0 or size > MAX_FRAME:
        raise ValueError("bad frame length")
    value = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frame is not an object")
    return value


def send_frame(sock: socket.socket, value: dict[str, Any]) -> None:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    if len(raw) > MAX_FRAME:
        raise ValueError("response too large")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def _cap_from_text(text: str) -> CapsuleCap:
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 72:
        raise ValueError("capability must be 72 bytes")
    return CapsuleCap.from_buffer_copy(raw)


def _cap_to_text(cap: CapsuleCap) -> str:
    return base64.b64encode(bytes(cap)).decode("ascii")


class ServiceState:
    def __init__(self, store_root: Path, principals_path: Path):
        config = json.loads(principals_path.read_text())
        self.principals: dict[str, dict[str, Any]] = {}
        for name, entry in config.items():
            key = bytes.fromhex(entry["key_hex"])
            if len(key) < 32:
                raise ValueError(f"principal {name}: key must be >=32 bytes")
            self.principals[name] = {
                "key": key,
                "domains": frozenset(entry.get("domains", [])),
            }
        store_root.mkdir(parents=True, exist_ok=True)
        os.chmod(store_root, 0o700)
        self.store = CapsuleStore(store_root, CAPSULE_STORE_CREATE)

    def authenticate(self, hello: dict[str, Any]) -> str | None:
        if hello.get("op") != "hello":
            return None
        name = str(hello.get("principal", ""))
        entry = self.principals.get(name)
        if not entry:
            return None
        try:
            nonce = base64.b64decode(hello["nonce"], validate=True)
            supplied = bytes.fromhex(hello["mac"])
        except Exception:
            return None
        message = AUTH_LABEL + name.encode() + b"\0" + nonce
        expected = hmac.new(entry["key"], message, hashlib.sha256).digest()
        return name if hmac.compare_digest(expected, supplied) else None

    def dispatch(self, principal: str, req: dict[str, Any]) -> dict[str, Any]:
        try:
            op = req.get("op")
            if op == "create":
                domain = str(req.get("domain", ""))
                if domain not in self.principals[principal]["domains"]:
                    return {"code": E_DOMAIN, "error": "domain denied"}
                path = str(req.get("path", ""))
                if not path.startswith("/") or ".." in path:
                    return {"code": CAPSULE_E_INVAL, "error": "bad path"}
                compat_text = req.get("compatibility_hash")
                if compat_text is None:
                    cap, cid = self.store.create(
                        f"/{principal}{path}", domain, int(req["bytes_per_token"])
                    )
                else:
                    compat = bytes.fromhex(str(compat_text))
                    cap, cid = self.store.create_compatible(
                        f"/{principal}{path}", domain,
                        int(req["bytes_per_token"]), compat
                    )
                return {"code": 0, "cap": _cap_to_text(cap), "id": cid}
            if op == "extend":
                cap = _cap_from_text(req["cap"])
                payload = base64.b64decode(req["data"], validate=True)
                self.store.extend(cap, payload, int(req["n_tokens"]))
                return {"code": 0}
            if op == "extend_binary":
                cap = _cap_from_text(req["cap"])
                payload = req.get("_binary_data")
                if not isinstance(payload, bytes):
                    return {"code": E_PROTOCOL, "error": "missing binary payload"}
                self.store.extend(cap, payload, int(req["n_tokens"]))
                return {"code": 0}
            if op == "derive":
                cap = _cap_from_text(req["cap"])
                out = self.store.derive(
                    cap, int(req.get("rights", CAP_R_ALL)), int(req.get("ttl_ms", 0))
                )
                return {"code": 0, "cap": _cap_to_text(out)}
            if op == "read":
                length = int(req["length"])
                if length < 0 or length > 16 * 1024 * 1024:
                    return {"code": CAPSULE_E_INVAL, "error": "bad length"}
                cap = _cap_from_text(req["cap"])
                data = self.store.read(cap, int(req.get("offset", 0)), length)
                return {"code": 0, "data": base64.b64encode(data).decode("ascii")}
            if op == "read_binary":
                length = int(req["length"])
                if length < 0 or length > 16 * 1024 * 1024:
                    return {"code": CAPSULE_E_INVAL, "error": "bad length"}
                cap = _cap_from_text(req["cap"])
                data = self.store.read(cap, int(req.get("offset", 0)), length)
                return {"code": 0, "_binary_data": data}
            if op == "stat":
                st = self.store.stat(_cap_from_text(req["cap"]))
                return {
                    "code": 0,
                    "id": int(st.capsule_id),
                    "n_tokens": int(st.n_tokens),
                    "referenced_bytes": int(st.referenced_bytes),
                    "epoch": int(st.epoch),
                    "flags": int(st.flags),
                    "compatibility_hash": bytes(st.compatibility_hash).hex(),
                }
            if op == "revoke":
                self.store.revoke(_cap_from_text(req["cap"]))
                return {"code": 0}
            if op == "seal":
                self.store.seal(_cap_from_text(req["cap"]))
                return {"code": 0}
            return {"code": E_PROTOCOL, "error": "unknown operation"}
        except CapsuleError as exc:
            return {"code": exc.code, "error": exc.name}
        except (KeyError, TypeError, ValueError) as exc:
            return {"code": E_PROTOCOL, "error": str(exc)}


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            hello = recv_frame(self.request)
            principal = self.server.state.authenticate(hello)  # type: ignore[attr-defined]
            if principal is None:
                send_frame(self.request, {"code": E_AUTH, "error": "authentication failed"})
                return
            send_frame(self.request, {"code": 0, "principal": principal})
            while True:
                req = recv_frame(self.request)
                if req.get("op") == "extend_binary":
                    length = int(req.get("length", -1))
                    if length < 0 or length > 16 * 1024 * 1024:
                        send_frame(self.request, {"code": E_PROTOCOL,
                                                  "error": "bad binary length"})
                        continue
                    req["_binary_data"] = _recv_exact(self.request, length)
                response = self.server.state.dispatch(principal, req)  # type: ignore[attr-defined]
                binary = response.pop("_binary_data", None)
                if binary is None:
                    send_frame(self.request, response)
                else:
                    response["length"] = len(binary)
                    send_frame(self.request, response)
                    self.request.sendall(binary)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # fail closed at the RPC boundary
            try:
                send_frame(self.request, {"code": E_PROTOCOL, "error": str(exc)})
            except Exception:
                pass


class CapsuleServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: Path, state: ServiceState):
        self.state = state
        super().__init__(str(socket_path), Handler)


class CapsuleClient:
    def __init__(self, socket_path: str | os.PathLike[str], principal: str, key: bytes):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(str(socket_path))
        nonce = os.urandom(32)
        msg = AUTH_LABEL + principal.encode() + b"\0" + nonce
        mac = hmac.new(key, msg, hashlib.sha256).hexdigest()
        send_frame(
            self.sock,
            {
                "op": "hello",
                "principal": principal,
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "mac": mac,
            },
        )
        response = recv_frame(self.sock)
        if response.get("code") != 0:
            self.sock.close()
            raise PermissionError(response)

    def close(self) -> None:
        self.sock.close()

    def call(self, op: str, **fields: Any) -> dict[str, Any]:
        send_frame(self.sock, {"op": op, **fields})
        return recv_frame(self.sock)

    def extend_binary(self, cap: str, data: bytes, n_tokens: int) -> dict[str, Any]:
        send_frame(self.sock, {"op": "extend_binary", "cap": cap,
                               "n_tokens": n_tokens, "length": len(data)})
        self.sock.sendall(data)
        return recv_frame(self.sock)

    def read_binary(self, cap: str, offset: int, length: int) -> tuple[dict[str, Any], bytes]:
        send_frame(self.sock, {"op": "read_binary", "cap": cap,
                               "offset": offset, "length": length})
        response = recv_frame(self.sock)
        if response.get("code") != 0:
            return response, b""
        n = int(response.get("length", -1))
        if n < 0 or n > MAX_FRAME:
            raise ValueError("bad binary response length")
        return response, _recv_exact(self.sock, n)


def serve(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    state = ServiceState(Path(args.store), Path(args.principals))
    with CapsuleServer(socket_path, state) as server:
        os.chmod(socket_path, 0o600)
        print(json.dumps({"ready": True, "socket": str(socket_path)}), flush=True)
        try:
            server.serve_forever(poll_interval=0.05)
        finally:
            state.store.close()
            if socket_path.exists():
                socket_path.unlink()
    return 0


def probe(args: argparse.Namespace) -> int:
    direct_read = True
    direct_errno = 0
    try:
        with open(Path(args.store) / "secret.key", "rb") as stream:
            stream.read(1)
    except OSError as exc:
        direct_read = False
        direct_errno = exc.errno or 0
    client = CapsuleClient(args.socket, args.principal, bytes.fromhex(args.key_hex))
    response = client.call("stat", cap=args.cap)
    client.close()
    print(
        json.dumps(
            {
                "direct_secret_read": direct_read,
                "direct_errno": direct_errno,
                "mediated_stat_code": response.get("code"),
            },
            sort_keys=True,
        )
    )
    return 0 if (not direct_read and response.get("code") == 0) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ps = sub.add_parser("serve")
    ps.add_argument("--store", required=True)
    ps.add_argument("--socket", required=True)
    ps.add_argument("--principals", required=True)
    pp = sub.add_parser("probe")
    pp.add_argument("--store", required=True)
    pp.add_argument("--socket", required=True)
    pp.add_argument("--principal", required=True)
    pp.add_argument("--key-hex", required=True)
    pp.add_argument("--cap", required=True)
    args = parser.parse_args()
    return serve(args) if args.command == "serve" else probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
