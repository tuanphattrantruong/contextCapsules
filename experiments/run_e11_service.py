#!/usr/bin/env python3
"""E11: isolated-service enforcement, denial semantics, and RPC cost."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentsim.capsule_ffi import CAP_R_READ  # noqa: E402
from service.capsuled import CapsuleClient  # noqa: E402


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * p)))]


def call_ok(client: CapsuleClient, op: str, **fields):
    response = client.call(op, **fields)
    if response.get("code") != 0:
        raise RuntimeError(f"{op}: {response}")
    return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out", default=str(ROOT / "experiments/results/e11_service.json")
    )
    args = parser.parse_args()

    scratch = ROOT / "experiments/tmp_service"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    store = scratch / "store"
    sock = scratch / "capsuled.sock"
    config = scratch / "principals.json"
    key_a = hashlib.sha256(b"capsule-e11-principal-a").digest()
    key_b = hashlib.sha256(b"capsule-e11-principal-b").digest()
    config.write_text(
        json.dumps(
            {
                "alice": {"key_hex": key_a.hex(), "domains": ["tenant:alice", "public"]},
                "bob": {"key_hex": key_b.hex(), "domains": ["tenant:bob", "public"]},
            }
        )
    )
    os.chmod(config, 0o600)

    cmd = [
        sys.executable,
        str(ROOT / "src/service/capsuled.py"),
        "serve",
        "--store",
        str(store),
        "--socket",
        str(sock),
        "--principals",
        str(config),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not sock.exists():
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read())
            time.sleep(0.01)
        if not sock.exists():
            raise RuntimeError("capsuled did not create its socket")

        alice = CapsuleClient(sock, "alice", key_a)
        bob = CapsuleClient(sock, "bob", key_b)
        created = call_ok(
            alice,
            "create",
            path="/object-1",
            domain="tenant:alice",
            bytes_per_token=1024,
        )
        owner = created["cap"]
        payload = bytes((i * 17 + 3) & 0xFF for i in range(16 * 1024))
        call_ok(
            alice,
            "extend",
            cap=owner,
            n_tokens=16,
            data=base64.b64encode(payload).decode("ascii"),
        )
        delegated = call_ok(alice, "derive", cap=owner, rights=CAP_R_READ, ttl_ms=0)["cap"]

        foreign_create = bob.call(
            "create",
            path="/forbidden",
            domain="tenant:alice",
            bytes_per_token=1024,
        )
        delegated_read = call_ok(bob, "read", cap=delegated, offset=0, length=4096)
        delegated_bytes = base64.b64decode(delegated_read["data"])
        forged_raw = bytearray(base64.b64decode(delegated))
        forged_raw[-1] ^= 1
        forged = base64.b64encode(forged_raw).decode("ascii")
        forged_read = bob.call("read", cap=forged, offset=0, length=16)

        created2 = call_ok(
            alice,
            "create",
            path="/object-2",
            domain="tenant:alice",
            bytes_per_token=1024,
        )
        owner2 = created2["cap"]
        call_ok(
            alice,
            "extend",
            cap=owner2,
            n_tokens=16,
            data=base64.b64encode(payload).decode("ascii"),
        )
        read2 = call_ok(alice, "derive", cap=owner2, rights=CAP_R_READ, ttl_ms=0)["cap"]

        warm = 50 if args.quick else 500
        trials = 500 if args.quick else 10000
        for _ in range(warm):
            call_ok(bob, "stat", cap=read2)
        stat_us: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter_ns()
            call_ok(bob, "stat", cap=read2)
            stat_us.append((time.perf_counter_ns() - t0) / 1000.0)

        read_trials = 50 if args.quick else 1000
        read_us: list[float] = []
        for _ in range(read_trials):
            t0 = time.perf_counter_ns()
            call_ok(bob, "read", cap=read2, offset=0, length=4096)
            read_us.append((time.perf_counter_ns() - t0) / 1000.0)

        sandbox = {"available": False, "passed": False}
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec:
            profile = (
                '(version 1)(allow default)'
                f'(deny file-read* file-write* (subpath "{store}"))'
            )
            probe_cmd = [
                sandbox_exec,
                "-p",
                profile,
                sys.executable,
                str(ROOT / "src/service/capsuled.py"),
                "probe",
                "--store",
                str(store),
                "--socket",
                str(sock),
                "--principal",
                "bob",
                "--key-hex",
                key_b.hex(),
                "--cap",
                read2,
            ]
            probe = subprocess.run(probe_cmd, text=True, capture_output=True)
            sandbox = {
                "available": True,
                "passed": probe.returncode == 0,
                "returncode": probe.returncode,
                "stdout": probe.stdout.strip(),
                "stderr": probe.stderr.strip(),
            }

        call_ok(alice, "revoke", cap=owner)
        after_revoke = bob.call("read", cap=delegated, offset=0, length=16)

        alice.close()
        bob.close()
        result = {
            "meta": {
                "experiment": "E11_isolated_service",
                "mode": "quick" if args.quick else "full",
                "machine": platform.platform(),
                "python": platform.python_version(),
                "service_command": cmd,
            },
            "security": {
                "foreign_domain_create_code": foreign_create.get("code"),
                "delegated_read_matches": delegated_bytes == payload[:4096],
                "forged_capability_code": forged_read.get("code"),
                "post_revoke_code": after_revoke.get("code"),
                "sandbox_bypass_probe": sandbox,
            },
            "performance": {
                "stat_rpc_trials": trials,
                "stat_rpc_us": {
                    "median": statistics.median(stat_us),
                    "p99": pct(stat_us, 0.99),
                    "mean": statistics.mean(stat_us),
                },
                "read_4k_rpc_trials": read_trials,
                "read_4k_rpc_us": {
                    "median": statistics.median(read_us),
                    "p99": pct(read_us, 0.99),
                    "mean": statistics.mean(read_us),
                },
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        checks = [
            foreign_create.get("code") == -101,
            delegated_bytes == payload[:4096],
            forged_read.get("code") == -1,
            after_revoke.get("code") == -3,
            (not sandbox["available"]) or sandbox["passed"],
        ]
        return 0 if all(checks) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
