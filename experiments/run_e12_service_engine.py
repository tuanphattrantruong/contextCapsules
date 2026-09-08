#!/usr/bin/env python3
"""E12: real llama.cpp sequence state through the isolated service.

A scheduler-side adapter owns the service credential and bearer capability.
The producer and consumer are distinct llama.cpp processes.  They receive a
temporary engine-state stream, never the service store path, HMAC secret,
manifests, epoch table, or principal key.  The service is SIGKILLed after a
successful seal and reopened before the consumer runs, testing recovery at the
documented operation boundary (not every internal fsync point).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from service.capsuled import CapsuleClient  # noqa: E402

BPT = 1024
CHUNK = 4 * 1024 * 1024
ALIGN = 16 * BPT
HDR = struct.Struct("=QQQQ")
LC_MAGIC = 0x4C43415053554C45


def start_service(store: Path, sock: Path, principals: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "src/service/capsuled.py"), "serve",
         "--store", str(store), "--socket", str(sock),
         "--principals", str(principals)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line or not json.loads(line).get("ready"):
        raise RuntimeError(f"service failed to start: {line!r}")
    return proc


def stop_service(proc: subprocess.Popen, hard: bool = False) -> None:
    if hard:
        proc.kill()
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def checked(client: CapsuleClient, op: str, **fields):
    response = client.call(op, **fields)
    if response.get("code") != 0:
        raise RuntimeError(f"{op}: {response}")
    return response


def percentile(values: list[int], q: float) -> int:
    xs = sorted(values)
    return xs[min(len(xs) - 1, int((len(xs) - 1) * q))]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(
        ROOT / "workloads/gguf/qwen2.5-0.5b-instruct-q8_0.gguf"))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--read-reps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "experiments/results/e12_service_engine.json"))
    args = ap.parse_args()
    engine = ROOT / "src/llamacap/llamacap"
    model = Path(args.model)
    if not engine.exists() or not model.exists():
        raise SystemExit("build llamacap and provide the GGUF model first")
    compatibility_descriptor = {
        "model_sha256": file_sha256(model),
        "engine": "llama.cpp-sequence-state",
        "state_schema": 1,
        "bytes_per_token": BPT,
        "chunk_tokens": ALIGN // BPT,
        "prompt_tokens": args.tokens,
    }
    compatibility_hash = hashlib.sha256(json.dumps(
        compatibility_descriptor, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()

    key = bytes.fromhex("42" * 32)
    with tempfile.TemporaryDirectory(prefix="capsule-e12-") as td_text:
        td = Path(td_text)
        store, sock = td / "service-store", td / "capsuled.sock"
        principals = td / "principals.json"
        principals.write_text(json.dumps({
            "scheduler": {"key_hex": key.hex(), "domains": ["tenant:1"]}
        }))
        export_blob, import_blob = td / "engine-export.bin", td / "engine-import.bin"

        export_cmd = [str(engine), "export", str(model), str(export_blob),
                      str(args.tokens), str(args.new_tokens), str(args.ngl)]
        producer = subprocess.run(export_cmd, cwd=ROOT, check=True,
                                  capture_output=True, text=True)
        export_meta = json.loads(producer.stdout.strip().splitlines()[-1])
        raw = export_blob.read_bytes()
        magic, blob_bytes, n_prompt, _ = HDR.unpack_from(raw)
        if magic != LC_MAGIC or n_prompt != args.tokens:
            raise RuntimeError("engine export header mismatch")
        padded = raw + bytes((-len(raw)) % ALIGN)
        source_sha = hashlib.sha256(raw).hexdigest()

        plain_ns: list[int] = []
        for _ in range(args.read_reps):
            t0 = time.perf_counter_ns()
            probe = export_blob.read_bytes()
            plain_ns.append(time.perf_counter_ns() - t0)
            if probe != raw:
                raise RuntimeError("plain file baseline changed bytes")

        proc = start_service(store, sock, principals)
        client = CapsuleClient(sock, "scheduler", key)
        t0 = time.perf_counter_ns()
        created = checked(client, "create", path="/engine/seq-json",
                          domain="tenant:1", bytes_per_token=BPT,
                          compatibility_hash=compatibility_hash)
        json_cap = created["cap"]
        for off in range(0, len(padded), CHUNK):
            part = padded[off:off + CHUNK]
            checked(client, "extend", cap=json_cap,
                    data=base64.b64encode(part).decode("ascii"),
                    n_tokens=len(part) // BPT)
        checked(client, "seal", cap=json_cap)
        json_put_ns = time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        created = checked(client, "create", path="/engine/seq-binary",
                          domain="tenant:1", bytes_per_token=BPT,
                          compatibility_hash=compatibility_hash)
        binary_cap = created["cap"]
        for off in range(0, len(padded), CHUNK):
            part = padded[off:off + CHUNK]
            response = client.extend_binary(binary_cap, part, len(part) // BPT)
            if response.get("code") != 0:
                raise RuntimeError(f"extend_binary: {response}")
        checked(client, "seal", cap=binary_cap)
        binary_put_ns = time.perf_counter_ns() - t0
        client.close()

        # A hard daemon stop after the acknowledged seal.  The restarted
        # service must preserve the object and capability epoch.
        stop_service(proc, hard=True)
        proc = start_service(store, sock, principals)
        client = CapsuleClient(sock, "scheduler", key)
        stat = checked(client, "stat", cap=binary_cap)
        if int(stat["referenced_bytes"]) != len(padded):
            raise RuntimeError("sealed object length changed after restart")
        if stat["compatibility_hash"] != compatibility_hash:
            raise RuntimeError("compatibility binding changed after restart")

        json_service_ns: list[int] = []
        binary_service_ns: list[int] = []
        recovered = b""
        for rep in range(args.read_reps):
            t0 = time.perf_counter_ns()
            parts = []
            for off in range(0, len(padded), CHUNK):
                response = checked(client, "read", cap=json_cap, offset=off,
                                   length=min(CHUNK, len(padded) - off))
                parts.append(base64.b64decode(response["data"], validate=True))
            candidate = b"".join(parts)
            json_service_ns.append(time.perf_counter_ns() - t0)
            if candidate != padded:
                raise RuntimeError("JSON service round-trip changed bytes")

            t0 = time.perf_counter_ns()
            parts = []
            for off in range(0, len(padded), CHUNK):
                response, data = client.read_binary(
                    binary_cap, off, min(CHUNK, len(padded) - off))
                if response.get("code") != 0:
                    raise RuntimeError(f"read_binary: {response}")
                parts.append(data)
            candidate = b"".join(parts)
            binary_service_ns.append(time.perf_counter_ns() - t0)
            if candidate != padded:
                raise RuntimeError("binary service round-trip changed bytes")
            if rep == 0:
                recovered = candidate

        recovered_raw = recovered[:HDR.size + blob_bytes]
        import_blob.write_bytes(recovered_raw)
        export_blob.unlink()
        import_cmd = [str(engine), "import", str(model), str(import_blob),
                      str(args.new_tokens), str(args.ngl)]
        consumer = subprocess.run(import_cmd, cwd=ROOT, check=True,
                                  capture_output=True, text=True)
        import_meta = json.loads(consumer.stdout.strip().splitlines()[-1])
        import_blob.unlink()
        token_match = export_meta["ref_tokens"] == import_meta["tokens"]

        checked(client, "revoke", cap=binary_cap)
        denied = client.call("stat", cap=binary_cap).get("code") != 0
        client.close()
        stop_service(proc)

        result = {
            "experiment": "E12_service_backed_llamacpp",
            "engine": "llama.cpp (Metal)",
            "model": model.name,
            "n_prompt": args.tokens,
            "n_generated": args.new_tokens,
            "engine_blob_bytes": blob_bytes,
            "service_object_bytes": len(padded),
            "source_sha256": source_sha,
            "compatibility_descriptor": compatibility_descriptor,
            "compatibility_hash": compatibility_hash,
            "post_restart_compatibility_match": True,
            "roundtrip_sha256": hashlib.sha256(recovered_raw).hexdigest(),
            "byte_exact": recovered_raw == raw,
            "generated_tokens_identical": token_match,
            "daemon_sigkill_after_seal": True,
            "post_restart_stat_ok": True,
            "post_revoke_denied": denied,
            "engine_argv_exposes_store_path": False,
            "engine_argv_exposes_principal_key": False,
            "json_service_put_ms": json_put_ns / 1e6,
            "binary_service_put_ms": binary_put_ns / 1e6,
            "plain_file_read_ms_p50": statistics.median(plain_ns) / 1e6,
            "plain_file_read_ms_p99": percentile(plain_ns, .99) / 1e6,
            "json_service_read_ms_p50": statistics.median(json_service_ns) / 1e6,
            "json_service_read_ms_p99": percentile(json_service_ns, .99) / 1e6,
            "binary_service_read_ms_p50": statistics.median(binary_service_ns) / 1e6,
            "binary_service_read_ms_p99": percentile(binary_service_ns, .99) / 1e6,
            "json_service_over_plain_p50": statistics.median(json_service_ns) /
                                           statistics.median(plain_ns),
            "binary_service_over_plain_p50": statistics.median(binary_service_ns) /
                                             statistics.median(plain_ns),
            "export": export_meta,
            "import": import_meta,
            "pass": bool(recovered_raw == raw and token_match and denied),
            "scope": ("sidecar-mediated sequence-state boundary; no scheduler "
                      "block manager, GPU-page capability, or mid-operation "
                      "crash-point claim"),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
