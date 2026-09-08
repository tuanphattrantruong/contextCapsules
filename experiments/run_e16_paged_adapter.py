#!/usr/bin/env python3
"""E16: executable paged-block adapter latency and concurrency study."""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentsim.paged_adapter import AdapterError, PagedCapsuleAdapter  # noqa: E402


def pct(xs: list[int], p: float) -> float:
    ys = sorted(xs); pos = (len(ys) - 1) * p
    lo = int(pos); hi = min(lo + 1, len(ys) - 1); f = pos - lo
    return ys[lo] * (1 - f) + ys[hi] * f


def main() -> int:
    # A 16 KiB block keeps the state modest (8 MiB) while matching the fact
    # that real KV blocks carry many tensor elements per allocator handle.
    block_bytes, blocks, iterations = 16384, 512, 300
    payload = [bytes(((i + j) & 0xff for j in range(block_bytes)))
               for i in range(blocks)]
    adapter = PagedCapsuleAdapter(); cap = adapter.create("tenant:7", payload)
    attach_ns, copy_ns = [], []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        with adapter.attach(cap, "tenant:7") as view:
            _ = view.read(0)[0] ^ view.read(-1)[-1]
        attach_ns.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        copied = [memoryview(x).tobytes() for x in payload]
        _ = copied[0][0] ^ copied[-1][-1]
        copy_ns.append(time.perf_counter_ns() - t0)

    workers, per_worker = 8, 500
    barrier = threading.Barrier(workers)
    errors: list[str] = []
    done = [0] * workers

    def worker(index: int) -> None:
        try:
            barrier.wait()
            for _ in range(per_worker):
                with adapter.attach(cap, "tenant:7") as view:
                    assert view.read(index)[0] == payload[index][0]
                done[index] += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    t0 = time.perf_counter_ns()
    for t in ts: t.start()
    for t in ts: t.join()
    concurrent_ns = time.perf_counter_ns() - t0

    child = adapter.fork(cap, "tenant:7")
    before = adapter.object_handles(cap, "tenant:7")[0]
    adapter.write_cow(child, "tenant:7", 0, b"C" * block_bytes)
    cow_isolated = adapter.arena.read(before) == payload[0]
    denied = 0
    for action in (lambda: adapter.attach(cap, "tenant:8"),
                   lambda: adapter.attach("bad-token", "tenant:7")):
        try: action()
        except AdapterError: denied += 1
    adapter.revoke(cap, "tenant:7")
    try: adapter.attach(cap, "tenant:7")
    except AdapterError: denied += 1
    adapter.destroy(cap, "tenant:7"); adapter.destroy(child, "tenant:7")

    out = {
        "experiment": "E16_paged_block_adapter",
        "configuration": {"blocks": blocks, "block_bytes": block_bytes,
                          "logical_state_mib": blocks * block_bytes / 2**20,
                          "iterations": iterations, "workers": workers,
                          "operations_per_worker": per_worker},
        "measured": {
            "attach_release_p50_us": statistics.median(attach_ns) / 1000,
            "attach_release_p95_us": pct(attach_ns, .95) / 1000,
            "materialize_copy_p50_us": statistics.median(copy_ns) / 1000,
            "materialize_copy_p95_us": pct(copy_ns, .95) / 1000,
            "copy_over_attach_p50": statistics.median(copy_ns) /
                                      statistics.median(attach_ns),
            "concurrent_operations": sum(done),
            "concurrent_ops_per_s": sum(done) / (concurrent_ns / 1e9),
        },
        "correctness": {"thread_errors": errors,
                        "cow_parent_unchanged": cow_isolated,
                        "unauthorized_or_revoked_denied": denied,
                        "expected_denials": 3,
                        "post_destroy": adapter.arena.stats()},
        "scope": ("CPU allocator-metadata prototype; measures handle-vector "
                  "attach/refcount/COW, not GPU DMA or production scheduling"),
    }
    path = ROOT / "experiments/results/e16_paged_adapter.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    ok = not errors and cow_isolated and denied == 3 and \
        out["correctness"]["post_destroy"]["live_blocks"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
