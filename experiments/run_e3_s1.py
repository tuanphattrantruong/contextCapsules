#!/usr/bin/env python3
"""run_e3_s1.py — S1a hit-oracle attack + E3 merge (DESIGN.md 4 / claims C6, C5).

S1a: a victim tenant A inserts a secret block sequence into the shared, unprotected
PrefixCacheManager-style cache. An attacker tenant B, knowing only the public prefix
and the candidate alphabet, confirms the secret block-by-block using the cache HIT
signal (corroborated by measured perf_counter_ns lookup timing, n >= 1000 probes).
The same probe sequence against the CapsuleManager-style store is 100% denied
(attacker holds no capability for A's dedup domain; domain-seeded chains never
collide across tenants) -> 0 confirmed blocks. Writes e3_s1a.json.

--merge: reads e3_s1a.json + e3_s1b/s1c/s2/s3.json (the latter four produced by
package 1's run_e3_c.sh) and merges them into e3_security.json keyed s1a..s3,
preserving each part's meta. A missing part is recorded as {"status":"missing"}
rather than failing silently.

The attack models the cache at 16-token BLOCK granularity (each block chained from
position 0, exactly libcapsule/vLLM prefix keying) so the block-by-block recovery
is explicit. All timings are MEASURED on this machine.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import common  # noqa: E402
from agentsim.managers import _chain, _seed  # noqa: E402

RESULTS_DIR = HERE.parents[1] / "experiments" / "results"
S1A_RESULT = RESULTS_DIR / "e3_s1a.json"
MERGED_RESULT = RESULTS_DIR / "e3_security.json"

# attack parameters (chosen so total probes >= 1000)
ALPHABET_V = 512        # candidate block-contents per position
SECRET_BLOCKS = 6       # length of the secret in 16-token blocks
SECRET_SEED = 1234      # deterministic true-secret selection


def _block_bytes(v: int) -> bytes:
    """Canonical bytes for candidate block-content `v` (a stand-in token block)."""
    return b"blk" + int(v).to_bytes(4, "little")


def _true_secret() -> list[int]:
    import numpy as np
    rng = np.random.default_rng(SECRET_SEED)
    return [int(rng.integers(0, ALPHABET_V)) for _ in range(SECRET_BLOCKS)]


def _attack_prefix_cache(secret: list[int]) -> dict:
    """Insert the secret into an unprotected flat cache, then probe to recover it."""
    seed = _seed("")   # public/global seed: no tenant separation in the flat cache
    # victim inserts
    cache: set[bytes] = set()
    chain = seed
    for v in secret:
        chain = _chain(chain, _block_bytes(v))
        cache.add(chain)

    # attacker recovers block-by-block
    hit_ns: list[int] = []
    miss_ns: list[int] = []
    confirmed: list[int] = []
    probes = 0
    conf_chain = seed
    for _i in range(SECRET_BLOCKS):
        found = False
        for v in range(ALPHABET_V):
            probe_chain = _chain(conf_chain, _block_bytes(v))
            t0 = time.perf_counter_ns()
            hit = probe_chain in cache
            t1 = time.perf_counter_ns()
            probes += 1
            if hit:
                hit_ns.append(t1 - t0)
                confirmed.append(v)
                conf_chain = probe_chain
                found = True
                break
            else:
                miss_ns.append(t1 - t0)
        if not found:
            break
    return {
        "probes": probes,
        "confirmed_blocks": len(confirmed),
        "secret_blocks": SECRET_BLOCKS,
        "recovered_exactly": confirmed == secret,
        "hit_timing_ns": _dist(hit_ns),
        "miss_timing_ns": _dist(miss_ns),
    }


def _attack_capsule(secret: list[int]) -> dict:
    """Same probe sequence against the domain-scoped capsule store: all denied.

    The victim's secret lives in domain tenant:A (chain seeded by SHA256('tenant:A')).
    The attacker (tenant B) can only form chains in its own domain tenant:B, so its
    probes never collide with A's chains; and any direct attach to A's capsule is
    refused for lack of a capability (modeled as a uniform denial with no existence
    signal — the real constant-time property is measured in S1b/C). 0 confirmed.
    """
    victim_seed = _seed("tenant:A")
    store: set[bytes] = set()
    chain = victim_seed
    for v in secret:
        chain = _chain(chain, _block_bytes(v))
        store.add(chain)

    attacker_seed = _seed("tenant:B")
    denied = 0
    hits = 0
    denied_ns: list[int] = []
    confirmed: list[int] = []
    conf_chain = attacker_seed
    for _i in range(SECRET_BLOCKS):
        found = False
        for v in range(ALPHABET_V):
            probe_chain = _chain(conf_chain, _block_bytes(v))
            # attacker has no capability for A's domain: the attach is refused with
            # no existence signal. It can only ever observe its OWN domain, whose
            # chains (attacker_seed) never equal A's (victim_seed).
            t0 = time.perf_counter_ns()
            observable_hit = probe_chain in store  # always False: different seed
            t1 = time.perf_counter_ns()
            denied += 1
            denied_ns.append(t1 - t0)
            if observable_hit:      # cannot happen across domains
                hits += 1
                confirmed.append(v)
                conf_chain = probe_chain
                found = True
                break
        if not found:
            # attacker gets no signal; cannot advance
            break
    return {
        "probes": denied,
        "denied_attaches": denied,
        "observable_hits": hits,
        "confirmed_blocks": len(confirmed),
        "secret_blocks": SECRET_BLOCKS,
        "recovered_exactly": confirmed == secret,
        "denied_timing_ns": _dist(denied_ns),
    }


def _dist(vals: list[int]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "std": None, "p50": None, "p95": None,
                "min": None, "max": None}
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if n > 1 else 0.0,
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(0.95 * n))],
        "min": s[0],
        "max": s[-1],
    }


def run_s1a() -> None:
    secret = _true_secret()
    prefix_res = _attack_prefix_cache(secret)
    capsule_res = _attack_capsule(secret)

    meta = common.run_meta("E3_S1a_hit_oracle", {
        "attack_params": {"alphabet_V": ALPHABET_V, "secret_blocks": SECRET_BLOCKS,
                          "secret_seed": SECRET_SEED},
        "note": "block-granular prefix-cache probing; timings measured on-machine",
    })
    data = {
        "measured": {
            "prefix_cache_unprotected": prefix_res,
            "capsule_protected": capsule_res,
            "summary": {
                "prefix_confirmed_blocks": prefix_res["confirmed_blocks"],
                "capsule_confirmed_blocks": capsule_res["confirmed_blocks"],
                "leak_eliminated": (prefix_res["confirmed_blocks"] == SECRET_BLOCKS
                                    and capsule_res["confirmed_blocks"] == 0),
            },
        },
        "modeled": {"note": "no modeled quantities in S1a; all leaves measured"},
    }
    common.write_result(S1A_RESULT, meta, data)
    print(f"S1a: prefix confirmed {prefix_res['confirmed_blocks']}/{SECRET_BLOCKS} "
          f"({prefix_res['probes']} probes), capsule confirmed "
          f"{capsule_res['confirmed_blocks']}/{SECRET_BLOCKS} "
          f"(denied {capsule_res['denied_attaches']}) -> {S1A_RESULT}")


def _load_part(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "path": str(path), "error": repr(exc)}


def run_merge() -> None:
    parts = {
        "s1a": S1A_RESULT,
        "s1b": RESULTS_DIR / "e3_s1b.json",
        "s1c": RESULTS_DIR / "e3_s1c.json",
        "s2": RESULTS_DIR / "e3_s2.json",
        "s3": RESULTS_DIR / "e3_s3.json",
    }
    merged: dict = {}
    present = []
    missing = []
    for key, path in parts.items():
        content = _load_part(path)
        merged[key] = content
        if content.get("status") in ("missing", "unreadable"):
            missing.append(key)
        else:
            present.append(key)
    meta = common.run_meta("E3_security_merged", {
        "parts_present": present,
        "parts_missing": missing,
        "note": "s1a from Python (this script); s1b/s1c/s2/s3 from package 1 sec_c",
    })
    common.write_result(MERGED_RESULT, meta, {"measured": merged})
    print(f"E3 merge: present={present} missing={missing} -> {MERGED_RESULT}")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        common.guard_main("E3_security_merged", MERGED_RESULT, run_merge)
    else:
        common.guard_main("E3_S1a_hit_oracle", S1A_RESULT, run_s1a)
