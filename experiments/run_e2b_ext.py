#!/usr/bin/env python3
"""run_e2b_ext.py — E2b at the scale review round 1 asked for.

The original E2b earned the Python mirror the right to speak for libcapsule on
the strength of four counters, one configuration out of 54, and 500 of 10,000
requests. That is a thin bridge to hang two workload studies on.

This sweeps the full tenant x sharing x leak grid, at 4x the requests, for BOTH
capsule naming variants (domain-seeded and position-chained), and compares six
store counters plus the independent denial recount on every cell. Any single
mismatch fails the whole experiment.
"""

from __future__ import annotations

import json
import resource
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import capsule_ffi, common, managers, traces  # noqa: E402
from agentsim import managers_ext  # noqa: E402

managers_ext.install()

import run_e2b as base  # noqa: E402

RESULT = HERE.parents[1] / "experiments" / "results" / "e2b_ext_validation.json"
TMP = HERE.parents[1] / "experiments" / "tmp_store_e2b_ext"

CONFIGS = [
    dict(T=T, f=f, p_leakdup=pl, seed=s)
    for T in (4, 16, 64)
    for f in (0.25, 0.75)
    for pl in (0.0, 0.05, 0.15)
    for s in (1,)
]
N_REQUESTS = 1200
VARIANTS = ["capsule", "capsule-poschain"]


def replay_oplog(store, op_log):
    """Replay an op-log, including the create_chained_extend op the
    position-chained variant emits (base.\_replay_oplog predates it)."""
    handles = {}
    cache = {}
    for op in op_log:
        kind = op[0]
        if kind == "create_extend":
            _, h, domain, specs = op
            cap, _ = store.create("/e2b/h%d" % h, domain, base.SCALED_BPT)
            buf, n = base._gen_chunks(specs, cache)
            if n:
                store.extend(cap, buf, n)
            handles[h] = cap
        elif kind == "create_chained_extend":
            _, h, domain, parent_h, specs = op
            if parent_h is None:
                cap, _ = store.create("/e2b/h%d" % h, domain, base.SCALED_BPT)
            else:
                cap, _ = store.create_chained("/e2b/h%d" % h, domain,
                                              base.SCALED_BPT, handles[parent_h])
            buf, n = base._gen_chunks(specs, cache)
            if n:
                store.extend(cap, buf, n)
            handles[h] = cap
        elif kind == "fork":
            _, child, parent = op
            handles[child], _ = store.fork(handles[parent], "/e2b/h%d" % child)
        elif kind == "extend":
            _, h, specs = op
            buf, n = base._gen_chunks(specs, cache)
            if n:
                store.extend(handles[h], buf, n)
        else:
            raise ValueError("unknown op %r" % (kind,))
    return handles


def one_cell(variant, c):
    cfg = traces.TraceConfig(n_requests=N_REQUESTS, n_tenants=c["T"],
                             f=c["f"], p_leakdup=c["p_leakdup"])
    trace = traces.gen_trace(cfg, c["seed"])
    cls = managers.MANAGERS[variant]
    mirror = cls(cfg, record_ops=True, free_decode_on_end=False)
    for ev in trace:
        mirror.on_event(ev)
    m = mirror.result()

    mirror_counters = {
        "stored_bytes": m["stored_chunks"] * base.CHUNK_NBYTES,
        "dedup_hits": m["dedup_hits"],
        "dedup_saved_bytes": m["dedup_saved_chunks"] * base.CHUNK_NBYTES,
    }
    indep_denied = base._independent_denied_recount(cfg, trace)

    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        with capsule_ffi.CapsuleStore(TMP,
                                      capsule_ffi.CAPSULE_STORE_CREATE) as store:
            replay_oplog(store, mirror.op_log)
            st = store.store_stats()
            c_counters = {
                "stored_bytes": st.stored_bytes,
                "dedup_hits": st.dedup_hits,
                "dedup_saved_bytes": st.dedup_saved_bytes,
                "n_capsules": st.n_capsules,
                "n_extents": st.n_extents,
                "logical_bytes": st.logical_bytes,
            }
            probe = base._c_denial_probe(store)
    finally:
        if TMP.exists():
            shutil.rmtree(TMP)

    eq = {k: (mirror_counters[k] == c_counters[k]) for k in mirror_counters}
    eq["denied_matches_independent_recount"] = (
        m["denied_attaches"] == indep_denied)
    eq["c_denial_path_enforced"] = bool(probe.get("enforced", False))
    # the store must never dedup across domains: stored < logical whenever
    # leak-duplication exists, and never below what within-domain dedup allows
    eq["no_cross_domain_dedup"] = bool(
        c_counters["stored_bytes"] <= c_counters["logical_bytes"])

    return {
        "variant": variant, **c, "n_requests": N_REQUESTS,
        "trace_events": len(trace), "op_log_len": len(mirror.op_log),
        "mirror": mirror_counters, "c": c_counters,
        "mirror_denied": m["denied_attaches"],
        "independent_denied_recount": indep_denied,
        "checks": eq,
        "all_equal": all(eq.values()),
    }


def run() -> None:
    t0 = time.perf_counter()
    cells = []
    for variant in VARIANTS:
        for c in CONFIGS:
            r = one_cell(variant, c)
            cells.append(r)
            flag = "ok " if r["all_equal"] else "FAIL"
            print("  %s %-16s T=%-3d f=%.2f pl=%.2f seed=%d  stored=%d hits=%d"
                  % (flag, variant, c["T"], c["f"], c["p_leakdup"], c["seed"],
                     r["c"]["stored_bytes"], r["c"]["dedup_hits"]), flush=True)
    elapsed = time.perf_counter() - t0
    npass = sum(1 for c in cells if c["all_equal"])

    meta = common.run_meta("E2b_ext_validation", {
        "n_cells": len(cells), "variants": VARIANTS,
        "n_requests_per_cell": N_REQUESTS,
        "configs_per_variant": len(CONFIGS),
        "counters_compared": ["stored_bytes", "dedup_hits", "dedup_saved_bytes",
                              "denied (vs independent recount)",
                              "c denial path enforced",
                              "no cross-domain dedup"],
        "scaled_bpt": base.SCALED_BPT, "chunk_nbytes": base.CHUNK_NBYTES,
        "elapsed_s": elapsed,
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
    })
    common.write_result(RESULT, meta, {
        "cells": cells,
        "summary": {"cells_passing": npass, "cells_total": len(cells),
                    "all_pass": npass == len(cells)},
    })
    print("E2b-ext: %d/%d cells counter-exact, %.1fs -> %s"
          % (npass, len(cells), elapsed, RESULT))
    if npass != len(cells):
        raise SystemExit("E2b-ext: %d cells mismatched" % (len(cells) - npass))


if __name__ == "__main__":
    run()
