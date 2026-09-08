#!/usr/bin/env python3
"""run_e4.py — lineage-aware vs LRU eviction on agent-tree traces (claim C10).

Trace: fork-heavy (p_fork=0.6, kids<=3, depth 3), T=16, f=0.5, p_leakdup=0,
10k requests. For each seed the no-eviction CapsuleManager peak resident is
measured first; budgets {10%,25%,50%} of THAT run's own peak are then enforced
under policies {LRU, LINEAGE}. 18 cells (3 budgets x 2 policies x 3 seeds).

Metric: recomputed_tokens (MEASURED counter; eviction of a shared run forces its
next reference to recompute) plus modeled bytes (x REAL_BPT). Lineage-aware
eviction protects widely-referenced prefixes (score = age/(1+log2(1+refcount)),
evict highest), so at equal budget it should recompute fewer tokens than LRU.
Budget < 5 min.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import common, kvmodel, managers, metrics, traces  # noqa: E402

RESULT = HERE.parents[1] / "experiments" / "results" / "e4_eviction.json"

N_REQUESTS = 10000
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BUDGETS = [0.10, 0.25, 0.50]
POLICIES = ["lru", "lineage"]
EVICT_EVERY = 50   # run the eviction pass every N events (throttled for speed)


def _e4_config() -> traces.TraceConfig:
    return traces.TraceConfig(n_requests=N_REQUESTS, n_tenants=16, f=0.5,
                              p_leakdup=0.0, p_fork=0.6, max_kids=3, max_depth=3)


def _run_no_evict(cfg, trace) -> dict:
    m = managers.CapsuleManager(cfg)
    for ev in trace:
        m.on_event(ev)
    r = m.result()
    r["_peak_blocks"] = m.peak_blocks
    return r


def _run_with_evict(cfg, trace, target_blocks: int, policy: str) -> dict:
    m = managers.CapsuleManager(cfg)
    for idx, ev in enumerate(trace):
        m.on_event(ev)
        if (idx % EVICT_EVERY == 0) and m.resident_blocks > target_blocks:
            m.evict_to(target_blocks, policy)
    # final pass
    if m.resident_blocks > target_blocks:
        m.evict_to(target_blocks, policy)
    r = m.result()
    r["_evicted_chunks"] = m.evicted_chunks
    return r


def run() -> None:
    t_start = time.perf_counter()
    cfg = _e4_config()

    # per-seed baselines and traces
    baselines = {}
    trace_cache = {}
    for seed in SEEDS:
        tr = traces.gen_trace(cfg, seed)
        trace_cache[seed] = tr
        baselines[seed] = _run_no_evict(cfg, tr)

    cells = []
    for budget in BUDGETS:
        for policy in POLICIES:
            per_seed = []
            for seed in SEEDS:
                tr = trace_cache[seed]
                peak_blocks = baselines[seed]["_peak_blocks"]
                target_blocks = int(budget * peak_blocks)
                r = _run_with_evict(cfg, tr, target_blocks, policy)
                base_rec = baselines[seed]["recomputed_prefix_tokens"]
                r["_eviction_induced_recompute_tokens"] = (
                    r["recomputed_prefix_tokens"] - base_rec)
                r["_target_blocks"] = target_blocks
                r["_baseline_peak_blocks"] = peak_blocks
                per_seed.append(r)
            # aggregate across seeds
            def agg(key):
                return metrics._mean_std([float(r[key]) for r in per_seed])
            cells.append({
                "budget_frac": budget,
                "policy": policy,
                "seeds": SEEDS,
                "measured": {
                    "recomputed_tokens": agg("recomputed_tokens"),
                    "recomputed_prefix_tokens": agg("recomputed_prefix_tokens"),
                    "eviction_induced_recompute_tokens": agg("_eviction_induced_recompute_tokens"),
                    "evicted_chunks": agg("_evicted_chunks"),
                    "peak_resident_tokens": agg("peak_resident_tokens"),
                    "final_resident_tokens": agg("final_resident_tokens"),
                    "prefix_hit_tokens": agg("prefix_hit_tokens"),
                    "target_blocks": agg("_target_blocks"),
                },
                "modeled": {
                    "recomputed_bytes": metrics._mean_std([
                        float(kvmodel.tokens_to_bytes(int(r["recomputed_tokens"]),
                                                      kvmodel.REAL_BPT))
                        for r in per_seed]),
                    "eviction_induced_recompute_bytes": metrics._mean_std([
                        float(kvmodel.tokens_to_bytes(
                            int(r["_eviction_induced_recompute_tokens"]),
                            kvmodel.REAL_BPT))
                        for r in per_seed]),
                },
                "per_seed": per_seed,
            })

    elapsed = time.perf_counter() - t_start

    # head-to-head lineage vs lru at each budget (eviction-induced recompute).
    # Honesty: a tie is reported as a tie ("tie": true, strictly_reduces false),
    # never as a reduction; per-seed win counts expose sign consistency.
    head_to_head = []
    for budget in BUDGETS:
        lru = next(c for c in cells if c["budget_frac"] == budget and c["policy"] == "lru")
        lin = next(c for c in cells if c["budget_frac"] == budget and c["policy"] == "lineage")
        lru_rec = lru["measured"]["eviction_induced_recompute_tokens"]["mean"]
        lin_rec = lin["measured"]["eviction_induced_recompute_tokens"]["mean"]
        lru_seed = [r["_eviction_induced_recompute_tokens"] for r in lru["per_seed"]]
        lin_seed = [r["_eviction_induced_recompute_tokens"] for r in lin["per_seed"]]
        wins = sum(1 for a, b in zip(lin_seed, lru_seed) if a < b)
        ties = sum(1 for a, b in zip(lin_seed, lru_seed) if a == b)
        head_to_head.append({
            "budget_frac": budget,
            "lru_eviction_recompute_tokens_mean": lru_rec,
            "lineage_eviction_recompute_tokens_mean": lin_rec,
            "lineage_strictly_reduces_recompute": lin_rec < lru_rec,
            "tie": lin_rec == lru_rec,
            "reduction_frac": (1.0 - lin_rec / lru_rec) if lru_rec > 0 else None,
            "per_seed_lineage_wins": wins,
            "per_seed_ties": ties,
            "n_seeds": len(lru_seed),
        })

    meta = common.run_meta("E4_eviction", {
        "trace": {"p_fork": 0.6, "max_kids": 3, "max_depth": 3, "T": 16, "f": 0.5,
                  "p_leakdup": 0.0, "n_requests": N_REQUESTS},
        "budgets_frac_of_own_peak": BUDGETS,
        "policies": POLICIES, "seeds": SEEDS,
        "evict_every_events": EVICT_EVERY,
        "elapsed_s": elapsed,
        "kv_geometry": {"REAL_BPT": kvmodel.REAL_BPT},
    })
    data = {
        "measured": {"cells": cells, "head_to_head": head_to_head,
                     "baseline_peaks_blocks": {str(s): baselines[s]["_peak_blocks"]
                                               for s in SEEDS}},
        "modeled": {"note": "recomputed_bytes = recomputed_tokens x REAL_BPT"},
    }
    common.write_result(RESULT, meta, data)
    print(f"E4 complete: {len(cells)} cells, {elapsed:.1f}s -> {RESULT}")
    for h in head_to_head:
        print(f"  budget={h['budget_frac']:.0%}  lru_recompute={h['lru_eviction_recompute_tokens_mean']:.0f}  "
              f"lineage_recompute={h['lineage_eviction_recompute_tokens_mean']:.0f}  "
              f"lineage_better={h['lineage_strictly_reduces_recompute']}")


if __name__ == "__main__":
    common.guard_main("E4_eviction", RESULT, run)
