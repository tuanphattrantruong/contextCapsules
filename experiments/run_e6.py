#!/usr/bin/env python3
"""run_e6.py — E6: the workload study with the baselines review round 1 demanded.

E2 compared capsules against exactly one alternative, a flat unprotected global
prefix cache, under two modeling choices that both favoured capsules. E6 fixes
all three problems at once:

  * adds `partitioned` -- per-tenant cache partitioning, the obvious deployed
    defense, which is as safe as capsules and pays for it in lost sharing;
  * adds `safekv` -- SafeKV-class selective isolation, swept over the
    sensitivity detector's false-negative rate;
  * adds `capsule-poschain` -- capsules with position-chained private naming,
    which E5e showed is the only sound choice for a dense transformer, so the
    hit-rate comparison becomes apples-to-apples;
  * adds `prefix-bounded` -- the baseline with capsule-equivalent lifecycle
    freeing, so peak-footprint numbers compare like with like;
  * raises seeds from 3 to 10 so single-digit-percent effects get a real
    confidence interval instead of a sign count.

Every number here is a MEASURED counter from a deterministic replay; byte
figures are modeled (x REAL_BPT = 131072), a constant E5a now confirms by
measurement on Llama-3.1-8B-Instruct.
"""

from __future__ import annotations

import json
import statistics as stats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import common, kvmodel, managers, metrics, traces  # noqa: E402
from agentsim import managers_ext  # noqa: E402

managers_ext.install()

RESULT = HERE.parents[1] / "experiments" / "results" / "e6_workload_ext.json"

TENANTS = [4, 16, 64]
FVALS = [0.25, 0.75]
LEAKDUP = [0.0, 0.05, 0.15]
SEEDS = list(range(1, 11))          # 10 seeds (was 3)
N_REQUESTS = 10000

# manager spec: (label, class-key, kwargs)
MANAGER_SPECS = [
    ("noshare", "noshare", {}),
    ("prefix", "prefix", {}),
    ("prefix-bounded", "prefix-bounded", {}),
    ("partitioned", "partitioned", {}),
    ("safekv-fn0.00", "safekv", {"fn_rate": 0.0}),
    ("safekv-fn0.05", "safekv", {"fn_rate": 0.05}),
    ("safekv-fn0.20", "safekv", {"fn_rate": 0.20}),
    ("capsule", "capsule", {}),
    ("capsule-poschain", "capsule-poschain", {}),
]


def build(key, cfg, kwargs):
    cls = managers.MANAGERS[key]
    return cls(cfg, **kwargs) if kwargs else cls(cfg)


def replay(key, cfg, trace, kwargs):
    m = build(key, cfg, kwargs)
    for ev in trace:
        m.on_event(ev)
    return m.result()


def hit_rate(r):
    h = r.get("prefix_hit_tokens", 0)
    d = h + r.get("recomputed_prefix_tokens", 0)
    return (h / d) if d else 0.0


def ci95(xs):
    """Mean and half-width of a 95% CI (t-approx via 1.96 for n>=10)."""
    if len(xs) < 2:
        return (float(xs[0]) if xs else 0.0), 0.0
    m = stats.mean(xs)
    sd = stats.stdev(xs)
    return m, 1.96 * sd / (len(xs) ** 0.5)


def run() -> None:
    t0 = time.perf_counter()
    cells = []
    n_traces = 0
    for T in TENANTS:
        for f in FVALS:
            for pl in LEAKDUP:
                cfg = traces.TraceConfig(n_requests=N_REQUESTS, n_tenants=T,
                                         f=f, p_leakdup=pl)
                per = {lbl: [] for lbl, _, _ in MANAGER_SPECS}
                trace_events = None
                for seed in SEEDS:
                    tr = traces.gen_trace(cfg, seed)
                    trace_events = len(tr)
                    n_traces += 1
                    for lbl, key, kw in MANAGER_SPECS:
                        per[lbl].append(replay(key, cfg, tr, kw))
                for lbl, key, kw in MANAGER_SPECS:
                    rs = per[lbl]
                    hrs = [hit_rate(r) for r in rs]
                    peaks = [r["peak_resident_tokens"] for r in rs]
                    recs = [r["recomputed_tokens"] for r in rs]
                    xten = [r.get("cross_tenant_reuse_events", 0) for r in rs]
                    den = [r.get("denied_attaches", 0) for r in rs]
                    hm, hc = ci95(hrs)
                    pm, pc = ci95([float(x) for x in peaks])
                    rm, rc = ci95([float(x) for x in recs])
                    cells.append({
                        "manager": lbl, "impl": key, "kwargs": kw,
                        "T": T, "f": f, "p_leakdup": pl,
                        "n_requests": N_REQUESTS, "trace_events": trace_events,
                        "seeds": SEEDS,
                        "hit_rate": {"mean": hm, "ci95": hc,
                                     "min": min(hrs), "max": max(hrs)},
                        "peak_resident_tokens": {"mean": pm, "ci95": pc},
                        "recomputed_tokens": {"mean": rm, "ci95": rc},
                        "cross_tenant_reuse_events": {
                            "mean": sum(xten) / len(xten), "max": max(xten)},
                        "denied_attaches": {
                            "mean": sum(den) / len(den), "max": max(den)},
                        "leaked_private_blocks": {
                            "mean": sum(r.get("leaked_private_blocks_in_shared_ns", 0)
                                        for r in rs) / len(rs)},
                        "per_seed_hit_rate": hrs,
                        "per_seed_peak_tokens": peaks,
                        "per_seed_recompute_tokens": recs,
                    })
    elapsed = time.perf_counter() - t0

    # ---- derived head-to-head comparisons the paper will quote --------------
    idx = {}
    for c in cells:
        idx[(c["manager"], c["T"], c["f"], c["p_leakdup"])] = c

    comps = []
    for T in TENANTS:
        for f in FVALS:
            for pl in LEAKDUP:
                cap = idx[("capsule-poschain", T, f, pl)]
                for other in ("prefix", "partitioned", "safekv-fn0.05"):
                    o = idx[(other, T, f, pl)]
                    # paired per-seed differences (same traces) -> paired CI
                    d = [a - b for a, b in zip(cap["per_seed_hit_rate"],
                                               o["per_seed_hit_rate"])]
                    dm, dc = ci95(d)
                    comps.append({
                        "T": T, "f": f, "p_leakdup": pl,
                        "capsule_poschain_vs": other,
                        "hit_rate_delta_mean_pp": dm * 100.0,
                        "hit_rate_delta_ci95_pp": dc * 100.0,
                        "capsule_wins_seeds": sum(1 for x in d if x > 0),
                        "ties": sum(1 for x in d if x == 0),
                        "losses": sum(1 for x in d if x < 0),
                        "n_seeds": len(d),
                    })

    summary = {
        "isolation_cost_of_partitioning": {
            "note": ("hit-rate points capsules retain over per-tenant "
                     "partitioning at equal safety (both leak zero)"),
            "per_config": [
                {"T": c["T"], "f": c["f"], "p_leakdup": c["p_leakdup"],
                 "delta_pp": c["hit_rate_delta_mean_pp"],
                 "ci95_pp": c["hit_rate_delta_ci95_pp"],
                 "wins": c["capsule_wins_seeds"], "n": c["n_seeds"]}
                for c in comps if c["capsule_poschain_vs"] == "partitioned"
            ],
        },
        "vs_unprotected_prefix_fair_naming": {
            "note": ("capsule-poschain vs the vLLM-faithful cache, both "
                     "position-chained: the apples-to-apples no-loss test"),
            "per_config": [
                {"T": c["T"], "f": c["f"], "p_leakdup": c["p_leakdup"],
                 "delta_pp": c["hit_rate_delta_mean_pp"],
                 "ci95_pp": c["hit_rate_delta_ci95_pp"],
                 "wins": c["capsule_wins_seeds"], "losses": c["losses"],
                 "n": c["n_seeds"]}
                for c in comps if c["capsule_poschain_vs"] == "prefix"
            ],
        },
        "leak_surface": {
            m: {"cross_tenant_reuse_mean": stats.mean(
                    [c["cross_tenant_reuse_events"]["mean"] for c in cells
                     if c["manager"] == m]),
                "cross_tenant_reuse_max": max(
                    [c["cross_tenant_reuse_events"]["max"] for c in cells
                     if c["manager"] == m]),
                "leaked_private_blocks_mean": stats.mean(
                    [c["leaked_private_blocks"]["mean"] for c in cells
                     if c["manager"] == m])}
            for m in [lbl for lbl, _, _ in MANAGER_SPECS]
        },
    }

    meta = common.run_meta("E6_workload_ext", {
        "grid": {"managers": [lbl for lbl, _, _ in MANAGER_SPECS],
                 "tenants": TENANTS, "f": FVALS, "p_leakdup": LEAKDUP,
                 "seeds": SEEDS, "n_requests": N_REQUESTS,
                 "n_cells": len(cells), "n_traces_generated": n_traces},
        "kv_geometry": {"REAL_BPT": kvmodel.REAL_BPT,
                        "label": ("modeled bytes use REAL_BPT=131072; "
                                  "E5a MEASURES this constant on "
                                  "Llama-3.1-8B-Instruct")},
        "elapsed_s": elapsed,
    })
    common.write_result(RESULT, meta,
                        {"cells": cells, "comparisons": comps,
                         "summary": summary})
    print("E6 complete: %d cells, %d traces, %.1fs -> %s"
          % (len(cells), n_traces, elapsed, RESULT))


if __name__ == "__main__":
    run()
