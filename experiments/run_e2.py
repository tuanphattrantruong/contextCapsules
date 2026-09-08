#!/usr/bin/env python3
"""run_e2.py — E2 workload study (DESIGN.md 6 / claims C4, C5).

Grid: managers{noshare, prefix, capsule} x T{4,16,64} x f{0.25,0.75}
      x p_leakdup{0,0.05,0.15} x seeds{1,2,3} = 162 runs x 10k requests,
unbounded memory. Per-cell mean +/- std across seeds -> e2_workload.json.

All token counts are MEASURED counters from real replays on this machine; byte
figures are MODELED (x REAL_BPT). Budget < 10 min (measured ~ tens of seconds).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import common, kvmodel, managers, metrics, traces  # noqa: E402

RESULT = HERE.parents[1] / "experiments" / "results" / "e2_workload.json"

# grid axes
TENANTS = [4, 16, 64]
FVALS = [0.25, 0.75]
LEAKDUP = [0.0, 0.05, 0.15]
SEEDS = [1, 2, 3]
MANAGERS = ["noshare", "prefix", "capsule"]
N_REQUESTS = 10000


def run() -> None:
    t_start = time.perf_counter()
    cells = []
    n_traces = 0
    for T in TENANTS:
        for f in FVALS:
            for pl in LEAKDUP:
                cfg = traces.TraceConfig(n_requests=N_REQUESTS, n_tenants=T, f=f,
                                         p_leakdup=pl)
                # generate one trace per seed, run all managers on each
                per_seed_results = {m: [] for m in MANAGERS}
                trace_events = None
                for seed in SEEDS:
                    tr = traces.gen_trace(cfg, seed)
                    trace_events = len(tr)
                    n_traces += 1
                    for m in MANAGERS:
                        res = managers.run_trace(m, cfg, tr)
                        per_seed_results[m].append(res)
                for m in MANAGERS:
                    agg = metrics.aggregate_seeds(per_seed_results[m])
                    cells.append({
                        "manager": m,
                        "T": T, "f": f, "p_leakdup": pl,
                        "n_requests": N_REQUESTS,
                        "trace_events": trace_events,
                        "seeds": SEEDS,
                        "aggregate": agg,
                        "per_seed": [
                            {k: v for k, v in r.items()}
                            for r in per_seed_results[m]
                        ],
                    })
    elapsed = time.perf_counter() - t_start

    meta = common.run_meta("E2_workload", {
        "grid": {
            "managers": MANAGERS, "tenants": TENANTS, "f": FVALS,
            "p_leakdup": LEAKDUP, "seeds": SEEDS, "n_requests": N_REQUESTS,
            "n_cells": len(cells), "n_traces_generated": n_traces,
        },
        "grid_unchanged": True,
        "kv_geometry": {"REAL_BPT": kvmodel.REAL_BPT,
                        "label": "modeled bytes use REAL_BPT=131072 (Llama-3-8B GQA)"},
        "elapsed_s": elapsed,
        "budget_note": "target < 10 min; full 162-run grid at 10k requests",
    })
    common.write_result(RESULT, meta, {"cells": cells})
    print(f"E2 complete: {len(cells)} cells, {n_traces} traces, {elapsed:.1f}s -> {RESULT}")


if __name__ == "__main__":
    common.guard_main("E2_workload", RESULT, run)
