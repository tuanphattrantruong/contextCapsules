#!/usr/bin/env python3
"""run_e8.py — E8: the workload study on REAL production traces.

Round-1 objection R2: the workload results came from a synthetic generator, so
the sharing structure being measured was the sharing structure we invented.

E8 replays the public traces released with Mooncake (FAST'25) through the same
seven managers E6 uses:

  conversation  12,031 requests, real arrivals, real block-level prefix hashes
  toolagent     23,608 requests, the agentic workload this paper motivates with
  synthetic      3,993 requests (Mooncake's own synthetic trace, for contrast)

Nothing about the traces is generated here: arrival times, input and output
lengths, and the prefix-block structure are the trace's own. The two modelling
choices -- which blocks count as published corpus (`corpus_min_refs`) and how
conversations map to tenants (`n_tenants`) -- are swept, and the resulting
cross-tenant duplication rate is REPORTED rather than dialled in.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import common, kvmodel, managers, mooncake  # noqa: E402
from agentsim import managers_ext  # noqa: E402

managers_ext.install()

RESULT = HERE.parents[1] / "experiments" / "results" / "e8_real_traces.json"
REAL = HERE.parents[1] / "workloads" / "real"

TRACES = [
    ("conversation", REAL / "mooncake_conversation.jsonl"),
    ("toolagent", REAL / "toolagent_trace.jsonl"),
    ("synthetic-mooncake", REAL / "synthetic_trace.jsonl"),
]
TENANTS = [4, 16, 64]
CORPUS_MIN_REFS = [10, 50]
SALTS = [1, 2, 3]                      # tenant-assignment seeds

MANAGER_SPECS = [
    ("noshare", "noshare", {}),
    ("prefix", "prefix", {}),
    ("partitioned", "partitioned", {}),
    ("safekv-fn0.05", "safekv", {"fn_rate": 0.05}),
    ("capsule-poschain", "capsule-poschain", {}),
]


def hit_rate(r):
    h = r.get("prefix_hit_tokens", 0)
    d = h + r.get("recomputed_prefix_tokens", 0)
    return (h / d) if d else 0.0


def replay(key, cfg, events, kwargs):
    cls = managers.MANAGERS[key]
    m = cls(cfg, **kwargs) if kwargs else cls(cfg)
    for ev in events:
        m.on_event(ev)
    return m.result()


def run() -> None:
    t0 = time.perf_counter()
    cells = []
    trace_stats = {}
    for tname, tpath in TRACES:
        if not tpath.exists():
            print("  skip %s (missing %s)" % (tname, tpath))
            continue
        rows = mooncake.load_rows(tpath)
        for cmr in CORPUS_MIN_REFS:
            for T in TENANTS:
                per = {lbl: [] for lbl, _, _ in MANAGER_SPECS}
                st = None
                for salt in SALTS:
                    events, st = mooncake.build_events(
                        rows, n_tenants=T, corpus_min_refs=cmr, salt=salt)
                    cfg = mooncake.TraceCfg(T, len(rows))
                    for lbl, key, kw in MANAGER_SPECS:
                        per[lbl].append(replay(key, cfg, events, kw))
                trace_stats["%s/cmr%d/T%d" % (tname, cmr, T)] = st
                for lbl, _key, _kw in MANAGER_SPECS:
                    rs = per[lbl]
                    hrs = [hit_rate(r) for r in rs]
                    cells.append({
                        "trace": tname, "corpus_min_refs": cmr, "T": T,
                        "manager": lbl, "salts": SALTS,
                        "hit_rate_mean": sum(hrs) / len(hrs),
                        "hit_rate_min": min(hrs), "hit_rate_max": max(hrs),
                        "per_salt_hit_rate": hrs,
                        "peak_resident_tokens_mean":
                            sum(r["peak_resident_tokens"] for r in rs) / len(rs),
                        "recomputed_tokens_mean":
                            sum(r["recomputed_tokens"] for r in rs) / len(rs),
                        "cross_tenant_reuse_mean":
                            sum(r.get("cross_tenant_reuse_events", 0)
                                for r in rs) / len(rs),
                        "denied_attaches_mean":
                            sum(r.get("denied_attaches", 0) for r in rs) / len(rs),
                        "leaked_private_blocks_mean":
                            sum(r.get("leaked_private_blocks_in_shared_ns", 0)
                                for r in rs) / len(rs),
                    })
                print("  %-20s cmr=%-3d T=%-3d  " % (tname, cmr, T)
                      + " ".join("%s=%.4f" % (lbl.split("-")[0][:4],
                                              [c for c in cells
                                               if c["trace"] == tname
                                               and c["corpus_min_refs"] == cmr
                                               and c["T"] == T
                                               and c["manager"] == lbl][0]["hit_rate_mean"])
                                 for lbl, _, _ in MANAGER_SPECS), flush=True)

    # --- head-to-head: capsules vs each alternative, paired by salt ---------
    idx = {(c["trace"], c["corpus_min_refs"], c["T"], c["manager"]): c
           for c in cells}
    comps = []
    for (tname, cmr, T, mgr), c in list(idx.items()):
        if mgr != "capsule-poschain":
            continue
        for other in ("prefix", "partitioned", "safekv-fn0.05"):
            o = idx.get((tname, cmr, T, other))
            if not o:
                continue
            d = [a - b for a, b in zip(c["per_salt_hit_rate"],
                                       o["per_salt_hit_rate"])]
            comps.append({
                "trace": tname, "corpus_min_refs": cmr, "T": T,
                "capsule_vs": other,
                "delta_pp_mean": 100.0 * sum(d) / len(d),
                "delta_pp_min": 100.0 * min(d),
                "delta_pp_max": 100.0 * max(d),
                "wins": sum(1 for x in d if x > 1e-12),
                "ties": sum(1 for x in d if abs(x) <= 1e-12),
                "losses": sum(1 for x in d if x < -1e-12),
                "n": len(d),
            })

    elapsed = time.perf_counter() - t0
    meta = common.run_meta("E8_real_traces", {
        "traces": [t for t, p in TRACES if p.exists()],
        "source": ("Mooncake FAST'25 public traces, "
                   "github.com/kvcache-ai/Mooncake/FAST25-release/traces"),
        "tenants": TENANTS, "corpus_min_refs": CORPUS_MIN_REFS,
        "tenant_salts": SALTS,
        "managers": [lbl for lbl, _, _ in MANAGER_SPECS],
        "n_cells": len(cells),
        "kv_geometry": {"REAL_BPT": kvmodel.REAL_BPT},
        "elapsed_s": elapsed,
        "note": ("arrival times, input/output lengths and prefix-block structure "
                 "are the trace's own; tenant assignment and the corpus "
                 "threshold are swept modelling choices"),
    })
    common.write_result(RESULT, meta, {
        "cells": cells, "comparisons": comps, "trace_stats": trace_stats})
    print("E8 complete: %d cells, %.1fs -> %s" % (len(cells), elapsed, RESULT))


if __name__ == "__main__":
    run()
