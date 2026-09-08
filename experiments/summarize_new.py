#!/usr/bin/env python3
"""summarize_new.py — consolidate the round-2 experiments into headline
numbers and the CSVs the figures are plotted from.

Reads only results/*.json produced by run_e5/e6/e7/e2b_ext; writes
data/*.csv and prints a claim-by-claim digest.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def section(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


out_lines = []


def emit(k, v):
    print("  %-52s %s" % (k, v))
    out_lines.append((k, v))


# ---------------------------------------------------------------- E5
e5 = load("e5_realmodel.json")
if e5:
    section("E5 — real model (Llama-3.1-8B-Instruct)")
    m = e5["meta"]
    emit("machine", "%s, %d cores, %.0f GiB"
         % (m["hw_model"], m["ncpu"], m["memsize_bytes"] / 2**30))
    g = e5["e5a_geometry"]
    emit("measured bytes/token", g["measured_bytes_per_token_from_packed_buffer"])
    emit("matches the previously-modeled 131072", g["matches_paper_constant"])
    emit("chunk bytes (16 tokens)",
         g["from_config"]["bytes_per_chunk_16tok"])

    rows = e5["e5c_ttft"]
    with open(DATA / "e5_ttft.csv", "w") as f:
        f.write("ntok,kv_mib,prefill_ms,restore_ms,attach_ms,unpack_ms,speedup,pack_ms,store_ms\n")
        for r in rows:
            f.write("%d,%.1f,%.2f,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f\n"
                    % (r["n_tokens"], r["kv_bytes"] / 2**20,
                       r["prefill_ms"]["p50"], r["restore_ms"]["p50"],
                       r["attach_ms_p50"], r["unpack_ms_p50"],
                       r["speedup_p50"], r["pack_ms"], r["store_put_ms"]))
    emit("TTFT rows -> data/e5_ttft.csv", len(rows))
    for r in rows:
        emit("  n=%d  prefill %.0f ms  restore %.0f ms" %
             (r["n_tokens"], r["prefill_ms"]["p50"], r["restore_ms"]["p50"]),
             "speedup %.1fx  exact=%s"
             % (r["speedup_p50"], r["restored_cache_bitwise_equal"]))
    emit("all restores bit-exact",
         all(r["restored_cache_bitwise_equal"] for r in rows))
    emit("speedup range", "%.1f-%.1fx"
         % (min(r["speedup_p50"] for r in rows),
            max(r["speedup_p50"] for r in rows)))

    d = e5["e5d_dedup"]
    emit("real-KV dedup, shared domain (stored/logical)",
         "%.3f" % d["shared_domain"]["dedup_ratio_vs_logical"])
    emit("real-KV dedup, per-tenant domains (stored/logical)",
         "%.3f" % d["per_tenant_domains"]["dedup_ratio_vs_logical"])
    emit("cross-domain dedup occurred", d["cross_domain_dedup_occurred"])
    emit("real prefix KV is suffix-independent",
         d["real_prefix_kv_is_suffix_independent"])

    e = e5["e5e_positiondep"]
    emit("position-independence: chunk-hash match frac",
         "%.4f" % e["mean_chunk_hash_match_frac"])
    emit("position-independence: mean relative L2",
         "%.3f" % e["mean_rel_l2"])

e5b = load("e5b_fidelity_cpu.json")
if e5b:
    r = e5b["e5b_fidelity"]
    section("E5b — cross-process fidelity (CPU, deterministic backend)")
    emit("device", r["device"])
    emit("prompt tokens / KV bytes",
         "%d / %.1f MiB" % (r["n_prompt_tokens"], r["kv_bytes"] / 2**20))
    emit("parent pid != restorer pid",
         r["parent_pid"] != r["restorer_pid"])
    emit("KV bytes bitwise identical across process",
         r["kv_bytes_bitwise_identical_across_process"])
    emit("chain hash matches", r["restore_chain_hash_matches"])
    emit("generated token ids identical", r["generated_ids_identical"])
    emit("logit sums identical", r["logit_sums_identical"])
    emit("revoke after restore denies", r["revoked_after_restore_denies"])

e5f = load("e5f_determinism_mps.json")
if e5f:
    r = e5f["e5f_determinism"]
    section("E5f — backend determinism probe (MPS)")
    emit("reps x new tokens", "%d x %d" % (r["reps"], r["n_new"]))
    emit("live decode self-consistent", r["live_decode_self_consistent"])
    emit("restored decode self-consistent", r["restored_decode_self_consistent"])

# ---------------------------------------------------------------- E6
e6 = load("e6_workload_ext.json")
if e6:
    d = e6["data"]
    cells, s = d["cells"], d["summary"]
    section("E6 — workload study with the missing baselines (10 seeds)")

    leak = s["leak_surface"]
    with open(DATA / "e6_leak.csv", "w") as f:
        f.write("idx,manager,xten_mean,xten_max,leaked_blocks\n")
        for i, (mgr, v) in enumerate(leak.items()):
            f.write("%d,%s,%.3f,%d,%.1f\n"
                    % (i, mgr, v["cross_tenant_reuse_mean"],
                       v["cross_tenant_reuse_max"],
                       v["leaked_private_blocks_mean"]))
    for mgr, v in leak.items():
        emit("leak surface: %s" % mgr,
             "xten mean %.2f max %d | leaked blocks %.0f"
             % (v["cross_tenant_reuse_mean"], v["cross_tenant_reuse_max"],
                v["leaked_private_blocks_mean"]))

    part = s["isolation_cost_of_partitioning"]["per_config"]
    with open(DATA / "e6_vs_partitioned.csv", "w") as f:
        f.write("x,T,f,p,delta_pp,ci_pp\n")
        for i, r in enumerate([q for q in part if q["p_leakdup"] == 0.05]):
            f.write("%d,%d,%.2f,%.2f,%.3f,%.3f\n"
                    % (i, r["T"], r["f"], r["p_leakdup"], r["delta_pp"],
                       r["ci95_pp"]))
    emit("vs partitioned: hit-rate advantage range",
         "+%.2f to +%.2f pp"
         % (min(r["delta_pp"] for r in part), max(r["delta_pp"] for r in part)))
    emit("vs partitioned: capsule wins",
         "%d/%d config-seed pairs"
         % (sum(r["wins"] for r in part), sum(r["n"] for r in part)))

    fair = s["vs_unprotected_prefix_fair_naming"]["per_config"]
    with open(DATA / "e6_vs_prefix.csv", "w") as f:
        f.write("x,T,f,p,delta_pp,ci_pp\n")
        for i, r in enumerate(fair):
            f.write("%d,%d,%.2f,%.2f,%.4f,%.4f\n"
                    % (i, r["T"], r["f"], r["p_leakdup"], r["delta_pp"],
                       r["ci95_pp"]))
    emit("vs unprotected prefix (fair naming): worst loss",
         "%.3f pp" % min(r["delta_pp"] for r in fair))
    emit("vs unprotected prefix at p_leakdup=0",
         "max |delta| %.4f pp"
         % max(abs(r["delta_pp"]) for r in fair if r["p_leakdup"] == 0.0))

    # hit rate + footprint table for the main figure
    want = ["prefix", "partitioned", "safekv-fn0.05", "capsule-poschain"]
    with open(DATA / "e6_hitrate.csv", "w") as f:
        f.write("x,label," + ",".join(w.replace("-", "_").replace(".", "")
                                      for w in want) + "\n")
        i = 0
        for T in (4, 16, 64):
            for fv in (0.25, 0.75):
                vals = []
                for w in want:
                    c = next(c for c in cells if c["manager"] == w
                             and c["T"] == T and c["f"] == fv
                             and c["p_leakdup"] == 0.05)
                    vals.append("%.4f" % c["hit_rate"]["mean"])
                f.write("%d,T%d/f%.2f,%s\n" % (i, T, fv, ",".join(vals)))
                i += 1
    emit("hit-rate CSV -> data/e6_hitrate.csv", "written")

# ---------------------------------------------------------------- E7
sh = load("e7_concurrency_sharded.json")
si = load("e7_concurrency_single.json")
if sh and si:
    section("E7 — concurrency (32-core M3 Ultra)")
    a_sh = {r["threads"]: r for r in sh["e7a_validate_stat"]}
    a_si = {r["threads"]: r for r in si["e7a_validate_stat"]}
    with open(DATA / "e7_validate.csv", "w") as f:
        f.write("threads,sharded_kops,single_kops,sharded_p50ns,single_p50ns\n")
        for t in sorted(a_sh):
            f.write("%d,%.1f,%.1f,%.1f,%.1f\n"
                    % (t, a_sh[t]["throughput_kops_s"],
                       a_si[t]["throughput_kops_s"],
                       a_sh[t]["p50_ns"], a_si[t]["p50_ns"]))
            emit("validate t=%d" % t,
                 "sharded %8.1f kops/s (p50 %6.1f ns) | single %8.1f kops/s (p50 %7.1f ns)"
                 % (a_sh[t]["throughput_kops_s"], a_sh[t]["p50_ns"],
                    a_si[t]["throughput_kops_s"], a_si[t]["p50_ns"]))
    tmax = max(a_sh)
    emit("peak validation throughput (sharded)",
         "%.1f Mops/s at %d threads" % (a_sh[tmax]["throughput_kops_s"] / 1000, tmax))
    emit("scaling 1 -> %d threads" % tmax,
         "%.1fx" % (a_sh[tmax]["throughput_kops_s"] / a_sh[1]["throughput_kops_s"]))
    emit("sharded advantage at %d threads" % tmax,
         "%.1fx" % (a_sh[tmax]["throughput_kops_s"] / a_si[tmax]["throughput_kops_s"]))

    ex = {r["threads"]: r for r in sh["e7c_extend_contended"]}
    with open(DATA / "e7_extend.csv", "w") as f:
        f.write("threads,kops,p50us\n")
        for t in sorted(ex):
            f.write("%d,%.3f,%.1f\n"
                    % (t, ex[t]["throughput_kops_s"], ex[t]["p50_ns"] / 1000))
    emit("extend throughput (serialized)",
         "%.2f -> %.2f kops/s across 1..%d threads"
         % (ex[1]["throughput_kops_s"], ex[tmax]["throughput_kops_s"], tmax))

    rv = {r["threads"]: r for r in sh["e7d_revoke_under_load"]}
    with open(DATA / "e7_revoke.csv", "w") as f:
        f.write("threads,p50us,p95us\n")
        for t in sorted(rv):
            f.write("%d,%.2f,%.2f\n"
                    % (t, rv[t]["p50_ns"] / 1000, rv[t]["p95_ns"] / 1000))
    emit("revocation p50 under load",
         "%.1f-%.1f us across 1..%d noise threads"
         % (min(r["p50_ns"] for r in rv.values()) / 1000,
            max(r["p50_ns"] for r in rv.values()) / 1000, tmax))

# ---------------------------------------------------------------- E2b ext
e2b = load("e2b_ext_validation.json")
if e2b:
    d = e2b["data"]
    section("E2b-ext — mirror vs libcapsule at grid scale")
    emit("cells counter-exact",
         "%d/%d" % (d["summary"]["cells_passing"], d["summary"]["cells_total"]))
    emit("requests per cell", e2b["meta"]["n_requests_per_cell"])
    emit("counters compared", len(e2b["meta"]["counters_compared"]))

print("\nCSV files in", DATA)
