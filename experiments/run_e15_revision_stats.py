#!/usr/bin/env python3
"""Independent-process replication for sub-microsecond security-path claims."""
from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def q(xs: list[float], p: float) -> float:
    ys = sorted(xs); x = (len(ys) - 1) * p
    a = int(x); b = min(a + 1, len(ys) - 1)
    return ys[a] * (b - x) + ys[b] * (x - a)


def summary(xs: list[float]) -> dict:
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "q1": q(xs, .25), "q3": q(xs, .75), "values": xs}


def main() -> int:
    subprocess.run(["make", "all"], cwd=ROOT / "src/libcapsule", check=True,
                   stdout=subprocess.DEVNULL)
    rows = []
    with tempfile.TemporaryDirectory(prefix="caps-e15-") as td:
        t = Path(td)
        for i in range(10):
            micro = t / f"micro-{i}.json"
            sec = t / f"sec-{i}.json"
            (t / f"store-m-{i}").mkdir()
            (t / f"store-s-{i}").mkdir()
            subprocess.run([str(ROOT / "src/libcapsule/bench_micro"),
                            str(t / f"store-m-{i}"), str(micro), "quick"],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            subprocess.run([str(ROOT / "src/libcapsule/sec_c"), "s1b",
                            str(t / f"store-s-{i}"), str(sec), "quick"],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            m = json.loads(micro.read_text())["data"]["measured"]["scalar_ns"]
            s = json.loads(sec.read_text())["data"]["measured"]
            rows.append({
                "process_run": i,
                "read4k_checked_p50_ns": m["read4k_checked"]["p50"],
                "read4k_unchecked_p50_ns": m["read4k_unchecked"]["p50"],
                "checked_minus_unchecked_p50_ns":
                    m["read4k_checked"]["p50"] - m["read4k_unchecked"]["p50"],
                "durable_revoke_p50_ns": m["revoke"]["p50"],
                "forged_existing_mean_ns": s["forged_existing_id_ns"]["mean"],
                "forged_nonexistent_mean_ns": s["forged_nonexistent_id_ns"]["mean"],
                "existence_timing_mean_diff_ns": s["mean_diff_ns"],
                "both_return_eperm": s["both_return_E_PERM"],
            })
    metrics = {}
    for key in rows[0]:
        if key == "process_run" or isinstance(rows[0][key], bool):
            continue
        metrics[key] = summary([float(r[key]) for r in rows])
    result = {
        "experiment": "E15_independent_process_timing_replication",
        "independent_process_runs": len(rows), "runs": rows,
        "summary": metrics,
        "all_forged_caps_return_eperm": all(r["both_return_eperm"] for r in rows),
        "protocol": "10 fresh processes and fresh stores; quick mode within-process distributions; report cross-process median/IQR/min/max",
        "scope": "Apple M3 Ultra/macOS arm64; quiet-host but unpinned; no Linux/x86 transfer claim",
    }
    out = ROOT / "experiments/results/e15_revision_stats.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "summary": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
