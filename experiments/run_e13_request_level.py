#!/usr/bin/env python3
"""Run E13 in a fresh temporary capsule store and preserve only the JSON."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(
        ROOT / "workloads/gguf/qwen2.5-0.5b-instruct-q8_0.gguf"))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--out", default=str(
        ROOT / "experiments/results/e13_request_level.json"))
    args = ap.parse_args()
    engine = ROOT / "src/llamacap/llamacap"
    with tempfile.TemporaryDirectory(prefix="capsule-e13-") as td:
        cmd = [str(engine), "requests", args.model, str(Path(td) / "store"),
               str(args.ngl), str(args.tokens), str(args.reps)]
        proc = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True,
                              text=True)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
