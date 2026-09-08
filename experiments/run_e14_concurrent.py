#!/usr/bin/env python3
"""Run the four-stream E14 request loop in a fresh store."""
from __future__ import annotations
import argparse, json, subprocess, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "workloads/gguf/qwen2.5-0.5b-instruct-q8_0.gguf"))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--out", default=str(ROOT / "experiments/results/e14_concurrent.json"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="capsule-e14-") as td:
        cmd = [str(ROOT / "src/llamacap/llamacap"), "requests-mt", a.model,
               str(Path(td) / "store"), str(a.ngl), str(a.tokens),
               str(a.streams), str(a.reps)]
        p = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads(p.stdout.strip().splitlines()[-1])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
