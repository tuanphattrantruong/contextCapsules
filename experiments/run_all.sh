#!/usr/bin/env bash
# run_all.sh — orchestrate the full Context Capsules experiment suite.
#
# Order (DESIGN.md 7): build+test libcapsule -> E1 (C micro) -> E2 (workload)
# -> E2b (mirror-vs-C) -> E3 C studies -> E3 S1a (Python) + merge -> E4 (eviction).
# Total wall-clock budget < ~25 min. Every results file is written under
# experiments/results/ with run metadata and measured/modeled labels.
#
# Honesty: a failing step aborts the run (set -e) rather than producing partial or
# fabricated results. Each Python experiment additionally writes a
# {"status":"failed",...} record on an internal error.
set -euo pipefail

# repo root = parent of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "=== Context Capsules full run ==="
echo "repo root: $ROOT"
date -u
uname -a
echo

echo "=== [1/8] build libcapsule + run test suites ==="
make -C src/libcapsule
make -C src/libcapsule test
python3 tests/run_tests.py

echo "=== [2/8] generate exemplar traces (workloads/) ==="
( cd "$ROOT/src" && python3 -m agentsim.traces )

echo "=== [3/8] E1 — C microbenchmarks ==="
bash experiments/run_e1.sh

echo "=== [4/8] E2 — workload study (162 runs) ==="
python3 experiments/run_e2.py

echo "=== [5/8] E2b — mirror vs libcapsule (exact) ==="
python3 experiments/run_e2b.py

echo "=== [6/8] E3 — C security studies (S1b/S1c/S2/S3) ==="
bash experiments/run_e3_c.sh

echo "=== [7/8] E3 — S1a hit-oracle (Python) + merge ==="
python3 experiments/run_e3_s1.py
python3 experiments/run_e3_s1.py --merge

echo "=== [8/8] E4 — lineage vs LRU eviction ==="
python3 experiments/run_e4.py

echo
echo "=== results ==="
ls -la experiments/results/
echo "=== done ==="
