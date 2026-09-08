#!/usr/bin/env bash
# run_e1.sh — E1 microbenchmarks (package 1 driver bench_micro).
#
# Produces experiments/results/e1_micro.json (claims C1/C2/C3). All numbers are
# measured on this machine by bench_micro (mach_absolute_time / proc_pid_rusage);
# the JSON carries literal `date -u` / `uname -a` in its meta block.
#
# Mode: E1_MODE=full (default) runs the full DESIGN 6 size sweep (up to 512 MiB,
# < 5 min); E1_MODE=quick runs a reduced sweep for a fast smoke check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODE="${E1_MODE:-full}"
BENCH="$ROOT/src/libcapsule/bench_micro"
STORE="$ROOT/experiments/tmp_store_c/e1"
OUT="$ROOT/experiments/results/e1_micro.json"

if [ ! -x "$BENCH" ]; then
  echo "bench_micro not built; run: make -C src/libcapsule" >&2
  exit 1
fi

mkdir -p "$ROOT/experiments/results" "$ROOT/experiments/tmp_store_c"
rm -rf "$STORE"
mkdir -p "$STORE"

echo "E1 microbenchmarks (mode=$MODE) -> $OUT"
"$BENCH" "$STORE" "$OUT" "$MODE"

# clean the package-1 temp store (each package cleans its own)
rm -rf "$STORE"
echo "E1 done."
