#!/usr/bin/env bash
# run_e3_c.sh — E3 security case studies in the real C library (package 1 sec_c).
#
# Runs S1b, S1c, S2, S3 and writes experiments/results/e3_s1b.json,
# e3_s1c.json, e3_s2.json, e3_s3.json. These are the four C-side parts that
# run_e3_s1.py --merge combines with the Python S1a part into e3_security.json.
# Package 1 does NOT write e3_security.json itself (the merge is package 2's).
#
# Mode: E3_MODE=full (default) uses the DESIGN 4 trial counts; E3_MODE=quick is a
# reduced smoke run. All timings measured on this machine (mach_absolute_time).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODE="${E3_MODE:-full}"
SEC="$ROOT/src/libcapsule/sec_c"
STORE="$ROOT/experiments/tmp_store_c/e3"
RES="$ROOT/experiments/results"

if [ ! -x "$SEC" ]; then
  echo "sec_c not built; run: make -C src/libcapsule" >&2
  exit 1
fi

mkdir -p "$RES" "$ROOT/experiments/tmp_store_c"
rm -rf "$STORE"
mkdir -p "$STORE"

for study in s1b s1c s2 s3; do
  echo "E3 $study (mode=$MODE)"
  "$SEC" "$study" "$STORE/$study" "$RES/e3_$study.json" "$MODE"
done

rm -rf "$STORE"
echo "E3 C studies done -> $RES/e3_{s1b,s1c,s2,s3}.json"
