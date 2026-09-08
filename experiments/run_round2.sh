#!/bin/sh
# run_round2.sh — re-baseline every experiment on ONE machine (the M3 Ultra).
#
# The machine is shared, so the timing-sensitive C benchmarks run TWICE and we
# report medians with the cross-run range, never one run's tail. Load at
# the start and end of each timing run is recorded alongside the results.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON:-python3}
LIB=src/libcapsule
RES=experiments/results

mkdir -p "$ROOT"/experiments/tmp_store_c/e1 \
         "$ROOT"/experiments/tmp_store_c/e3 \
         "$RES"

log_load() { echo "[$1] load: $(uptime | sed 's/.*load averages*://')"; }

echo "### waiting for any e2b-ext to finish"
while pgrep -f run_e2b_ext >/dev/null 2>&1; do sleep 5; done

for RUN in ; do
  log_load "E1 run $RUN start"
  rm -rf "$ROOT"/experiments/tmp_store_c/e1
  mkdir -p "$ROOT"/experiments/tmp_store_c/e1
  "$LIB"/bench_micro "$ROOT"/experiments/tmp_store_c/e1 \
      "$RES"/e1_micro_run$RUN.json full > experiments/e1_run$RUN.log 2>&1
  log_load "E1 run $RUN end"
done
[ -f "$RES"/e1_micro_run1.json ] && cp "$RES"/e1_micro_run1.json "$RES"/e1_micro.json

for RUN in ; do
  log_load "E7 run $RUN start"
  CAPSULE_BR_SHARDS=64 "$LIB"/bench_conc /tmp/cc_sh$RUN \
    "$RES"/e7_concurrency_sharded_run$RUN.json 20000 \
    > experiments/e7_sharded_run$RUN.log 2>&1
  CAPSULE_BR_SHARDS=1  "$LIB"/bench_conc /tmp/cc_si$RUN \
    "$RES"/e7_concurrency_single_run$RUN.json  20000 \
    > experiments/e7_single_run$RUN.log  2>&1
  log_load "E7 run $RUN end"
done
[ -f "$RES"/e7_concurrency_sharded_run1.json ] && cp "$RES"/e7_concurrency_sharded_run1.json "$RES"/e7_concurrency_sharded.json
[ -f "$RES"/e7_concurrency_single_run1.json ] && cp "$RES"/e7_concurrency_single_run1.json "$RES"/e7_concurrency_single.json

echo "### E3 security studies (C parts + merge)"
log_load "E3 start"
for TAG in s1b s1c s2 s3; do
  rm -rf "$ROOT"/experiments/tmp_store_c/e3/$TAG
  mkdir -p "$ROOT"/experiments/tmp_store_c/e3/$TAG
  "$LIB"/sec_c "$TAG" "$ROOT"/experiments/tmp_store_c/e3/$TAG \
      "$RES"/e3_$TAG.json full >> experiments/e3_round2.log 2>&1
done
$PY experiments/run_e3_s1.py >> experiments/e3_round2.log 2>&1
$PY experiments/run_e3_s1.py --merge >> experiments/e3_round2.log 2>&1
log_load "E3 end"

echo "### E2 (continuity) and E4 (10 seeds)"
$PY experiments/run_e2.py  > experiments/e2_round2.log 2>&1
$PY experiments/run_e4.py  > experiments/e4_round2.log 2>&1
$PY experiments/run_e2b.py > experiments/e2b_round2.log 2>&1

echo "### done"
