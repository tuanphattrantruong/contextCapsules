#!/bin/sh
# run_e10.sh — E10: capsules mounted under a real inference engine (llama.cpp).
#
# Builds llama.cpp, links libcapsule against it, and runs three things:
#   1. cross-process resume: process A prefills and stores; process B, given
#      only a 72-byte capability, attaches, pushes the state back into the
#      engine, and continues generation. Tokens must be identical.
#   2. restore-vs-prefill across context lengths, timed inside the engine.
#   3. the same on two model sizes.
#
# Usage: sh experiments/run_e10.sh [llama_dir]
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
LLAMA_DIR=${1:-/tmp/llama.cpp}
RES=experiments/results
GGUF=workloads/gguf
PY=${PYTHON:-python3}

if [ ! -d "$LLAMA_DIR" ]; then
  echo "### cloning llama.cpp -> $LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
if [ ! -f "$LLAMA_DIR/build/bin/libllama.dylib" ]; then
  echo "### building llama.cpp (Metal)"
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DGGML_METAL=ON \
        -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON
  cmake --build "$LLAMA_DIR/build" --config Release -j
fi

mkdir -p "$GGUF" "$RES"
[ -f "$GGUF/qwen2.5-0.5b-instruct-q8_0.gguf" ] || curl -sSL -o \
  "$GGUF/qwen2.5-0.5b-instruct-q8_0.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
[ -f "$GGUF/llama-3.1-8b-instruct-q4km.gguf" ] || curl -sSL -o \
  "$GGUF/llama-3.1-8b-instruct-q4km.gguf" \
  "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

echo "### building llamacap"
make -C src/llamacap LLAMA_DIR="$LLAMA_DIR"

echo "### E10a cross-process resume (Llama-3.1-8B, 2048 tokens)"
rm -rf /tmp/e10_store
./src/llamacap/llamacap save "$GGUF/llama-3.1-8b-instruct-q4km.gguf" \
    /tmp/e10_store 2048 12 99 2>/dev/null | grep '^{' > "$RES"/e10_save.json
CAP=$($PY -c "import json;print(json.load(open('$RES/e10_save.json'))['cap'])")
./src/llamacap/llamacap restore "$GGUF/llama-3.1-8b-instruct-q4km.gguf" \
    /tmp/e10_store 2048 12 99 "$CAP" 2>/dev/null | grep '^{' > "$RES"/e10_restore.json
$PY - <<'EOF'
import json
s = json.load(open('experiments/results/e10_save.json'))
r = json.load(open('experiments/results/e10_restore.json'))
out = {
    "engine": "llama.cpp (Metal)",
    "model": "Llama-3.1-8B-Instruct Q4_K_M",
    "n_prompt": s["n_prompt"], "n_generated": s["n_new"],
    "engine_blob_bytes": s["engine_blob_bytes"],
    "chain_hash_matches": s["chain"] == r["chain"],
    "generated_tokens_identical": s["ref_tokens"] == r["tokens"],
    "revoke_after_restore_denies": r["revoke_denies"],
    "save": s, "restore": r,
}
json.dump(out, open('experiments/results/e10_crossproc.json', 'w'), indent=2)
print("  identical tokens: %s | chain match: %s | revoke denies: %s"
      % (out["generated_tokens_identical"], out["chain_hash_matches"],
         out["revoke_after_restore_denies"]))
EOF

echo "### E10b restore vs engine prefill"
./src/llamacap/llamacap bench "$GGUF/qwen2.5-0.5b-instruct-q8_0.gguf" \
    /tmp/e10_bq 99 5 512,1024,2048,4096,8192 > "$RES"/e10_llamacpp_qwen05b.json
./src/llamacap/llamacap bench "$GGUF/llama-3.1-8b-instruct-q4km.gguf" \
    /tmp/e10_b8 99 5 512,1024,2048,4096,8192 > "$RES"/e10_llamacpp_llama8b.json
echo "### done"
