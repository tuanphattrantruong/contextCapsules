#!/usr/bin/env python3
"""run_e9.py — E9: does the real-model result generalize beyond one model?

Round-1 objection R3: E5 measured one model from one family, so nothing said
whether the geometry, the restore-versus-recompute win, or the
position-dependence finding are properties of capsules or of Llama-3.1-8B.

E9 repeats the three model-dependent arms of E5 across model families and
sizes, on both the Metal and CPU backends:

  geometry      bytes/token derived from each model's own live cache
  restore       prefill-from-scratch vs restore-from-capsule, bit-exactness
  positiondep   chunk-hash agreement for identical tokens after a different
                prefix -- the measurement that forced the naming correction

Families: Qwen2.5 (1.5B, 7B), Llama-3.2 (1B), Gemma-3 (1B), OLMo-2 (1B),
SmolLM2 (360M), plus the Llama-3.1-8B run of E5.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_e5  # noqa: E402
from kvbridge import KVGeometry, CapsuleKVStore  # noqa: E402
from kvbridge.pack import pack_kv, unpack_kv, cache_equal  # noqa: E402

MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "meta-llama/Llama-3.2-1B",
    "allenai/OLMo-2-0425-1B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-1b-pt",
]


def one_model(model_id, device, lengths, reps, scratch):
    run_e5.set_device(device)
    out = {"model": model_id, "device": device}
    t0 = time.perf_counter()
    tok, model = run_e5.load_model(model_id)
    out["load_s"] = time.perf_counter() - t0

    geo, geom = run_e5.e5a_geometry(model, model_id, tok)
    out["geometry"] = geo["from_config"]
    out["bytes_per_token"] = geo["measured_bytes_per_token_from_packed_buffer"]
    out["geometry_agrees_with_config"] = geo["agrees_with_config"]

    rows = []
    text = run_e5.corpus_text(1500000, seed=11)
    for n in lengths:
        try:
            ids = run_e5.tokens_for(tok, text, n)
        except ValueError:
            continue
        run_e5.prefill(model, ids)
        pre = []
        for _ in range(reps):
            t = run_e5.now()
            o = run_e5.prefill(model, ids)
            pre.append((run_e5.now() - t) * 1e3)
        buf = pack_kv(o.past_key_values)
        st = CapsuleKVStore(str(Path(scratch) / ("%s_n%d" %
                            (model_id.replace("/", "_"), n))))
        cap, _ = st.put_kv("/t/s", "tenant:1", geom, buf, n)
        res = []
        for _ in range(reps):
            t = time.perf_counter()
            m = st.attach(cap)
            arr = CapsuleKVStore.map_to_array(m)
            cache = unpack_kv(arr[:buf.nbytes], n, geom, device=device)
            run_e5.sync()
            res.append((time.perf_counter() - t) * 1e3)
            st.detach(m)
            del cache
        m = st.attach(cap)
        arr = CapsuleKVStore.map_to_array(m)
        eq, worst = cache_equal(o.past_key_values,
                                unpack_kv(arr[:buf.nbytes], n, geom, device=device))
        st.detach(m)
        st.close()
        rows.append({
            "n_tokens": n, "kv_bytes": int(buf.nbytes),
            "prefill_ms_p50": float(np.median(pre)),
            "restore_ms_p50": float(np.median(res)),
            "speedup_p50": float(np.median(pre) / np.median(res)),
            "bitwise_equal": bool(eq),
        })
        print("    n=%-5d prefill %8.1f ms  restore %7.1f ms  %5.1fx  exact=%s"
              % (n, np.median(pre), np.median(res),
                 np.median(pre) / np.median(res), eq), flush=True)
    out["restore"] = rows

    pd = run_e5.e5e_positiondep(model, tok, geom, n_doc=256, n_prefixes=3,
                               pre_len=256)
    out["positiondep"] = {
        "mean_chunk_hash_match_frac": pd["mean_chunk_hash_match_frac"],
        "mean_rel_l2": pd["mean_rel_l2"],
    }
    print("    position-independence: chunk-hash match %.4f, rel L2 %.3f"
          % (pd["mean_chunk_hash_match_frac"], pd["mean_rel_l2"]), flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--lengths", default="512,1024,2048")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=str(ROOT / "experiments/results/e9_multimodel.json"))
    a = ap.parse_args()

    lengths = [int(x) for x in a.lengths.split(",")]
    scratch = ROOT / "experiments" / "tmp_e9"
    scratch.mkdir(parents=True, exist_ok=True)

    res = {"meta": {
        "experiment": "E9_multimodel",
        "device": a.device, "lengths": lengths, "reps": a.reps,
        "torch": torch.__version__,
        "date_u": subprocess.run(["date", "-u"], capture_output=True,
                                 text=True).stdout.strip(),
        "hw_model": subprocess.run(["sysctl", "-n", "hw.model"],
                                   capture_output=True, text=True).stdout.strip(),
        "note": "all KV bytes are real tensors; no modeled constants",
    }, "models": []}

    for mid in a.models.split(","):
        print("[%s]" % mid, flush=True)
        try:
            res["models"].append(one_model(mid, a.device, lengths, a.reps, scratch))
        except Exception as e:                      # noqa: BLE001
            print("    FAILED: %s: %s" % (type(e).__name__, e), flush=True)
            res["models"].append({"model": mid, "device": a.device,
                                  "error": "%s: %s" % (type(e).__name__, e)})
        Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
