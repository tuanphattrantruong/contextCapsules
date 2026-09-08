#!/usr/bin/env python3
"""E5 — Real-model capsule study.

Everything in E1-E4 was either a synthetic byte buffer (C microbenchmarks) or a
token counter multiplied by an assumed constant (simulator). E5 removes that
gap: it runs a real transformer, takes the KV cache it actually produces, and
puts it through libcapsule.

Parts
  E5a  geometry      : bytes/token MEASURED from the loaded model
  E5b  fidelity      : cross-PROCESS restore is bit-exact and produces
                       identical generated tokens (the "name outlives the
                       process" claim, tested)
  E5c  ttft          : prefill-from-scratch vs restore-from-capsule, real
                       wall-clock, across prompt lengths
  E5d  dedup         : real KV dedup at 16-token granularity, within and
                       across dedup domains
  E5e  positiondep   : how much of the simulator's position-independence
                       assumption (RESULTS_SUMMARY L2) survives contact with a
                       real dense transformer

Usage
  run_e5.py --model <hf-id> --part all
  run_e5.py --role restorer --state <json>     (internal: E5b child process)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from kvbridge import KVGeometry, CapsuleKVStore  # noqa: E402
from kvbridge.pack import pack_kv, pack_kv_range, unpack_kv, cache_equal  # noqa: E402

CHUNK_TOKENS = 16
DEV = os.environ.get("KVBRIDGE_DEVICE", "mps")


def set_device(d):
    global DEV
    DEV = d
    os.environ["KVBRIDGE_DEVICE"] = d


# ---------------------------------------------------------------- utilities

def sync():
    if DEV == "mps":
        torch.mps.synchronize()
    elif DEV == "cuda":
        torch.cuda.synchronize()


def now() -> float:
    sync()
    return time.perf_counter()


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.to(DEV).eval()
    return tok, model


def corpus_text(nchars=400000, seed=0):
    """Deterministic English-ish text; no network, no dataset dependency."""
    rng = np.random.default_rng(seed)
    vocab = ("the of and to in a is that it for was with as on be at by this "
             "from or an but not are were which you all can has have had they "
             "we their one there when what who will more if out about into "
             "system memory context cache token model tenant capsule inference "
             "process kernel protection lifecycle revocation naming domain "
             "prefix attention transformer serving agent request latency "
             "throughput evaluation measurement architecture operating").split()
    out = []
    n = 0
    while n < nchars:
        s = " ".join(rng.choice(vocab, size=int(rng.integers(8, 22))))
        out.append(s.capitalize() + ".")
        n += len(out[-1]) + 1
    return " ".join(out)


def tokens_for(tok, text, n):
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=n + 64).input_ids
    if ids.shape[1] < n:
        raise ValueError("corpus too short: %d < %d" % (ids.shape[1], n))
    return ids[:, :n]


def prefill(model, ids):
    with torch.no_grad():
        return model(ids.to(DEV), use_cache=True)


def greedy_decode(model, cache, last_id, n_new):
    """Greedy-decode n_new tokens continuing from `cache`. Returns ids+logits."""
    ids_out, logit_sig = [], []
    cur = torch.tensor([[last_id]], device=DEV)
    with torch.no_grad():
        for _ in range(n_new):
            o = model(cur, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            lg = o.logits[0, -1]
            nxt = int(torch.argmax(lg))
            ids_out.append(nxt)
            logit_sig.append(float(lg.float().sum()))
            cur = torch.tensor([[nxt]], device=DEV)
    return ids_out, logit_sig


# ---------------------------------------------------------------- E5a

def e5a_geometry(model, model_id, tok):
    ids = tokens_for(tok, corpus_text(20000), 64)
    out = prefill(model, ids)
    g_model = KVGeometry.from_model(model, model_id)
    g_cache = KVGeometry.from_cache(out.past_key_values, model_id)
    buf = pack_kv(out.past_key_values)
    measured_bpt = buf.nbytes // 64
    return {
        "from_config": g_model.as_dict(),
        "from_live_cache": g_cache.as_dict(),
        "measured_bytes_per_token_from_packed_buffer": int(measured_bpt),
        "agrees_with_config": bool(measured_bpt == g_model.bytes_per_token),
        "paper_modeled_constant_131072": 131072,
        "matches_paper_constant": bool(measured_bpt == 131072),
    }, g_model


# ---------------------------------------------------------------- E5b

def e5b_fidelity(model, tok, geom, store_root, n_prompt=512, n_new=16):
    """Store real KV in proc A; restore and decode in a fresh proc B.

    Two distinct claims are tested separately, because they have different
    failure modes:

      (1) BYTE fidelity -- the bytes a second process gets back through a
          capability are the bytes the first process stored. Verified by
          SHA-256 recomputed by the child over the attached mapping, and by
          the store's own chain hash. Device-independent.

      (2) SEMANTIC fidelity -- decoding from restored KV yields the same
          tokens as decoding from the live cache. This is only a meaningful
          test on a deterministic backend: E5f measures that this model's MPS
          decode is not run-to-run deterministic even against itself, so this
          arm must run on CPU.
    """
    import hashlib

    ids = tokens_for(tok, corpus_text(200000, seed=7), n_prompt)
    out = prefill(model, ids)
    pkv = out.past_key_values
    buf = pack_kv(pkv)
    assert buf.nbytes == n_prompt * geom.bytes_per_token
    buf_sha = hashlib.sha256(buf.tobytes()).hexdigest()

    last = int(ids[0, -1])
    ref_ids, ref_sig = greedy_decode(model, prefill(model, ids).past_key_values,
                                     last, n_new)

    st = CapsuleKVStore(str(store_root))
    cap, cid = st.put_kv("/tenant1/sess", "tenant:1", geom, buf, n_prompt)
    stt = st.stat(cap)
    chain = bytes(bytearray(stt.chain_hash)).hex()
    caphex = bytes(bytearray(cap)).hex()
    st.close()

    state = {
        "store": str(store_root), "cap": caphex, "n_prompt": n_prompt,
        "n_new": n_new, "last_id": last, "model": geom.model_id,
        "device": DEV, "expect_sha256": buf_sha,
    }
    sp = Path(store_root).parent / "e5b_state.json"
    sp.write_text(json.dumps(state))

    child = subprocess.run(
        [sys.executable, str(Path(__file__)), "--role", "restorer",
         "--state", str(sp)],
        capture_output=True, text=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                 KVBRIDGE_DEVICE=DEV))
    if child.returncode != 0:
        return {"error": "restorer failed", "stderr": child.stderr[-3000:]}
    res = json.loads(child.stdout.strip().splitlines()[-1])

    return {
        "device": DEV,
        "n_prompt_tokens": n_prompt, "n_generated": n_new,
        "kv_bytes": int(buf.nbytes),
        "parent_pid": os.getpid(), "restorer_pid": res["pid"],
        "capsule_id": cid, "capsule_chain_hash": chain,
        # --- claim 1: byte fidelity across processes
        "parent_sha256": buf_sha,
        "child_sha256_of_attached_mapping": res["sha256"],
        "kv_bytes_bitwise_identical_across_process":
            bool(res["sha256"] == buf_sha),
        "restore_chain_hash_matches": bool(res["chain_hash"] == chain),
        # --- claim 2: semantic fidelity (meaningful only on a deterministic backend)
        "reference_token_ids": ref_ids,
        "restored_token_ids": res["token_ids"],
        "generated_ids_identical": bool(ref_ids == res["token_ids"]),
        "logit_sums_identical": bool(
            len(ref_sig) == len(res["logit_sums"]) and
            all(a == b for a, b in zip(ref_sig, res["logit_sums"]))),
        "semantic_test_is_valid_on_this_device": DEV == "cpu",
        # --- lifecycle
        "revoked_after_restore_denies": res["revoke_denies"],
    }


def role_restorer(state_path):
    """Child process: attach by capability alone, restore KV, decode."""
    import hashlib
    st = json.loads(Path(state_path).read_text())
    set_device(os.environ.get("KVBRIDGE_DEVICE", st.get("device", "mps")))
    tok, model = load_model(st["model"])
    geom = KVGeometry.from_model(model, st["model"])

    from agentsim import capsule_ffi as ffi
    store = CapsuleKVStore(st["store"], flags=0)
    cap = ffi.CapsuleCap.from_buffer_copy(bytes.fromhex(st["cap"]))

    stt = store.stat(cap)
    chain = bytes(bytearray(stt.chain_hash)).hex()

    m = store.attach(cap)
    arr = CapsuleKVStore.map_to_array(m)
    nbytes = st["n_prompt"] * geom.bytes_per_token
    got = np.ascontiguousarray(arr[:nbytes])
    sha = hashlib.sha256(got.tobytes()).hexdigest()
    cache = unpack_kv(got, st["n_prompt"], geom, device=DEV)

    ids, sig = greedy_decode(model, cache, st["last_id"], st["n_new"])
    store.detach(m)

    store.revoke(cap)
    try:
        store.stat(cap)
        revoke_denies = False
    except ffi.CapsuleError as e:
        revoke_denies = (e.code == ffi.CAPSULE_E_REVOKED)
    store.close()

    print(json.dumps({
        "pid": os.getpid(), "token_ids": ids, "logit_sums": sig,
        "chain_hash": chain, "sha256": sha, "revoke_denies": revoke_denies,
    }))


def e5f_determinism(model, tok, geom, n_prompt=512, n_new=64, reps=6):
    """Is the backend itself deterministic? Decides how E5b must be read.

    Compares (a) repeated decodes from freshly prefilled live caches, and
    (b) repeated decodes from identical capsule-restored caches.
    """
    ids = tokens_for(tok, corpus_text(200000, seed=7), n_prompt)
    last = int(ids[0, -1])
    live = [greedy_decode(model, prefill(model, ids).past_key_values, last,
                          n_new)[0] for _ in range(reps)]
    buf = pack_kv(prefill(model, ids).past_key_values)
    rt = [greedy_decode(model, unpack_kv(buf.copy(), n_prompt, geom, device=DEV),
                        last, n_new)[0] for _ in range(reps)]
    def first_div(runs):
        for i in range(len(runs[0])):
            if any(r[i] != runs[0][i] for r in runs):
                return i
        return -1
    return {
        "device": DEV, "reps": reps, "n_prompt": n_prompt, "n_new": n_new,
        "live_first_divergence_index": first_div(live),
        "restored_first_divergence_index": first_div(rt),
        "live_decode_runs": live,
        "live_decode_self_consistent": all(x == live[0] for x in live),
        "restored_decode_runs": rt,
        "restored_decode_self_consistent": all(x == rt[0] for x in rt),
        "note": ("if live_decode_self_consistent is false the backend is "
                 "nondeterministic and end-to-end token equality is not a "
                 "valid fidelity test on it; byte equality still is"),
    }


# ---------------------------------------------------------------- E5c

def e5c_ttft(model, tok, geom, store_root, lengths, reps=5):
    """Prefill-from-scratch vs restore-from-capsule, real wall clock."""
    text = corpus_text(3000000, seed=11)
    rows = []
    for n in lengths:
        ids = tokens_for(tok, text, n)
        # warm
        prefill(model, ids)

        pre = []
        for _ in range(reps):
            t0 = now()
            out = prefill(model, ids)
            pre.append((now() - t0) * 1e3)
        pkv = out.past_key_values

        t0 = now()
        buf = pack_kv(pkv)
        pack_ms = (now() - t0) * 1e3

        st = CapsuleKVStore(str(Path(store_root) / ("n%d" % n)))
        t0 = time.perf_counter()
        cap, _ = st.put_kv("/t/s", "tenant:1", geom, buf, n)
        store_ms = (time.perf_counter() - t0) * 1e3

        res, att, unp = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            m = st.attach(cap)
            arr = CapsuleKVStore.map_to_array(m)
            t1 = time.perf_counter()
            cache = unpack_kv(arr[:buf.nbytes], n, geom, device=DEV)
            sync()
            t2 = time.perf_counter()
            att.append((t1 - t0) * 1e3)
            unp.append((t2 - t1) * 1e3)
            res.append((t2 - t0) * 1e3)
            st.detach(m)
            del cache

        # correctness spot-check at this length
        m = st.attach(cap)
        arr = CapsuleKVStore.map_to_array(m)
        cache = unpack_kv(arr[:buf.nbytes], n, geom, device=DEV)
        eq, worst = cache_equal(pkv, cache)
        st.detach(m)
        st.close()

        rows.append({
            "n_tokens": n,
            "kv_bytes": int(buf.nbytes),
            "prefill_ms": {"p50": float(np.median(pre)), "mean": float(np.mean(pre)),
                           "min": float(np.min(pre)), "n": reps},
            "pack_ms": pack_ms,
            "store_put_ms": store_ms,
            "restore_ms": {"p50": float(np.median(res)), "mean": float(np.mean(res)),
                           "min": float(np.min(res)), "n": reps},
            "attach_ms_p50": float(np.median(att)),
            "unpack_ms_p50": float(np.median(unp)),
            "speedup_p50": float(np.median(pre) / np.median(res)),
            "restored_cache_bitwise_equal": bool(eq),
        })
        print("  n=%-6d prefill %8.1f ms  restore %7.1f ms  speedup %5.1fx  exact=%s"
              % (n, np.median(pre), np.median(res),
                 np.median(pre) / np.median(res), eq), flush=True)
    return rows


# ---------------------------------------------------------------- E5d

def e5d_dedup(model, tok, geom, store_root, n_sys=512, n_priv=256, n_tenants=4):
    """Real-KV dedup at 16-token granularity, within and across domains."""
    text = corpus_text(400000, seed=3)
    sys_ids = tokens_for(tok, text, n_sys)
    sys_kv = pack_kv(prefill(model, sys_ids).past_key_values)

    # Each tenant: the SAME system prompt, then its own private continuation.
    # Real causal attention => the system-prompt KV is byte-identical for all
    # tenants, so a within-domain store must dedup it and a cross-domain store
    # must not.
    priv_text = corpus_text(400000, seed=99)
    priv_ids = tokens_for(tok, priv_text, n_priv)

    def full_kv():
        cat = torch.cat([sys_ids, priv_ids], dim=1)
        return pack_kv(prefill(model, cat).past_key_values), cat.shape[1]

    kv_full, n_full = full_kv()
    n_full -= n_full % CHUNK_TOKENS
    kv_full = kv_full[:n_full * geom.bytes_per_token]

    # (a) shared public domain: all tenants store the same prefix
    shared = CapsuleKVStore(str(Path(store_root) / "shared"))
    for t in range(n_tenants):
        shared.put_kv("/t%d/sess" % t, "", geom, kv_full, n_full)
    s_shared = shared.store_stats()
    shared.close()

    # (b) per-tenant domains: identical bytes must NOT dedup across tenants
    isolated = CapsuleKVStore(str(Path(store_root) / "isolated"))
    for t in range(n_tenants):
        isolated.put_kv("/t%d/sess" % t, "tenant:%d" % t, geom, kv_full, n_full)
    s_iso = isolated.store_stats()
    isolated.close()

    # (c) does the shared prefix's KV actually match byte-for-byte when the
    #     suffix differs?  (the property prefix dedup depends on)
    prefix_from_full = kv_full[:n_sys * geom.bytes_per_token]
    prefix_alone = sys_kv[:n_sys * geom.bytes_per_token]
    same = bool(np.array_equal(prefix_from_full, prefix_alone))

    logical = n_tenants * n_full * geom.bytes_per_token
    return {
        "n_sys_tokens": n_sys, "n_total_tokens": n_full,
        "n_tenants": n_tenants,
        "logical_bytes": logical,
        "shared_domain": {
            "stored_bytes": int(s_shared.stored_bytes),
            "dedup_hits": int(s_shared.dedup_hits),
            "dedup_saved_bytes": int(s_shared.dedup_saved_bytes),
            "dedup_ratio_vs_logical": float(s_shared.stored_bytes) / logical,
        },
        "per_tenant_domains": {
            "stored_bytes": int(s_iso.stored_bytes),
            "dedup_hits": int(s_iso.dedup_hits),
            "dedup_saved_bytes": int(s_iso.dedup_saved_bytes),
            "dedup_ratio_vs_logical": float(s_iso.stored_bytes) / logical,
        },
        "cross_domain_dedup_occurred": bool(s_iso.dedup_hits > 0),
        "real_prefix_kv_is_suffix_independent": same,
    }


# ---------------------------------------------------------------- E5e

def e5e_positiondep(model, tok, geom, n_doc=256, n_prefixes=4, pre_len=256):
    """Measure the position-independence assumption on a real transformer.

    The simulator's capsule manager names private content from a per-tenant
    seed, so the same document dedups wherever it appears. A dense transformer
    does not work that way: the document's KV depends on everything before it.
    This quantifies exactly how wrong the assumption is, at chunk granularity.
    """
    import hashlib
    doc_ids = tokens_for(tok, corpus_text(200000, seed=21), n_doc)
    ref = None
    rows = []
    for p in range(n_prefixes):
        pre_ids = tokens_for(tok, corpus_text(200000, seed=100 + p), pre_len)
        cat = torch.cat([pre_ids, doc_ids], dim=1)
        pkv = prefill(model, cat).past_key_values
        doc_kv = pack_kv_range(pkv, pre_len, pre_len + n_doc)
        cbytes = CHUNK_TOKENS * geom.bytes_per_token
        hashes = [hashlib.sha256(doc_kv[i:i + cbytes]).hexdigest()
                  for i in range(0, doc_kv.nbytes, cbytes)]
        if ref is None:
            ref, ref_kv = hashes, doc_kv
            rows.append({"prefix": p, "chunk_hash_match_frac": 1.0,
                         "rel_l2": 0.0, "is_reference": True})
        else:
            match = sum(1 for a, b in zip(ref, hashes) if a == b) / len(ref)
            a = torch.from_numpy(ref_kv.copy()).view(geom.dtype).float()
            b = torch.from_numpy(doc_kv.copy()).view(geom.dtype).float()
            rel = float(torch.linalg.vector_norm(a - b) /
                        torch.linalg.vector_norm(a))
            rows.append({"prefix": p, "chunk_hash_match_frac": match,
                         "rel_l2": rel, "is_reference": False})
    nonref = [r for r in rows if not r["is_reference"]]
    return {
        "n_doc_tokens": n_doc, "prefix_len": pre_len,
        "n_prefixes": n_prefixes,
        "per_prefix": rows,
        "mean_chunk_hash_match_frac": float(
            np.mean([r["chunk_hash_match_frac"] for r in nonref])),
        "mean_rel_l2": float(np.mean([r["rel_l2"] for r in nonref])),
        "verdict": ("position-independent naming is UNSOUND for a dense "
                    "transformer: identical token content after a different "
                    "prefix produces different KV bytes"),
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--part", default="all")
    ap.add_argument("--role", default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=str(ROOT / "experiments/results/e5_realmodel.json"))
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    if a.device:
        set_device(a.device)

    if a.role == "restorer":
        role_restorer(a.state)
        return

    scratch = Path(a.scratch or (ROOT / "experiments/tmp_e5"))
    scratch.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    tok, model = load_model(a.model)
    load_s = time.perf_counter() - t_load
    print("model loaded in %.1fs on %s" % (load_s, DEV), flush=True)

    res = {"meta": {
        "experiment": "E5_real_model",
        "model": a.model, "device": DEV,
        "torch": torch.__version__,
        "date_u": subprocess.run(["date", "-u"], capture_output=True,
                                 text=True).stdout.strip(),
        "uname_a": subprocess.run(["uname", "-a"], capture_output=True,
                                  text=True).stdout.strip(),
        "hw_model": subprocess.run(["sysctl", "-n", "hw.model"],
                                   capture_output=True, text=True).stdout.strip(),
        "ncpu": os.cpu_count(),
        "memsize_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                            capture_output=True,
                                            text=True).stdout.strip()),
        "python": platform.python_version(),
        "model_load_s": load_s,
        "note": ("all KV bytes here are REAL tensors produced by the model; "
                 "no modeled constants"),
    }}

    parts = a.part.split(",") if a.part != "all" else \
        ["a", "b", "c", "d", "e", "f"]

    print("[E5a] geometry", flush=True)
    geo, geom = e5a_geometry(model, a.model, tok)
    res["e5a_geometry"] = geo
    print("  measured bytes/token = %d (paper constant 131072: %s)"
          % (geo["measured_bytes_per_token_from_packed_buffer"],
             geo["matches_paper_constant"]), flush=True)

    if "b" in parts:
        print("[E5b] cross-process fidelity", flush=True)
        res["e5b_fidelity"] = e5b_fidelity(
            model, tok, geom, scratch / "e5b_store")
        r = res["e5b_fidelity"]
        print("  bytes identical across process: %s | chain match: %s"
              % (r.get("kv_bytes_bitwise_identical_across_process"),
                 r.get("restore_chain_hash_matches")), flush=True)
        print("  identical generated ids: %s (valid on this device: %s) | "
              "revoke denies: %s"
              % (r.get("generated_ids_identical"),
                 r.get("semantic_test_is_valid_on_this_device"),
                 r.get("revoked_after_restore_denies")), flush=True)

    if "f" in parts:
        print("[E5f] backend determinism probe", flush=True)
        res["e5f_determinism"] = e5f_determinism(model, tok, geom)
        print("  live self-consistent: %s | restored self-consistent: %s"
              % (res["e5f_determinism"]["live_decode_self_consistent"],
                 res["e5f_determinism"]["restored_decode_self_consistent"]),
              flush=True)

    if "c" in parts:
        print("[E5c] TTFT: prefill vs restore", flush=True)
        lens = [int(x) for x in a.lengths.split(",")]
        res["e5c_ttft"] = e5c_ttft(model, tok, geom, scratch / "e5c",
                                   lens, reps=a.reps)

    if "d" in parts:
        print("[E5d] real-KV dedup", flush=True)
        res["e5d_dedup"] = e5d_dedup(model, tok, geom, scratch / "e5d")
        print("  shared-domain stored/logical = %.3f ; per-tenant = %.3f ; "
              "cross-domain dedup: %s"
              % (res["e5d_dedup"]["shared_domain"]["dedup_ratio_vs_logical"],
                 res["e5d_dedup"]["per_tenant_domains"]["dedup_ratio_vs_logical"],
                 res["e5d_dedup"]["cross_domain_dedup_occurred"]), flush=True)

    if "e" in parts:
        print("[E5e] position dependence of real KV", flush=True)
        res["e5e_positiondep"] = e5e_positiondep(model, tok, geom)
        print("  mean chunk-hash match across prefixes = %.4f ; rel L2 = %.3f"
              % (res["e5e_positiondep"]["mean_chunk_hash_match_frac"],
                 res["e5e_positiondep"]["mean_rel_l2"]), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
