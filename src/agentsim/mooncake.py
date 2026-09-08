"""mooncake.py — drive the managers from the REAL Mooncake production trace.

Round-1 review objection R2: every workload number came from a synthetic
Zipf/log-normal generator, so the sharing structure the study measures was the
sharing structure we invented. This module replaces the generator with the
public conversation trace released with Mooncake (FAST'25): 12,031 real
requests, real arrival timestamps, real input/output lengths, and -- the part
that matters here -- real block-level prefix hashes, so the prefix-reuse
structure is measured rather than modelled.

Trace schema (one JSON object per line):
    {"timestamp": ms, "input_length": tok, "output_length": tok,
     "hash_ids": [block ids along this request's prefix]}

Mapping to the manager event schema
-----------------------------------
* Block size is ~512 tokens (sum(input_length)/sum(len(hash_ids)) = 501.9); the
  last block of a request carries the remainder.
* CORPUS blocks are those whose global reference count reaches
  `corpus_min_refs`: content that many requests share, i.e. the system and tool
  prompts an operator would publish. They map to kind "sys", dedup domain "".
  In the tool-agent trace 18 such blocks carry 33.8% of all block references.
* A *session* is a conversation, identified by the first NON-corpus block along
  the request's prefix -- by construction a block few requests touch, so it
  names the branch where this conversation leaves the shared prompt.
* Sessions are assigned to tenants deterministically. The trace carries no
  tenant labels; this is the one modelling choice, and both it and
  `corpus_min_refs` are swept.
* Every non-corpus block is tenant-private (kind "tdoc"), owned by the first
  tenant that used it.
* Consequently a private block that a *second* tenant also uses is a genuine,
  MEASURED cross-tenant duplicate -- the leak surface. The synthetic study had
  to inject this with a p_leakdup knob; here it falls out of the trace.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import kvmodel
from . import traces

BLOCK_TOKENS = 512


def load_rows(path, max_requests: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_requests and len(rows) >= max_requests:
                break
    return rows


def _session_of(hash_ids: List[int]) -> int:
    """Legacy heuristic (first block after the universal prefix)."""
    return hash_ids[1] if len(hash_ids) > 1 else hash_ids[0]


def _refcounts(rows: List[dict]) -> Dict[int, int]:
    c: Dict[int, int] = defaultdict(int)
    for r in rows:
        for b in r["hash_ids"]:
            c[b] += 1
    return c


def _session_key(hash_ids: List[int], refs: Dict[int, int],
                 corpus_min_refs: int) -> int:
    """The conversation branch: first block that is not shared corpus."""
    for b in hash_ids:
        if refs[b] < corpus_min_refs:
            return b
    return hash_ids[-1]


def _assign_tenant(session: int, n_tenants: int, salt: int) -> int:
    h = hashlib.sha256(b"mc-tenant:%d:%d" % (session, salt)).digest()
    return int.from_bytes(h[:8], "little") % n_tenants


def _block_tokens(input_length: int, n_blocks: int, i: int) -> int:
    """Tokens carried by block i, so the per-request total is input_length."""
    if i < n_blocks - 1:
        return BLOCK_TOKENS
    return max(1, input_length - BLOCK_TOKENS * (n_blocks - 1))


def build_events(rows: List[dict], n_tenants: int = 16,
                 corpus_min_refs: int = 10, salt: int = 1,
                 ) -> Tuple[List[dict], Dict[str, Any]]:
    """Convert raw Mooncake rows into manager events + measured statistics."""
    refs = _refcounts(rows)
    public = {b for b, c in refs.items() if c >= corpus_min_refs}

    # ---- pass 1: tenants, per-block tenant sets, first user ----------------
    tenant_of_req: List[int] = []
    block_tenants: Dict[int, set] = defaultdict(set)
    sessions: Dict[int, int] = {}
    first_user: Dict[int, int] = {}
    for r in rows:
        sk = _session_key(r["hash_ids"], refs, corpus_min_refs)
        t = _assign_tenant(sk, n_tenants, salt)
        sessions[sk] = t
        tenant_of_req.append(t)
        for b in r["hash_ids"]:
            block_tenants[b].add(t)
            if b not in first_user:
                first_user[b] = t

    # dense per-owner local ids for private blocks
    local_of: Dict[int, int] = {}
    per_owner_count: Dict[int, int] = defaultdict(int)
    for b in sorted(block_tenants):
        if b in public:
            continue
        o = first_user[b]
        local_of[b] = per_owner_count[o]
        per_owner_count[o] += 1
    max_local = max(per_owner_count.values()) if per_owner_count else 0
    if max_local >= traces.TDOC_STRIDE:
        raise ValueError(
            "TDOC_STRIDE=%d too small for %d private blocks owned by one tenant; "
            "raise traces.TDOC_STRIDE" % (traces.TDOC_STRIDE, max_local))

    # ---- pass 2: emit events ----------------------------------------------
    timed: List[Tuple[float, int, int]] = []   # (t_ms, is_end, req)
    per_req: List[dict] = []
    n_priv_refs = n_xtenant_refs = n_pub_refs = 0
    for i, r in enumerate(rows):
        t = tenant_of_req[i]
        hids = r["hash_ids"]
        nb = len(hids)
        segments = []
        for j, b in enumerate(hids):
            tok = _block_tokens(r["input_length"], nb, j)
            if b in public:
                segments.append(["sys", int(b), int(tok)])
                n_pub_refs += 1
            else:
                owner = first_user[b]
                seg_id = owner * traces.TDOC_STRIDE + local_of[b]
                segments.append(["tdoc", int(seg_id), int(tok)])
                n_priv_refs += 1
                if owner != t:
                    n_xtenant_refs += 1
        per_req.append({
            "req": i, "tenant": t, "segments": segments,
            "new_tokens": int(r["output_length"]),
            "session": _session_key(hids, refs, corpus_min_refs),
        })
        start = float(r["timestamp"])
        # a request occupies the engine for at least its decode; the exact
        # duration only affects when decode state is freed, so use the trace's
        # own output_length as the duration proxy (1 ms/token).
        timed.append((start, 0, i))
        timed.append((start + float(r["output_length"]), 1, i))

    timed.sort(key=lambda x: (x[0], x[1], x[2]))
    events: List[dict] = []
    for idx, (t_ms, is_end, req) in enumerate(timed):
        d = per_req[req]
        events.append({
            "i": idx, "t_ms": round(t_ms, 6),
            "op": "end" if is_end else "start",
            "req": req, "parent": -1, "tenant": d["tenant"],
            "agent": 0, "session": d["session"],
            "segments": [] if is_end else d["segments"],
            "new_tokens": 0 if is_end else d["new_tokens"],
        })

    stats = {
        "n_requests": len(rows),
        "n_events": len(events),
        "n_tenants": n_tenants,
        "corpus_min_refs": corpus_min_refs,
        "n_sessions": len(sessions),
        "tenant_salt": salt,
        "block_tokens": BLOCK_TOKENS,
        "n_distinct_blocks": len(block_tenants),
        "n_corpus_blocks": len(public),
        "n_private_blocks": len(block_tenants) - len(public),
        "n_private_blocks_multi_tenant": sum(
            1 for b, ts in block_tenants.items() if b not in public and len(ts) > 1),
        "block_refs_total": n_pub_refs + n_priv_refs,
        "block_refs_corpus": n_pub_refs,
        "block_refs_private": n_priv_refs,
        "block_refs_cross_tenant_private": n_xtenant_refs,
        # the measured analogue of the synthetic study's p_leakdup knob
        "measured_leakdup_rate": (n_xtenant_refs / n_priv_refs) if n_priv_refs else 0.0,
        "input_tokens_total": sum(r["input_length"] for r in rows),
        "output_tokens_total": sum(r["output_length"] for r in rows),
        "max_private_blocks_per_owner": max_local,
    }
    return events, stats


class TraceCfg:
    """Minimal stand-in for traces.TraceConfig (managers only read n_tenants)."""

    def __init__(self, n_tenants: int, n_requests: int):
        self.n_tenants = n_tenants
        self.n_requests = n_requests
        self.f = float("nan")
        self.p_leakdup = float("nan")


def default_trace_path() -> Path:
    return (Path(__file__).resolve().parents[2] / "workloads" / "real"
            / "mooncake_conversation.jsonl")
