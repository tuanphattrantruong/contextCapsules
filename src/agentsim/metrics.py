"""metrics.py — aggregate per-seed manager results and derive modeled bytes.

Token counts are MEASURED counters produced by the managers on real runs. Byte
figures are MODELED (token count × a documented KV-geometry constant) and are kept
under a separate "modeled" group so results JSON can label every leaf honestly
(DESIGN.md 8).

hit rate (DESIGN.md metrics note) = prefix_hit_tokens / (prefix_hit_tokens +
recomputed_prefix_tokens): reuse of shared PREFIX (corpus) content only. Freshly
generated decode tokens are excluded from the denominator (they are generation,
not a cache miss).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from . import kvmodel

# The measured counters every manager reports (token/event counts).
COUNTER_KEYS = [
    "peak_resident_tokens",
    "final_resident_tokens",
    "prefix_hit_tokens",
    "recomputed_prefix_tokens",
    "recomputed_tokens",
    "generated_tokens",
    "cross_tenant_reuse_events",
    "denied_attaches",
    "dedup_hits",
]

# Token counters that we also express as modeled bytes (× REAL_BPT).
TOKEN_TO_BYTES_KEYS = [
    "peak_resident_tokens",
    "final_resident_tokens",
    "prefix_hit_tokens",
    "recomputed_prefix_tokens",
    "recomputed_tokens",
    "generated_tokens",
]


def hit_rate(res: dict[str, Any]) -> float:
    """Prefix (corpus) hit rate; 0.0 if no prefix tokens were touched."""
    hit = res.get("prefix_hit_tokens", 0)
    miss = res.get("recomputed_prefix_tokens", 0)
    denom = hit + miss
    return (hit / denom) if denom > 0 else 0.0


def _mean_std(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample std
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"mean": mean, "std": std, "n": n}


def aggregate_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of per-seed manager result dicts into mean/std.

    Returns {"measured": {counter: {mean,std,n}, "hit_rate": {...}},
             "modeled":  {counter_bytes: {mean,std,n}}}.
    Extra store counters present (stored_chunks etc.) are aggregated too, under
    measured.
    """
    if not per_seed:
        return {"measured": {}, "modeled": {}}

    # collect union of numeric keys across seeds (skip the string "manager")
    keys: list[str] = []
    seen = set()
    for r in per_seed:
        for k, v in r.items():
            if k == "manager":
                continue
            if isinstance(v, (int, float)) and k not in seen:
                seen.add(k)
                keys.append(k)

    measured: dict[str, Any] = {}
    for k in keys:
        measured[k] = _mean_std([float(r.get(k, 0)) for r in per_seed])
    measured["hit_rate"] = _mean_std([hit_rate(r) for r in per_seed])

    modeled: dict[str, Any] = {}
    for k in TOKEN_TO_BYTES_KEYS:
        if k in seen:
            bkey = k.replace("_tokens", "_bytes")
            modeled[bkey] = _mean_std([
                float(kvmodel.tokens_to_bytes(int(round(r.get(k, 0))), kvmodel.REAL_BPT))
                for r in per_seed
            ])
    return {"measured": measured, "modeled": modeled}


def per_seed_view(res: dict[str, Any]) -> dict[str, Any]:
    """Split a single result dict into measured/modeled groups with hit_rate."""
    measured = {k: v for k, v in res.items()
                if k != "manager" and isinstance(v, (int, float))}
    measured["hit_rate"] = hit_rate(res)
    modeled = {}
    for k in TOKEN_TO_BYTES_KEYS:
        if k in res:
            modeled[k.replace("_tokens", "_bytes")] = kvmodel.tokens_to_bytes(
                int(round(res[k])), kvmodel.REAL_BPT)
    return {"manager": res.get("manager"), "measured": measured, "modeled": modeled}
