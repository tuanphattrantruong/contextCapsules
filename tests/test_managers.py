"""test_managers.py — invariants of the three KV-state managers (DESIGN.md 3.2).

The correctness-critical claim of the paper is the leak-vs-deny surface (C5/C6):
on a workload with coincidental cross-tenant private duplication (p_leakdup > 0),
the unprotected prefix cache reuses that content across tenants (a leak), while
the capsule manager refuses it (a denied attach) and never reports a cross-tenant
reuse. These tests assert exactly that split, plus determinism and a compact
mirror-vs-real-C accounting equality (the E2b contract, in miniature).
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import sys
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from agentsim import managers, traces  # noqa: E402
from agentsim import capsule_ffi as F   # noqa: E402


def _leak_trace(seed: int = 1):
    # The prefix cache keys on the FULL position-0 chain, so a leak-dup tdoc only
    # produces a cross-tenant HIT when the entire preceding prefix also collides.
    # That happens most at low tenant count / low sharing factor (few distinct
    # prefixes). T=3, f=0.25, p_leakdup=0.15 at 2k requests reliably exhibits it
    # (cross_tenant_reuse 10/9/5 for seeds 1/2/3) — matching the full-grid E2
    # finding that the leak concentrates in the low-T, low-f cells.
    cfg = traces.TraceConfig(n_requests=2000, n_tenants=3, f=0.25, p_leakdup=0.15)
    return cfg, traces.gen_trace(cfg, seed)


def _independent_denied_recount(trace) -> int:
    seen = set()
    denied = 0
    for ev in trace:
        if ev["op"] != "start":
            continue
        tenant = ev["tenant"]
        for kind, seg_id, _tok in ev["segments"]:
            if kind != "tdoc":
                continue
            owner, local = traces.decode_tdoc_id(seg_id)
            if owner == tenant:
                continue
            key = (tenant, owner, local)
            if key not in seen:
                seen.add(key)
                denied += 1
    return denied


def test_prefix_cache_leaks_cross_tenant():
    cfg, trace = _leak_trace()
    r = managers.run_trace("prefix", cfg, trace)
    assert r["cross_tenant_reuse_events"] > 0, \
        "unprotected prefix cache must leak on p_leakdup>0"


def test_capsule_denies_cross_tenant():
    cfg, trace = _leak_trace()
    r = managers.run_trace("capsule", cfg, trace)
    assert r["cross_tenant_reuse_events"] == 0, "capsules never reuse cross-tenant"
    assert r["denied_attaches"] > 0, "capsules must record denied cross-tenant attaches"
    # denial count is exactly the independent trace recount (accounting is honest)
    assert r["denied_attaches"] == _independent_denied_recount(trace), \
        "denied_attaches must match an independent recount"


def test_noshare_never_shares():
    cfg, trace = _leak_trace()
    r = managers.run_trace("noshare", cfg, trace)
    assert r["prefix_hit_tokens"] == 0, "no-share never serves a prefix hit"
    assert r["denied_attaches"] == 0
    assert r["cross_tenant_reuse_events"] == 0


def test_public_corpus_is_shared_without_denial():
    """With no leak-dup, public corpus still dedups and nothing is denied."""
    cfg = traces.TraceConfig(n_requests=500, n_tenants=8, f=0.75, p_leakdup=0.0)
    trace = traces.gen_trace(cfg, 1)
    r = managers.run_trace("capsule", cfg, trace)
    assert r["denied_attaches"] == 0, "no leak-dup -> no denials"
    assert r["prefix_hit_tokens"] > 0, "public corpus is shared (prefix hits)"


def test_determinism_same_seed():
    cfg, trace = _leak_trace(seed=2)
    r1 = managers.run_trace("capsule", cfg, trace)
    r2 = managers.run_trace("capsule", cfg, traces.gen_trace(cfg, 2))
    assert r1 == r2, "same seed -> identical manager result"


def test_mirror_matches_real_c():
    """Compact E2b: mirror's predicted stored/dedup exactly equals the real store."""
    # load run_e2b's faithful op-log replay helpers (same code the driver uses)
    spec = importlib.util.spec_from_file_location(
        "run_e2b", str(_HERE.parents[1] / "experiments" / "run_e2b.py"))
    e2b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2b)

    cfg = traces.TraceConfig(n_requests=150, n_tenants=8, f=0.75, p_leakdup=0.1)
    trace = traces.gen_trace(cfg, 1)
    mirror = managers.CapsuleManager(cfg, record_ops=True, free_decode_on_end=False)
    for ev in trace:
        mirror.on_event(ev)
    mres = mirror.result()

    chunk_nbytes = e2b.CHUNK_NBYTES
    exp_stored = mres["stored_chunks"] * chunk_nbytes
    exp_saved = mres["dedup_saved_chunks"] * chunk_nbytes

    root = _HERE.parents[1] / "experiments" / "tmp_store_py" / "itest_mirror"
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        with F.CapsuleStore(root, F.CAPSULE_STORE_CREATE) as store:
            e2b._replay_oplog(store, mirror.op_log)
            st = store.store_stats()
        assert st.stored_bytes == exp_stored, \
            f"stored mismatch: C={st.stored_bytes} mirror={exp_stored}"
        assert st.dedup_hits == mres["dedup_hits"], \
            f"dedup_hits mismatch: C={st.dedup_hits} mirror={mres['dedup_hits']}"
        assert st.dedup_saved_bytes == exp_saved, \
            f"dedup_saved mismatch: C={st.dedup_saved_bytes} mirror={exp_saved}"
    finally:
        if root.exists():
            shutil.rmtree(root)
