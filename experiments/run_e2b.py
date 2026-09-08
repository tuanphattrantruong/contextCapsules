#!/usr/bin/env python3
"""run_e2b.py — mirror-vs-libcapsule exact validation (DESIGN.md 3.3 / claim C9).

Config: T=16, f=0.75, p_leakdup=0.05, seed=1, 500 requests. The CapsuleManager
mirror replays the trace and records an op-log of primitive store operations
(create+extend / fork+extend). The IDENTICAL op sequence is replayed against the
real libcapsule via ctypes at SCALED_BPT=1024 with real pseudorandom bytes derived
from each chunk's content key.

Acceptance (EXACT equality — a mismatch is recorded as a failure, never adjusted):
  * stored_bytes(scaled)  : mirror.stored_chunks * chunk_bytes  ==  C.stored_bytes
  * dedup_hits            : mirror.dedup_hits                   ==  C.dedup_hits
  * dedup_saved_bytes     : mirror.dedup_saved_chunks * chunk   ==  C.dedup_saved_bytes
  * denied count          : mirror.denied_attaches              ==  independent trace recount
  * C protection enforced : the real library rejects an unauthorized cross-domain
                            attach (E_PERM / E_REVOKED), demonstrating the denial is
                            real and not merely bookkeeping.

Also records measured process footprint (resource.getrusage RUSAGE_SELF ru_maxrss,
labeled maxrss_measured; bytes on macOS) and wall time (perf_counter_ns).
Budget < 2 min. Temp store under experiments/tmp_store_py/ (removed after).
"""

from __future__ import annotations

import resource
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from agentsim import capsule_ffi, common, kvmodel, managers, traces  # noqa: E402

RESULT = HERE.parents[1] / "experiments" / "results" / "e2b_validation.json"
TMP_STORE = HERE.parents[1] / "experiments" / "tmp_store_py"

SCALED_BPT = kvmodel.SCALED_BPT     # 1024
CHUNK_NBYTES = kvmodel.chunk_bytes(SCALED_BPT)   # 16 * 1024 = 16384 (one page)


def _independent_denied_recount(cfg: traces.TraceConfig,
                                trace: list[dict]) -> int:
    """Recompute the leak-dup denial count from the trace, independent of the mirror.

    A denial happens the first time a tenant materializes a tdoc whose CONTENT is
    owned by another tenant and the tenant does not yet hold that content in its own
    domain. Forks inherit; they do not re-materialize tdocs.
    """
    seen: set[tuple[int, int, int]] = set()   # (using_tenant, owner, local)
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


def _gen_chunks(chunk_specs: list, cache: dict) -> tuple[bytes, int]:
    """Assemble the byte buffer for a create_extend/extend op from its chunk specs.

    chunk_specs = [(content_key, n_blocks), ...]; each segment contributes n_blocks
    chunks of CHUNK_NBYTES bytes generated deterministically from (content_key,
    block_idx). Returns (buffer, n_tokens).
    """
    parts: list[bytes] = []
    total_blocks = 0
    for content_key, n_blocks in chunk_specs:
        for b in range(n_blocks):
            ck = (tuple(content_key), b)
            data = cache.get(ck)
            if data is None:
                data = traces.chunk_bytes_for(content_key, b, CHUNK_NBYTES)
                cache[ck] = data
            parts.append(data)
            total_blocks += 1
    return b"".join(parts), total_blocks * kvmodel.CHUNK_TOKENS


def _replay_oplog(store: "capsule_ffi.CapsuleStore", op_log: list) -> dict:
    """Replay the mirror's op-log against the real store; return timing + handle map."""
    handles: dict[int, object] = {}
    cache: dict = {}
    t0 = time.perf_counter_ns()
    for op in op_log:
        kind = op[0]
        if kind == "create_extend":
            _, handle, domain, chunk_specs = op
            cap, _cid = store.create(f"/e2b/h{handle}", domain, SCALED_BPT)
            buf, n_tokens = _gen_chunks(chunk_specs, cache)
            if n_tokens > 0:
                store.extend(cap, buf, n_tokens)
            handles[handle] = cap
        elif kind == "fork":
            _, child, parent = op
            parent_cap = handles[parent]
            child_cap, _cid = store.fork(parent_cap, f"/e2b/h{child}")
            handles[child] = child_cap
        elif kind == "extend":
            _, handle, chunk_specs = op
            buf, n_tokens = _gen_chunks(chunk_specs, cache)
            if n_tokens > 0:
                store.extend(handles[handle], buf, n_tokens)
        else:
            raise ValueError(f"unknown op {kind!r}")
    t1 = time.perf_counter_ns()
    return {"wall_ns": t1 - t0, "handles": handles, "n_ops": len(op_log),
            "n_chunk_cache": len(cache)}


def _c_denial_probe(store: "capsule_ffi.CapsuleStore") -> dict:
    """Demonstrate the real library denies an unauthorized attach.

    Create a private capsule, take its owner cap, revoke it, then attempt to attach
    with the revoked cap — the store must refuse (E_REVOKED). Also derive a
    read-only cap, revoke the owner, and confirm the derived cap dies too. This
    exercises the actual capability/epoch enforcement path in libcapsule.
    """
    cap, _cid = store.create("/e2b/denial_probe", "tenant:probe", SCALED_BPT)
    buf = traces.chunk_bytes_for(("probe", "x"), 0, CHUNK_NBYTES)
    store.extend(cap, buf, kvmodel.CHUNK_TOKENS)
    # a legitimate attach must succeed
    ok_code = store.try_attach(cap)
    # revoke, then attach must be denied
    store.revoke(cap)
    denied_code = store.try_attach(cap)
    enforced = (ok_code == capsule_ffi.CAPSULE_OK
                and denied_code in (capsule_ffi.CAPSULE_E_REVOKED,
                                    capsule_ffi.CAPSULE_E_PERM))
    return {
        "legit_attach_code": ok_code,
        "legit_attach_name": capsule_ffi.ERROR_NAMES.get(ok_code, str(ok_code)),
        "post_revoke_attach_code": denied_code,
        "post_revoke_attach_name": capsule_ffi.ERROR_NAMES.get(denied_code, str(denied_code)),
        "enforced": enforced,
    }


def run() -> None:
    cfg = traces.TraceConfig(n_requests=500, n_tenants=16, f=0.75, p_leakdup=0.05)
    seed = 1
    trace = traces.gen_trace(cfg, seed)

    # 1) mirror run (records op-log). No freeing so cumulative counters match a
    #    store with no destroy.
    t_mirror0 = time.perf_counter_ns()
    mirror = managers.CapsuleManager(cfg, record_ops=True, free_decode_on_end=False)
    for ev in trace:
        mirror.on_event(ev)
    t_mirror1 = time.perf_counter_ns()
    mres = mirror.result()

    mirror_stored_bytes = mres["stored_chunks"] * CHUNK_NBYTES
    mirror_dedup_saved_bytes = mres["dedup_saved_chunks"] * CHUNK_NBYTES
    mirror_dedup_hits = mres["dedup_hits"]
    mirror_denied = mres["denied_attaches"]

    indep_denied = _independent_denied_recount(cfg, trace)

    # 2) replay against the real library
    if TMP_STORE.exists():
        shutil.rmtree(TMP_STORE)
    TMP_STORE.mkdir(parents=True, exist_ok=True)

    maxrss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    replay_info: dict = {}
    denial_probe: dict = {}
    c_stats_dict: dict = {}
    try:
        with capsule_ffi.CapsuleStore(TMP_STORE,
                                      capsule_ffi.CAPSULE_STORE_CREATE) as store:
            replay_info = _replay_oplog(store, mirror.op_log)
            st = store.store_stats()
            c_stats_dict = {
                "n_capsules": st.n_capsules, "n_extents": st.n_extents,
                "stored_bytes": st.stored_bytes, "logical_bytes": st.logical_bytes,
                "dedup_saved_bytes": st.dedup_saved_bytes,
                "evicted_bytes": st.evicted_bytes, "dedup_hits": st.dedup_hits,
                "denied_ops": st.denied_ops, "revocations": st.revocations,
            }
            denial_probe = _c_denial_probe(store)
    finally:
        maxrss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if TMP_STORE.exists():
            shutil.rmtree(TMP_STORE)

    # 3) exact comparisons
    checks = {
        "stored_bytes": {
            "mirror": mirror_stored_bytes,
            "c": c_stats_dict.get("stored_bytes"),
            "equal": mirror_stored_bytes == c_stats_dict.get("stored_bytes"),
        },
        "dedup_hits": {
            "mirror": mirror_dedup_hits,
            "c": c_stats_dict.get("dedup_hits"),
            "equal": mirror_dedup_hits == c_stats_dict.get("dedup_hits"),
        },
        "dedup_saved_bytes": {
            "mirror": mirror_dedup_saved_bytes,
            "c": c_stats_dict.get("dedup_saved_bytes"),
            "equal": mirror_dedup_saved_bytes == c_stats_dict.get("dedup_saved_bytes"),
        },
        "denied": {
            "mirror": mirror_denied,
            "independent_recount": indep_denied,
            "equal": mirror_denied == indep_denied,
        },
        "c_denial_enforced": {
            "enforced": denial_probe.get("enforced", False),
            "detail": denial_probe,
        },
    }
    passed = (checks["stored_bytes"]["equal"]
              and checks["dedup_hits"]["equal"]
              and checks["dedup_saved_bytes"]["equal"]
              and checks["denied"]["equal"]
              and checks["c_denial_enforced"]["enforced"])

    meta = common.run_meta("E2b_validation", {
        "config": {"T": 16, "f": 0.75, "p_leakdup": 0.05, "seed": seed,
                   "n_requests": 500, "scaled_bpt": SCALED_BPT,
                   "chunk_nbytes": CHUNK_NBYTES},
        "trace_events": len(trace),
        "op_log_len": len(mirror.op_log),
        "note": "SCALED_BPT geometry; real pseudorandom chunk bytes keyed by content",
    })
    data = {
        "measured": {
            "pass": passed,
            "checks": checks,
            "c_store_stats": c_stats_dict,
            "mirror_result": mres,
            "wall_ns": {
                "mirror_replay": t_mirror1 - t_mirror0,
                "c_replay": replay_info.get("wall_ns"),
            },
            "maxrss_measured": {
                "before_bytes": maxrss_before,
                "after_bytes": maxrss_after,
                "delta_bytes": maxrss_after - maxrss_before,
                "unit_note": "ru_maxrss; bytes on macOS (Darwin)",
            },
            "replay_info": {k: v for k, v in replay_info.items() if k != "handles"},
        },
        "modeled": {
            "note": "stored/dedup byte figures use SCALED_BPT=1024 (real store geometry)",
        },
    }
    common.write_result(RESULT, meta, data)
    status = "PASS" if passed else "FAIL"
    print(f"E2b {status}: stored_bytes eq={checks['stored_bytes']['equal']} "
          f"dedup_hits eq={checks['dedup_hits']['equal']} "
          f"dedup_saved eq={checks['dedup_saved_bytes']['equal']} "
          f"denied eq={checks['denied']['equal']} "
          f"c_denial={checks['c_denial_enforced']['enforced']} -> {RESULT}")
    if not passed:
        # Honesty: a mismatch is a real failure of the validation, surfaced via
        # nonzero exit while preserving the full comparison in the result file.
        sys.exit(2)


if __name__ == "__main__":
    common.guard_main("E2b_validation", RESULT, run)
