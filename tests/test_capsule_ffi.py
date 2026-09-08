"""test_capsule_ffi.py — adversarial cross-package tests against the REAL library.

Package 2 talks to libcapsule only through the frozen ABI (capsule_ffi). These
tests exercise the correctness-critical mechanisms of DESIGN.md end-to-end via
ctypes on the actual dylib:

  * capability enforcement   — forged caps, rights attenuation, expiry,
                               cross-process-safe revocation, cross-domain isolation
  * contract accounting      — dedup exactness within a domain, fork = O(metadata)
                               (no byte copy / no new extents), store_stats coherence
  * transactional atomicity  — a rejected op leaves store counters unchanged;
                               committed state survives close/reopen (tmp+rename
                               manifests + prefix-index rebuild) and still dedups
  * lifecycle / erasure      — destroy removes the capsule; reopen no longer sees it

Every store lives under experiments/tmp_store_py/itests/* (inside the project) and
is removed by the harness. Requires the dylib built: make -C src/libcapsule.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import sys
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from agentsim import capsule_ffi as F  # noqa: E402

BPT = 1024                       # scaled: chunk = 16 KiB, satisfies %1024
CB = 16 * BPT                    # chunk bytes = 16384
TMP = _HERE.parents[1] / "experiments" / "tmp_store_py" / "itests"


def _store(name: str, flags: int = F.CAPSULE_STORE_CREATE) -> F.CapsuleStore:
    root = TMP / name
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    return F.CapsuleStore(root, flags)


def _content(nchunks: int, seed: int) -> bytes:
    """Deterministic distinct bytes: identical (nchunks, seed) -> identical bytes."""
    out = bytearray()
    x = (seed * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for _ in range(nchunks * CB):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        out.append(x & 0xFF)
    return bytes(out)


def _forge(cap: F.CapsuleCap, byte_idx: int) -> F.CapsuleCap:
    raw = bytearray(bytes(cap))
    raw[byte_idx] ^= 0x01
    return F.CapsuleCap.from_buffer_copy(bytes(raw))


# --------------------------------------------------------------------------- #
#  capability enforcement
# --------------------------------------------------------------------------- #

def test_forged_cap_denied():
    s = _store("forge")
    try:
        cap, cid = s.create("/a", "", BPT)
        s.extend(cap, _content(1, 1), 16)
        # flip a byte inside the HMAC (offset 40..71) -> HMAC-first rejection
        forged = _forge(cap, 45)
        try:
            s.read(forged, 0, 16)
            assert False, "forged cap must not read"
        except F.CapsuleError as e:
            assert e.code == F.CAPSULE_E_PERM, e.name
        # a fully random cap for a NONEXISTENT id also fails at the HMAC step
        bogus = F.CapsuleCap.from_buffer_copy(bytes([0xAB] * 72))
        assert s.try_attach(bogus) == F.CAPSULE_E_PERM
    finally:
        s.close()


def test_rights_attenuation():
    s = _store("rights")
    try:
        owner, _ = s.create("/a", "", BPT)
        s.extend(owner, _content(1, 2), 16)
        ro = s.derive(owner, F.CAP_R_READ, 0)          # read-only
        # read allowed, extend/seal denied
        s.read(ro, 0, 16)
        for op in (lambda: s.extend(ro, _content(1, 2), 16), lambda: s.seal(ro)):
            try:
                op(); assert False, "RO cap must not mutate"
            except F.CapsuleError as e:
                assert e.code == F.CAPSULE_E_PERM, e.name
        # cannot escalate rights, and cannot derive from a cap lacking SHARE
        for rights in (F.CAP_R_ALL, F.CAP_R_READ):
            try:
                s.derive(ro, rights, 0); assert False, "derive from RO must fail"
            except F.CapsuleError as e:
                assert e.code == F.CAPSULE_E_PERM, e.name
    finally:
        s.close()


def test_expiry():
    s = _store("expiry")
    try:
        owner, _ = s.create("/a", "", BPT)
        s.extend(owner, _content(1, 3), 16)
        shortlived = s.derive(owner, F.CAP_R_READ, 1)   # 1 ms ttl
        time.sleep(0.01)                                # 10 ms
        try:
            s.read(shortlived, 0, 16); assert False, "expired cap must fail"
        except F.CapsuleError as e:
            assert e.code == F.CAPSULE_E_EXPIRED, e.name
        s.read(owner, 0, 16)                            # owner (no expiry) still ok
    finally:
        s.close()


def test_revocation_kills_outstanding_caps():
    s = _store("revoke")
    try:
        owner, _ = s.create("/a", "", BPT)
        s.extend(owner, _content(1, 4), 16)
        ro = s.derive(owner, F.CAP_R_READ, 0)
        s.read(ro, 0, 16)                               # ok before revoke
        s.revoke(owner)
        for cap in (ro, owner):
            try:
                s.read(cap, 0, 16); assert False, "revoked cap must fail"
            except F.CapsuleError as e:
                assert e.code == F.CAPSULE_E_REVOKED, e.name
    finally:
        s.close()


def test_cross_domain_isolation():
    """Identical bytes in different dedup domains must NOT share (core protection)."""
    s = _store("domain")
    try:
        data = _content(2, 42)
        a, _ = s.create("/a", "", BPT)
        s.extend(a, data, 32)
        st = s.store_stats()
        base_stored = st.stored_bytes
        assert base_stored == 2 * CB

        # same bytes, SAME domain -> full dedup, no new stored bytes
        c, _ = s.create("/c", "", BPT)
        s.extend(c, data, 32)
        st = s.store_stats()
        assert st.dedup_hits == 2, st.dedup_hits
        assert st.stored_bytes == base_stored, "same-domain identical content shares"

        # same bytes, DIFFERENT domain -> no dedup, stored doubles
        b, _ = s.create("/b", "tenant:X", BPT)
        s.extend(b, data, 32)
        st = s.store_stats()
        assert st.dedup_hits == 2, "cross-domain must not dedup"
        assert st.stored_bytes == base_stored + 2 * CB, "isolated copy stored"
    finally:
        s.close()


# --------------------------------------------------------------------------- #
#  contract accounting
# --------------------------------------------------------------------------- #

def test_fork_is_metadata_only():
    s = _store("fork")
    try:
        p, _ = s.create("/p", "", BPT)
        s.extend(p, _content(4, 7), 64)
        before = s.store_stats()
        child, cid = s.fork(p, "/p/c")
        after = s.store_stats()
        assert after.stored_bytes == before.stored_bytes, "fork copies no bytes"
        assert after.n_extents == before.n_extents, "fork adds no extents"
        sp, sc = s.stat(p), s.stat(child)
        assert sc.n_tokens == sp.n_tokens == 64
        assert bytes(sc.chain_hash) == bytes(sp.chain_hash), "child inherits name"
        # extending the child does not touch the parent
        s.extend(child, _content(1, 99), 16)
        assert s.stat(p).n_tokens == 64 and s.stat(child).n_tokens == 80
    finally:
        s.close()


def test_rejected_op_leaves_state_unchanged():
    """Transactional atomicity: a rejected extend must not mutate counters."""
    s = _store("atomic")
    try:
        cap, _ = s.create("/a", "", BPT)
        s.extend(cap, _content(2, 8), 32)
        before = s.store_stats()
        n_before = s.stat(cap).n_tokens
        # invalid n_tokens (not a multiple of 16) -> E_INVAL, no state change
        try:
            s.extend(cap, _content(1, 8), 8); assert False
        except F.CapsuleError as e:
            assert e.code == F.CAPSULE_E_INVAL, e.name
        # extend after seal -> E_SEALED, no state change
        s.seal(cap)
        try:
            s.extend(cap, _content(1, 8), 16); assert False
        except F.CapsuleError as e:
            assert e.code == F.CAPSULE_E_SEALED, e.name
        after = s.store_stats()
        assert after.stored_bytes == before.stored_bytes
        assert after.n_extents == before.n_extents
        assert s.stat(cap).n_tokens == n_before
    finally:
        s.close()


# --------------------------------------------------------------------------- #
#  persistence / lifecycle (atomic manifests + prefix-index rebuild)
# --------------------------------------------------------------------------- #

def test_reopen_persists_and_dedups():
    root = TMP / "persist"
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    data = _content(3, 55)
    chain_before = None
    with F.CapsuleStore(root, F.CAPSULE_STORE_CREATE) as s:
        cap, _ = s.create("/a", "dom", BPT)
        s.extend(cap, data, 48)
        s.seal(cap)
        chain_before = bytes(s.stat(cap).chain_hash)
    # reopen: manifests + prefix index rebuilt from disk under LOCK_SH
    with F.CapsuleStore(root, 0) as s:
        b, _ = s.create("/b", "dom", BPT)
        s.extend(b, data, 48)
        assert bytes(s.stat(b).chain_hash) == chain_before, "content name stable"
        st = s.store_stats()
        assert st.dedup_hits == 3, "fully dedups against rebuilt index"
    shutil.rmtree(root)


def test_destroy_removes_capsule():
    root = TMP / "destroy"
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with F.CapsuleStore(root, F.CAPSULE_STORE_CREATE) as s:
        cap, cid = s.create("/d", "solo", BPT)
        s.extend(cap, _content(2, 0xDEAD), 32)
        n_before = s.store_stats().n_capsules
        s.destroy(cap)
        # the destroyed cap is dead (epoch bumped) — no existence signal leak
        try:
            s.read(cap, 0, 16); assert False
        except F.CapsuleError as e:
            assert e.code in (F.CAPSULE_E_REVOKED, F.CAPSULE_E_NOENT), e.name
        assert s.store_stats().n_capsules == n_before - 1
    # reopen: destroyed capsule's manifest is gone
    with F.CapsuleStore(root, 0) as s:
        assert s.store_stats().n_capsules == 0
    shutil.rmtree(root)
