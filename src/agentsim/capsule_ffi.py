"""capsule_ffi.py — ctypes bridge to libcapsule (mirrors DESIGN.md 2.7 EXACTLY).

Package 2 codes ONLY against the frozen ABI in DESIGN.md 2.7 and the dylib at
src/libcapsule/libcapsule.dylib (override with env CAPSULE_LIB). It never reads
package 1's C sources. Every struct here is a field-for-field mirror of the frozen
header; every function has restype c_int and a negative return is an error mapped
to CapsuleError with the DESIGN.md 2.7 name.

A thin pythonic wrapper (CapsuleStore) raises on negative returns and manages the
opaque store handle.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER, Structure, c_char_p, c_int, c_uint8, c_uint32, c_uint64, c_void_p,
    byref,
)
from pathlib import Path
from typing import Any

# --- ABI constants (DESIGN.md 2.7) ------------------------------------------

CAPSULE_ABI_VERSION = 2
CAPSULE_CHUNK_TOKENS = 16
CAPSULE_MAX_EXTENTS = 128

# store flags
CAPSULE_STORE_CREATE = 0x1
CAPSULE_STORE_RDONLY = 0x2
CAPSULE_STORE_PARANOID = 0x4

# rights
CAP_R_READ = 0x01
CAP_R_EXTEND = 0x02
CAP_R_SHARE = 0x04
CAP_R_SEAL = 0x08
CAP_R_REVOKE = 0x10
CAP_R_ALL = 0x1F

# errors (negative)
CAPSULE_OK = 0
CAPSULE_E_PERM = -1
CAPSULE_E_EXPIRED = -2
CAPSULE_E_REVOKED = -3
CAPSULE_E_SEALED = -4
CAPSULE_E_EVICTED = -5
CAPSULE_E_NOENT = -6
CAPSULE_E_EXISTS = -7
CAPSULE_E_IO = -8
CAPSULE_E_INVAL = -9
CAPSULE_E_BUDGET = -10
CAPSULE_E_LOCK = -11

ERROR_NAMES = {
    CAPSULE_OK: "CAPSULE_OK",
    CAPSULE_E_PERM: "CAPSULE_E_PERM",
    CAPSULE_E_EXPIRED: "CAPSULE_E_EXPIRED",
    CAPSULE_E_REVOKED: "CAPSULE_E_REVOKED",
    CAPSULE_E_SEALED: "CAPSULE_E_SEALED",
    CAPSULE_E_EVICTED: "CAPSULE_E_EVICTED",
    CAPSULE_E_NOENT: "CAPSULE_E_NOENT",
    CAPSULE_E_EXISTS: "CAPSULE_E_EXISTS",
    CAPSULE_E_IO: "CAPSULE_E_IO",
    CAPSULE_E_INVAL: "CAPSULE_E_INVAL",
    CAPSULE_E_BUDGET: "CAPSULE_E_BUDGET",
    CAPSULE_E_LOCK: "CAPSULE_E_LOCK",
}

# eviction policies
POLICY_LRU = 0
POLICY_LINEAGE = 1


# --- opaque token -----------------------------------------------------------

# capsule_cap_t is { uint8_t bytes[72]; }
CapsuleCap = c_uint8 * 72


class CapsuleMap(Structure):
    """capsule_map_t (caller-allocated)."""
    _fields_ = [
        ("nsegs", c_uint32),
        ("_pad", c_uint32),
        ("total_bytes", c_uint64),
        ("seg_base", c_void_p * CAPSULE_MAX_EXTENTS),
        ("seg_len", c_uint64 * CAPSULE_MAX_EXTENTS),
        ("_internal", c_uint64 * 4),
    ]


class CapsuleStat(Structure):
    """capsule_stat_t (DESIGN.md field order)."""
    _fields_ = [
        ("capsule_id", c_uint64),
        ("bytes_per_token", c_uint32),
        ("chunk_tokens", c_uint32),
        ("n_tokens", c_uint64),
        ("referenced_bytes", c_uint64),
        ("owned_bytes", c_uint64),
        ("n_extents", c_uint32),
        ("epoch", c_uint32),
        ("flags", c_uint32),
        ("live_attaches", c_uint32),
        ("chain_hash", c_uint8 * 32),
        ("compatibility_hash", c_uint8 * 32),
        ("created_us", c_uint64),
        ("last_access_us", c_uint64),
        ("pin_deadline_us", c_uint64),
    ]


class CapsuleStoreStats(Structure):
    """capsule_store_stats_t — 9 x uint64 in the frozen order."""
    _fields_ = [
        ("n_capsules", c_uint64),
        ("n_extents", c_uint64),
        ("stored_bytes", c_uint64),
        ("logical_bytes", c_uint64),
        ("dedup_saved_bytes", c_uint64),
        ("evicted_bytes", c_uint64),
        ("dedup_hits", c_uint64),
        ("denied_ops", c_uint64),
        ("revocations", c_uint64),
    ]


# --- errors -----------------------------------------------------------------

class CapsuleError(Exception):
    def __init__(self, code: int, name: str | None = None):
        self.code = code
        self.name = name or ERROR_NAMES.get(code, f"CAPSULE_E_UNKNOWN({code})")
        super().__init__(f"{self.name} (code {code})")


class CapsuleLibraryMissing(RuntimeError):
    pass


# --- library loading & prototypes -------------------------------------------

def _dylib_path() -> Path:
    env = os.environ.get("CAPSULE_LIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "libcapsule" / "libcapsule.dylib"


_LIB: ctypes.CDLL | None = None


def load_library() -> ctypes.CDLL:
    """Load and prototype libcapsule.dylib (cached). Raises a clear error if absent."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _dylib_path()
    if not path.exists():
        raise CapsuleLibraryMissing(
            f"libcapsule dylib not found at {path}. Build package 1 first:\n"
            f"    make -C src/libcapsule\n"
            f"(or set CAPSULE_LIB to the dylib path)."
        )
    lib = ctypes.CDLL(str(path))
    _prototype(lib)
    _LIB = lib
    return lib


def _prototype(lib: ctypes.CDLL) -> None:
    P = POINTER
    lib.capsule_store_open.argtypes = [c_char_p, c_uint32, P(c_void_p)]
    lib.capsule_store_open.restype = c_int
    lib.capsule_store_close.argtypes = [c_void_p]
    lib.capsule_store_close.restype = c_int
    lib.capsule_store_stats.argtypes = [c_void_p, P(CapsuleStoreStats)]
    lib.capsule_store_stats.restype = c_int

    lib.capsule_create.argtypes = [c_void_p, c_char_p, c_char_p, c_uint32,
                                   P(CapsuleCap), P(c_uint64)]
    lib.capsule_create.restype = c_int
    lib.capsule_create_compatible.argtypes = [c_void_p, c_char_p, c_char_p,
        c_uint32, POINTER(c_uint8), P(CapsuleCap), P(c_uint64)]
    lib.capsule_create_compatible.restype = c_int
    lib.capsule_create_chained.argtypes = [c_void_p, c_char_p, c_char_p, c_uint32,
                                           P(CapsuleCap), P(CapsuleCap), P(c_uint64)]
    lib.capsule_create_chained.restype = c_int
    lib.capsule_extend.argtypes = [c_void_p, P(CapsuleCap), c_void_p, c_uint32]
    lib.capsule_extend.restype = c_int
    lib.capsule_seal.argtypes = [c_void_p, P(CapsuleCap)]
    lib.capsule_seal.restype = c_int
    lib.capsule_fork.argtypes = [c_void_p, P(CapsuleCap), c_char_p,
                                 P(CapsuleCap), P(c_uint64)]
    lib.capsule_fork.restype = c_int
    lib.capsule_attach.argtypes = [c_void_p, P(CapsuleCap), P(CapsuleMap)]
    lib.capsule_attach.restype = c_int
    lib.capsule_attach_compatible.argtypes = [c_void_p, P(CapsuleCap),
        POINTER(c_uint8), P(CapsuleMap)]
    lib.capsule_attach_compatible.restype = c_int
    lib.capsule_detach.argtypes = [c_void_p, P(CapsuleMap)]
    lib.capsule_detach.restype = c_int
    lib.capsule_read.argtypes = [c_void_p, P(CapsuleCap), c_uint64, c_uint64, c_void_p]
    lib.capsule_read.restype = c_int
    lib.capsule_derive.argtypes = [c_void_p, P(CapsuleCap), c_uint32, c_uint64,
                                   P(CapsuleCap)]
    lib.capsule_derive.restype = c_int
    lib.capsule_revoke.argtypes = [c_void_p, P(CapsuleCap)]
    lib.capsule_revoke.restype = c_int
    lib.capsule_pin.argtypes = [c_void_p, P(CapsuleCap), c_uint64]
    lib.capsule_pin.restype = c_int
    lib.capsule_destroy.argtypes = [c_void_p, P(CapsuleCap)]
    lib.capsule_destroy.restype = c_int
    lib.capsule_stat.argtypes = [c_void_p, P(CapsuleCap), P(CapsuleStat)]
    lib.capsule_stat.restype = c_int
    lib.capsule_evict.argtypes = [c_void_p, c_uint64, c_int]
    lib.capsule_evict.restype = c_int


def _check(code: int) -> int:
    if code < 0:
        raise CapsuleError(code, ERROR_NAMES.get(code))
    return code


# --- pythonic wrapper -------------------------------------------------------

class CapsuleStore:
    """Thin RAII wrapper over a libcapsule store handle.

    All methods raise CapsuleError on negative return codes. Capabilities are
    returned/accepted as CapsuleCap (ctypes uint8*72) buffers.
    """

    def __init__(self, root: str | os.PathLike[str], flags: int = CAPSULE_STORE_CREATE):
        self.lib = load_library()
        self._handle = c_void_p()
        root_b = str(root).encode("utf-8")
        _check(self.lib.capsule_store_open(root_b, c_uint32(flags),
                                           byref(self._handle)))
        self._closed = False

    # context manager
    def __enter__(self) -> "CapsuleStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed and self._handle:
            self.lib.capsule_store_close(self._handle)
            self._closed = True

    # -- operations ---------------------------------------------------------

    def create(self, path: str, dedup_domain: str | None, bytes_per_token: int):
        cap = CapsuleCap()
        cid = c_uint64()
        dom = None if dedup_domain is None else dedup_domain.encode("utf-8")
        _check(self.lib.capsule_create(self._handle, path.encode("utf-8"), dom,
                                       c_uint32(bytes_per_token), byref(cap),
                                       byref(cid)))
        return cap, cid.value

    def create_chained(self, path: str, dedup_domain, bytes_per_token: int,
                       chain_parent):
        """capsule_create_chained: chain_0 = H(domain_seed || parent.chain)."""
        cap = CapsuleCap()
        cid = c_uint64()
        dom = None if dedup_domain is None else dedup_domain.encode("utf-8")
        _check(self.lib.capsule_create_chained(
            self._handle, path.encode("utf-8"), dom, c_uint32(bytes_per_token),
            byref(chain_parent), byref(cap), byref(cid)))
        return cap, cid.value

    def create_compatible(self, path: str, dedup_domain, bytes_per_token: int,
                          compatibility_hash: bytes):
        if len(compatibility_hash) != 32:
            raise ValueError("compatibility_hash must be exactly 32 bytes")
        cap = CapsuleCap()
        cid = c_uint64()
        digest = (c_uint8 * 32).from_buffer_copy(compatibility_hash)
        dom = None if dedup_domain is None else dedup_domain.encode("utf-8")
        _check(self.lib.capsule_create_compatible(
            self._handle, path.encode("utf-8"), dom, c_uint32(bytes_per_token),
            digest, byref(cap), byref(cid)))
        return cap, cid.value

    def extend(self, cap: CapsuleCap, kv: bytes, n_tokens: int) -> None:
        buf = (c_uint8 * len(kv)).from_buffer_copy(kv)
        _check(self.lib.capsule_extend(self._handle, byref(cap),
                                       ctypes.cast(buf, c_void_p),
                                       c_uint32(n_tokens)))

    def seal(self, cap: CapsuleCap) -> None:
        _check(self.lib.capsule_seal(self._handle, byref(cap)))

    def fork(self, cap: CapsuleCap, child_path: str):
        child = CapsuleCap()
        cid = c_uint64()
        _check(self.lib.capsule_fork(self._handle, byref(cap),
                                     child_path.encode("utf-8"), byref(child),
                                     byref(cid)))
        return child, cid.value

    def attach(self, cap: CapsuleCap) -> CapsuleMap:
        m = CapsuleMap()
        _check(self.lib.capsule_attach(self._handle, byref(cap), byref(m)))
        return m

    def attach_compatible(self, cap: CapsuleCap,
                          expected_hash: bytes) -> CapsuleMap:
        if len(expected_hash) != 32:
            raise ValueError("expected_hash must be exactly 32 bytes")
        digest = (c_uint8 * 32).from_buffer_copy(expected_hash)
        m = CapsuleMap()
        _check(self.lib.capsule_attach_compatible(
            self._handle, byref(cap), digest, byref(m)))
        return m

    def detach(self, m: CapsuleMap) -> None:
        _check(self.lib.capsule_detach(self._handle, byref(m)))

    def read(self, cap: CapsuleCap, off: int, length: int) -> bytes:
        buf = (c_uint8 * length)()
        _check(self.lib.capsule_read(self._handle, byref(cap), c_uint64(off),
                                     c_uint64(length), ctypes.cast(buf, c_void_p)))
        return bytes(buf)

    def derive(self, parent: CapsuleCap, rights: int, ttl_ms: int) -> CapsuleCap:
        out = CapsuleCap()
        _check(self.lib.capsule_derive(self._handle, byref(parent), c_uint32(rights),
                                       c_uint64(ttl_ms), byref(out)))
        return out

    def revoke(self, cap: CapsuleCap) -> None:
        _check(self.lib.capsule_revoke(self._handle, byref(cap)))

    def pin(self, cap: CapsuleCap, ttl_ms: int) -> None:
        _check(self.lib.capsule_pin(self._handle, byref(cap), c_uint64(ttl_ms)))

    def destroy(self, cap: CapsuleCap) -> None:
        _check(self.lib.capsule_destroy(self._handle, byref(cap)))

    def stat(self, cap: CapsuleCap) -> CapsuleStat:
        st = CapsuleStat()
        _check(self.lib.capsule_stat(self._handle, byref(cap), byref(st)))
        return st

    def store_stats(self) -> CapsuleStoreStats:
        st = CapsuleStoreStats()
        _check(self.lib.capsule_store_stats(self._handle, byref(st)))
        return st

    def evict(self, target_resident_bytes: int, policy: int) -> int:
        return _check(self.lib.capsule_evict(self._handle,
                                             c_uint64(target_resident_bytes),
                                             c_int(policy)))

    # -- non-raising variant for expected denials ---------------------------

    def try_attach(self, cap: CapsuleCap) -> int:
        """Attach returning the raw code (does not raise). For denial tests."""
        m = CapsuleMap()
        code = self.lib.capsule_attach(self._handle, byref(cap), byref(m))
        if code == CAPSULE_OK:
            self.detach(m)
        return code


def library_available() -> bool:
    """True if the dylib exists and loads (without raising)."""
    try:
        load_library()
        return True
    except Exception:
        return False
