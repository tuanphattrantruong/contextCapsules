"""store.py — put/get real KV state through libcapsule.

Adds two things the generic ctypes wrapper does not have, both of which
matter once the payload is a real half-gigabyte KV cache rather than a
synthetic buffer:

  * zero-copy extend: the numpy buffer's own pointer is handed to
    capsule_extend, so storing KV costs one hash+write, not a Python copy;
  * zero-copy attach: the capsule's mmap'd extents are wrapped as numpy
    views, so restoring KV reads straight out of the page cache through the
    MAP_PRIVATE mapping the capability granted.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import byref, c_uint8, c_uint32, c_uint64, c_void_p
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agentsim import capsule_ffi as ffi  # noqa: E402

CHUNK_TOKENS = ffi.CAPSULE_CHUNK_TOKENS


class CapsuleKVStore:
    """A libcapsule store that speaks real KV byte buffers."""

    def __init__(self, root, flags: int = ffi.CAPSULE_STORE_CREATE):
        self.lib = ffi.load_library()
        if flags & ffi.CAPSULE_STORE_CREATE:
            Path(root).parent.mkdir(parents=True, exist_ok=True)
        self._h = c_void_p()
        rc = self.lib.capsule_store_open(str(root).encode(), c_uint32(flags),
                                         byref(self._h))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        self.root = str(root)
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        if not self._closed:
            self.lib.capsule_store_close(self._h)
            self._closed = True

    # -- lifecycle ---------------------------------------------------------

    def create(self, path: str, domain: Optional[str], bytes_per_token: int):
        cap = ffi.CapsuleCap()
        cid = c_uint64()
        dom = None if domain is None else domain.encode()
        rc = self.lib.capsule_create(self._h, path.encode(), dom,
                                     c_uint32(bytes_per_token), byref(cap),
                                     byref(cid))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return cap, cid.value

    def extend_zero_copy(self, cap, buf: np.ndarray, n_tokens: int) -> int:
        """capsule_extend straight from the numpy buffer's own memory."""
        assert buf.flags["C_CONTIGUOUS"], "extend needs a contiguous buffer"
        ptr = buf.ctypes.data_as(c_void_p)
        rc = self.lib.capsule_extend(self._h, byref(cap), ptr,
                                     c_uint32(n_tokens))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return rc

    def seal(self, cap):
        rc = self.lib.capsule_seal(self._h, byref(cap))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))

    def fork(self, cap, child_path: str):
        ccap = ffi.CapsuleCap()
        cid = c_uint64()
        rc = self.lib.capsule_fork(self._h, byref(cap), child_path.encode(),
                                   byref(ccap), byref(cid))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return ccap, cid.value

    def derive(self, cap, rights: int, ttl_ms: int = 0):
        out = ffi.CapsuleCap()
        rc = self.lib.capsule_derive(self._h, byref(cap), c_uint32(rights),
                                     c_uint64(ttl_ms), byref(out))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return out

    def revoke(self, cap):
        rc = self.lib.capsule_revoke(self._h, byref(cap))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))

    def stat(self, cap) -> ffi.CapsuleStat:
        st = ffi.CapsuleStat()
        rc = self.lib.capsule_stat(self._h, byref(cap), byref(st))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return st

    def store_stats(self) -> ffi.CapsuleStoreStats:
        st = ffi.CapsuleStoreStats()
        rc = self.lib.capsule_store_stats(self._h, byref(st))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return st

    # -- data path ---------------------------------------------------------

    def attach(self, cap) -> ffi.CapsuleMap:
        m = ffi.CapsuleMap()
        rc = self.lib.capsule_attach(self._h, byref(cap), byref(m))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return m

    def detach(self, m: ffi.CapsuleMap):
        self.lib.capsule_detach(self._h, byref(m))

    @staticmethod
    def map_to_array(m: ffi.CapsuleMap) -> np.ndarray:
        """Wrap an attached capsule's extents as one uint8 array.

        Single-extent capsules (the common case for a sealed context) are a
        true zero-copy view of the MAP_PRIVATE mapping; multi-extent capsules
        are concatenated, which costs one copy across extent boundaries.
        """
        segs = []
        for i in range(m.nsegs):
            base, ln = m.seg_base[i], int(m.seg_len[i])
            arr_t = c_uint8 * ln
            segs.append(np.ctypeslib.as_array(
                ctypes.cast(base, ctypes.POINTER(arr_t)).contents))
        if len(segs) == 1:
            return segs[0]
        return np.concatenate(segs)

    def read(self, cap, off: int, length: int) -> np.ndarray:
        """Fully mediated read: capability validated on every call."""
        out = np.empty(length, dtype=np.uint8)
        rc = self.lib.capsule_read(self._h, byref(cap), c_uint64(off),
                                   c_uint64(length),
                                   out.ctypes.data_as(c_void_p))
        if rc < 0:
            raise ffi.CapsuleError(rc, ffi.ERROR_NAMES.get(rc))
        return out

    # -- convenience -------------------------------------------------------

    def put_kv(self, path: str, domain: Optional[str], geom, buf: np.ndarray,
               n_tokens: int, seal: bool = True,
               chunk_tokens_per_call: int = 256) -> Tuple[object, int]:
        """create + extend (in whole-chunk batches) + seal. Returns (cap, id)."""
        if n_tokens % CHUNK_TOKENS:
            raise ValueError(
                "n_tokens must be a multiple of %d (got %d)"
                % (CHUNK_TOKENS, n_tokens))
        cap, cid = self.create(path, domain, geom.bytes_per_token)
        bpt = geom.bytes_per_token
        step = max(CHUNK_TOKENS, chunk_tokens_per_call - (chunk_tokens_per_call % CHUNK_TOKENS))
        for t0 in range(0, n_tokens, step):
            t1 = min(t0 + step, n_tokens)
            sub = buf[t0 * bpt:t1 * bpt]
            self.extend_zero_copy(cap, sub, t1 - t0)
        if seal:
            self.seal(cap)
        return cap, cid


def dylib_mtime() -> float:
    p = ffi._dylib_path()
    return os.path.getmtime(p) if p.exists() else 0.0
