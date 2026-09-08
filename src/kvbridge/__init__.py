"""kvbridge — real LLM KV state <-> libcapsule.

This package is the bridge that lets context capsules hold the actual
key/value tensors a transformer produces, rather than modeled byte counts.
It is deliberately small and dependency-light: torch + transformers for the
model side, the frozen libcapsule ABI (via agentsim.capsule_ffi) for the
store side.

Layout contract (see geometry.KVGeometry): capsule bytes are TOKEN-MAJOR,

    [token][layer][K,V][kv_head][head_dim]

so the bytes of any contiguous token range are contiguous in the capsule.
That is what makes 16-token chunking, content addressing, and prefix
deduplication meaningful on real KV state; the native transformers layout
([layer] -> [batch, kv_head, seq, head_dim]) is seq-strided and would make
a chunk a gather.
"""

from .geometry import KVGeometry
from .pack import pack_kv, unpack_kv, pack_kv_range
from .store import CapsuleKVStore

__all__ = [
    "KVGeometry",
    "pack_kv",
    "pack_kv_range",
    "unpack_kv",
    "CapsuleKVStore",
]
