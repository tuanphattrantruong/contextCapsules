"""pack.py — token-major (de)serialization of a real transformer KV cache.

Native transformers layout is a list of L layers, each a pair of tensors
shaped [batch, kv_heads, seq, head_dim]. A single token's KV is therefore
scattered across 2L tensors at stride head_dim. Capsules need the opposite:
the bytes of tokens [t, t+16) contiguous, so that a chunk is a byte range
that can be hashed, deduplicated, stored as an extent, and mmap'd.

pack_kv produces the token-major buffer

    [seq][layer][K,V][kv_head][head_dim]

and unpack_kv inverts it exactly. Both are lossless: pack->unpack is bitwise
identity (verified in tests/test_kvbridge.py and in E5's fidelity study).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch


def _cache_layers(past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Normalize any transformers cache object into a list of (K, V) pairs."""
    return [(past_key_values[i][0], past_key_values[i][1])
            for i in range(len(past_key_values))]


def pack_kv_range(past_key_values, t0: int, t1: int) -> np.ndarray:
    """Serialize tokens [t0, t1) of a KV cache into one token-major byte array.

    Returns a C-contiguous uint8 numpy array of length
    (t1 - t0) * bytes_per_token. The tensors are moved to CPU once, as a
    single stacked copy, and transposed there.
    """
    layers = _cache_layers(past_key_values)
    n_layers = len(layers)
    if n_layers == 0:
        return np.empty(0, dtype=np.uint8)

    # [layer, 2, kv_heads, seq_slice, head_dim] on the accelerator, one stack.
    stacked = torch.stack(
        [torch.stack((k[0, :, t0:t1, :], v[0, :, t0:t1, :]), dim=0)
         for k, v in layers],
        dim=0,
    )
    # -> [seq, layer, 2, kv_heads, head_dim], token-major and contiguous.
    tokmajor = stacked.permute(3, 0, 1, 2, 4).contiguous()
    host = tokmajor.to("cpu")
    # bf16/f16 have no numpy dtype on all builds; view as raw bytes.
    flat = host.reshape(-1)
    buf = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    return buf.numpy().copy()


def pack_kv(past_key_values) -> np.ndarray:
    """Serialize the whole cache (all tokens) token-major."""
    layers = _cache_layers(past_key_values)
    seq = layers[0][0].shape[2]
    return pack_kv_range(past_key_values, 0, seq)


def unpack_kv(
    buf: np.ndarray,
    n_tokens: int,
    geom,
    device: str = "cpu",
    cache_cls: Optional[Any] = None,
):
    """Inverse of pack_kv: rebuild a transformers cache from capsule bytes.

    `geom` is a KVGeometry. Returns a DynamicCache (or `cache_cls`) holding
    tensors on `device`, bitwise identical to the ones pack_kv consumed.
    """
    from transformers.cache_utils import DynamicCache

    raw = torch.from_numpy(np.ascontiguousarray(buf, dtype=np.uint8))
    typed = raw.view(geom.dtype).reshape(
        n_tokens, geom.n_layers, 2, geom.n_kv_heads, geom.head_dim
    )
    # -> [layer, 2, kv_heads, seq, head_dim]
    per_layer = typed.permute(1, 2, 3, 0, 4).contiguous()

    legacy = []
    for li in range(geom.n_layers):
        k = per_layer[li, 0].unsqueeze(0).to(device)
        v = per_layer[li, 1].unsqueeze(0).to(device)
        legacy.append((k, v))

    cls = cache_cls or DynamicCache
    return cls.from_legacy_cache(tuple(legacy))


def cache_equal(a, b) -> Tuple[bool, float]:
    """Bitwise comparison of two caches. Returns (equal, max_abs_diff)."""
    la, lb = _cache_layers(a), _cache_layers(b)
    if len(la) != len(lb):
        return False, float("inf")
    worst = 0.0
    for (ka, va), (kb, vb) in zip(la, lb):
        if ka.shape != kb.shape or va.shape != vb.shape:
            return False, float("inf")
        ka_c, kb_c = ka.to("cpu"), kb.to("cpu")
        va_c, vb_c = va.to("cpu"), vb.to("cpu")
        if not (torch.equal(ka_c, kb_c) and torch.equal(va_c, vb_c)):
            worst = max(
                worst,
                float((ka_c.float() - kb_c.float()).abs().max()),
                float((va_c.float() - vb_c.float()).abs().max()),
            )
    return worst == 0.0, worst
