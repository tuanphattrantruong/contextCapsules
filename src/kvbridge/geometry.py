"""geometry.py — KV geometry derived from a real model config.

Everything here is *measured from the loaded model*, not assumed. The paper's
former modeled constant (131072 B/token for a Llama-3-8B-class model) is
reproduced by this code as an output, not an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
}


@dataclass(frozen=True)
class KVGeometry:
    """Physical KV geometry of one model, as loaded."""

    model_id: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype: Any
    itemsize: int

    @property
    def bytes_per_token(self) -> int:
        """K and V, every layer, every KV head."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.itemsize

    @property
    def bytes_per_chunk(self) -> int:
        """One 16-token capsule chunk (CAPSULE_CHUNK_TOKENS)."""
        return 16 * self.bytes_per_token

    @property
    def elems_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
            "itemsize": self.itemsize,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_chunk_16tok": self.bytes_per_chunk,
        }

    @classmethod
    def from_model(cls, model, model_id: str = "") -> "KVGeometry":
        """Derive geometry from a loaded HF model (config + real parameter dtype)."""
        cfg = model.config
        n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
        n_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
        dtype = next(model.parameters()).dtype
        return cls(
            model_id=model_id or getattr(cfg, "_name_or_path", "unknown"),
            n_layers=int(n_layers),
            n_kv_heads=int(n_kv),
            head_dim=int(head_dim),
            dtype=dtype,
            itemsize=_DTYPE_BYTES[dtype],
        )

    @classmethod
    def from_cache(cls, past_key_values, model_id: str = "") -> "KVGeometry":
        """Derive geometry from an actual populated cache (the ground truth)."""
        n_layers = len(past_key_values)
        k, _v = past_key_values[0]
        _b, n_kv, _s, head_dim = k.shape
        return cls(
            model_id=model_id,
            n_layers=int(n_layers),
            n_kv_heads=int(n_kv),
            head_dim=int(head_dim),
            dtype=k.dtype,
            itemsize=_DTYPE_BYTES[k.dtype],
        )
