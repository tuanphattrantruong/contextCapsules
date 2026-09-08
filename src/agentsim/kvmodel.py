"""kvmodel.py — KV geometry constants (documented model constants, NOT measurements).

These are the sole source of the token->bytes mapping used throughout agentsim.
Any number derived by multiplying a token count by one of these constants is
labeled "modeled" in results JSON (DESIGN.md section 1.1 / section 8). Nothing in
this module is a measurement.

REAL_BPT geometry (Llama-3-8B-class GQA):
    32 layers x 2 (K,V) x 8 KV heads x 128 head_dim x 2 bytes (fp16)
      = 32 * 2 * 8 * 128 * 2
      = 131072 bytes / token.
    Reference: Meta Llama-3-8B config — hidden_size 4096, num_attention_heads 32,
    num_key_value_heads 8 (GQA), head_dim 128, num_hidden_layers 32, fp16 weights.
    (See the Llama 3 model card / config.json: num_key_value_heads=8, so the KV
    cache stores 8 heads, not 32 — this is the GQA reduction.)

    chunk = CHUNK_TOKENS (16) tokens -> real chunk = 16 * 131072 = 2 MiB.
    A 4096-token context = 4096 * 131072 = 512 MiB (the brief's "512 MB capsule").

SCALED_BPT geometry (high-volume validation runs, E2b):
    1024 bytes / token -> chunk = 16 * 1024 = 16384 bytes = exactly one 16 KiB page.

Constraint (DESIGN.md 1.1): chunk bytes (16 * bytes_per_token) must be a multiple
of the 16384-byte page size, i.e. bytes_per_token % 1024 == 0, else the C library
returns CAPSULE_E_INVAL at create. Both constants below satisfy this.
"""

# --- documented model constants (bytes per token) ---------------------------

# Llama-3-8B-class GQA: 32 layers * 2 (K,V) * 8 kv_heads * 128 head_dim * 2 B (fp16)
REAL_BPT = 131072
assert REAL_BPT == 32 * 2 * 8 * 128 * 2, "REAL_BPT geometry drift"

# Scaled geometry for high-volume validation: 1 chunk == 1 page (16 KiB).
SCALED_BPT = 1024

# Token-block granularity (matches CAPSULE_CHUNK_TOKENS in the frozen ABI).
CHUNK_TOKENS = 16

# Page size on this machine (Apple Silicon). Documented for the constraint check.
PAGE_BYTES = 16384

# Both constants must keep chunk bytes a multiple of the page size.
assert (CHUNK_TOKENS * REAL_BPT) % PAGE_BYTES == 0, "REAL_BPT chunk not page-aligned"
assert (CHUNK_TOKENS * SCALED_BPT) % PAGE_BYTES == 0, "SCALED_BPT chunk not page-aligned"


def tokens_to_bytes(tokens: int, bpt: int = REAL_BPT) -> int:
    """Modeled byte size of `tokens` tokens at `bpt` bytes/token.

    Result is a MODELED quantity (token count * documented constant), never a
    measurement. Callers labeling results must place this under a "modeled" key.
    """
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative, got {tokens}")
    return int(tokens) * int(bpt)


def round_up_chunks(tokens: int) -> int:
    """Round a token count up to the next multiple of CHUNK_TOKENS (16).

    The trace model and all managers account KV in whole 16-token blocks, so every
    segment / decode length is rounded up here before use.
    """
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative, got {tokens}")
    t = int(tokens)
    rem = t % CHUNK_TOKENS
    if rem:
        t += CHUNK_TOKENS - rem
    return t


def n_blocks(tokens: int) -> int:
    """Number of whole 16-token blocks covering `tokens` (rounds up)."""
    return round_up_chunks(tokens) // CHUNK_TOKENS


def chunk_bytes(bpt: int) -> int:
    """Bytes in one 16-token chunk at `bpt` bytes/token."""
    return CHUNK_TOKENS * int(bpt)
