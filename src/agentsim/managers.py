"""managers.py — the three KV-state managers (DESIGN.md 3.2).

All three consume the same trace events (m.on_event(ev)) and expose m.result() ->
dict of MEASURED counters. Byte conversions (× REAL_BPT / SCALED_BPT, labeled
"modeled") happen in metrics.py, not here.

Accounting unit: the 16-token block. A logical *segment* (a specific sys prompt,
gdoc, tdoc, or a request's decode) is an atomic run of blocks: its content is
fixed, so two references to the same content produce byte-identical chunks and
either fully deduplicate or not at all. This lets every manager account at
segment-run granularity (O(segments/event)) while remaining exact at the chunk
level (proven below), so 162 × 10k requests finish in minutes.

Content identity vs. dedup domain
---------------------------------
content key of a block = (content-owner marker, kind, content-id, block-idx):
  sys / gdoc  -> owner marker "*" (public corpus)
  tdoc        -> owner marker = the CONTENT owner tenant (recovered from the
                 segment id via traces.decode_tdoc_id); leak-dup tdocs therefore
                 share content keys with the owning tenant's original.
  decode      -> owner marker = the using tenant, id = req (genuinely per-request).
The *dedup domain* (capsule admission scope) is a separate axis:
  public kinds -> ""            (shared across all tenants)
  tdoc/decode  -> "tenant:<k>"  where k is the USING tenant.
So leak-dup content has identical bytes (same content key) but a different dedup
domain from the owner's original (using tenant k != owner m) — the exact split the
study turns on: the flat prefix cache keys by content only and reuses it across
tenants (leak); capsules key by (domain-seeded chain) and refuse (denied attach).

Chain hashing (both position-chained and domain-seeded variants) mirrors DESIGN.md
2.5: chain_0 = SHA256(seed), chain_{i+1} = SHA256(chain_i || content-id-bytes_i).
Because distinct segment contents differ from block 0 (pseudorandom bytes keyed by
content key), a run of whole segments deduplicates exactly as libcapsule's
per-chunk extend would within a single extend call (dedup mode adopts the matching
leading segments, then stores from the first divergent segment onward).

Modeling choices (documented for honesty; see workloads/README.md):
  * PrefixCacheManager is vLLM/RadixAttention-faithful: one flat cache keyed by the
    cumulative chain FROM POSITION 0 over all segments (public + private), no tenant
    checks, never frees in E2 (records peak).
  * CapsuleManager is libcapsule-faithful: each request composes a shared PUBLIC
    capsule (domain "", chain over its public segments) and a PRIVATE capsule
    (domain "tenant:k", chain over its private segments = tdocs then decode).
    Fork is an O(1) reference to the parent's capsules plus the child's own decode.
    Cross-tenant private dedup is impossible (domain-seeded chains never collide),
    so leak-dup content is stored per tenant and counted as a denied attach.
  * NoShareManager materializes every request's full context privately and frees it
    at end; forks re-materialize (recompute) the inherited context.
These are genuinely different systems; the results report each honestly rather than
forcing an identical sharing granularity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from . import kvmodel
from . import traces

# --- chain seeds ------------------------------------------------------------

def _seed(domain: str) -> bytes:
    return hashlib.sha256(domain.encode("utf-8")).digest()

_PUBLIC_SEED = _seed("")                       # capsule public domain
_PREFIX_SEED = hashlib.sha256(b"__prefixcache__").digest()  # flat cache seed


def _tenant_domain(tenant: int) -> str:
    return f"tenant:{tenant}"


def _chain(prev: bytes, seg_id_bytes: bytes) -> bytes:
    return hashlib.sha256(prev + seg_id_bytes).digest()


# --- segment descriptors ----------------------------------------------------

@dataclass(slots=True)
class Seg:
    """One atomic content run within a request's context."""
    kind: str            # 'sys' | 'gdoc' | 'tdoc' | 'decode'
    id_bytes: bytes      # canonical content-id bytes (content-owner based)
    n_blocks: int
    is_public: bool
    is_corpus: bool      # corpus (reusable) vs decode (generated)
    is_leakdup: bool     # tdoc whose content owner != using tenant


def _sys_seg(seg_id: int, tokens: int) -> Seg:
    return Seg("sys", b"s" + int(seg_id).to_bytes(4, "little"),
               kvmodel.n_blocks(tokens), True, True, False)


def _gdoc_seg(seg_id: int, tokens: int) -> Seg:
    return Seg("gdoc", b"g" + int(seg_id).to_bytes(4, "little"),
               kvmodel.n_blocks(tokens), True, True, False)


def _tdoc_seg(seg_id: int, tokens: int, using_tenant: int) -> Seg:
    owner, local = traces.decode_tdoc_id(seg_id)
    # content-owner-based id bytes so leak-dup shares content keys with the owner
    idb = b"t" + int(owner).to_bytes(4, "little") + int(local).to_bytes(4, "little")
    return Seg("tdoc", idb, kvmodel.n_blocks(tokens), False, True,
               owner != using_tenant)


def _decode_seg(req: int, tenant: int, new_tokens: int) -> Seg:
    idb = b"d" + int(tenant).to_bytes(4, "little") + int(req).to_bytes(8, "little")
    return Seg("decode", idb, kvmodel.n_blocks(new_tokens), False, False, False)


def _expand_structural(ev: dict[str, Any]) -> tuple[list[Seg], list[Seg]]:
    """Split a request's structural segments into (public, private) descriptors."""
    tenant = ev["tenant"]
    public: list[Seg] = []
    private: list[Seg] = []
    for kind, seg_id, tokens in ev["segments"]:
        if kind == "sys":
            public.append(_sys_seg(seg_id, tokens))
        elif kind == "gdoc":
            public.append(_gdoc_seg(seg_id, tokens))
        elif kind == "tdoc":
            private.append(_tdoc_seg(seg_id, tokens, tenant))
        else:
            raise ValueError(f"unknown structural segment kind {kind!r}")
    return public, private


# ===========================================================================
# NoShareManager
# ===========================================================================

class NoShareManager:
    """No sharing: every request materializes its full context privately.

    Forks re-materialize (recompute) the inherited context. Everything is freed at
    end. Establishes the worst-case footprint/recompute baseline.
    """

    name = "noshare"

    def __init__(self, cfg: traces.TraceConfig):
        self.cfg = cfg
        self.resident_blocks = 0
        self.peak_blocks = 0
        self.recomputed_prefix_tokens = 0
        self.generated_tokens = 0
        # per-request: (blocks_held, corpus_blocks_incl_inherited, total_blocks)
        self._req_blocks: dict[int, int] = {}
        self._req_total: dict[int, int] = {}   # total context blocks (for fork inherit)
        self._req_corpus: dict[int, int] = {}

    def on_event(self, ev: dict[str, Any]) -> None:
        op = ev["op"]
        if op == "end":
            self._on_end(ev)
            return
        req = ev["req"]
        if op == "start":
            public, private = _expand_structural(ev)
            corpus_blocks = sum(s.n_blocks for s in public) + sum(s.n_blocks for s in private)
            decode = _decode_seg(req, ev["tenant"], ev["new_tokens"])
            self.recomputed_prefix_tokens += corpus_blocks * kvmodel.CHUNK_TOKENS
            self.generated_tokens += decode.n_blocks * kvmodel.CHUNK_TOKENS
            total = corpus_blocks + decode.n_blocks
            self._req_total[req] = total
            self._req_corpus[req] = corpus_blocks
        else:  # fork
            parent = ev["parent"]
            inherited_total = self._req_total.get(parent, 0)
            inherited_corpus = self._req_corpus.get(parent, 0)
            decode = _decode_seg(req, ev["tenant"], ev["new_tokens"])
            # no CoW: re-materialize the entire inherited context (recompute)
            self.recomputed_prefix_tokens += inherited_total * kvmodel.CHUNK_TOKENS
            self.generated_tokens += decode.n_blocks * kvmodel.CHUNK_TOKENS
            total = inherited_total + decode.n_blocks
            self._req_total[req] = total
            # inherited context becomes this request's prefix; its own decode adds
            self._req_corpus[req] = inherited_total
        self._req_blocks[req] = total
        self.resident_blocks += total
        if self.resident_blocks > self.peak_blocks:
            self.peak_blocks = self.resident_blocks

    def _on_end(self, ev: dict[str, Any]) -> None:
        req = ev["req"]
        held = self._req_blocks.pop(req, 0)
        self.resident_blocks -= held
        if self.resident_blocks < 0:
            self.resident_blocks = 0

    def result(self) -> dict[str, Any]:
        cu = kvmodel.CHUNK_TOKENS
        return {
            "manager": self.name,
            "peak_resident_tokens": self.peak_blocks * cu,
            "final_resident_tokens": self.resident_blocks * cu,
            "prefix_hit_tokens": 0,
            "recomputed_prefix_tokens": self.recomputed_prefix_tokens,
            "recomputed_tokens": self.recomputed_prefix_tokens,
            "generated_tokens": self.generated_tokens,
            "cross_tenant_reuse_events": 0,
            "denied_attaches": 0,
            "dedup_hits": 0,
        }


# ===========================================================================
# PrefixCacheManager
# ===========================================================================

@dataclass(slots=True)
class _CacheRun:
    n_blocks: int
    inserter: int
    kind: str
    is_public: bool


class PrefixCacheManager:
    """vLLM-style flat global hash-block cache, position-chained, no tenant checks.

    Block/segment key = cumulative chain from position 0 over all content keys.
    Unbounded in E2 (never frees; records peak). Counts a cross_tenant_reuse_event
    on every hit of a PRIVATE (tdoc/decode) run whose inserter tenant differs from
    the consumer — the leak surface.
    """

    name = "prefix"

    def __init__(self, cfg: traces.TraceConfig):
        self.cfg = cfg
        self.cache: dict[bytes, _CacheRun] = {}
        self.resident_blocks = 0
        self.peak_blocks = 0
        self.prefix_hit_tokens = 0
        self.recomputed_prefix_tokens = 0
        self.generated_tokens = 0
        self.cross_tenant_reuse_events = 0
        self.dedup_hits = 0
        # per-request final chain + cumulative corpus blocks + tenant (for forks)
        self._req_chain: dict[int, bytes] = {}
        self._req_corpus_blocks: dict[int, int] = {}

    def _consume_run(self, chain: bytes, seg: Seg, tenant: int) -> bytes:
        chain = _chain(chain, seg.id_bytes)
        key = chain
        cu = kvmodel.CHUNK_TOKENS
        run = self.cache.get(key)
        if run is not None:
            # hit
            if seg.is_corpus:
                self.prefix_hit_tokens += seg.n_blocks * cu
                self.dedup_hits += seg.n_blocks
            if (not seg.is_public) and run.inserter != tenant:
                self.cross_tenant_reuse_events += 1
        else:
            # miss -> insert (never freed in E2)
            self.cache[key] = _CacheRun(seg.n_blocks, tenant, seg.kind, seg.is_public)
            self.resident_blocks += seg.n_blocks
            if self.resident_blocks > self.peak_blocks:
                self.peak_blocks = self.resident_blocks
            if seg.is_corpus:
                self.recomputed_prefix_tokens += seg.n_blocks * cu
            else:
                self.generated_tokens += seg.n_blocks * cu
        return chain

    def on_event(self, ev: dict[str, Any]) -> None:
        op = ev["op"]
        if op == "end":
            return  # unbounded cache: nothing freed
        req = ev["req"]
        tenant = ev["tenant"]
        cu = kvmodel.CHUNK_TOKENS
        if op == "start":
            public, private = _expand_structural(ev)
            chain = _PREFIX_SEED
            corpus_blocks = 0
            for seg in public:
                chain = self._consume_run(chain, seg, tenant)
                corpus_blocks += seg.n_blocks
            for seg in private:
                chain = self._consume_run(chain, seg, tenant)
                corpus_blocks += seg.n_blocks
            decode = _decode_seg(req, tenant, ev["new_tokens"])
            chain = self._consume_run(chain, decode, tenant)
            self._req_chain[req] = chain
            self._req_corpus_blocks[req] = corpus_blocks
        else:  # fork: inherit parent's chain + all its runs (already resident hits)
            parent = ev["parent"]
            chain = self._req_chain.get(parent, _PREFIX_SEED)
            inherited_corpus = self._req_corpus_blocks.get(parent, 0)
            # inherited corpus runs are resident -> served as hits
            self.prefix_hit_tokens += inherited_corpus * cu
            self.dedup_hits += inherited_corpus
            decode = _decode_seg(req, tenant, ev["new_tokens"])
            chain = self._consume_run(chain, decode, tenant)
            self._req_chain[req] = chain
            self._req_corpus_blocks[req] = inherited_corpus

    def result(self) -> dict[str, Any]:
        cu = kvmodel.CHUNK_TOKENS
        return {
            "manager": self.name,
            "peak_resident_tokens": self.peak_blocks * cu,
            "final_resident_tokens": self.resident_blocks * cu,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "recomputed_prefix_tokens": self.recomputed_prefix_tokens,
            "recomputed_tokens": self.recomputed_prefix_tokens,
            "generated_tokens": self.generated_tokens,
            "cross_tenant_reuse_events": self.cross_tenant_reuse_events,
            "denied_attaches": 0,
            "dedup_hits": self.dedup_hits,
        }


# ===========================================================================
# CapsuleManager
# ===========================================================================

@dataclass(slots=True)
class _Run:
    """A stored segment run (chunk range) in the capsule store mirror."""
    n_blocks: int
    kind: str
    is_public: bool
    domain: str
    live_refs: int          # live requests referencing (for freeing decode)
    total_refs: int         # cumulative manifests referencing (eviction lineage)
    last_access: int        # seq (for LRU)


class CapsuleManager:
    """Python mirror of libcapsule: dedup domains, fork = O(1), refcounted extents.

    Store model (per request):
      * PUBLIC capsule  — domain "", chain over the request's public segments.
      * PRIVATE capsule — domain "tenant:k", chain over private segments
                          (tdocs in order, then the request's own decode).
    A fork inherits the parent's public + private capsules by reference (O(1)) and
    extends the private capsule with its own decode.

    Cross-tenant private dedup is impossible by construction (domain-seeded chains),
    so leak-dup tdoc references are counted as denied_attaches and materialized in
    the using tenant's own domain (recompute).

    Supports byte-budget eviction ('lru' / 'lineage') for E4. When record_ops=True
    an op-log of primitive store operations (create/extend/fork) is captured so
    run_e2b can replay the IDENTICAL sequence against the real C library and check
    the mirror's predicted {stored, dedup_hits, dedup_saved, denied} exactly.
    """

    name = "capsule"

    def __init__(self, cfg: traces.TraceConfig, *, record_ops: bool = False,
                 free_decode_on_end: bool = True):
        self.cfg = cfg
        self.record_ops = record_ops
        self.free_decode_on_end = free_decode_on_end
        self.index: dict[bytes, _Run] = {}     # run-key(chain) -> _Run
        self.resident_blocks = 0
        self.peak_blocks = 0
        self.prefix_hit_tokens = 0
        self.recomputed_prefix_tokens = 0
        self.generated_tokens = 0
        self.denied_attaches = 0
        self._denied_seen: set = set()          # distinct (domain, content) denials
        self.dedup_hits = 0                     # store counter (extend adoptions)
        self.stored_chunks = 0                  # cumulative chunks physically stored
        self.dedup_saved_chunks = 0
        self.evicted_chunks = 0
        self._seq = 0
        # per-request lineage
        self._req_public_keys: dict[int, list[bytes]] = {}
        self._req_private_keys: dict[int, list[bytes]] = {}
        self._req_private_chain: dict[int, bytes] = {}
        self._req_decode_keys: dict[int, list[bytes]] = {}  # decode runs to decref
        self._req_corpus_blocks: dict[int, int] = {}        # inherited corpus size
        # op-log for E2b (list of tuples). See run_e2b for the replay grammar.
        self.op_log: list[tuple] = []
        self._op_handles: dict[int, int] = {}   # req -> private-capsule handle idx
        self._op_pub_handles: dict[bytes, int] = {}  # public run-key -> handle idx
        self._next_handle = 0

    # -- store primitives ---------------------------------------------------

    def _touch(self, run: _Run) -> None:
        self._seq += 1
        run.last_access = self._seq

    def _extend_capsule(self, seed: bytes, segs: list[Seg], domain: str,
                        req_tenant: int, count_denied: bool) -> tuple[list[bytes], bytes]:
        """Simulate one capsule_extend call over `segs` (dedup mode per call).

        Returns (run_keys, final_chain). Updates dedup/stored/hit/recompute/resident
        counters exactly as libcapsule would per chunk (segments are atomic).
        """
        chain = seed
        dedup_mode = True
        run_keys: list[bytes] = []
        cu = kvmodel.CHUNK_TOKENS
        for seg in segs:
            chain = _chain(chain, seg.id_bytes)
            key = chain
            run_keys.append(key)
            existing = self.index.get(key)
            if dedup_mode and existing is not None:
                # dedup hit: adopt existing chunks (the using tenant already holds
                # this content in its OWN domain — not a cross-tenant reuse)
                self.dedup_hits += seg.n_blocks
                self.dedup_saved_chunks += seg.n_blocks
                existing.live_refs += 1
                existing.total_refs += 1
                self._touch(existing)
                if seg.is_corpus:
                    self.prefix_hit_tokens += seg.n_blocks * cu
            else:
                dedup_mode = False
                if existing is None:
                    if count_denied and seg.is_leakdup:
                        # cross-tenant private content: the owner's capsule lives in
                        # a different dedup domain, so the capsule refuses the reuse
                        # and materializes a private copy (recompute). Counted once
                        # per distinct (using-domain, content) pair — position in the
                        # private chain does not create extra denials.
                        dkey = (domain, seg.id_bytes)
                        if dkey not in self._denied_seen:
                            self._denied_seen.add(dkey)
                            self.denied_attaches += 1
                    self._seq += 1
                    self.index[key] = _Run(seg.n_blocks, seg.kind, seg.is_public,
                                           domain, 1, 1, self._seq)
                    self.resident_blocks += seg.n_blocks
                    if self.resident_blocks > self.peak_blocks:
                        self.peak_blocks = self.resident_blocks
                else:
                    # rare: chain exists but dedup mode already left; C re-appends.
                    existing.live_refs += 1
                    existing.total_refs += 1
                    self._touch(existing)
                self.stored_chunks += seg.n_blocks
                if seg.is_corpus:
                    self.recomputed_prefix_tokens += seg.n_blocks * cu
                else:
                    self.generated_tokens += seg.n_blocks * cu
        return run_keys, chain

    def _reference_run(self, key: bytes, seg_is_corpus: bool) -> None:
        """Fork inheritance: reference an existing run (O(1)); recompute if evicted."""
        run = self.index.get(key)
        cu = kvmodel.CHUNK_TOKENS
        if run is not None:
            run.live_refs += 1
            run.total_refs += 1
            self._touch(run)
            if seg_is_corpus:
                self.prefix_hit_tokens += run.n_blocks * cu

    def _alloc_handle(self) -> int:
        h = self._next_handle
        self._next_handle += 1
        return h

    # -- event dispatch -----------------------------------------------------

    def on_event(self, ev: dict[str, Any]) -> None:
        op = ev["op"]
        if op == "end":
            self._on_end(ev)
            return
        req = ev["req"]
        tenant = ev["tenant"]
        if op == "start":
            self._on_start(ev, req, tenant)
        else:
            self._on_fork(ev, req, tenant)

    def _on_start(self, ev: dict[str, Any], req: int, tenant: int) -> None:
        public, private = _expand_structural(ev)
        decode = _decode_seg(req, tenant, ev["new_tokens"])
        priv_segs = private + [decode]
        dom = _tenant_domain(tenant)

        pub_keys, _ = self._extend_capsule(_PUBLIC_SEED, public, "", tenant,
                                           count_denied=False)
        priv_keys, priv_chain = self._extend_capsule(_seed(dom), priv_segs, dom,
                                                      tenant, count_denied=True)

        corpus_blocks = sum(s.n_blocks for s in public) + sum(s.n_blocks for s in private)
        self._req_public_keys[req] = pub_keys
        self._req_private_keys[req] = priv_keys
        self._req_private_chain[req] = priv_chain
        self._req_decode_keys[req] = [priv_keys[-1]]  # own decode run
        self._req_corpus_blocks[req] = corpus_blocks

        if self.record_ops:
            self._log_start(ev, req, tenant, public, priv_segs, pub_keys, priv_keys)

    def _on_fork(self, ev: dict[str, Any], req: int, tenant: int) -> None:
        parent = ev["parent"]
        pub_keys = self._req_public_keys.get(parent, [])
        parent_priv_keys = self._req_private_keys.get(parent, [])
        parent_priv_chain = self._req_private_chain.get(parent, _seed(_tenant_domain(tenant)))
        inherited_corpus = self._req_corpus_blocks.get(parent, 0)

        # O(1) fork: reference inherited public + private runs
        for k in pub_keys:
            r = self.index.get(k)
            if r is not None:
                r.live_refs += 1
                r.total_refs += 1
                self._touch(r)
        for k in parent_priv_keys:
            r = self.index.get(k)
            if r is not None:
                r.live_refs += 1
                r.total_refs += 1
                self._touch(r)
        # inherited corpus served without recompute (fork reuse, C1)
        self.prefix_hit_tokens += inherited_corpus * kvmodel.CHUNK_TOKENS

        # extend the (forked) private capsule with the child's own decode
        decode = _decode_seg(req, tenant, ev["new_tokens"])
        dom = _tenant_domain(tenant)
        dec_keys, priv_chain = self._extend_capsule(parent_priv_chain, [decode], dom,
                                                    tenant, count_denied=False)

        self._req_public_keys[req] = pub_keys
        self._req_private_keys[req] = parent_priv_keys + dec_keys
        self._req_private_chain[req] = priv_chain
        # decode runs this request should decref at end: ancestor decodes + own
        parent_decodes = self._req_decode_keys.get(parent, [])
        self._req_decode_keys[req] = list(parent_decodes) + [dec_keys[-1]]
        self._req_corpus_blocks[req] = inherited_corpus

        if self.record_ops:
            self._log_fork(ev, req, parent, tenant, decode, dec_keys)

    def _on_end(self, ev: dict[str, Any]) -> None:
        req = ev["req"]
        # decrement live_refs on every referenced run; free decode runs at 0.
        # Lineage dicts are KEPT (not popped): a fork whose parent has already ended
        # must still be able to inherit the parent's persistent corpus context.
        pub_keys = self._req_public_keys.get(req, [])
        priv_keys = self._req_private_keys.get(req, [])
        decode_keys = set(self._req_decode_keys.get(req, []))
        for k in pub_keys:
            r = self.index.get(k)
            if r is not None and r.live_refs > 0:
                r.live_refs -= 1
        for k in priv_keys:
            r = self.index.get(k)
            if r is None:
                continue
            if r.live_refs > 0:
                r.live_refs -= 1
            if (self.free_decode_on_end and k in decode_keys
                    and r.kind == "decode" and r.live_refs == 0):
                # ephemeral decode reclaimed by lifecycle
                self.resident_blocks -= r.n_blocks
                del self.index[k]

    # -- E4 eviction --------------------------------------------------------

    def evict_to(self, target_blocks: int, policy: str) -> int:
        """Evict unpinned, not-live-referenced runs until resident <= target.

        Returns blocks evicted this call. Policy 'lru' (oldest last_access first)
        or 'lineage' (score = age/(1+log2(1+total_refs)); evict highest score, so
        widely-shared prefixes are protected). Evicted runs -> future references
        recompute (the simulator counts that recompute on the next reference).
        """
        import math
        if self.resident_blocks <= target_blocks:
            return 0
        # candidates: runs with no live references (safe to evict)
        cands = [(k, r) for k, r in self.index.items() if r.live_refs == 0]
        now = self._seq + 1
        if policy == "lru":
            cands.sort(key=lambda kr: kr[1].last_access)  # oldest first
        elif policy == "lineage":
            def score(kr):
                r = kr[1]
                age = now - r.last_access
                return age / (1.0 + math.log2(1.0 + r.total_refs))
            cands.sort(key=score, reverse=True)  # highest score first
        else:
            raise ValueError(f"unknown eviction policy {policy!r}")
        evicted = 0
        for k, r in cands:
            if self.resident_blocks <= target_blocks:
                break
            del self.index[k]
            self.resident_blocks -= r.n_blocks
            self.evicted_chunks += r.n_blocks
            evicted += r.n_blocks
        return evicted

    # -- op-log for E2b -----------------------------------------------------

    def _log_start(self, ev, req, tenant, public, priv_segs, pub_keys, priv_keys):
        # One create+extend per request per capsule (public then private). Public
        # content is re-extended every start so the C store deduplicates it exactly
        # as the mirror does (repeat public content -> dedup_hits in both). Forks
        # reuse the parent's private capsule via capsule_fork (see _log_fork), so
        # the shared prefix is never re-extended along an agent tree.
        if public:
            pub_handle = self._alloc_handle()
            self.op_log.append((
                "create_extend", pub_handle, "",
                [(_seg_content_key(s, tenant, "*"), s.n_blocks) for s in public],
            ))
        priv_handle = self._alloc_handle()
        self._op_handles[req] = priv_handle
        dom = _tenant_domain(tenant)
        self.op_log.append((
            "create_extend", priv_handle, dom,
            [(_seg_content_key(s, tenant, tenant), s.n_blocks) for s in priv_segs],
        ))

    def _log_fork(self, ev, req, parent, tenant, decode, dec_keys):
        parent_handle = self._op_handles.get(parent)
        child_handle = self._alloc_handle()
        self._op_handles[req] = child_handle
        dom = _tenant_domain(tenant)
        self.op_log.append(("fork", child_handle, parent_handle))
        self.op_log.append((
            "extend", child_handle,
            [(_seg_content_key(decode, tenant, tenant), decode.n_blocks)],
        ))

    # -- result -------------------------------------------------------------

    def result(self) -> dict[str, Any]:
        cu = kvmodel.CHUNK_TOKENS
        return {
            "manager": self.name,
            "peak_resident_tokens": self.peak_blocks * cu,
            "final_resident_tokens": self.resident_blocks * cu,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "recomputed_prefix_tokens": self.recomputed_prefix_tokens,
            "recomputed_tokens": self.recomputed_prefix_tokens,
            "generated_tokens": self.generated_tokens,
            "cross_tenant_reuse_events": 0,
            "denied_attaches": self.denied_attaches,
            "dedup_hits": self.dedup_hits,
            # store-level counters (for E2b exact validation vs libcapsule)
            "stored_chunks": self.stored_chunks,
            "dedup_saved_chunks": self.dedup_saved_chunks,
            "evicted_chunks": self.evicted_chunks,
            "n_runs": len(self.index),
        }


def _seg_content_key(seg: Seg, using_tenant: int, owner_marker: Any) -> tuple:
    """Content key for byte generation in E2b (content-owner based).

    owner_marker "*" for public; for private the CONTENT owner is embedded in
    seg.id_bytes already, so we pass a stable, JSON-able key derived from it.
    """
    return (seg.kind, seg.id_bytes.hex())


# --- registry ---------------------------------------------------------------

MANAGERS = {
    "noshare": NoShareManager,
    "prefix": PrefixCacheManager,
    "capsule": CapsuleManager,
}


def run_trace(manager_name: str, cfg: traces.TraceConfig,
              trace: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Instantiate a manager, replay a trace, return its result() dict."""
    cls = MANAGERS[manager_name]
    m = cls(cfg, **kwargs) if manager_name == "capsule" else cls(cfg)
    for ev in trace:
        m.on_event(ev)
    return m.result()
