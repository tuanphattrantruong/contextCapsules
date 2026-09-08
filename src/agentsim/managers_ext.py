"""managers_ext.py — the baselines and ablations round 1 of review demanded.

`managers.py` is the audited artifact and is not modified. This module adds
four managers that answer three specific reviewer objections:

  partitioned      The obvious, deployed defense the original study never
                   compared against: the same vLLM-style prefix cache, but
                   keyed by (tenant, chain). It is exactly as safe as capsules
                   against the content oracle -- and that is the point. It also
                   destroys every cross-tenant sharing opportunity, which is
                   what capsules keep. Without this baseline the security claim
                   is "we beat a design nobody would ship".

  safekv           SafeKV-style selective isolation (Chu et al.): a sensitivity
                   detector routes "sensitive" content to a per-tenant
                   namespace and everything else to the shared one. Detection
                   is a heuristic, so it has a false-negative rate; the leak
                   surface is therefore a *statistic*, not a guarantee. We
                   sweep that rate.

  capsule-poschain Capsules whose private chain is seeded by the domain AND
                   continues over the public prefix, i.e. position-chained
                   exactly like the baseline. E5e measured that a real dense
                   transformer's KV for identical tokens differs completely
                   after a different prefix (chunk-hash match 0.000), so
                   position-independent naming is unsound and this variant --
                   not the original -- is the honest configuration.

  prefix-bounded   The vLLM-style cache with the same lifecycle freeing that
                   capsules get (dead decode state reclaimed at request end),
                   removing the never-frees asymmetry from the footprint
                   comparison.
"""

from __future__ import annotations

import hashlib
from typing import Any

from . import kvmodel
from . import traces
from .managers import (
    CapsuleManager,
    PrefixCacheManager,
    Seg,
    _CacheRun,
    _chain,
    _decode_seg,
    _expand_structural,
    _seed,
    _tenant_domain,
)


# ===========================================================================
# Partitioned prefix cache -- the obvious defense
# ===========================================================================

class PartitionedPrefixCacheManager(PrefixCacheManager):
    """Per-tenant isolated prefix caches: cache key = (tenant, position chain).

    Structurally immune to the cross-tenant content oracle (an attacker can
    only name entries in its own partition), and structurally incapable of
    sharing a system prompt between two tenants. This is the price capsules
    are claimed to avoid paying.
    """

    name = "partitioned"

    def __init__(self, cfg: traces.TraceConfig):
        super().__init__(cfg)
        self._seeds: dict[int, bytes] = {}

    def _tenant_seed(self, tenant: int) -> bytes:
        s = self._seeds.get(tenant)
        if s is None:
            s = _seed(f"__partition__:{tenant}")
            self._seeds[tenant] = s
        return s

    def on_event(self, ev: dict[str, Any]) -> None:
        op = ev["op"]
        if op == "end":
            return
        req = ev["req"]
        tenant = ev["tenant"]
        cu = kvmodel.CHUNK_TOKENS
        if op == "start":
            public, private = _expand_structural(ev)
            chain = self._tenant_seed(tenant)
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
        else:
            parent = ev["parent"]
            chain = self._req_chain.get(parent, self._tenant_seed(tenant))
            inherited = self._req_corpus_blocks.get(parent, 0)
            self.prefix_hit_tokens += inherited * cu
            self.dedup_hits += inherited
            decode = _decode_seg(req, tenant, ev["new_tokens"])
            chain = self._consume_run(chain, decode, tenant)
            self._req_chain[req] = chain
            self._req_corpus_blocks[req] = inherited

    def result(self) -> dict[str, Any]:
        r = super().result()
        r["manager"] = self.name
        # cross-tenant reuse is impossible by construction; assert it held.
        r["cross_tenant_reuse_events"] = self.cross_tenant_reuse_events
        r["structurally_isolated"] = True
        return r


# ===========================================================================
# SafeKV-style selective isolation
# ===========================================================================

class SafeKVPrefixCacheManager(PrefixCacheManager):
    """Sensitivity-detector-driven selective sharing (SafeKV-class policy).

    Content the detector calls sensitive goes to the tenant's own namespace;
    everything else shares the global one. Public corpus content is never
    sensitive. Private content is detected with recall (1 - fn_rate); the
    misses land in the shared namespace and are exactly the leak surface.

    The detector is deterministic per content id (hash-derived) so runs are
    reproducible and the same content is classified the same way every time --
    which is how a real content-based classifier behaves.
    """

    name = "safekv"

    def __init__(self, cfg: traces.TraceConfig, fn_rate: float = 0.05):
        super().__init__(cfg)
        self.fn_rate = fn_rate
        self._seeds: dict[int, bytes] = {}
        self.leaked_blocks = 0
        self.misclassified_contents: set = set()

    def _tenant_seed(self, tenant: int) -> bytes:
        s = self._seeds.get(tenant)
        if s is None:
            s = _seed(f"__safekv__:{tenant}")
            self._seeds[tenant] = s
        return s

    def _detect_sensitive(self, seg: Seg) -> bool:
        """True => isolate. Public content is never sensitive."""
        if seg.is_public:
            return False
        if self.fn_rate <= 0.0:
            return True
        h = hashlib.sha256(b"detector" + seg.id_bytes).digest()
        u = int.from_bytes(h[:8], "little") / float(1 << 64)
        missed = u < self.fn_rate
        if missed:
            self.misclassified_contents.add(seg.id_bytes)
        return not missed

    def on_event(self, ev: dict[str, Any]) -> None:
        op = ev["op"]
        if op == "end":
            return
        req = ev["req"]
        tenant = ev["tenant"]
        cu = kvmodel.CHUNK_TOKENS
        if op == "start":
            public, private = _expand_structural(ev)
            # two chains: the shared one and the tenant-private one. A segment
            # advances whichever chain its classification routes it to.
            shared_chain = _seed("__safekv_shared__")
            priv_chain = self._tenant_seed(tenant)
            corpus = 0
            for seg in public + private:
                if self._detect_sensitive(seg):
                    priv_chain = self._consume_run(priv_chain, seg, tenant)
                else:
                    shared_chain = self._consume_run(shared_chain, seg, tenant)
                    if not seg.is_public:
                        self.leaked_blocks += seg.n_blocks
                corpus += seg.n_blocks
            decode = _decode_seg(req, tenant, ev["new_tokens"])
            if self._detect_sensitive(decode):
                priv_chain = self._consume_run(priv_chain, decode, tenant)
            else:
                shared_chain = self._consume_run(shared_chain, decode, tenant)
                self.leaked_blocks += decode.n_blocks
            self._req_chain[req] = priv_chain
            self._req_corpus_blocks[req] = corpus
        else:
            parent = ev["parent"]
            priv_chain = self._req_chain.get(parent, self._tenant_seed(tenant))
            inherited = self._req_corpus_blocks.get(parent, 0)
            self.prefix_hit_tokens += inherited * cu
            self.dedup_hits += inherited
            decode = _decode_seg(req, tenant, ev["new_tokens"])
            priv_chain = self._consume_run(priv_chain, decode, tenant)
            self._req_chain[req] = priv_chain
            self._req_corpus_blocks[req] = inherited

    def result(self) -> dict[str, Any]:
        r = super().result()
        r["manager"] = self.name
        r["detector_fn_rate"] = self.fn_rate
        r["leaked_private_blocks_in_shared_ns"] = self.leaked_blocks
        r["n_misclassified_contents"] = len(self.misclassified_contents)
        r["structurally_isolated"] = False
        return r


# ===========================================================================
# Position-chained capsules -- the honest configuration after E5e
# ===========================================================================

class CapsulePosChainManager(CapsuleManager):
    """Capsules whose private chain is domain-seeded AND position-chained.

    Original design: private chain starts at H(domain), so a document dedups
    wherever it appears -- which E5e shows a dense transformer does not
    support. Here the private chain is seeded by

        H( H(domain) || public_chain )

    so it is still impossible for two domains to collide (the security
    property is untouched), but content only dedups when the whole preceding
    context matches, exactly as the vLLM-style baseline requires and exactly as
    real KV behaves.
    """

    name = "capsule-poschain"

    def _on_start(self, ev: dict[str, Any], req: int, tenant: int) -> None:
        public, private = _expand_structural(ev)
        decode = _decode_seg(req, tenant, ev["new_tokens"])
        priv_segs = private + [decode]
        dom = _tenant_domain(tenant)

        pub_keys, pub_chain = self._extend_capsule(
            _seed(""), public, "", tenant, count_denied=False)

        # domain-seeded, but continuing over the public prefix
        priv_seed = hashlib.sha256(_seed(dom) + pub_chain).digest()
        priv_keys, priv_chain = self._extend_capsule(
            priv_seed, priv_segs, dom, tenant, count_denied=True)

        corpus_blocks = (sum(s.n_blocks for s in public)
                         + sum(s.n_blocks for s in private))
        self._req_public_keys[req] = pub_keys
        self._req_private_keys[req] = priv_keys
        self._req_private_chain[req] = priv_chain
        self._req_decode_keys[req] = [priv_keys[-1]]
        self._req_corpus_blocks[req] = corpus_blocks

        if self.record_ops:
            self._log_start_chained(ev, req, tenant, public, priv_segs)

    def _log_start_chained(self, ev, req, tenant, public, priv_segs):
        """Op-log for the position-chained rule.

        The private capsule is created with capsule_create_chained against the
        public capsule, so the C store reproduces exactly the seeding the mirror
        uses. Without this the replay silently falls back to plain create and
        dedups more than the mirror predicts -- which is how the extended E2b
        caught the discrepancy in the first place.
        """
        from .managers import _seg_content_key
        pub_handle = None
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
            "create_chained_extend", priv_handle, dom, pub_handle,
            [(_seg_content_key(s, tenant, tenant), s.n_blocks) for s in priv_segs],
        ))

    def result(self) -> dict[str, Any]:
        r = super().result()
        r["manager"] = self.name
        r["naming"] = "domain-seeded, position-chained"
        return r


# ===========================================================================
# Prefix cache with capsule-equivalent lifecycle freeing
# ===========================================================================

class BoundedPrefixCacheManager(PrefixCacheManager):
    """vLLM-style cache that frees dead decode state at request end.

    Removes the never-frees asymmetry (RESULTS_SUMMARY L3) from the footprint
    comparison, so peak-resident numbers compare like with like.
    """

    name = "prefix-bounded"

    def __init__(self, cfg: traces.TraceConfig):
        super().__init__(cfg)
        self._req_decode_key: dict[int, bytes] = {}
        self._decode_live: dict[bytes, int] = {}

    def _consume_run(self, chain: bytes, seg: Seg, tenant: int) -> bytes:
        out = super()._consume_run(chain, seg, tenant)
        if seg.kind == "decode":
            self._decode_live[out] = self._decode_live.get(out, 0) + 1
        return out

    def on_event(self, ev: dict[str, Any]) -> None:
        if ev["op"] == "end":
            req = ev["req"]
            k = self._req_decode_key.get(req)
            if k is not None:
                n = self._decode_live.get(k, 0) - 1
                self._decode_live[k] = n
                if n <= 0:
                    run = self.cache.pop(k, None)
                    if run is not None:
                        self.resident_blocks -= run.n_blocks
                    self._decode_live.pop(k, None)
            return
        super().on_event(ev)
        self._req_decode_key[ev["req"]] = self._req_chain.get(ev["req"])

    def result(self) -> dict[str, Any]:
        r = super().result()
        r["manager"] = self.name
        r["frees_dead_decode_state"] = True
        return r


EXT_MANAGERS = {
    "partitioned": PartitionedPrefixCacheManager,
    "safekv": SafeKVPrefixCacheManager,
    "capsule-poschain": CapsulePosChainManager,
    "prefix-bounded": BoundedPrefixCacheManager,
}


def install() -> None:
    """Register the extension managers in the shared registry."""
    from . import managers as _m
    _m.MANAGERS.update(EXT_MANAGERS)
