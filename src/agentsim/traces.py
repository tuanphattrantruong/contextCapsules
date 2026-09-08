"""traces.py — deterministic multi-tenant agent-swarm trace generator (DESIGN.md 3.1).

A trace is a list of event dicts (deterministic given a seed via
numpy.random.default_rng(seed)). Events are emitted in causal, non-decreasing
timestamp order under a slot-based concurrency model (M live sessions).

Event schema (exactly DESIGN.md 3.1):
    {"i":int, "t_ms":float, "op":"start"|"fork"|"end",
     "req":int, "parent":int (or -1), "tenant":int, "agent":int, "session":int,
     "segments":[[kind,id,tokens], ...], "new_tokens":int}

Segment kinds and their sharing/protection semantics:
    "sys"  : global system prompts (public, dedup domain "", read-cap to all tenants)
    "gdoc" : global RAG corpus docs (public, dedup domain "")
    "tdoc" : tenant-private docs  (dedup domain "tenant:<k>")
    (session-private decode tokens are modeled as a per-request private segment
     inside the managers, not carried as an explicit trace segment.)

TDOC id encoding — self-describing content identity (no shared state needed
between the generator and the managers): a tdoc segment's `id` field encodes the
CONTENT owner and the content-local id as

    id = content_owner_tenant * TDOC_STRIDE + content_local_id

so a manager recovers (owner, local) = divmod(id, TDOC_STRIDE) and compares the
owner against the event's `tenant` (the *using* tenant). When owner != tenant the
tdoc is a leak-duplicate: identical CONTENT bytes owned by another tenant but
referenced by this one. The unprotected prefix cache deduplicates such content
across tenants (the leak surface); capsules keep it in the using tenant's dedup
domain and refuse the cross-tenant reuse.

p_leakdup realizes the leak surface: that fraction of each tenant's tdocs have
their content identity aliased to another tenant's tdoc (distinct segment id and
owner-of-reference, identical content bytes).

Content bytes for real-C runs (E2b) are a deterministic function of the content
key alone (NOT the using tenant):
    chunk b of content_key -> default_rng(stable_hash((content_key, b))).bytes(chunk_bytes)
where stable_hash is hashlib.sha256 (builtin hash() is NOT stable across runs).
Identical content keys therefore yield identical bytes, so leak-dup tdocs produce
byte-for-byte identical chunks — exactly the condition under which a same-domain
store would dedup and a cross-domain store must not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import kvmodel

# Encodes content-owner tenant into a tdoc segment id (owner*STRIDE + local).
# The synthetic generator uses n_tdoc <= 200 per tenant; the Mooncake replay
# (agentsim/mooncake.py) owns up to ~10^5 private blocks per tenant, so the
# stride is sized for that. Segment CONTENT identity is derived from the decoded
# (owner, local) pair, not from the raw id, so widening the stride leaves every
# existing result bit-identical (verified by re-running E2/E6).
TDOC_STRIDE = 10_000_000


@dataclass
class TraceConfig:
    """All trace-generator parameters (defaults per DESIGN.md 3.1 table)."""
    n_requests: int = 10000        # root sessions + forks combined
    n_tenants: int = 16            # grid axis T
    f: float = 0.75               # P(a doc segment is a global gdoc vs a tdoc)
    p_leakdup: float = 0.05        # fraction of a tenant's tdocs aliasing another tenant's
    # corpus sizes / skews
    n_sys: int = 8
    sys_zipf_s: float = 1.2
    sys_tokens: int = 512
    n_gdoc: int = 500
    gdoc_zipf_s: float = 1.1
    n_tdoc_per_tenant: int = 200
    tdoc_zipf_s: float = 1.1
    doc_len_mu: float = float(np.log(384.0))   # LogNormal(ln 384, 0.6)
    doc_len_sigma: float = 0.6
    new_tokens_mu: float = float(np.log(192.0))  # LogNormal(ln 192, 0.7)
    new_tokens_sigma: float = 0.7
    min_docs: int = 1
    max_docs: int = 3
    # agent-tree fan-out
    p_fork: float = 0.35
    max_kids: int = 3
    max_depth: int = 2
    # concurrency
    concurrency_M: int = 64

    def key(self) -> dict[str, Any]:
        """Serializable snapshot of the config (for results meta)."""
        return asdict(self)


def stable_hash(obj: Any) -> int:
    """Deterministic, cross-run-stable hash of a (JSON-able) object via SHA-256.

    Builtin hash() is randomized per process (PYTHONHASHSEED); this is not. Used
    to seed per-chunk content-byte RNGs so identical content keys give identical
    bytes on every run and in every process.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # 64-bit stable seed (numpy default_rng accepts Python int of any width, but
    # keep it in a comfortable range).
    return int.from_bytes(digest[:8], "little")


def chunk_bytes_for(content_key: tuple, block_idx: int, chunk_nbytes: int) -> bytes:
    """Deterministic pseudorandom bytes for chunk `block_idx` of `content_key`.

    Depends only on (content_key, block_idx): identical content keys -> identical
    bytes. `chunk_nbytes` = kvmodel.chunk_bytes(bpt) for the run's geometry.
    """
    rng = np.random.default_rng(stable_hash((list(content_key), int(block_idx))))
    return rng.bytes(int(chunk_nbytes))


def _zipf_weights(n: int, s: float) -> np.ndarray:
    """Bounded, deterministic Zipf(s) probability vector over ranks 1..n.

    Preferred over rng.zipf (unbounded, can exceed the corpus range). Returns
    normalized 1/k^s weights.
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()


def _round_len(x: float) -> int:
    return max(kvmodel.CHUNK_TOKENS, kvmodel.round_up_chunks(int(np.ceil(x))))


@dataclass
class _Corpus:
    """Precomputed corpus for one (config, seed): sizes, skews, leak-dup aliasing."""
    sys_tokens: np.ndarray          # [n_sys] token counts (all == sys_tokens)
    sys_w: np.ndarray               # [n_sys] selection weights
    gdoc_tokens: np.ndarray         # [n_gdoc] token counts
    gdoc_w: np.ndarray              # [n_gdoc] selection weights
    tdoc_tokens: np.ndarray         # [n_tenants, n_tdoc] token counts (per owner)
    tdoc_w: np.ndarray              # [n_tdoc] selection weights (shared shape)
    # per (tenant, local tdoc id) -> (content_owner, content_local_id)
    tdoc_content: dict[tuple[int, int], tuple[int, int]]


def _build_corpus(cfg: TraceConfig, rng: np.random.Generator) -> _Corpus:
    sys_tokens = np.full(cfg.n_sys, cfg.sys_tokens, dtype=np.int64)
    sys_w = _zipf_weights(cfg.n_sys, cfg.sys_zipf_s)

    gdoc_raw = rng.lognormal(cfg.doc_len_mu, cfg.doc_len_sigma, size=cfg.n_gdoc)
    gdoc_tokens = np.array([_round_len(x) for x in gdoc_raw], dtype=np.int64)
    gdoc_w = _zipf_weights(cfg.n_gdoc, cfg.gdoc_zipf_s)

    tdoc_tokens = np.empty((cfg.n_tenants, cfg.n_tdoc_per_tenant), dtype=np.int64)
    for k in range(cfg.n_tenants):
        raw = rng.lognormal(cfg.doc_len_mu, cfg.doc_len_sigma,
                            size=cfg.n_tdoc_per_tenant)
        tdoc_tokens[k] = [_round_len(x) for x in raw]
    tdoc_w = _zipf_weights(cfg.n_tdoc_per_tenant, cfg.tdoc_zipf_s)

    # Leak-dup aliasing: for each tenant, a fraction p_leakdup of its tdocs have
    # their CONTENT identity aliased to another tenant's tdoc. The segment id
    # stays local (k, j); content becomes (m, i) with m != k.
    tdoc_content: dict[tuple[int, int], tuple[int, int]] = {}
    n_leak = int(round(cfg.p_leakdup * cfg.n_tdoc_per_tenant))
    for k in range(cfg.n_tenants):
        # default: own content
        for j in range(cfg.n_tdoc_per_tenant):
            tdoc_content[(k, j)] = (k, j)
        if n_leak > 0 and cfg.n_tenants > 1:
            leak_ids = rng.choice(cfg.n_tdoc_per_tenant, size=n_leak, replace=False)
            for j in leak_ids:
                m = int(rng.integers(0, cfg.n_tenants))
                while m == k:
                    m = int(rng.integers(0, cfg.n_tenants))
                i = int(rng.integers(0, cfg.n_tdoc_per_tenant))
                # Alias k's tdoc j to the CONTENT of tenant m's tdoc i, and align
                # the reference length to the owner's real content length so the
                # aliased bytes cover identical block ranges.
                tdoc_content[(k, int(j))] = (m, i)
                tdoc_tokens[k, int(j)] = tdoc_tokens[m, i]

    return _Corpus(sys_tokens, sys_w, gdoc_tokens, gdoc_w,
                   tdoc_tokens, tdoc_w, tdoc_content)


@dataclass
class _Node:
    """One request node in an agent tree (pre-scheduling)."""
    req: int
    parent: int
    tenant: int
    agent: int
    session: int
    depth: int
    segments: list[list[Any]]   # structural segments [[kind,id,tokens],...]
    new_tokens: int
    children: list[int] = field(default_factory=list)


def _make_request_segments(cfg: TraceConfig, corpus: _Corpus, tenant: int,
                           rng: np.random.Generator) -> tuple[list[list[Any]], int]:
    """Build the structural segments and decode length for a root request."""
    segments: list[list[Any]] = []
    # 1 sys segment
    s = int(rng.choice(cfg.n_sys, p=corpus.sys_w))
    segments.append(["sys", s, int(corpus.sys_tokens[s])])
    # 1..max_docs doc segments
    ndocs = int(rng.integers(cfg.min_docs, cfg.max_docs + 1))
    for _ in range(ndocs):
        if rng.random() < cfg.f:
            g = int(rng.choice(cfg.n_gdoc, p=corpus.gdoc_w))
            segments.append(["gdoc", g, int(corpus.gdoc_tokens[g])])
        else:
            j = int(rng.choice(cfg.n_tdoc_per_tenant, p=corpus.tdoc_w))
            tok = int(corpus.tdoc_tokens[tenant, j])
            # encode owner-of-reference (the using tenant is the reference owner;
            # content ownership is recovered by the manager from tdoc_content only
            # for byte generation — the *segment* id encodes the reference owner so
            # leak-dup content still shares content keys). We encode the CONTENT
            # owner into the id so managers can detect cross-tenant content.
            owner, local = corpus.tdoc_content[(tenant, j)]
            seg_id = owner * TDOC_STRIDE + local
            segments.append(["tdoc", seg_id, tok])
    new_tokens = _round_len(rng.lognormal(cfg.new_tokens_mu, cfg.new_tokens_sigma))
    return segments, new_tokens


def _grow_tree(cfg: TraceConfig, corpus: _Corpus, rng: np.random.Generator,
               nodes: dict[int, _Node], counters: dict[str, int],
               tenant: int, session: int, parent: int, depth: int,
               remaining: list[int]) -> None:
    """Recursively add fork children under the node just created (id = parent).

    `remaining[0]` is the global budget of nodes still allowed; decremented as
    nodes are created. Children inherit the parent's structural segments for
    readability; managers use the parent linkage for the actual shared context.
    """
    if remaining[0] <= 0 or depth >= cfg.max_depth:
        return
    if rng.random() >= cfg.p_fork:
        return
    nkids = int(rng.integers(1, cfg.max_kids + 1))
    parent_node = nodes[parent]
    for _ in range(nkids):
        if remaining[0] <= 0:
            return
        req = counters["req"]
        counters["req"] += 1
        counters["agent"] += 1
        remaining[0] -= 1
        child = _Node(
            req=req, parent=parent, tenant=tenant, agent=counters["agent"],
            session=session, depth=depth + 1,
            # fork carries the parent's structural segments for a self-contained
            # exemplar; its private decode is new_tokens.
            segments=[list(s) for s in parent_node.segments],
            new_tokens=_round_len(rng.lognormal(cfg.new_tokens_mu,
                                                cfg.new_tokens_sigma)),
        )
        nodes[req] = child
        parent_node.children.append(req)
        _grow_tree(cfg, corpus, rng, nodes, counters, tenant, session, req,
                   depth + 1, remaining)


def _generate_forest(cfg: TraceConfig, corpus: _Corpus,
                     rng: np.random.Generator) -> tuple[dict[int, _Node], list[int]]:
    """Generate root sessions and their fork subtrees until n_requests nodes."""
    nodes: dict[int, _Node] = {}
    roots: list[int] = []
    counters = {"req": 0, "agent": 0}
    session = 0
    remaining = [cfg.n_requests]
    while remaining[0] > 0:
        tenant = int(rng.integers(0, cfg.n_tenants))
        segments, new_tokens = _make_request_segments(cfg, corpus, tenant, rng)
        req = counters["req"]
        counters["req"] += 1
        counters["agent"] = 0
        remaining[0] -= 1
        root = _Node(req=req, parent=-1, tenant=tenant, agent=0, session=session,
                     depth=0, segments=segments, new_tokens=new_tokens)
        nodes[req] = root
        roots.append(req)
        _grow_tree(cfg, corpus, rng, nodes, counters, tenant, session, req, 0,
                   remaining)
        session += 1
    return nodes, roots


def _session_nodes_dfs(nodes: dict[int, _Node], root: int) -> list[int]:
    """DFS order of a session's tree (parent always before its children)."""
    order: list[int] = []
    stack = [root]
    while stack:
        r = stack.pop()
        order.append(r)
        # push children reversed so they are visited in generation order
        for c in reversed(nodes[r].children):
            stack.append(c)
    return order


def _schedule(cfg: TraceConfig, nodes: dict[int, _Node],
              roots: list[int]) -> list[dict[str, Any]]:
    """Session-slot interleave to steady-state concurrency M; emit start/fork/end.

    A *session* (an agent tree: its root plus all forks) holds one of M slots for
    its whole lifetime and frees it when the tree completes. This guarantees every
    fork is processed while its ancestors are still live (fork references a live
    parent), and reclamation happens at session end. Within a session's active
    window every node's start/fork is emitted (parent before child, DFS order),
    then every node's end. Across the M concurrent sessions, events interleave by
    timestamp — producing realistic cross-session cache sharing and churn.
    """
    import heapq
    M = cfg.concurrency_M
    eps = 1e-4          # spacing between a session's own events
    end_heap: list[tuple[float, int]] = []   # active session end times (heap)
    now = 0.0

    timed: list[tuple[float, int, int, str, int]] = []  # (t_ms, is_end, ssn, o, req)
    for ssn, root in enumerate(roots):
        # acquire a slot: if M sessions are live, advance to the earliest end
        while len(end_heap) >= M:
            done, _ = heapq.heappop(end_heap)
            if done > now:
                now = done
        order = _session_nodes_dfs(nodes, root)
        total_tokens = sum(nodes[r].new_tokens for r in order)
        dur = float(max(len(order), total_tokens))  # slot held ~ tree size
        session_start = now
        session_end = session_start + dur
        for o, req in enumerate(order):
            node = nodes[req]
            op = "start" if node.parent < 0 else "fork"
            start_t = session_start + o * eps
            end_t = session_end + o * eps
            timed.append((start_t, 0, ssn, o, req))   # is_end=0
            timed.append((end_t, 1, ssn, o, req))     # is_end=1
        heapq.heappush(end_heap, (session_end, ssn))
        now = session_start + eps  # next session may start immediately if slot free

    # global event order: by time, starts before ends at ties, then session/order
    timed.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    events: list[dict[str, Any]] = []
    for idx, (t_ms, is_end, _ssn, _o, req) in enumerate(timed):
        node = nodes[req]
        op = "end" if is_end else ("start" if node.parent < 0 else "fork")
        events.append({
            "i": idx,
            "t_ms": round(t_ms, 6),
            "op": op,
            "req": req,
            "parent": node.parent,
            "tenant": node.tenant,
            "agent": node.agent,
            "session": node.session,
            "segments": [list(s) for s in node.segments],
            "new_tokens": node.new_tokens,
        })
    return events


def gen_trace(cfg: TraceConfig, seed: int) -> list[dict[str, Any]]:
    """Deterministically generate a trace for `cfg` and `seed`."""
    rng = np.random.default_rng(seed)
    corpus = _build_corpus(cfg, rng)
    nodes, roots = _generate_forest(cfg, corpus, rng)
    events = _schedule(cfg, nodes, roots)
    return events


def dump_jsonl(trace: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write a trace as JSON Lines (one event object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for ev in trace:
            fh.write(json.dumps(ev, separators=(",", ":")))
            fh.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a trace previously written by dump_jsonl."""
    out: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def decode_tdoc_id(seg_id: int) -> tuple[int, int]:
    """Recover (content_owner_tenant, content_local_id) from a tdoc segment id."""
    return divmod(int(seg_id), TDOC_STRIDE)


if __name__ == "__main__":
    # Dump a small exemplar trace (E2 default cell) and the E4 fork-heavy trace to
    # workloads/ as reproducible artifacts.
    here = Path(__file__).resolve()
    workloads = here.parents[2] / "workloads"
    workloads.mkdir(parents=True, exist_ok=True)

    exemplar_cfg = TraceConfig(n_requests=2000, n_tenants=16, f=0.75,
                               p_leakdup=0.05)
    tr = gen_trace(exemplar_cfg, seed=1)
    dump_jsonl(tr, workloads / "exemplar_e2_T16_f75_leak05_seed1.jsonl")

    e4_cfg = TraceConfig(n_requests=2000, n_tenants=16, f=0.5, p_leakdup=0.0,
                         p_fork=0.6, max_kids=3, max_depth=3)
    tr4 = gen_trace(e4_cfg, seed=1)
    dump_jsonl(tr4, workloads / "exemplar_e4_forkheavy_T16_f50_seed1.jsonl")

    print(f"wrote exemplar traces to {workloads}")
    print(f"  e2 exemplar events: {len(tr)}")
    print(f"  e4 exemplar events: {len(tr4)}")
