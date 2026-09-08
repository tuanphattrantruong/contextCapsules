# Workloads — agent-swarm trace format & generator

Traces drive the Context Capsules workload study (E2), the mirror validation
(E2b), and the eviction study (E4). They are **deterministic artifacts**: every
trace is fully reproducible from `(TraceConfig, seed)` via
`numpy.random.default_rng(seed)`. The `*.jsonl` files here are exemplars dumped by
`python3 -m agentsim.traces`; the experiments regenerate their own traces from
config+seed rather than reading these files.

## Trace JSONL schema (DESIGN.md 3.1)

One JSON object per line, in causal, non-decreasing `t_ms` order:

```json
{"i":0, "t_ms":0.0, "op":"start", "req":0, "parent":-1,
 "tenant":11, "agent":0, "session":0,
 "segments":[["sys",2,512],["tdoc",11009,336],["gdoc",2,480]], "new_tokens":224}
```

| field | meaning |
|---|---|
| `i` | event index (emission order) |
| `t_ms` | virtual timestamp (ms); non-decreasing across the trace |
| `op` | `start` (new root session), `fork` (agent-tree child), `end` (request finished) |
| `req` | request id (unique per node in the agent forest) |
| `parent` | parent `req` for a fork, else `-1` |
| `tenant` | owning tenant of the session (0..T-1) |
| `agent` | agent id within the session (root = 0, forks increment) |
| `session` | session (agent-tree) id |
| `segments` | ordered structural segments `[kind, id, tokens]` |
| `new_tokens` | session-private decode tokens appended by this request |

### Segment kinds and protection semantics

| kind | corpus | dedup domain | notes |
|---|---|---|---|
| `sys` | 8 global system prompts (public) | `""` | read-cap granted to all tenants |
| `gdoc` | 500 global RAG docs (public) | `""` | shared corpus |
| `tdoc` | 200 per-tenant private docs | `tenant:<using-tenant>` | id encodes the CONTENT owner |
| *decode* | session-private generated tokens | `tenant:<using-tenant>` | modeled inside the managers, not a trace segment |

**tdoc id encoding (self-describing).** A `tdoc` segment's `id` is
`content_owner_tenant * 1000 + content_local_id` (`TDOC_STRIDE = 1000`, and
`n_tdoc_per_tenant = 200 < 1000`). A manager recovers `(owner, local) =
divmod(id, 1000)` and compares `owner` with the event's `tenant`:

* `owner == tenant` — the tenant's own private doc.
* `owner != tenant` — a **leak-duplicate**: identical content bytes owned by
  another tenant. The unprotected prefix cache deduplicates such content across
  tenants (the leak surface); capsules keep it in the using tenant's domain and
  refuse the cross-tenant reuse (a *denied attach*).

`p_leakdup` is the fraction of each tenant's tdocs whose content identity is
aliased to another tenant's tdoc.

### Content bytes (real-C runs, E2b)

Chunk `b` of a content key is
`default_rng(stable_hash((content_key, b))).bytes(chunk_bytes)`, where
`stable_hash` is **SHA-256** (the builtin `hash()` is not stable across runs).
Bytes depend only on the content key, so identical content (including leak-dup
aliases) produces byte-identical chunks — the exact condition under which a
same-domain store deduplicates and a cross-domain store must not.

## Generator parameters (`agentsim.traces.TraceConfig`)

| param | default | E2 grid / E4 | meaning |
|---|---|---|---|
| `n_requests` | 10000 | 10000 | root sessions + forks combined |
| `n_tenants` (T) | 16 | {4, 16, 64} | tenant count |
| `f` | 0.75 | {0.25, 0.75} | P(a doc segment is a public `gdoc` vs a `tdoc`) |
| `p_leakdup` | 0.05 | {0, 0.05, 0.15} | fraction of tdocs aliasing another tenant's content |
| `n_sys` / skew / len | 8 / Zipf 1.2 / 512 | — | shared system prompts |
| `n_gdoc` / skew / len | 500 / Zipf 1.1 / LogNormal(ln 384, 0.6) | — | global docs |
| `n_tdoc_per_tenant` / skew / len | 200 / Zipf 1.1 / LogNormal(ln 384, 0.6) | — | tenant docs |
| `new_tokens` | LogNormal(ln 192, 0.7) | — | decode length |
| `p_fork` / `max_kids` / `max_depth` | 0.35 / 3 / 2 | E4: 0.6 / 3 / 3 | agent-tree fan-out |
| `concurrency_M` | 64 | 64 | steady-state live sessions |
| seeds | — | {1, 2, 3} | `numpy.random.default_rng(seed)` |

All token counts are rounded up to multiples of 16 (`CHUNK_TOKENS`). Zipf skews use
a bounded categorical over normalized `1/k^s` weights (deterministic; never the
unbounded `rng.zipf`, which can exceed the corpus range).

### Concurrency / interleaving

A **session** (an agent tree: its root plus every fork) holds one of `M` slots for
its whole lifetime and frees it when the tree completes. This guarantees every
fork is processed while its ancestors are still live, and reclamation (`end`)
happens at session end. Across the `M` concurrent sessions, events interleave by
timestamp, producing realistic cross-session cache sharing and churn.

## Exemplar files

| file | config |
|---|---|
| `exemplar_e2_T16_f75_leak05_seed1.jsonl` | E2 default cell, 2000 requests, seed 1 |
| `exemplar_e4_forkheavy_T16_f50_seed1.jsonl` | E4 fork-heavy trace, 2000 requests, seed 1 |

(The full experiments use 10000 requests; the exemplars are trimmed for size.)

## Measured vs. modeled labeling rule (DESIGN.md 8)

Every numeric leaf in a results JSON is grouped under **`measured`** or
**`modeled`**:

* **measured** — produced by code run on this machine: token/event counters from
  the managers, `time.perf_counter_ns` timings, `resource.getrusage` footprint,
  and the real `libcapsule` store counters.
* **modeled** — a token count multiplied by a documented KV-geometry constant
  (`REAL_BPT = 131072` B/token, Llama-3-8B GQA; `SCALED_BPT = 1024` for the real-C
  store). These constants are model assumptions, not measurements.

No TTFT seconds are ever fabricated: recompute is reported in tokens and modeled
bytes only. A failed run records `{"status":"failed","error":...}` rather than a
substituted number.
