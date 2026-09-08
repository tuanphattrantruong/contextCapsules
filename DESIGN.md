# DESIGN.md — Context Capsules prototype (binding design)

> **Note.** This is the design contract as written *before* the measurement
> campaign. Where a predicted magnitude here disagrees with a measured one, the
> numbers of record are the JSON files in `experiments/results/`.

This document is the contract between the two
implementation packages. Package boundaries and the ABI in §2.7 are frozen; internal
details may be refined by the implementing engineer only where marked "impl choice".

Platform facts verified on this machine before writing this design (probe programs, 2026-08-29):

- `CommonCrypto` `CC_SHA256` / `CCHmac(kCCHmacAlgSHA256)` compile and produce correct
  digests with plain `clang -std=c11` (SHA-256("abc") = `ba7816bf...` confirmed). No installs needed.
- `mach_absolute_time` timebase on this machine is 125/3 (41.67 ns/tick).
- Apple Silicon **page size is 16384 bytes** — all mmap offsets must be 16 KiB-aligned (§2.4).
- `mmap(MAP_PRIVATE)` of a file gives hardware CoW: read-faulting 64 MiB of clean file-backed
  pages added **+0 MB** to `phys_footprint`; dirtying half added **+32 MB**. So per-process
  `phys_footprint` (via `proc_pid_rusage(RUSAGE_INFO_V4)` or `task_info(TASK_VM_INFO)`, both
  verified working and agreeing) counts only *dirty/anonymous* pages. This is exactly the metric
  we want for E1 sharing: N attachers of a sealed capsule cost ~0 each; N copiers cost full size each.
- **Benchmark hazard (verified)**: clang -O2 dead-store-eliminates `malloc`+`memset` whose result
  is unused — the footprint probe showed 0 growth until the buffer was consumed. Every benchmark
  MUST consume buffers (volatile accumulator or checksum folded into JSON output).
- Fill benchmark buffers with **pseudorandom bytes, never memset patterns** — macOS memory
  compression would otherwise shrink the copy baseline's real footprint.
- zlib 1.2.12 present (`-lz`); only used if an engineer wants crc32 for cheap integrity in tests —
  all naming/protection hashing is SHA-256/HMAC-SHA-256 via CommonCrypto.

---

## 1. System overview

**Capsule** = a named, content-addressed, capability-protected sequence of KV chunks
representing the KV state of one token prefix. The C library `libcapsule` is the real,
measured artifact: it owns naming (chain-hashed chunk sequences + hierarchical debug paths),
protection (HMAC capabilities + epoch revocation), lifecycle (extend/seal/fork/pin/evict/
revoke/destroy-with-erasure), and physical storage (immutable extent files, `MAP_PRIVATE`
CoW attach, prefix-index dedup with *dedup domains*).

The Python `agentsim` replays multi-tenant agent-swarm traces against three managers
(no-share, vLLM-style prefix cache, capsules) for the workload study, and drives the real
C library via ctypes for validation and for the security studies' counterparts.

Trust model: the store library is the trusted reference monitor
(stand-in for a daemon/kernel service); "tenants" are processes/actors restricted to the
public API. Capabilities protect against referencing objects not granted; the experiments'
attacker processes hold no secret and interact only through the API.

### 1.1 KV geometry constants (documented model constants, NOT measurements)

`REAL_BPT = 131072` bytes/token = 32 layers × 2 (K,V) × 8 KV heads × 128 head_dim × 2 B (fp16)
— Llama-3-8B-class GQA geometry (cite the Llama 3 model card/config in code comments).
Chunk = `CHUNK_TOKENS = 16` tokens → real chunk = 2 MiB; a 4096-token context = 512 MiB
(the target "512 MB capsule"). Scaled geometry for high-volume validation runs:
`SCALED_BPT = 1024` B/token → chunk = 16 KiB (= exactly one page). Constraint: chunk bytes
(`16 × bytes_per_token`) must be a multiple of 16384, i.e. `bytes_per_token % 1024 == 0`
(else `CAPSULE_E_INVAL` at create). Any number derived by multiplying token counts by these
constants is labeled `"modeled"` in results JSON; anything from timers/footprint APIs is `"measured"`.

---

## 2. libcapsule (C11, package 1)

### 2.1 Store directory layout (on disk)

```
<root>/
  store.meta        # text, one line per field: version=1 chunk_tokens=16 created_us=...
  secret.key        # 32 random bytes (arc4random_buf), chmod 0600, created once
  lock              # flock file: LOCK_EX for metadata mutation, LOCK_SH for open/scan
  epochs.tbl        # binary: {u64 magic 'CAPEPOCH', u64 capacity} + u32 epoch[capacity]
                    #   slot index = capsule_id (ids are dense, assigned from 0);
                    #   mmap'd MAP_SHARED by every opener; bump = __atomic_fetch_add;
                    #   grown by ftruncate+remap when full (under LOCK_EX)
  manifests/<id>.mf # one binary manifest per capsule (format §2.2)
  extents/<eid>.ext # raw chunk bytes; immutable once a later extent exists or capsule sealed
```

The prefix index (chain-hash → location) is **in-memory only**, rebuilt at
`capsule_store_open` by scanning manifests (impl choice: open-addressing table,
~64 B/entry; 1M entries ≈ 64 MB worst case, fine).

### 2.2 Manifest binary format (`manifests/<id>.mf`)

Little-endian, packed:

```
mf_header {
  u64 magic = 0x0031464D53504143 ("CAPSMF1\0")
  u64 capsule_id
  u32 bytes_per_token
  u32 chunk_tokens            // always 16 in v1
  u64 n_tokens                // committed tokens (multiple of chunk_tokens)
  u32 flags                   // bit0 SEALED, bit1 EVICTED, bit2 NODEDUP(unused v1)
  u32 n_extents
  u64 parent_id               // ~0ULL if root
  u64 created_us, last_access_us, pin_deadline_us   // CLOCK_REALTIME µs
  u8  chain_hash[32]          // chain over committed chunks
  u8  domain_seed[32]         // SHA-256(dedup_domain string), chain seed (§2.5)
  u16 path_len;  char path[path_len]   // debug namespace, e.g. "/t3/a12/s5"
}
n_extents × mf_extent { u64 extent_id; u64 byte_off; u64 byte_len }   // ordered runs
(n_tokens/chunk_tokens) × u8 chunk_hash[32]
```

Manifest updates are atomic: write `<id>.mf.tmp`, `rename(2)`. Rewrite cost is
O(n_chunks) per extend — acceptable at prototype scale; documented as a
prototype simplification.

### 2.3 Capability token (wire format, 72 bytes, opaque to clients)

```
cap_token {                      // packed little-endian
  u64 magic = 0x0031504143 ("CAP1\0")
  u64 capsule_id
  u32 rights                     // CAP_R_* bitmask
  u32 epoch                      // epoch at mint time
  u64 expiry_us                  // absolute CLOCK_REALTIME µs; 0 = no expiry
  u64 nonce                      // arc4random
  u8  hmac[32]                   // HMAC-SHA256(secret, first 40 bytes)
}
```

Rights: `CAP_R_READ 0x01, CAP_R_EXTEND 0x02, CAP_R_SHARE 0x04, CAP_R_SEAL 0x08,
CAP_R_REVOKE 0x10, CAP_R_ALL 0x1F`. Validation order (constant with respect to
capsule existence — this is a measured security property, S1b):
**(1) HMAC verify with constant-time comparison (`timingsafe_bcmp`, available on macOS;
else a hand-rolled volatile-accumulator compare) — before ANY store lookup; (2) rights ⊇ op;
(3) expiry vs `gettimeofday`; (4) epoch == `epochs.tbl[capsule_id]`; (5) manifest lookup.**
A forged cap for a nonexistent id and a forged cap for an existing id must take the
same path (fail at step 1). `capsule_derive` attenuates: child rights must be ⊆ parent
rights and parent must hold `CAP_R_SHARE`; child expiry = min(parent expiry, now+ttl).

### 2.4 Extents, attach, CoW

- An **extent** is an immutable run of whole chunks in one file. Byte offsets of chunk
  boundaries are multiples of chunk bytes, which are multiples of the 16 KiB page size,
  so any `mf_extent.byte_off` is mmap-able directly.
- A capsule has at most one **open tail extent** (its own, growing via `O_APPEND` writes).
  `capsule_seal` and `capsule_fork` freeze the tail.
- `capsule_attach` validates the cap (READ), then maps each extent run
  `mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, byte_off)` → hardware CoW;
  the engine may scribble on its private view without affecting the store (S3 verifies).
  Mapping reflects tokens committed at attach time (manifest read under LOCK_SH).
- `capsule_read(cap, off, len, buf)` is the API-mediated path: full cap validation
  **per call** (this is the measured per-access protection overhead), then `pread` from
  extent files. Revocation is observed by `capsule_read` at the next call (epoch check);
  raw attached mappings are revoked-at-next-validation (documented, standard capability
  semantics; `capsule_detach` unmaps).

### 2.5 Naming, chain hashing, dedup domains

- `chunk_hash = SHA256(chunk_bytes)`; `chain_0 = domain_seed`;
  `chain_{i+1} = SHA256(chain_i || chunk_hash_{i+1})`. The capsule's content name is the
  chain hash over its committed chunks; self-verifying and prefix-composable.
- **Dedup domain**: `domain_seed = SHA256(domain_string)`. Capsules created with the same
  domain string can dedup against each other; different domains never collide by
  construction. Convention in experiments: `""` (global/public corpus), `"tenant:<k>"`
  (tenant-private). This makes cross-tenant sharing an explicit policy, not an accident —
  the core protection claim.
- Prefix index: `chain_i → {extent_id, byte_off_of_next... , origin}`; concretely each entry
  maps a chain value to the storage location of the chunk that *produced* it, so an extend
  whose running chain matches can adopt existing bytes.

**Extend algorithm** (`capsule_extend(cap, kv, n_tokens)`, n_tokens % 16 == 0):

```
validate cap (EXTEND) ; fail E_SEALED if sealed, E_EVICTED if evicted
for each 16-token chunk c:
  h = SHA256(c); chain' = SHA256(chain || h)
  if in dedup mode and prefix_index[chain'] exists:
      extend current adopted run (coalesce contiguous refs into one mf_extent)
      stats.dedup_hits++, stats.dedup_saved_bytes += chunk_bytes
  else:
      leave dedup mode permanently for this call;
      append chunk to tail extent (create tail if last extent is shared/frozen);
      insert prefix_index[chain'] = location
  chain = chain'
append chunk_hashes, update n_tokens/chain_hash, atomic manifest rewrite
```

Return value and timing-channel note: return code is identical for dedup and store paths.
The *timing* differs (dedup skips the write) — this is a real residual channel within a
domain, quantified in S1c. `CAPSULE_STORE_PARANOID` store flag: on dedup hit, still write
the chunk to a scratch extent (marked garbage, reclaimable) so extend timing is
write-dominated in both paths; S1c measures the distribution collapse. Between domains the
channel does not exist (no admission).

### 2.6 Lifecycle

- **fork**: validate READ on parent; freeze parent tail; create child manifest whose
  extent list = parent's (refcounts++), `parent_id` set, child gets its own domain_seed
  (= parent's) and chain (= parent's); returns owner cap for child. Cost is O(n_extents)
  manifest copy — never O(bytes). This is claim C1.
- **seal**: validate SEAL; set SEALED flag; freeze tail. Extends thereafter → `E_SEALED`.
- **pin**: validate READ; `pin_deadline_us = now + ttl_ms*1000` (ttl 0 clears). Eviction
  skips pinned-and-unexpired capsules' exclusively-owned extents.
- **revoke**: validate REVOKE; `__atomic_fetch_add(&epochs[capsule_id], 1)`. Every
  outstanding cap for the capsule dies at its holder's next validation (S2 measures
  cross-process propagation latency through the shared-mmap epoch table).
- **destroy**: validate REVOKE (destroy implies revoke); bump epoch; for each extent with
  refcount 1: mmap MAP_SHARED, `memset(0)` (via a `volatile unsigned char *` loop or
  `memset_s` so it cannot be eliminated), `msync(MS_SYNC)`, munmap, unlink; shared extents
  refcount--. Remove manifest. Erase-on-last-reference; compliance-delete = revoke (cuts
  readers) + destroy (erases owned bytes). Unit test verifies the extent file is gone and
  a pre-destroy `pread` fd sees zeros before unlink takes effect.
- **evict** (`capsule_evict(store, target_resident_bytes, policy)`): store-owner operation
  (no cap in v1; documented). Evicts whole unpinned, unreferenced-by-live-attach extents
  until resident ≤ target; a capsule with any evicted extent gets EVICTED flag; subsequent
  attach/read/extend → `E_EVICTED` (the client must recompute — the simulator counts this).
  Policies: `0=LRU` (oldest `last_access_us` of owning capsules first) and `1=LINEAGE`
  (score = age_us / (1 + log2(1 + extent_refcount)); evict highest score → shared prefixes
  with many referencing manifests are protected). E4's quantitative comparison runs in the
  Python mirror at 10k-request scale; the C implementation is functionally tested.

### 2.7 Public ABI (`src/libcapsule/capsule.h`) — FROZEN, both packages code to this

```c
#define CAPSULE_ABI_VERSION 1
#define CAPSULE_CHUNK_TOKENS 16
#define CAPSULE_MAX_EXTENTS 128
typedef struct capsule_store capsule_store_t;               /* opaque */
typedef struct { uint8_t bytes[72]; } capsule_cap_t;        /* opaque token */

typedef struct {
  uint32_t nsegs; uint32_t _pad; uint64_t total_bytes;
  void    *seg_base[CAPSULE_MAX_EXTENTS];
  uint64_t seg_len[CAPSULE_MAX_EXTENTS];
  uint64_t _internal[4];
} capsule_map_t;                                            /* caller-allocated */

typedef struct {
  uint64_t capsule_id; uint32_t bytes_per_token; uint32_t chunk_tokens;
  uint64_t n_tokens; uint64_t referenced_bytes;  /* sum of extent ranges referenced */
  uint64_t owned_bytes;                          /* extents where sole reference    */
  uint32_t n_extents; uint32_t epoch; uint32_t flags;   /* 1 SEALED, 2 EVICTED, 4 PINNED */
  uint32_t live_attaches;
  uint8_t  chain_hash[32];
  uint64_t created_us, last_access_us, pin_deadline_us;
} capsule_stat_t;

typedef struct {
  uint64_t n_capsules, n_extents, stored_bytes, logical_bytes,
           dedup_saved_bytes, evicted_bytes, dedup_hits, denied_ops, revocations;
} capsule_store_stats_t;

/* store flags */
#define CAPSULE_STORE_CREATE   0x1
#define CAPSULE_STORE_RDONLY   0x2
#define CAPSULE_STORE_PARANOID 0x4
/* rights */
#define CAP_R_READ 0x01
#define CAP_R_EXTEND 0x02
#define CAP_R_SHARE 0x04
#define CAP_R_SEAL 0x08
#define CAP_R_REVOKE 0x10
#define CAP_R_ALL 0x1F
/* errors (negative) */
#define CAPSULE_OK 0
#define CAPSULE_E_PERM -1
#define CAPSULE_E_EXPIRED -2
#define CAPSULE_E_REVOKED -3
#define CAPSULE_E_SEALED -4
#define CAPSULE_E_EVICTED -5
#define CAPSULE_E_NOENT -6
#define CAPSULE_E_EXISTS -7
#define CAPSULE_E_IO -8
#define CAPSULE_E_INVAL -9
#define CAPSULE_E_BUDGET -10
#define CAPSULE_E_LOCK -11

int  capsule_store_open (const char *root, uint32_t flags, capsule_store_t **out);
int  capsule_store_close(capsule_store_t *s);
int  capsule_store_stats(capsule_store_t *s, capsule_store_stats_t *out);

int  capsule_create (capsule_store_t*, const char *path, const char *dedup_domain, /* NULL = "" */
                     uint32_t bytes_per_token, capsule_cap_t *owner_cap_out, uint64_t *id_out);
int  capsule_extend (capsule_store_t*, const capsule_cap_t*, const void *kv, uint32_t n_tokens);
int  capsule_seal   (capsule_store_t*, const capsule_cap_t*);
int  capsule_fork   (capsule_store_t*, const capsule_cap_t*, const char *child_path,
                     capsule_cap_t *child_cap_out, uint64_t *child_id_out);
int  capsule_attach (capsule_store_t*, const capsule_cap_t*, capsule_map_t *out);
int  capsule_detach (capsule_store_t*, capsule_map_t*);
int  capsule_read   (capsule_store_t*, const capsule_cap_t*, uint64_t off, uint64_t len, void *buf);
int  capsule_derive (capsule_store_t*, const capsule_cap_t *parent, uint32_t rights,
                     uint64_t ttl_ms, capsule_cap_t *out);
int  capsule_revoke (capsule_store_t*, const capsule_cap_t*);
int  capsule_pin    (capsule_store_t*, const capsule_cap_t*, uint64_t ttl_ms);
int  capsule_destroy(capsule_store_t*, const capsule_cap_t*);
int  capsule_stat   (capsule_store_t*, const capsule_cap_t*, capsule_stat_t*);
int  capsule_evict  (capsule_store_t*, uint64_t target_resident_bytes, int policy);
```

Build: `make -C src/libcapsule` produces `src/libcapsule/libcapsule.dylib`
(`clang -O2 -std=c11 -Wall -Wextra -fPIC -dynamiclib`, link nothing extra —
CommonCrypto is in libSystem) plus static-linked binaries `test_capsule`,
`bench_micro`, `sec_c`. All structs are plain C layout, little-endian arm64;
ctypes mirrors them field-for-field (§3.4). No capsule id is discoverable through
the API without a capability; ids travel out-of-band from create/fork returns.

---

## 3. agentsim (Python 3.14 + numpy, package 2)

### 3.1 Trace model

A trace is a list of events (deterministic given a seed) over logical **segments**;
segment identity, not raw bytes, drives the Python managers. Event schema (JSONL when dumped):

```json
{"i":0, "t_ms":12.5, "op":"start|fork|end", "req":41, "parent":17,
 "tenant":3, "agent":2, "session":9,
 "segments":[["sys",4,512],["gdoc",17,384],["tdoc",88,256]], "new_tokens":192}
```

- `start`: new session; prefix = listed segments (kind, id, tokens); then `new_tokens`
  session-private decode tokens are appended.
- `fork`: agent tool-call child of `parent` req at the parent's current length,
  appending its own `new_tokens`.
- `end`: request finishes; its private KV becomes dead (manager may free/evict).
- Segment kinds → sharing/protection semantics:
  `sys` (global corpus, public: capsule domain `""`, read-cap granted to all tenants),
  `gdoc` (global RAG corpus, public, domain `""`),
  `tdoc` (tenant-private doc, domain `tenant:<k>`),
  private decode tokens (session-private, domain `tenant:<k>`).
- Token counts are rounded up to multiples of 16.

Generator parameters (defaults; `traces.py`):

| param | default | meaning |
|---|---|---|
| n_requests | 10000 | root sessions + forks combined |
| tenants T | {4,16,64} | grid axis |
| sharing factor f | {0.25,0.75} | P(doc segment is global `gdoc` vs `tdoc`) |
| p_leakdup | {0,0.05,0.15} | P(a `tdoc`'s content duplicates another tenant's `tdoc`) |
| n_sys | 8, Zipf s=1.2, 512 tok each | shared system prompts |
| n_gdoc | 500, Zipf s=1.1 | global docs, len ~ LogNormal(ln 384, 0.6) |
| n_tdoc | 200/tenant, Zipf s=1.1 | tenant docs, same length dist |
| new_tokens | LogNormal(ln 192, 0.7) | decode length |
| p_fork / max kids / depth | 0.35 / 3 / 2 | agent tree fan-out |
| concurrency M | 64 | steady-state live sessions (arrival interleave) |
| seeds | 1, 2, 3 | numpy `default_rng(seed)` |

`p_leakdup` is realized by aliasing a `tdoc`'s *content identity* to another tenant's
`tdoc` while keeping distinct segment ids — identical bytes, different owners. This is
the surface where the unprotected baseline dedups-and-leaks and capsules deliberately
do not (the "price of safety" axis).

### 3.2 The three managers (`managers.py`)

Common accounting unit: 16-token blocks; content key = (content-domain, kind, content_id,
block_index) where content-domain is `"*"` for public kinds and the *content owner* for
tdocs (so leak-dup blocks share keys). Metrics per run: peak/final resident_bytes
(blocks × REAL_BPT, labeled modeled), prefix_hit_tokens, recomputed_tokens,
cross_tenant_reuse_events (reuse where consumer tenant ≠ producer tenant and kind is
private), denied_attaches, per-manager notes.

1. `NoShareManager` — every request materializes its full context privately; free at `end`.
2. `PrefixCacheManager` — faithful vLLM-style global hash-block cache: block key =
   chain hash of content keys from position 0; single flat cache, **no tenant checks**
   (hits across tenants on leak-dup content count as `cross_tenant_reuse_events` = the
   leak surface); LRU at byte budget (E2: unbounded, records peak; E4: not used).
3. `CapsuleManager` — Python mirror of libcapsule semantics: dedup domains as §2.5
   (public kinds domain `""` → shared; tdoc/private domain per-tenant → never
   cross-admitted), fork = O(1) reference, refcounted extents, capability table
   (grants: every tenant holds read caps for public corpus capsules; tenant-private
   capsules granted only within tenant). Cross-tenant attach attempts (leak-dup probes)
   are counted as `denied_attaches`. Supports eviction budgets + both policies for E4.

### 3.3 Mirror validation against the real library (E2b)

`run_e2b.py` replays one fixed config (T=16, f=0.75, p_leakdup=0.05, seed=1,
500 requests) twice: once through `CapsuleManager` (mirror), once through libcapsule
via ctypes with `SCALED_BPT=1024` and real bytes (chunk bytes generated from
`default_rng(stable_hash(content_key))`). Acceptance: mirror's
{stored_bytes(scaled), dedup_hits, dedup_saved_bytes, denied count} must **exactly equal**
`capsule_store_stats` / accumulated API results; measured process `phys_footprint` and
wall time recorded. Mismatch = bug; the result file records pass/fail honestly.

### 3.4 ctypes bridge (`capsule_ffi.py`)

Loads `Path(__file__).resolve().parents[1] / "libcapsule" / "libcapsule.dylib"`
(override with env `CAPSULE_LIB`). Mirrors §2.7 exactly:
`CapsuleCap = c_uint8 * 72`; `CapsuleMap`, `CapsuleStat`, `CapsuleStoreStats` as
`ctypes.Structure` with the exact field order/types above; all functions
`restype = c_int`, negative = error (raise `CapsuleError(code)` with the §2.7 names).
`capsule_store_t*` handled as `c_void_p`.

---

## 4. Security / lifecycle case studies (E3)

- **S1a — hit-oracle on the unprotected prefix cache** (Python, `run_e3_s1.py`):
  victim tenant A inserts a secret prefix; attacker tenant B probes candidate 16-token
  blocks through the shared `PrefixCacheManager` and uses the hit signal (plus
  `perf_counter_ns` lookup timing) to confirm content block-by-block. Report: probes
  needed, confirmed blocks, hit vs miss timing distributions (mean±std, n≥1000 probes).
  Then the same probe sequence against `CapsuleManager`: attacker holds no cap for A's
  domain → 100% denied; 0 confirmed blocks.
- **S1b — constant-time denial in the real library** (C, `sec_c s1b`): attacker process
  (no secret) calls `capsule_attach` with forged caps for (i) an existing capsule id and
  (ii) a nonexistent id; 10k trials each; report both latency distributions — claim:
  they overlap (HMAC-first validation, §2.3). If they do not overlap, that is a bug to fix
  before recording results, not a result to hide.
- **S1c — residual dedup-timing channel, quantified and closed** (C, `sec_c s1c`):
  within one domain, measure `capsule_extend` latency dedup-hit vs miss (500 each,
  2 MiB chunks): normal store (distributions separate → residual channel exists,
  quantified) vs `CAPSULE_STORE_PARANOID` (distributions collapse). Cost of paranoid
  mode = extra write bandwidth, reported.
- **S2 — revocation propagation** (C, `sec_c s2`): 1 revoker + 4 reader processes; readers
  spin on `capsule_read` recording `mach_absolute_time` per call (global monotonic clock,
  comparable across processes); revoker records t_revoke then bumps the epoch. Latency =
  reader's first `E_REVOKED` timestamp − t_revoke; 50 trials; report per-reader
  p50/p95/max.
- **S3 — seal-then-share integrity** (C, `sec_c s3`): owner seals; sharee (read-only
  derived cap) `capsule_extend` → `E_PERM`; owner extend → `E_SEALED`; sharee scribbles
  over its `capsule_attach` MAP_PRIVATE view, detaches; re-read via `capsule_read` and
  recompute chain hash → matches manifest `chain_hash` (bitwise). Pass/fail JSON.

---

## 5. Claims-to-evidence map

| # | Claim | Experiment | Metric | Results file |
|---|---|---|---|---|
| C1 | Fork is O(metadata), not O(bytes): capsule fork latency is flat in capsule size while copy grows linearly (~10⁴× at 512 MiB) | E1 | fork ns vs memcpy ns across {16..4096} tokens | `e1_micro.json` |
| C2 | CoW attach makes N-way sharing ~1× physical: 8 processes attaching one 512 MiB sealed capsule add ~0 dirty footprint each vs 512 MiB each for copies | E1 | per-process `phys_footprint` deltas + sum, attach vs copy | `e1_micro.json` |
| C3 | Protection is cheap: per-op capability validation (HMAC+epoch) is sub-µs and a small fraction of a chunk extend | E1 | validate-only ns; `capsule_read` with-check vs no-check ns | `e1_micro.json` |
| C4 | Capsules match unprotected prefix caching on hit rate/footprint for legitimately shared content (within-tenant + granted public), across sharing factors and tenant counts | E2 | prefix_hit_tokens, recomputed_tokens, peak resident (modeled) — capsule vs prefix-cache vs no-share | `e2_workload.json` |
| C5 | The price of safety is small and explicit: capsules forgo only coincidental cross-tenant private duplication (p_leakdup axis), which is precisely the baseline's leak surface | E2 + S1a | footprint/recompute delta vs p_leakdup; baseline cross_tenant_reuse_events vs capsule denied_attaches | `e2_workload.json`, `e3_security.json` |
| C6 | The unprotected cache is a content oracle; capsules eliminate it (denied without capability, constant-time in target existence), and the residual dedup-timing channel is quantified and closable (paranoid mode) | E3 S1a/b/c | confirmed blocks; denial latency distributions existing vs nonexistent; extend-timing distributions normal vs paranoid | `e3_security.json` |
| C7 | Revocation propagates to concurrent cross-process readers in bounded time | E3 S2 | revoke→first-E_REVOKED latency p50/p95/max, 4 readers × 50 trials | `e3_security.json` |
| C8 | Seal gives tamper-proof immutable sharing; reader CoW scribbles never reach the store (self-verifying names) | E3 S3 | pass/fail + chain-hash equality | `e3_security.json` |
| C9 | Destroy-with-erasure and lifecycle contract are real: erasure verified, mirror matches the C artifact exactly | unit tests + E2b | test pass counts; mirror-vs-C stat equality; measured footprint of real run | `e2b_validation.json` |
| C10 | Lineage-aware eviction beats LRU on agent-tree traces at equal memory budget | E4 | recomputed tokens (→ modeled bytes) at budgets {10,25,50}% of peak | `e4_eviction.json` |

Honesty labels: every metric above is tagged `measured` (timer/footprint/counters from real
runs) or `modeled` (token counts × documented constants). No TTFT seconds are fabricated —
recompute is reported in tokens and modeled bytes only.

## 6. Experiment matrix (exact)

All runs single-threaded unless stated; every results file carries
`meta = {experiment, cmd, date (\`date -u\`), uname, arch, timer, iterations/seeds, git-less content note}`.
Warmup 10 iterations before every timed C loop; report p50/p95/mean±std.

**E1 — microbenchmarks** (`bench_micro`, real geometry 131072 B/token; store on SSD under `experiments/tmp_store_c/`):
- Latency vs size, sizes = {16, 64, 256, 1024, 4096} tokens (2 MiB…512 MiB):
  create+destroy (1000 iters), extend-per-2MiB-chunk (200), seal (1000), attach lazy (200;
  4096-tok: 50), attach+full-touch (20, volatile-sum consumed), detach, fork (1000),
  derive (10000), revoke (1000), stat (10000), capsule_read 4 KiB (10000) with-cap-check
  vs internal no-check build flag, cap-validate-only (100000).
- memcpy-copy baseline same sizes (50 iters; 4096-tok: 20), buffers pseudorandom, checksummed.
- 8-process sharing: sealed 4096-token capsule (512 MiB, pseudorandom); 8 forked child
  processes attach + touch all pages, report each child's `phys_footprint` delta and sum;
  baseline: 8 children each private-copy 512 MiB (footprint ~4 GiB total — safe in 16 GiB);
  3 repetitions.
- Runtime budget: < 5 min.

**E2 — workload study** (`run_e2.py`): grid = managers{noshare, prefix, capsule} ×
T{4,16,64} × f{0.25,0.75} × p_leakdup{0,0.05,0.15} × seeds{1,2,3} = 162 runs ×
10k requests, unbounded memory, metrics of §3.2, mean±std across seeds. Budget: < 10 min.

**E2b — mirror validation** (`run_e2b.py`): §3.3, one config, 500 requests, real C store
at SCALED_BPT. Budget: < 2 min.

**E3 — security** (`run_e3_s1.py` + `sec_c s1b|s1c|s2|s3` via `run_e3_c.sh`): trial counts
in §4; merged into `e3_security.json` (Python writes `e3_s1a.json`; the shell script merges
C outputs; final merge done by `run_e3_s1.py --merge` reading both). Budget: < 5 min.

**E4 — eviction** (`run_e4.py`): CapsuleManager mirror, fork-heavy trace
(p_fork=0.6, kids≤3, depth 3, T=16, f=0.5, p_leakdup=0), 10k requests;
budgets {10%, 25%, 50%} of the run's own no-eviction peak; policies {LRU, LINEAGE};
seeds {1,2,3} → 18 runs; metric recomputed_tokens (+ modeled bytes). Budget: < 5 min.

## 7. Module breakdown & ownership

Package 1 "libcapsule-core" (C): `src/libcapsule/{capsule.h, store_internal.h, capsule.c,
cap.c, hash.c, evict.c, bench_micro.c, test_capsule.c, sec_c.c, Makefile}`,
`experiments/run_e1.sh`, `experiments/run_e3_c.sh`. Temp stores under
`experiments/tmp_store_c/` (created/removed by its runners).

Package 2 "agentsim-experiments" (Python): `src/agentsim/{__init__.py, kvmodel.py,
traces.py, managers.py, capsule_ffi.py, metrics.py, common.py}`,
`experiments/{run_e2.py, run_e2b.py, run_e3_s1.py, run_e4.py, run_all.sh}`,
`workloads/README.md` (+ dumped exemplar traces `workloads/*.jsonl`). Temp stores under
`experiments/tmp_store_py/`.

No file appears in both lists. Package 2 codes against §2.7/§3.4 only; it never reads C
sources. `run_all.sh` orchestrates: `make -C src/libcapsule test` → `run_e1.sh` →
`run_e2.py` → `run_e2b.py` → `run_e3_c.sh` → `run_e3_s1.py` (incl. merge) → `run_e4.py`.

## 8. Results conventions

`experiments/results/e{1,2,2b,4}_*.json`, `e3_security.json`. Every file:
`{"meta": {...run metadata, includes literal output of `date -u` and `uname -a`...},
"data": {...}}`, every numeric leaf grouped under `measured` or `modeled` keys.
Failures are recorded as `{"status": "failed", "error": "..."}` — never replaced with
plausible numbers.
