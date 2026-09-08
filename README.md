# Context Capsules — reproduction artifact

**Context capsules** make reusable LLM key–value (KV) state a first-class object below
the serving engine: a capsule has a content-derived name scoped by a deduplication
domain, every operation presents an HMAC capability that attenuates on delegation and
fails after an epoch bump, and its lifecycle (extend, seal, fork, pin, evict, destroy)
is explicit in the interface.

This artifact contains everything needed to build the prototype, run its test suites,
and reproduce every experiment: the C11 store library, the authenticated service, the
two engine bridges, the paged-block adapter, the policy simulator, the experiment
drivers, the input traces, and the result files from our own runs.

---

## 1. Layout

```
src/libcapsule/     the store: naming, capabilities, extents, eviction, locking (C11)
                    + test_capsule (unit tests), bench_micro (E1), sec_c (E3),
                      bench_conc (E7)
src/service/        capsuled.py — authenticated Unix-socket service (E11, E12)
src/kvbridge/       transformers KV cache <-> capsule ABI, token-major packing (E5, E9)
src/llamacap/       capsules under llama.cpp's per-sequence state API (E10, E13, E14)
src/agentsim/       trace generator, the seven KV-state managers, the Python mirror
                    of the store, and the paged-block adapter (E2/E4/E6/E8/E16)
tests/              Python integration tests over the library and the managers
experiments/        one driver per experiment + results/ (our runs, JSON)
workloads/          input traces; exemplar generated traces; GGUF models are NOT
                    shipped (see workloads/gguf/README.md)
data/               output directory for summarize_new.py (empty until you run it)
DESIGN.md           the binding design contract the sources refer to by section
```

## 2. Requirements

The store is deliberately dependency-light, but it is **macOS/Apple Silicon specific**:
it uses CommonCrypto for SHA-256/HMAC, `mach_absolute_time` for timing, and relies on
16 KiB page alignment and `MAP_PRIVATE` copy-on-write semantics. Nothing else is
needed to build it — no third-party C libraries.

| Component | Needs |
|---|---|
| `libcapsule` + C benchmarks | clang with C11, `make` |
| Python experiments, tests | Python ≥ 3.10 (3.12 used), `numpy` |
| E5, E9 (real-model KV) | `torch`, `transformers`, and HF model access |
| E10, E12, E13, E14 (engine) | a built llama.cpp tree + two GGUF checkpoints |

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # torch/transformers only needed for E5, E9
```

Every command below is run from this directory. Scripts that need an interpreter honour
`PYTHON` (default `python3`).

**Unpack this artifact somewhere shallow** (e.g. `~/capsules`, `/tmp/capsules`). The
service experiments bind a Unix domain socket inside `experiments/`, and macOS caps a
socket path at ~104 bytes: from a deeply nested directory E11 and E12 abort with
`OSError: AF_UNIX path too long`. Nothing else is path-sensitive.

## 3. Build and test

```sh
make -C src/libcapsule          # libcapsule.dylib + test_capsule, bench_micro, sec_c, bench_conc
make -C src/libcapsule test     # C unit tests
python3 tests/run_tests.py      # Python integration tests over the built dylib
```

Both suites must pass before any experiment is meaningful: they cover chained naming,
capability attenuation, epoch revocation across a fork/`_exit` reopen, seal semantics,
backing-store zero/unlink, attach/revoke revalidation, and the paged adapter's
generation/refcount invariants.

## 4. Reproduce the experiments

Every driver writes a JSON file under `experiments/results/` carrying its own run
metadata (host, date, timer, and per-number `measured` vs `modeled` labels). Our result
files are already there — move them aside if you want a clean comparison.

| # | What it measures | Command | Writes | Extra needs | Time |
|---|---|---|---|---|---|
| E1 | per-operation cost, fork vs copy, CoW footprint | `bash experiments/run_e1.sh` | `e1_micro.json` | — | ~3 min |
| E2 | first workload study (3 managers) | `python3 experiments/run_e2.py` | `e2_workload.json` | numpy | ~20 s |
| E2b | Python mirror vs the C store, counter-exact | `python3 experiments/run_e2b.py`<br>`python3 experiments/run_e2b_ext.py` | `e2b_validation.json`<br>`e2b_ext_validation.json` | numpy | ~4 min |
| E3 | dedup/hit oracle, denial timing, residual channel | `bash experiments/run_e3_c.sh`<br>`python3 experiments/run_e3_s1.py`<br>`python3 experiments/run_e3_s1.py --merge` | `e3_s1b/s1c/s2/s3.json`,<br>`e3_s1a.json`, `e3_security.json` | numpy | ~5 min |
| E4 | lineage-aware vs LRU eviction | `python3 experiments/run_e4.py` | `e4_eviction.json` | numpy | ~10 s |
| E5 | real KV geometry, cross-process resume, dedup, position dependence | `python3 experiments/run_e5.py` | `e5_realmodel.json` | torch, transformers | model-bound |
| E6 | full workload study, 9 manager configs × 18 cells × 10 seeds | `python3 experiments/run_e6.py` | `e6_workload_ext.json` | numpy | ~2 min |
| E7 | validation/read/extend/revoke under 1–32 threads | `CAPSULE_BR_SHARDS=64 src/libcapsule/bench_conc /tmp/cc_sh experiments/results/e7_concurrency_sharded.json 20000`<br>`CAPSULE_BR_SHARDS=1 src/libcapsule/bench_conc /tmp/cc_si experiments/results/e7_concurrency_single.json 20000` | `e7_concurrency_{sharded,single}.json` | — | ~2 min |
| E8 | the same study on the public traces in `workloads/real/` | `python3 experiments/run_e8.py` | `e8_real_traces.json` | numpy | ~4 min |
| E9 | does the real-model result hold across model families | `python3 experiments/run_e9.py` | `e9_multimodel.json` | torch, transformers | model-bound |
| E10 | capsules inside llama.cpp: cross-process resume, restore vs prefill | `sh experiments/run_e10.sh [llama_dir]` | `e10_*.json` | llama.cpp, GGUF | build-bound |
| E11 | service isolation, denial semantics, RPC cost | `python3 experiments/run_e11_service.py` | `e11_service.json` | — | ~2 min |
| E12 | engine state through the service + daemon hard restart | `python3 experiments/run_e12_service_engine.py` | `e12_service_engine.json` | llamacap, GGUF | ~3 min |
| E13 | request-level TTFT: recompute / native / session file / capsule | `python3 experiments/run_e13_request_level.py` | `e13_request_level.json` | llamacap, GGUF | ~2 min |
| E14 | four concurrent contexts sharing one model | `python3 experiments/run_e14_concurrent.py` | `e14_concurrent.json` | llamacap, GGUF | ~3 min |
| E15 | ten fresh processes for the sub-microsecond paths | `python3 experiments/run_e15_revision_stats.py` | `e15_revision_stats.json` | — | ~2 min |
| E16 | paged block-manager adapter: handles, CoW, revoke, races | `python3 experiments/run_e16_paged_adapter.py` | `e16_paged_adapter.json` | numpy | ~1 min |

Convenience drivers:

- `bash experiments/run_all.sh` — build, test, then E1 → E4 end to end.
- `sh experiments/run_round2.sh` — re-baseline the timing-sensitive experiments on one
  machine, running each C benchmark twice so a median and a cross-run range can be
  reported instead of a single run's tail.
- `python3 experiments/summarize_new.py` — digest of the headline numbers, and the CSVs
  behind the figures, written to `data/`.
- `python3 experiments/summarize_e12.py` — merges the three E12 runs.

E10/E12/E13/E14 need `src/llamacap/llamacap`, which links against a built llama.cpp:

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp -DGGML_METAL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON
cmake --build /tmp/llama.cpp/build --config Release -j
make -C src/llamacap LLAMA_DIR=/tmp/llama.cpp
```

`experiments/run_e10.sh` does all of the above itself, including downloading the two
GGUF checkpoints, if they are absent.

## 5. Inputs

- `workloads/real/` — the three public traces released with Mooncake
  (`mooncake_conversation.jsonl`, `toolagent_trace.jsonl`, `synthetic_trace.jsonl`),
  used by E8. Shipped.
- `workloads/exemplar_*.jsonl` — two generated traces, kept so a reader can inspect the
  trace format without rerunning the generator. Regenerate all of them with
  `cd src && python3 -m agentsim.traces`.
- `workloads/gguf/` — the two GGUF checkpoints for the engine experiments. **Not
  shipped** (~5.6 GB); `workloads/gguf/README.md` gives the URLs, sizes and SHA-256s.

## 6. What reproduces exactly, and what does not

- **Bit-identical from seeds**: the workload studies (E2, E6, E8), eviction (E4), the
  hit-oracle attack (E3 S1a) and the mirror-vs-store validation (E2b). Same seeds, same
  counters, on any machine.
- **Token-identical**: the fidelity checks in E5, E10, E12, E13, E14 compare generated
  token ids and SHA-256 digests of restored KV bytes, not timings. E5's token-equality
  arm runs on CPU because that backend is deterministic.
- **Machine-specific**: every latency, throughput and footprint number. Ours come from
  one Apple M3 Ultra (32 cores, 512 GiB unified memory) running macOS 26.5 that was
  shared with unrelated load, which is why the timing-sensitive experiments are run
  twice and reported as medians with a cross-run range. Expect different absolute
  values; the relationships (flat versus linear, restore versus recompute, sharded
  versus single-counter locking) are what should carry over.

Result files record their own provenance in a `meta` block, so a rerun can be compared
against ours run-by-run rather than number-by-number.

## 7. Known limits of the prototype

Stated up front so nobody reproduces something the artifact does not claim:

- The store is userspace. `capsuled` provides authenticated isolation, but without OS
  confinement of the store namespace a same-user process can bypass the library.
- Manifest rewrites on `extend` are `O(chunks)`, which dominates fork cost, and
  `extend` is serialized by the store write lock.
- Metadata persistence is not a full crash-consistency protocol: epoch revocation is
  explicitly flushed before return (and tested across a fork/`_exit` reopen), but
  manifests, refcounts and directory entries are not fault-injected.
- Capsules sit at llama.cpp's state-serialization boundary, not inside a scheduler or a
  GPU block manager. E16 exercises the paged handle lifecycle as CPU allocator metadata
  only — no GPU DMA, no continuous batching.
- Bytes already disclosed to a holder cannot be retracted: revoke and destroy stop
  future mediated calls, they do not unmap an existing attachment or erase a copy.
