/* store_internal.h — libcapsule internal structures and helpers.
 * Internal to package 1. The Python side never sees this file; it codes
 * against capsule.h (DESIGN.md §2.7) only.
 *
 * On-disk layout (DESIGN.md §2.1/§2.2):
 *   <root>/store.meta        text: version/chunk_tokens/created_us/next ids
 *   <root>/secret.key        32 random bytes, 0600
 *   <root>/lock              flock file (LOCK_EX metadata mutation, LOCK_SH open)
 *   <root>/epochs.tbl        {u64 magic, u64 capacity} + u32 epoch[capacity], mmap MAP_SHARED
 *   <root>/manifests/<id>.mf binary manifest (mf_header_t + path + runs + chunk hashes)
 *   <root>/extents/<eid>.ext raw chunk payload bytes
 */
#ifndef CAPSULE_STORE_INTERNAL_H
#define CAPSULE_STORE_INTERNAL_H

#include "capsule.h"
#include <stddef.h>
#include <pthread.h>
#include <stdint.h>

#include "brlock.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CAPS_MF_MAGIC    0x0032464D53504143ULL  /* "CAPSMF2\0" little-endian */
#define CAPS_CAP_MAGIC   0x0000000031504143ULL  /* "CAP1\0" little-endian    */
#define CAPS_EPOCH_MAGIC 0x484F435045504143ULL  /* documented epoch-table magic */
#define CAPS_MAX_PATH    512                    /* manifest debug-path cap (incl. NUL) */

#define CAPS_FLAG_SEALED  0x1u
#define CAPS_FLAG_EVICTED 0x2u

/* ---- on-disk records (packed little-endian; arm64 host is LE) ---- */

typedef struct __attribute__((packed)) {
  uint64_t magic;             /* CAPS_MF_MAGIC */
  uint64_t capsule_id;
  uint32_t bytes_per_token;
  uint32_t chunk_tokens;      /* always 16 in v1 */
  uint64_t n_tokens;          /* committed tokens, multiple of chunk_tokens */
  uint32_t flags;             /* bit0 SEALED, bit1 EVICTED */
  uint32_t n_extents;         /* number of mf_run_t records that follow the path */
  uint64_t parent_id;         /* ~0ULL if root */
  uint64_t created_us, last_access_us, pin_deadline_us;
  uint8_t  chain_hash[32];
  uint8_t  initial_chain_hash[32];
  uint8_t  domain_seed[32];
  uint8_t  compatibility_hash[32];
  uint16_t path_len;          /* path bytes follow header, then runs, then hashes */
} mf_header_t;

typedef struct {              /* naturally packed: 3 × u64 */
  uint64_t extent_id;
  uint64_t byte_off;
  uint64_t byte_len;
} mf_run_t;

typedef struct __attribute__((packed)) {
  uint64_t magic;             /* CAPS_CAP_MAGIC */
  uint64_t capsule_id;
  uint32_t rights;
  uint32_t epoch;
  uint64_t expiry_us;         /* absolute CLOCK_REALTIME µs; 0 = never */
  uint64_t nonce;
  uint8_t  hmac[32];          /* HMAC-SHA256(secret, first 40 bytes) */
} cap_wire_t;

typedef struct __attribute__((packed)) {
  uint64_t magic;             /* CAPS_EPOCH_MAGIC */
  uint64_t capacity;
} epoch_hdr_t;

/* ---- in-memory records ---- */

typedef struct {
  uint8_t  present;           /* extent id known (payload file or referenced) */
  uint8_t  evicted;           /* payload unlinked (by evict or destroy)       */
  uint8_t  garbage;           /* paranoid scratch / orphan: excluded from stored_bytes */
  uint8_t  frozen;            /* no further appends allowed                   */
  uint8_t  has_owner;         /* owner_id valid: capsule that may append      */
  uint32_t refcount;          /* number of manifests referencing this extent  */
  uint64_t owner_id;
  uint64_t file_bytes;        /* current payload file size                    */
  int      fd_rw;             /* cached fd (O_RDWR, or O_RDONLY fallback); -1 closed */
} extent_rec_t;

typedef struct {
  uint8_t  present;
  uint64_t id;
  uint32_t bytes_per_token, chunk_tokens;
  uint64_t n_tokens;
  uint32_t flags;             /* CAPS_FLAG_* */
  uint64_t parent_id;
  uint64_t created_us, last_access_us, pin_deadline_us;
  uint8_t  chain_hash[32];
  uint8_t  initial_chain_hash[32];
  uint8_t  domain_seed[32];
  uint8_t  compatibility_hash[32];
  char     path[CAPS_MAX_PATH];
  uint32_t n_runs, runs_cap;
  mf_run_t *runs;
  uint64_t n_chunks, chunks_cap;
  uint8_t  *chunk_hashes;     /* n_chunks × 32 */
  uint32_t live_attaches;     /* in-process count */
} capsule_rec_t;

typedef struct {              /* prefix-index entry: chain value -> chunk location */
  uint8_t  state;             /* 0 empty, 1 used, 2 tombstone */
  uint8_t  chain[32];
  uint64_t eid, off, len;
} pi_entry_t;

struct capsule_store {
  /* Intra-process reader/writer lock. flock() serializes *processes* but is
   * per-open-file-description, so it does not serialize threads sharing one
   * store handle. Mutating entry points take the write lock; read-only ones
   * take the read lock, so validation and reads run concurrently. */
  caps_brlock_t rw;
  char     root[1024];
  uint32_t flags;
  int      lock_fd;
  uint8_t  secret[32];
  /* epoch table (mmap MAP_SHARED) */
  int      epoch_fd;
  void    *epoch_base;
  size_t   epoch_map_len;
  uint64_t epoch_capacity;
  uint32_t *epochs;           /* epoch_base + sizeof(epoch_hdr_t) */
  /* capsules, dense by id */
  capsule_rec_t *recs; uint64_t recs_cap;
  /* extents, dense by extent id */
  extent_rec_t *exts; uint64_t exts_cap;
  uint64_t next_capsule_id, next_extent_id;
  uint64_t meta_created_us;
  /* prefix index (in-memory only, rebuilt at open) */
  pi_entry_t *pi; uint64_t pi_cap, pi_used;
  /* counters — process-local, reset at each capsule_store_open (documented) */
  uint64_t dedup_hits, dedup_saved_bytes, denied_ops, revocations, evicted_bytes;
  uint64_t paranoid_garbage_bytes;   /* bytes written to paranoid scratch extents */
  uint64_t paranoid_scratch_eid;
  uint8_t  has_scratch;
};

/* ---- hash.c ---- */
typedef struct { uint8_t opaque[128]; } caps_sha256_ctx_t;  /* wraps CC_SHA256_CTX (104 B) */
void caps_sha256(const void *data, size_t n, uint8_t out[32]);
void caps_sha256_init(caps_sha256_ctx_t *c);
void caps_sha256_update(caps_sha256_ctx_t *c, const void *data, size_t n);
void caps_sha256_final(caps_sha256_ctx_t *c, uint8_t out[32]);
void caps_hmac_sha256(const uint8_t key[32], const void *data, size_t n, uint8_t out[32]);
void caps_chain_step(const uint8_t chain[32], const uint8_t chunk_hash[32], uint8_t out[32]);
void caps_domain_seed(const char *domain /* NULL = "" */, uint8_t out[32]);

/* ---- cap.c ---- */
int caps_mint(capsule_store_t *s, uint64_t capsule_id, uint32_t rights,
              uint64_t expiry_us, capsule_cap_t *out);
/* Full 5-step validation (DESIGN §2.3): HMAC-first constant-time, rights,
 * expiry, epoch, manifest lookup. rec_out may be NULL. */
int caps_validate(capsule_store_t *s, const capsule_cap_t *cap, uint32_t need,
                  capsule_rec_t **rec_out);

/* ---- capsule.c helpers shared with evict.c / binaries ---- */
uint64_t caps_now_us(void);
int  caps_lock_ex(capsule_store_t *s);
int  caps_lock_sh(capsule_store_t *s);
void caps_unlock(capsule_store_t *s);
int  caps_manifest_write(capsule_store_t *s, capsule_rec_t *rec);
void caps_extent_path(const capsule_store_t *s, uint64_t eid, char *buf, size_t n);
void caps_pi_remove_extent(capsule_store_t *s, uint64_t eid);
void caps_ext_close_fd(extent_rec_t *e);

/* ---- bench/diagnostic entry points (exported from the library objects;
 * NOT part of the frozen public ABI in capsule.h; used by bench_micro/sec_c
 * which link the objects directly) ---- */
int      capsule_internal_validate(capsule_store_t *s, const capsule_cap_t *cap, uint32_t need);
int      capsule_read_unchecked(capsule_store_t *s, uint64_t capsule_id,
                                uint64_t off, uint64_t len, void *buf);
uint64_t capsule_internal_paranoid_garbage_bytes(const capsule_store_t *s);

#ifdef __cplusplus
}
#endif

#endif /* CAPSULE_STORE_INTERNAL_H */
