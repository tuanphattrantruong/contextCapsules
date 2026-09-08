/* capsule.h — libcapsule public ABI (Context Capsules prototype).
 *
 * ABI v3. v2 added capsule_create_chained. v3 binds capsules to a caller-defined
 * 256-bit compatibility descriptor and adds checked create/attach calls. The
 * addition was forced by measurement: naming a tenant's private
 * content from the domain seed alone is position-independent, which real
 * dense-transformer KV does not support (identical tokens after a different
 * prefix produce different bytes). A capsule must therefore be able to start
 * its chain from another capsule's chain while living in its own dedup domain.
 * All structs are plain C layout, little-endian arm64.
 */
#ifndef CAPSULE_H
#define CAPSULE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAPSULE_ABI_VERSION 3
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
  uint8_t  compatibility_hash[32];
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
/* compatibility_hash is an opaque SHA-256 digest over the caller's complete
 * compatibility descriptor (model/weights, engine schema, tensor geometry,
 * precision and layout). It is incorporated into the naming seed, preventing
 * cross-configuration deduplication. */
int  capsule_create_compatible(capsule_store_t*, const char *path, const char *dedup_domain,
                     uint32_t bytes_per_token, const uint8_t compatibility_hash[32],
                     capsule_cap_t *owner_cap_out, uint64_t *id_out);
/* As capsule_create, but chain_0 = H(domain_seed || chain_parent.chain) rather
 * than chain_0 = domain_seed. The new capsule deduplicates within `dedup_domain`
 * only against content that shares BOTH the domain and the parent's entire
 * committed prefix. `chain_parent` needs READ rights; it is not otherwise
 * referenced (no extent sharing, no refcount, no lineage edge). */
int  capsule_create_chained(capsule_store_t*, const char *path, const char *dedup_domain,
                     uint32_t bytes_per_token, const capsule_cap_t *chain_parent,
                     capsule_cap_t *owner_cap_out, uint64_t *id_out);
int  capsule_extend (capsule_store_t*, const capsule_cap_t*, const void *kv, uint32_t n_tokens);
int  capsule_seal   (capsule_store_t*, const capsule_cap_t*);
int  capsule_fork   (capsule_store_t*, const capsule_cap_t*, const char *child_path,
                     capsule_cap_t *child_cap_out, uint64_t *child_id_out);
int  capsule_attach (capsule_store_t*, const capsule_cap_t*, capsule_map_t *out);
/* Fails with CAPSULE_E_INVAL before mapping if expected_hash differs from the
 * descriptor recorded at create time. */
int  capsule_attach_compatible(capsule_store_t*, const capsule_cap_t*,
                     const uint8_t expected_hash[32], capsule_map_t *out);
int  capsule_detach (capsule_store_t*, capsule_map_t*);
int  capsule_read   (capsule_store_t*, const capsule_cap_t*, uint64_t off, uint64_t len, void *buf);
int  capsule_derive (capsule_store_t*, const capsule_cap_t *parent, uint32_t rights,
                     uint64_t ttl_ms, capsule_cap_t *out);
int  capsule_revoke (capsule_store_t*, const capsule_cap_t*);
int  capsule_pin    (capsule_store_t*, const capsule_cap_t*, uint64_t ttl_ms);
int  capsule_destroy(capsule_store_t*, const capsule_cap_t*);
int  capsule_stat   (capsule_store_t*, const capsule_cap_t*, capsule_stat_t*);
int  capsule_evict  (capsule_store_t*, uint64_t target_resident_bytes, int policy);

#ifdef __cplusplus
}
#endif

#endif /* CAPSULE_H */
