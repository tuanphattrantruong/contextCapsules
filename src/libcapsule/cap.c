/* cap.c — 72-byte HMAC-SHA-256 capability tokens (DESIGN.md §2.3).
 *
 * Validation order is a MEASURED security property (S1b):
 *   (1) HMAC verify FIRST, constant-time compare (timingsafe_bcmp from
 *       libSystem — probe-verified on this machine), BEFORE any store or
 *       manifest lookup. A forged cap for an existing capsule id and a forged
 *       cap for a nonexistent id take the identical code path (fail here).
 *   (2) rights superset of the requested operation
 *   (3) expiry vs gettimeofday
 *   (4) epoch equals atomic load of the shared-mmap epochs.tbl slot
 *   (5) manifest (in-memory record) lookup
 * Every validation failure increments stats.denied_ops.
 */
#include "store_internal.h"

#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(cap_wire_t) == 72, "cap wire format must be 72 bytes");
_Static_assert(sizeof(capsule_cap_t) == 72, "public cap token must be 72 bytes");

/* Present in libSystem on macOS (no public prototype header); verified by a
 * link+run probe on this machine before this file was written. */
extern int timingsafe_bcmp(const void *b1, const void *b2, size_t n);

int caps_mint(capsule_store_t *s, uint64_t capsule_id, uint32_t rights,
              uint64_t expiry_us, capsule_cap_t *out) {
  if (!s || !out) return CAPSULE_E_INVAL;
  cap_wire_t w;
  memset(&w, 0, sizeof w);
  w.magic = CAPS_CAP_MAGIC;
  w.capsule_id = capsule_id;
  w.rights = rights & CAP_R_ALL;
  w.epoch = (capsule_id < s->epoch_capacity)
              ? __atomic_load_n(&s->epochs[capsule_id], __ATOMIC_SEQ_CST)
              : 0;
  if (w.epoch == UINT32_MAX) return CAPSULE_E_REVOKED;
  w.expiry_us = expiry_us;
  arc4random_buf(&w.nonce, sizeof w.nonce);
  caps_hmac_sha256(s->secret, &w, 40, w.hmac);
  memcpy(out->bytes, &w, 72);
  return CAPSULE_OK;
}

int caps_validate(capsule_store_t *s, const capsule_cap_t *cap, uint32_t need,
                  capsule_rec_t **rec_out) {
  if (!s || !cap) return CAPSULE_E_INVAL;

  /* Step 1: HMAC over the first 40 bytes, constant-time compare, before ANY
   * lookup that depends on capsule existence. */
  uint8_t mac[32];
  caps_hmac_sha256(s->secret, cap->bytes, 40, mac);
  if (timingsafe_bcmp(mac, cap->bytes + 40, 32) != 0) {
    __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
    return CAPSULE_E_PERM;
  }

  cap_wire_t w;
  memcpy(&w, cap->bytes, 72);
  if (w.magic != CAPS_CAP_MAGIC) {          /* cannot be reached by a forger:
                                             * the magic is HMAC-covered */
    __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
    return CAPSULE_E_PERM;
  }

  /* Step 2: rights. */
  if ((w.rights & need) != need) {
    __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
    return CAPSULE_E_PERM;
  }

  /* Step 3: expiry. */
  if (w.expiry_us != 0 && caps_now_us() > w.expiry_us) {
    __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
    return CAPSULE_E_EXPIRED;
  }

  /* Step 4: epoch (lock-free atomic load from the shared table). */
  if (w.capsule_id < s->epoch_capacity) {
    uint32_t cur = __atomic_load_n(&s->epochs[w.capsule_id], __ATOMIC_SEQ_CST);
    if (cur == UINT32_MAX || cur != w.epoch) {
      __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
      return CAPSULE_E_REVOKED;
    }
  }

  /* Step 5: manifest lookup. */
  if (w.capsule_id >= s->recs_cap || !s->recs[w.capsule_id].present) {
    __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
    return CAPSULE_E_NOENT;
  }
  if (rec_out) *rec_out = &s->recs[w.capsule_id];
  return CAPSULE_OK;
}

int capsule_derive(capsule_store_t *s, const capsule_cap_t *parent, uint32_t rights,
                   uint64_t ttl_ms, capsule_cap_t *out) {
  if (!s || !parent || !out) return CAPSULE_E_INVAL;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, parent, CAP_R_SHARE, &rec);
  if (r != CAPSULE_OK) return r;

  cap_wire_t pw;
  memcpy(&pw, parent->bytes, 72);

  /* Monotonic attenuation: child rights must be a subset of parent rights. */
  if ((rights & ~CAP_R_ALL) != 0 || (rights & ~pw.rights) != 0) {
    __atomic_fetch_add(&s->denied_ops, 1, __ATOMIC_RELAXED);
    return CAPSULE_E_PERM;
  }

  /* child expiry = min(parent expiry (0 = infinity), now + ttl_ms*1000).
   * Impl choice (documented): ttl_ms == 0 means "no additional TTL", i.e. the
   * child inherits the parent expiry, rather than expiring immediately. */
  uint64_t exp = pw.expiry_us;
  if (ttl_ms != 0) {
    uint64_t e2 = caps_now_us() + ttl_ms * 1000ULL;
    exp = (exp == 0 || e2 < exp) ? e2 : exp;
  }
  return caps_mint(s, pw.capsule_id, rights, exp, out);
}

/* Bench entry point (not in the frozen public ABI): full validation only. */
int capsule_internal_validate(capsule_store_t *s, const capsule_cap_t *cap, uint32_t need) {
  return caps_validate(s, cap, need, NULL);
}
