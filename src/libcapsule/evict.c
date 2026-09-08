/* evict.c — whole-extent eviction under a resident-byte target (DESIGN §2.6).
 *
 * Store-owner operation (no capability in v1; documented). Evicts whole
 * extents (unlink payload, keep manifests, set EVICTED on every referencing
 * capsule) until stored_bytes <= target. Skips extents referenced by a
 * pinned-and-unexpired capsule or one with live attaches.
 *
 * Policies:
 *   0 = LRU:     candidates ordered by max(last_access_us) over referencing
 *                capsules, oldest first.
 *   1 = LINEAGE: score = age_us / (1 + log2(1 + refcount)), evict highest
 *                score first — widely shared prefix extents are protected.
 *
 * The C implementation is functionally tested; the quantitative E4 comparison
 * runs in the Python mirror.
 */
#include "store_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  uint64_t eid;
  uint64_t max_last_access_us;
  uint32_t refcount;
  double   score;      /* LINEAGE */
} cand_t;

static int cmp_lru(const void *a, const void *b) {
  const cand_t *x = a, *y = b;
  if (x->max_last_access_us < y->max_last_access_us) return -1;  /* oldest first */
  if (x->max_last_access_us > y->max_last_access_us) return 1;
  return (x->eid < y->eid) ? -1 : (x->eid > y->eid);
}

static int cmp_lineage(const void *a, const void *b) {
  const cand_t *x = a, *y = b;
  if (x->score > y->score) return -1;  /* highest score first */
  if (x->score < y->score) return 1;
  return (x->eid < y->eid) ? -1 : (x->eid > y->eid);
}

static int capsule_evict_inner(capsule_store_t *s, uint64_t target_resident_bytes, int policy) {
  if (!s) return CAPSULE_E_INVAL;
  if (policy != 0 && policy != 1) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;

  int r = caps_lock_ex(s);
  if (r != CAPSULE_OK) return r;

  uint64_t now = caps_now_us();
  uint64_t stored = 0;
  for (uint64_t eid = 0; eid < s->exts_cap; eid++) {
    extent_rec_t *e = &s->exts[eid];
    if (e->present && !e->evicted && !e->garbage) stored += e->file_bytes;
  }
  if (stored <= target_resident_bytes) { caps_unlock(s); return CAPSULE_OK; }

  cand_t *cands = malloc((s->exts_cap ? s->exts_cap : 1) * sizeof *cands);
  if (!cands) { caps_unlock(s); return CAPSULE_E_IO; }
  size_t ncand = 0;

  for (uint64_t eid = 0; eid < s->exts_cap; eid++) {
    extent_rec_t *e = &s->exts[eid];
    if (!e->present || e->evicted || e->garbage || e->refcount == 0) continue;
    int skip = 0;
    uint64_t max_la = 0;
    for (uint64_t id = 0; id < s->next_capsule_id && id < s->recs_cap; id++) {
      capsule_rec_t *rec = &s->recs[id];
      if (!rec->present) continue;
      int refs = 0;
      for (uint32_t i = 0; i < rec->n_runs; i++)
        if (rec->runs[i].extent_id == eid) { refs = 1; break; }
      if (!refs) continue;
      if (rec->live_attaches > 0 ||
          (rec->pin_deadline_us != 0 && rec->pin_deadline_us > now)) {
        skip = 1;
        break;
      }
      if (rec->last_access_us > max_la) max_la = rec->last_access_us;
    }
    if (skip) continue;
    cand_t *c = &cands[ncand++];
    c->eid = eid;
    c->max_last_access_us = max_la;
    c->refcount = e->refcount;
    double age = (now > max_la) ? (double)(now - max_la) : 0.0;
    c->score = age / (1.0 + log2(1.0 + (double)e->refcount));
  }

  qsort(cands, ncand, sizeof *cands, policy == 0 ? cmp_lru : cmp_lineage);

  int rr = CAPSULE_OK;
  for (size_t k = 0; k < ncand && stored > target_resident_bytes; k++) {
    uint64_t eid = cands[k].eid;
    extent_rec_t *e = &s->exts[eid];
    uint64_t fb = e->file_bytes;

    char p[1200];
    caps_extent_path(s, eid, p, sizeof p);
    caps_ext_close_fd(e);
    unlink(p);
    e->evicted = 1;
    e->frozen = 1;
    e->has_owner = 0;
    e->file_bytes = 0;
    s->evicted_bytes += fb;
    stored -= fb;
    caps_pi_remove_extent(s, eid);

    /* mark and persist EVICTED on every referencing capsule */
    for (uint64_t id = 0; id < s->next_capsule_id && id < s->recs_cap; id++) {
      capsule_rec_t *rec = &s->recs[id];
      if (!rec->present) continue;
      int refs = 0;
      for (uint32_t i = 0; i < rec->n_runs; i++)
        if (rec->runs[i].extent_id == eid) { refs = 1; break; }
      if (!refs) continue;
      if (!(rec->flags & CAPS_FLAG_EVICTED)) {
        rec->flags |= CAPS_FLAG_EVICTED;
        int w = caps_manifest_write(s, rec);
        if (w != CAPSULE_OK) rr = w;
      }
    }
  }

  free(cands);
  caps_unlock(s);
  if (rr != CAPSULE_OK) return rr;
  return stored <= target_resident_bytes ? CAPSULE_OK : CAPSULE_E_BUDGET;
}


/* ---- thread-safety wrappers ------------------------------------------
 * Every public entry point acquires the store rwlock: writers exclusive,
 * readers shared. The _inner functions are the original bodies, unchanged.
 */

int capsule_evict(capsule_store_t *s, uint64_t target_resident_bytes, int policy) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_evict_inner(s, target_resident_bytes, policy);
  caps_br_wunlock(&s->rw);
  return _r;
}
