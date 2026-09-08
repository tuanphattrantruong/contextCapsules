/* capsule.c — capsule store: directory layout, manifests, epoch table,
 * prefix index, and the core lifecycle ops (DESIGN.md §2.1–§2.6).
 *
 * Concurrency model (documented simplifications, per DESIGN §2.1/§2.4):
 *  - single-writer / multi-reader across processes; all on-disk metadata
 *    mutation happens under flock(LOCK_EX) on <root>/lock; store open scans
 *    under LOCK_SH.
 *  - each process's in-memory view (manifests, prefix index) is a snapshot as
 *    of its capsule_store_open; ONLY the epoch table is live-shared across
 *    processes (mmap MAP_SHARED + atomic ops) — that is exactly what
 *    revocation (S2) needs. Extent payloads are append-only/immutable, so a
 *    snapshot's committed ranges never change under it.
 *  - stats counters (dedup_hits, denied_ops, ...) are process-local since
 *    open; the library is single-threaded per store handle.
 *  - capability epochs are synchronously persisted before revoke returns.
 *    Manifest/meta tmp+rename updates are not yet fsync'd, so this prototype
 *    claims crash-durable revocation, not general crash-durable metadata.
 *  - last_access_us is updated in memory on read/attach and persisted on the
 *    next metadata mutation of that capsule (a per-read manifest rewrite would
 *    dominate the measured read path).
 */
#include "store_internal.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ utils */

uint64_t caps_now_us(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

int caps_lock_ex(capsule_store_t *s) {
  return flock(s->lock_fd, LOCK_EX) == 0 ? CAPSULE_OK : CAPSULE_E_LOCK;
}
int caps_lock_sh(capsule_store_t *s) {
  return flock(s->lock_fd, LOCK_SH) == 0 ? CAPSULE_OK : CAPSULE_E_LOCK;
}
void caps_unlock(capsule_store_t *s) { (void)flock(s->lock_fd, LOCK_UN); }

void caps_extent_path(const capsule_store_t *s, uint64_t eid, char *buf, size_t n) {
  snprintf(buf, n, "%s/extents/%llu.ext", s->root, (unsigned long long)eid);
}
static void manifest_path(const capsule_store_t *s, uint64_t id, char *buf, size_t n) {
  snprintf(buf, n, "%s/manifests/%llu.mf", s->root, (unsigned long long)id);
}

void caps_ext_close_fd(extent_rec_t *e) {
  if (e->fd_rw >= 0) { close(e->fd_rw); e->fd_rw = -1; }
}

static int ext_fd(capsule_store_t *s, uint64_t eid) {
  extent_rec_t *e = &s->exts[eid];
  if (e->fd_rw >= 0) return e->fd_rw;
  char p[1200];
  caps_extent_path(s, eid, p, sizeof p);
  int fd = open(p, O_RDWR);
  if (fd < 0) fd = open(p, O_RDONLY);
  e->fd_rw = fd;
  return fd;
}

static int write_full(int fd, const void *buf, size_t n, off_t off) {
  const char *p = buf;
  size_t done = 0;
  while (done < n) {
    ssize_t w = pwrite(fd, p + done, n - done, off + (off_t)done);
    if (w < 0) { if (errno == EINTR) continue; return -1; }
    done += (size_t)w;
  }
  return 0;
}

static int read_full(int fd, void *buf, size_t n, off_t off) {
  char *p = buf;
  size_t done = 0;
  while (done < n) {
    ssize_t r = pread(fd, p + done, n - done, off + (off_t)done);
    if (r < 0) { if (errno == EINTR) continue; return -1; }
    if (r == 0) return -1;
    done += (size_t)r;
  }
  return 0;
}

/* ---------------------------------------------------------------- meta io */

static int meta_write(capsule_store_t *s) {
  char tmp[1200], fin[1200];
  snprintf(tmp, sizeof tmp, "%s/store.meta.tmp", s->root);
  snprintf(fin, sizeof fin, "%s/store.meta", s->root);
  FILE *f = fopen(tmp, "w");
  if (!f) return CAPSULE_E_IO;
  fprintf(f,
          "version=1\nchunk_tokens=%d\ncreated_us=%llu\n"
          "next_capsule_id=%llu\nnext_extent_id=%llu\n",
          CAPSULE_CHUNK_TOKENS, (unsigned long long)s->meta_created_us,
          (unsigned long long)s->next_capsule_id,
          (unsigned long long)s->next_extent_id);
  if (fclose(f) != 0) return CAPSULE_E_IO;
  if (rename(tmp, fin) != 0) return CAPSULE_E_IO;
  return CAPSULE_OK;
}

static int meta_read(capsule_store_t *s) {
  char fin[1200];
  snprintf(fin, sizeof fin, "%s/store.meta", s->root);
  FILE *f = fopen(fin, "r");
  if (!f) return CAPSULE_E_NOENT;
  char line[256];
  int ok_version = 0;
  while (fgets(line, sizeof line, f)) {
    unsigned long long v;
    if (sscanf(line, "version=%llu", &v) == 1) ok_version = (v == 1);
    else if (sscanf(line, "created_us=%llu", &v) == 1) s->meta_created_us = v;
    else if (sscanf(line, "next_capsule_id=%llu", &v) == 1) s->next_capsule_id = v;
    else if (sscanf(line, "next_extent_id=%llu", &v) == 1) s->next_extent_id = v;
  }
  fclose(f);
  return ok_version ? CAPSULE_OK : CAPSULE_E_IO;
}

/* ------------------------------------------------------------ epoch table */

static int epoch_map(capsule_store_t *s) {
  struct stat st;
  if (fstat(s->epoch_fd, &st) != 0) return CAPSULE_E_IO;
  if ((size_t)st.st_size < sizeof(epoch_hdr_t)) return CAPSULE_E_IO;
  int prot = PROT_READ | ((s->flags & CAPSULE_STORE_RDONLY) ? 0 : PROT_WRITE);
  void *base = mmap(NULL, (size_t)st.st_size, prot, MAP_SHARED, s->epoch_fd, 0);
  if (base == MAP_FAILED) return CAPSULE_E_IO;
  epoch_hdr_t hdr;
  memcpy(&hdr, base, sizeof hdr);
  if (hdr.magic != CAPS_EPOCH_MAGIC) { munmap(base, (size_t)st.st_size); return CAPSULE_E_IO; }
  s->epoch_base = base;
  s->epoch_map_len = (size_t)st.st_size;
  s->epoch_capacity = hdr.capacity;
  s->epochs = (uint32_t *)((char *)base + sizeof(epoch_hdr_t));
  return CAPSULE_OK;
}

static void epoch_unmap(capsule_store_t *s) {
  if (s->epoch_base) { munmap(s->epoch_base, s->epoch_map_len); s->epoch_base = NULL; }
  s->epochs = NULL;
}

/* Caller holds LOCK_EX. Grows the table via ftruncate + remap. */
static int epoch_ensure(capsule_store_t *s, uint64_t need_capacity) {
  if (need_capacity <= s->epoch_capacity) return CAPSULE_OK;
  uint64_t nc = s->epoch_capacity ? s->epoch_capacity : 64;
  while (nc < need_capacity) nc *= 2;
  epoch_unmap(s);
  if (ftruncate(s->epoch_fd, (off_t)(sizeof(epoch_hdr_t) + 4 * nc)) != 0) return CAPSULE_E_IO;
  epoch_hdr_t hdr = { CAPS_EPOCH_MAGIC, nc };
  if (write_full(s->epoch_fd, &hdr, sizeof hdr, 0) != 0) return CAPSULE_E_IO;
  return epoch_map(s);
}

/* Persist a completed epoch transition before reporting revoke success.  The
 * mmap write is the live cross-process invalidation mechanism; msync+fsync is
 * the crash boundary for future opens. */
static int epoch_sync(capsule_store_t *s) {
  if (msync(s->epoch_base, s->epoch_map_len, MS_SYNC) != 0)
    return CAPSULE_E_IO;
  if (fsync(s->epoch_fd) != 0)
    return CAPSULE_E_IO;
  return CAPSULE_OK;
}

static int epoch_advance_durable(capsule_store_t *s, uint64_t id) {
  if (id >= s->epoch_capacity) return CAPSULE_E_IO;
  uint32_t old = __atomic_load_n(&s->epochs[id], __ATOMIC_SEQ_CST);
  for (;;) {
    /* UINT32_MAX is a retired identifier, not an epoch that may wrap to zero. */
    if (old == UINT32_MAX) return CAPSULE_E_REVOKED;
    uint32_t next = old + 1;
    if (__atomic_compare_exchange_n(&s->epochs[id], &old, next, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
      break;
  }
  return epoch_sync(s);
}

/* ------------------------------------------------------------ array growth */

static int recs_ensure(capsule_store_t *s, uint64_t need) {
  if (need <= s->recs_cap) return CAPSULE_OK;
  uint64_t nc = s->recs_cap ? s->recs_cap : 16;
  while (nc < need) nc *= 2;
  capsule_rec_t *n = realloc(s->recs, nc * sizeof *n);
  if (!n) return CAPSULE_E_IO;
  memset(n + s->recs_cap, 0, (nc - s->recs_cap) * sizeof *n);
  s->recs = n;
  s->recs_cap = nc;
  return CAPSULE_OK;
}

static int exts_ensure(capsule_store_t *s, uint64_t need) {
  if (need <= s->exts_cap) return CAPSULE_OK;
  uint64_t nc = s->exts_cap ? s->exts_cap : 16;
  while (nc < need) nc *= 2;
  extent_rec_t *n = realloc(s->exts, nc * sizeof *n);
  if (!n) return CAPSULE_E_IO;
  for (uint64_t i = s->exts_cap; i < nc; i++) {
    memset(&n[i], 0, sizeof n[i]);
    n[i].fd_rw = -1;
  }
  s->exts = n;
  s->exts_cap = nc;
  return CAPSULE_OK;
}

static int rec_runs_ensure(capsule_rec_t *rec, uint32_t need) {
  if (need <= rec->runs_cap) return CAPSULE_OK;
  uint32_t nc = rec->runs_cap ? rec->runs_cap : 8;
  while (nc < need) nc *= 2;
  mf_run_t *n = realloc(rec->runs, (size_t)nc * sizeof *n);
  if (!n) return CAPSULE_E_IO;
  rec->runs = n;
  rec->runs_cap = nc;
  return CAPSULE_OK;
}

static int rec_chunks_ensure(capsule_rec_t *rec, uint64_t need) {
  if (need <= rec->chunks_cap) return CAPSULE_OK;
  uint64_t nc = rec->chunks_cap ? rec->chunks_cap : 32;
  while (nc < need) nc *= 2;
  uint8_t *n = realloc(rec->chunk_hashes, nc * 32);
  if (!n) return CAPSULE_E_IO;
  rec->chunk_hashes = n;
  rec->chunks_cap = nc;
  return CAPSULE_OK;
}

/* ------------------------------------------------------------ prefix index */

static uint64_t pi_slot_hash(const uint8_t chain[32]) {
  uint64_t h;
  memcpy(&h, chain, 8);
  return h;
}

static int pi_grow(capsule_store_t *s, uint64_t min_cap);

static int pi_insert(capsule_store_t *s, const uint8_t chain[32],
                     uint64_t eid, uint64_t off, uint64_t len) {
  if (s->pi_cap == 0 || (s->pi_used + 1) * 10 >= s->pi_cap * 7) {
    int r = pi_grow(s, s->pi_cap ? s->pi_cap * 2 : 1024);
    if (r != CAPSULE_OK) return r;
  }
  uint64_t mask = s->pi_cap - 1;
  uint64_t i = pi_slot_hash(chain) & mask;
  int64_t first_tomb = -1;
  for (;;) {
    pi_entry_t *e = &s->pi[i];
    if (e->state == 0) {
      if (first_tomb >= 0) e = &s->pi[first_tomb];
      e->state = 1;
      memcpy(e->chain, chain, 32);
      e->eid = eid; e->off = off; e->len = len;
      s->pi_used++;
      return CAPSULE_OK;
    }
    if (e->state == 2) {
      if (first_tomb < 0) first_tomb = (int64_t)i;
    } else if (memcmp(e->chain, chain, 32) == 0) {
      return CAPSULE_OK;  /* already present (same content, same domain) */
    }
    i = (i + 1) & mask;
  }
}

static pi_entry_t *pi_lookup(capsule_store_t *s, const uint8_t chain[32]) {
  if (s->pi_cap == 0) return NULL;
  uint64_t mask = s->pi_cap - 1;
  uint64_t i = pi_slot_hash(chain) & mask;
  for (;;) {
    pi_entry_t *e = &s->pi[i];
    if (e->state == 0) return NULL;
    if (e->state == 1 && memcmp(e->chain, chain, 32) == 0) return e;
    i = (i + 1) & mask;
  }
}

static int pi_grow(capsule_store_t *s, uint64_t min_cap) {
  uint64_t nc = 1024;
  while (nc < min_cap) nc *= 2;
  pi_entry_t *np = calloc(nc, sizeof *np);
  if (!np) return CAPSULE_E_IO;
  pi_entry_t *old = s->pi;
  uint64_t oldc = s->pi_cap;
  s->pi = np; s->pi_cap = nc; s->pi_used = 0;
  for (uint64_t i = 0; i < oldc; i++)
    if (old && old[i].state == 1)
      pi_insert(s, old[i].chain, old[i].eid, old[i].off, old[i].len);
  free(old);
  return CAPSULE_OK;
}

void caps_pi_remove_extent(capsule_store_t *s, uint64_t eid) {
  for (uint64_t i = 0; i < s->pi_cap; i++) {
    if (s->pi[i].state == 1 && s->pi[i].eid == eid) {
      s->pi[i].state = 2;  /* tombstone */
      if (s->pi_used) s->pi_used--;
    }
  }
}

/* ------------------------------------------------------------ manifest io */

int caps_manifest_write(capsule_store_t *s, capsule_rec_t *rec) {
  size_t plen = strlen(rec->path);
  size_t sz = sizeof(mf_header_t) + plen +
              (size_t)rec->n_runs * sizeof(mf_run_t) + (size_t)rec->n_chunks * 32;
  uint8_t *buf = malloc(sz ? sz : 1);
  if (!buf) return CAPSULE_E_IO;
  mf_header_t h;
  memset(&h, 0, sizeof h);
  h.magic = CAPS_MF_MAGIC;
  h.capsule_id = rec->id;
  h.bytes_per_token = rec->bytes_per_token;
  h.chunk_tokens = rec->chunk_tokens;
  h.n_tokens = rec->n_tokens;
  h.flags = rec->flags;
  h.n_extents = rec->n_runs;
  h.parent_id = rec->parent_id;
  h.created_us = rec->created_us;
  h.last_access_us = rec->last_access_us;
  h.pin_deadline_us = rec->pin_deadline_us;
  memcpy(h.chain_hash, rec->chain_hash, 32);
  memcpy(h.initial_chain_hash, rec->initial_chain_hash, 32);
  memcpy(h.domain_seed, rec->domain_seed, 32);
  memcpy(h.compatibility_hash, rec->compatibility_hash, 32);
  h.path_len = (uint16_t)plen;
  size_t o = 0;
  memcpy(buf + o, &h, sizeof h); o += sizeof h;
  memcpy(buf + o, rec->path, plen); o += plen;
  memcpy(buf + o, rec->runs, (size_t)rec->n_runs * sizeof(mf_run_t));
  o += (size_t)rec->n_runs * sizeof(mf_run_t);
  memcpy(buf + o, rec->chunk_hashes, (size_t)rec->n_chunks * 32);
  o += (size_t)rec->n_chunks * 32;

  char tmp[1200], fin[1200];
  snprintf(tmp, sizeof tmp, "%s/manifests/%llu.mf.tmp", s->root,
           (unsigned long long)rec->id);
  manifest_path(s, rec->id, fin, sizeof fin);
  int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) { free(buf); return CAPSULE_E_IO; }
  int w = write_full(fd, buf, o, 0);
  close(fd);
  free(buf);
  if (w != 0 || rename(tmp, fin) != 0) return CAPSULE_E_IO;
  return CAPSULE_OK;
}

static int manifest_read(capsule_store_t *s, uint64_t id, capsule_rec_t *rec) {
  char fin[1200];
  manifest_path(s, id, fin, sizeof fin);
  int fd = open(fin, O_RDONLY);
  if (fd < 0) return CAPSULE_E_NOENT;
  struct stat st;
  if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(mf_header_t)) {
    close(fd); return CAPSULE_E_IO;
  }
  mf_header_t h;
  if (read_full(fd, &h, sizeof h, 0) != 0 || h.magic != CAPS_MF_MAGIC ||
      h.capsule_id != id || h.path_len >= CAPS_MAX_PATH ||
      h.chunk_tokens != CAPSULE_CHUNK_TOKENS) {
    close(fd); return CAPSULE_E_IO;
  }
  size_t need = sizeof h + h.path_len +
                (size_t)h.n_extents * sizeof(mf_run_t) +
                (size_t)(h.n_tokens / h.chunk_tokens) * 32;
  if ((size_t)st.st_size < need) { close(fd); return CAPSULE_E_IO; }

  memset(rec, 0, sizeof *rec);
  rec->present = 1;
  rec->id = id;
  rec->bytes_per_token = h.bytes_per_token;
  rec->chunk_tokens = h.chunk_tokens;
  rec->n_tokens = h.n_tokens;
  rec->flags = h.flags;
  rec->parent_id = h.parent_id;
  rec->created_us = h.created_us;
  rec->last_access_us = h.last_access_us;
  rec->pin_deadline_us = h.pin_deadline_us;
  memcpy(rec->chain_hash, h.chain_hash, 32);
  memcpy(rec->initial_chain_hash, h.initial_chain_hash, 32);
  memcpy(rec->domain_seed, h.domain_seed, 32);
  memcpy(rec->compatibility_hash, h.compatibility_hash, 32);

  off_t o = (off_t)sizeof h;
  if (h.path_len) {
    if (read_full(fd, rec->path, h.path_len, o) != 0) goto io_err;
  }
  rec->path[h.path_len] = '\0';
  o += h.path_len;

  if (h.n_extents) {
    if (rec_runs_ensure(rec, h.n_extents) != CAPSULE_OK) goto io_err;
    if (read_full(fd, rec->runs, (size_t)h.n_extents * sizeof(mf_run_t), o) != 0)
      goto io_err;
  }
  rec->n_runs = h.n_extents;
  o += (off_t)((size_t)h.n_extents * sizeof(mf_run_t));

  uint64_t nch = h.n_tokens / h.chunk_tokens;
  if (nch) {
    if (rec_chunks_ensure(rec, nch) != CAPSULE_OK) goto io_err;
    if (read_full(fd, rec->chunk_hashes, (size_t)nch * 32, o) != 0) goto io_err;
  }
  rec->n_chunks = nch;
  close(fd);
  return CAPSULE_OK;

io_err:
  free(rec->runs); free(rec->chunk_hashes);
  memset(rec, 0, sizeof *rec);
  close(fd);
  return CAPSULE_E_IO;
}

/* --------------------------------------------------------------- run utils */

static int rec_references(const capsule_rec_t *rec, uint64_t eid) {
  for (uint32_t i = 0; i < rec->n_runs; i++)
    if (rec->runs[i].extent_id == eid) return 1;
  return 0;
}

/* Append one chunk-location to rec's run list, coalescing with the last run
 * when contiguous in the same extent. */
static int rec_add_loc(capsule_rec_t *rec, uint64_t eid, uint64_t off, uint64_t len) {
  if (rec->n_runs > 0) {
    mf_run_t *last = &rec->runs[rec->n_runs - 1];
    if (last->extent_id == eid && last->byte_off + last->byte_len == off) {
      last->byte_len += len;
      return CAPSULE_OK;
    }
  }
  int r = rec_runs_ensure(rec, rec->n_runs + 1);
  if (r != CAPSULE_OK) return r;
  rec->runs[rec->n_runs].extent_id = eid;
  rec->runs[rec->n_runs].byte_off = off;
  rec->runs[rec->n_runs].byte_len = len;
  rec->n_runs++;
  return CAPSULE_OK;
}

/* Map logical chunk index -> (eid, byte_off). Returns 0 on success. */
static int rec_chunk_loc(const capsule_rec_t *rec, uint64_t chunk_idx,
                         uint64_t chunk_bytes, uint64_t *eid, uint64_t *off) {
  uint64_t want = chunk_idx * chunk_bytes;
  uint64_t pos = 0;
  for (uint32_t i = 0; i < rec->n_runs; i++) {
    if (want < pos + rec->runs[i].byte_len) {
      *eid = rec->runs[i].extent_id;
      *off = rec->runs[i].byte_off + (want - pos);
      return 0;
    }
    pos += rec->runs[i].byte_len;
  }
  return -1;
}

/* ---------------------------------------------------------------- open/close */

static int mkdir_ok(const char *p) {
  if (mkdir(p, 0755) != 0 && errno != EEXIST) return -1;
  return 0;
}

int capsule_store_open(const char *root, uint32_t flags, capsule_store_t **out) {
  if (!root || !out || strlen(root) >= 1000) return CAPSULE_E_INVAL;
  *out = NULL;
  capsule_store_t *s = calloc(1, sizeof *s);
  if (!s) return CAPSULE_E_IO;
  strncpy(s->root, root, sizeof s->root - 1);
  s->flags = flags;
  s->lock_fd = s->epoch_fd = -1;
  caps_br_init(&s->rw);
  int r = CAPSULE_E_IO;
  char p[1200];

  if (flags & CAPSULE_STORE_CREATE) {
    if (mkdir_ok(s->root) != 0) goto fail;
    snprintf(p, sizeof p, "%s/manifests", s->root);
    if (mkdir_ok(p) != 0) goto fail;
    snprintf(p, sizeof p, "%s/extents", s->root);
    if (mkdir_ok(p) != 0) goto fail;
  } else {
    struct stat st;
    if (stat(s->root, &st) != 0 || !S_ISDIR(st.st_mode)) { r = CAPSULE_E_NOENT; goto fail; }
  }

  snprintf(p, sizeof p, "%s/lock", s->root);
  s->lock_fd = open(p, O_RDWR | O_CREAT, 0644);
  if (s->lock_fd < 0 && (flags & CAPSULE_STORE_RDONLY))
    s->lock_fd = open(p, O_RDONLY);
  if (s->lock_fd < 0) goto fail;

  if ((r = caps_lock_ex(s)) != CAPSULE_OK) goto fail;

  /* store.meta / secret.key / epochs.tbl — created once under LOCK_EX. */
  snprintf(p, sizeof p, "%s/store.meta", s->root);
  struct stat st;
  if (stat(p, &st) != 0) {
    if (!(flags & CAPSULE_STORE_CREATE)) { caps_unlock(s); r = CAPSULE_E_NOENT; goto fail; }
    s->meta_created_us = caps_now_us();
    s->next_capsule_id = 0;
    s->next_extent_id = 0;
    if ((r = meta_write(s)) != CAPSULE_OK) { caps_unlock(s); goto fail; }
  } else {
    if ((r = meta_read(s)) != CAPSULE_OK) { caps_unlock(s); goto fail; }
  }

  snprintf(p, sizeof p, "%s/secret.key", s->root);
  if (stat(p, &st) != 0) {
    if (!(flags & CAPSULE_STORE_CREATE)) { caps_unlock(s); r = CAPSULE_E_NOENT; goto fail; }
    uint8_t key[32];
    arc4random_buf(key, 32);
    int kfd = open(p, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (kfd < 0 || write_full(kfd, key, 32, 0) != 0) {
      if (kfd >= 0) close(kfd);
      caps_unlock(s); r = CAPSULE_E_IO; goto fail;
    }
    close(kfd);
  }
  {
    int kfd = open(p, O_RDONLY);
    if (kfd < 0 || read_full(kfd, s->secret, 32, 0) != 0) {
      if (kfd >= 0) close(kfd);
      caps_unlock(s); r = CAPSULE_E_IO; goto fail;
    }
    close(kfd);
  }

  snprintf(p, sizeof p, "%s/epochs.tbl", s->root);
  if (stat(p, &st) != 0) {
    if (!(flags & CAPSULE_STORE_CREATE)) { caps_unlock(s); r = CAPSULE_E_NOENT; goto fail; }
    int efd = open(p, O_RDWR | O_CREAT | O_EXCL, 0644);
    uint64_t cap0 = 64;
    while (cap0 < s->next_capsule_id) cap0 *= 2;
    epoch_hdr_t hdr = { CAPS_EPOCH_MAGIC, cap0 };
    if (efd < 0 || ftruncate(efd, (off_t)(sizeof hdr + 4 * cap0)) != 0 ||
        write_full(efd, &hdr, sizeof hdr, 0) != 0) {
      if (efd >= 0) close(efd);
      caps_unlock(s); r = CAPSULE_E_IO; goto fail;
    }
    close(efd);
  }
  s->epoch_fd = open(p, (flags & CAPSULE_STORE_RDONLY) ? O_RDONLY : O_RDWR);
  if (s->epoch_fd < 0) { caps_unlock(s); r = CAPSULE_E_IO; goto fail; }
  if ((r = epoch_map(s)) != CAPSULE_OK) { caps_unlock(s); goto fail; }

  /* scan under the (still held) exclusive lock; downgrade is not needed at
   * prototype scale — open is rare and short. */
  if ((r = recs_ensure(s, s->next_capsule_id ? s->next_capsule_id : 1)) != CAPSULE_OK ||
      (r = exts_ensure(s, s->next_extent_id ? s->next_extent_id : 1)) != CAPSULE_OK) {
    caps_unlock(s); goto fail;
  }

  /* manifests */
  snprintf(p, sizeof p, "%s/manifests", s->root);
  DIR *d = opendir(p);
  if (!d) { caps_unlock(s); r = CAPSULE_E_IO; goto fail; }
  struct dirent *de;
  uint64_t total_chunks = 0;
  while ((de = readdir(d)) != NULL) {
    unsigned long long id;
    char suffix[8];
    if (sscanf(de->d_name, "%llu.%7s", &id, suffix) != 2) continue;
    if (strcmp(suffix, "mf") != 0) continue;   /* skip .tmp leftovers */
    if (id >= s->next_capsule_id) continue;    /* not committed in meta */
    if ((r = manifest_read(s, id, &s->recs[id])) != CAPSULE_OK) {
      closedir(d); caps_unlock(s); goto fail;
    }
    total_chunks += s->recs[id].n_chunks;
  }
  closedir(d);

  /* extents on disk */
  snprintf(p, sizeof p, "%s/extents", s->root);
  d = opendir(p);
  if (!d) { caps_unlock(s); r = CAPSULE_E_IO; goto fail; }
  while ((de = readdir(d)) != NULL) {
    unsigned long long eid;
    char suffix[8];
    if (sscanf(de->d_name, "%llu.%7s", &eid, suffix) != 2) continue;
    if (strcmp(suffix, "ext") != 0) continue;
    if (eid >= s->exts_cap && exts_ensure(s, eid + 1) != CAPSULE_OK) {
      closedir(d); caps_unlock(s); r = CAPSULE_E_IO; goto fail;
    }
    char ep[1200];
    caps_extent_path(s, eid, ep, sizeof ep);
    struct stat est;
    if (stat(ep, &est) == 0) {
      s->exts[eid].present = 1;
      s->exts[eid].file_bytes = (uint64_t)est.st_size;
      s->exts[eid].frozen = 1;   /* refined below */
    }
  }
  closedir(d);

  /* refcounts (distinct capsules per extent) + evicted detection */
  for (uint64_t id = 0; id < s->next_capsule_id; id++) {
    capsule_rec_t *rec = &s->recs[id];
    if (!rec->present) continue;
    for (uint32_t i = 0; i < rec->n_runs; i++) {
      uint64_t eid = rec->runs[i].extent_id;
      int counted = 0;
      for (uint32_t j = 0; j < i; j++)
        if (rec->runs[j].extent_id == eid) { counted = 1; break; }
      if (counted) continue;
      if (eid >= s->exts_cap && exts_ensure(s, eid + 1) != CAPSULE_OK) {
        caps_unlock(s); r = CAPSULE_E_IO; goto fail;
      }
      extent_rec_t *e = &s->exts[eid];
      if (!e->present) { e->present = 1; e->evicted = 1; e->frozen = 1; }
      e->refcount++;
      if (e->evicted) rec->flags |= CAPS_FLAG_EVICTED;
    }
  }

  /* garbage (paranoid scratch / orphan tails): payload with no referents */
  for (uint64_t eid = 0; eid < s->exts_cap; eid++) {
    extent_rec_t *e = &s->exts[eid];
    if (e->present && !e->evicted && e->refcount == 0) {
      e->garbage = 1;
      s->paranoid_garbage_bytes += e->file_bytes;
    }
  }

  /* open-tail reconstruction: appendable iff sole referent, sole run, run is
   * that capsule's last run, covers [0, file_bytes), capsule not sealed. */
  for (uint64_t id = 0; id < s->next_capsule_id; id++) {
    capsule_rec_t *rec = &s->recs[id];
    if (!rec->present || (rec->flags & CAPS_FLAG_SEALED) || rec->n_runs == 0) continue;
    mf_run_t *last = &rec->runs[rec->n_runs - 1];
    extent_rec_t *e = &s->exts[last->extent_id];
    if (e->present && !e->evicted && !e->garbage && e->refcount == 1 &&
        last->byte_off == 0 && last->byte_len == e->file_bytes) {
      /* ensure no other run of this rec references the same extent */
      int sole = 1;
      for (uint32_t i = 0; i + 1 < rec->n_runs; i++)
        if (rec->runs[i].extent_id == last->extent_id) { sole = 0; break; }
      if (sole) { e->frozen = 0; e->has_owner = 1; e->owner_id = id; }
    }
  }

  /* prefix index: replay chains from each manifest's chunk-hash list */
  if ((r = pi_grow(s, total_chunks * 2 + 16)) != CAPSULE_OK) { caps_unlock(s); goto fail; }
  for (uint64_t id = 0; id < s->next_capsule_id; id++) {
    capsule_rec_t *rec = &s->recs[id];
    if (!rec->present) continue;
    uint64_t cb = (uint64_t)CAPSULE_CHUNK_TOKENS * rec->bytes_per_token;
    uint8_t chain[32];
    memcpy(chain, rec->initial_chain_hash, 32);
    for (uint64_t c = 0; c < rec->n_chunks; c++) {
      uint8_t nx[32];
      caps_chain_step(chain, rec->chunk_hashes + c * 32, nx);
      memcpy(chain, nx, 32);
      uint64_t eid, off;
      if (rec_chunk_loc(rec, c, cb, &eid, &off) == 0 && !s->exts[eid].evicted)
        pi_insert(s, chain, eid, off, cb);
    }
  }

  caps_unlock(s);
  *out = s;
  return CAPSULE_OK;

fail:
  if (s->epoch_base) epoch_unmap(s);
  if (s->epoch_fd >= 0) close(s->epoch_fd);
  if (s->lock_fd >= 0) close(s->lock_fd);
  caps_br_destroy(&s->rw);
  for (uint64_t i = 0; i < s->recs_cap; i++) {
    if (s->recs && s->recs[i].present) { free(s->recs[i].runs); free(s->recs[i].chunk_hashes); }
  }
  free(s->recs); free(s->exts); free(s->pi); free(s);
  return r;
}

int capsule_store_close(capsule_store_t *s) {
  if (!s) return CAPSULE_E_INVAL;
  for (uint64_t i = 0; i < s->exts_cap; i++) caps_ext_close_fd(&s->exts[i]);
  epoch_unmap(s);
  if (s->epoch_fd >= 0) close(s->epoch_fd);
  if (s->lock_fd >= 0) close(s->lock_fd);
  caps_br_destroy(&s->rw);
  for (uint64_t i = 0; i < s->recs_cap; i++) {
    if (s->recs[i].present) { free(s->recs[i].runs); free(s->recs[i].chunk_hashes); }
  }
  free(s->recs); free(s->exts); free(s->pi);
  free(s);
  return CAPSULE_OK;
}

static int capsule_store_stats_inner(capsule_store_t *s, capsule_store_stats_t *out) {
  if (!s || !out) return CAPSULE_E_INVAL;
  memset(out, 0, sizeof *out);
  for (uint64_t id = 0; id < s->next_capsule_id && id < s->recs_cap; id++) {
    if (!s->recs[id].present) continue;
    out->n_capsules++;
    out->logical_bytes += s->recs[id].n_tokens * (uint64_t)s->recs[id].bytes_per_token;
  }
  for (uint64_t eid = 0; eid < s->exts_cap; eid++) {
    extent_rec_t *e = &s->exts[eid];
    if (e->present && !e->evicted && !e->garbage) {
      out->n_extents++;
      out->stored_bytes += e->file_bytes;
    }
  }
  out->dedup_saved_bytes = s->dedup_saved_bytes;
  out->evicted_bytes = s->evicted_bytes;
  out->dedup_hits = s->dedup_hits;
  out->denied_ops = s->denied_ops;
  out->revocations = s->revocations;
  return CAPSULE_OK;
}

uint64_t capsule_internal_paranoid_garbage_bytes(const capsule_store_t *s) {
  return s ? s->paranoid_garbage_bytes : 0;
}

/* ------------------------------------------------------------------ create */

static int create_common(capsule_store_t *s, const char *path,
                   const char *dedup_domain, uint32_t bytes_per_token,
                   const uint8_t *parent_chain /* NULL = plain create */,
                   const uint8_t *compatibility_hash /* NULL = legacy zero */,
                   capsule_cap_t *owner_cap_out, uint64_t *id_out) {
  if (!s || !owner_cap_out || !id_out) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  if (bytes_per_token == 0 || bytes_per_token % 1024 != 0) return CAPSULE_E_INVAL;
  const char *pp = path ? path : "";
  if (strlen(pp) >= CAPS_MAX_PATH) return CAPSULE_E_INVAL;

  int r = caps_lock_ex(s);
  if (r != CAPSULE_OK) return r;

  uint64_t id = s->next_capsule_id;
  if ((r = recs_ensure(s, id + 1)) != CAPSULE_OK) { caps_unlock(s); return r; }
  if ((r = epoch_ensure(s, id + 1)) != CAPSULE_OK) { caps_unlock(s); return r; }
  s->next_capsule_id = id + 1;
  if ((r = meta_write(s)) != CAPSULE_OK) { s->next_capsule_id = id; caps_unlock(s); return r; }

  capsule_rec_t *rec = &s->recs[id];
  memset(rec, 0, sizeof *rec);
  rec->present = 1;
  rec->id = id;
  rec->bytes_per_token = bytes_per_token;
  rec->chunk_tokens = CAPSULE_CHUNK_TOKENS;
  rec->parent_id = ~0ULL;
  rec->created_us = rec->last_access_us = caps_now_us();
  caps_domain_seed(dedup_domain, rec->domain_seed);
  if (compatibility_hash)
    memcpy(rec->compatibility_hash, compatibility_hash, 32);
  /* Every v3 name starts with H(domain_seed || compatibility_hash), so equal
   * bytes in incompatible runtime configurations cannot alias. */
  uint8_t compatible_seed[32];
  {
    caps_sha256_ctx_t hc;
    caps_sha256_init(&hc);
    caps_sha256_update(&hc, rec->domain_seed, 32);
    caps_sha256_update(&hc, rec->compatibility_hash, 32);
    caps_sha256_final(&hc, compatible_seed);
  }
  if (parent_chain) {
    /* Chained names additionally bind the complete parent prefix. */
    caps_sha256_ctx_t hc;
    caps_sha256_init(&hc);
    caps_sha256_update(&hc, compatible_seed, 32);
    caps_sha256_update(&hc, parent_chain, 32);
    caps_sha256_final(&hc, rec->chain_hash);
  } else {
    memcpy(rec->chain_hash, compatible_seed, 32);
  }
  memcpy(rec->initial_chain_hash, rec->chain_hash, 32);
  strncpy(rec->path, pp, CAPS_MAX_PATH - 1);

  if ((r = caps_manifest_write(s, rec)) != CAPSULE_OK) {
    rec->present = 0;
    caps_unlock(s);
    return r;
  }
  caps_unlock(s);

  *id_out = id;
  return caps_mint(s, id, CAP_R_ALL, 0, owner_cap_out);
}

static int capsule_create_inner(capsule_store_t *s, const char *path,
                   const char *dedup_domain, uint32_t bytes_per_token,
                   capsule_cap_t *owner_cap_out, uint64_t *id_out) {
  return create_common(s, path, dedup_domain, bytes_per_token, NULL, NULL,
                       owner_cap_out, id_out);
}

static int capsule_create_compatible_inner(capsule_store_t *s, const char *path,
                   const char *dedup_domain, uint32_t bytes_per_token,
                   const uint8_t compatibility_hash[32],
                   capsule_cap_t *owner_cap_out, uint64_t *id_out) {
  if (!compatibility_hash) return CAPSULE_E_INVAL;
  return create_common(s, path, dedup_domain, bytes_per_token, NULL,
                       compatibility_hash, owner_cap_out, id_out);
}

static int capsule_create_chained_inner(capsule_store_t *s, const char *path,
                   const char *dedup_domain, uint32_t bytes_per_token,
                   const capsule_cap_t *chain_parent,
                   capsule_cap_t *owner_cap_out, uint64_t *id_out) {
  if (!s || !chain_parent || !owner_cap_out || !id_out) return CAPSULE_E_INVAL;
  capsule_rec_t *prec = NULL;
  int r = caps_validate(s, chain_parent, CAP_R_READ, &prec);
  if (r != CAPSULE_OK) return r;
  uint8_t pchain[32];
  memcpy(pchain, prec->chain_hash, 32);
  return create_common(s, path, dedup_domain, bytes_per_token, pchain,
                       prec->compatibility_hash,
                       owner_cap_out, id_out);
}

/* ------------------------------------------------------------------ extend */

static int scratch_append(capsule_store_t *s, const void *chunk, uint64_t cb,
                          int *meta_dirty) {
  if (!s->has_scratch) {
    uint64_t eid = s->next_extent_id;
    int r = exts_ensure(s, eid + 1);
    if (r != CAPSULE_OK) return r;
    s->next_extent_id = eid + 1;
    *meta_dirty = 1;
    char p[1200];
    caps_extent_path(s, eid, p, sizeof p);
    int fd = open(p, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return CAPSULE_E_IO;
    extent_rec_t *e = &s->exts[eid];
    e->present = 1; e->garbage = 1; e->frozen = 1; e->fd_rw = fd;
    e->file_bytes = 0; e->refcount = 0;
    s->paranoid_scratch_eid = eid;
    s->has_scratch = 1;
  }
  extent_rec_t *e = &s->exts[s->paranoid_scratch_eid];
  int fd = ext_fd(s, s->paranoid_scratch_eid);
  if (fd < 0) return CAPSULE_E_IO;
  if (write_full(fd, chunk, cb, (off_t)e->file_bytes) != 0) return CAPSULE_E_IO;
  e->file_bytes += cb;
  s->paranoid_garbage_bytes += cb;
  return CAPSULE_OK;
}

static int capsule_extend_inner(capsule_store_t *s, const capsule_cap_t *cap,
                   const void *kv, uint32_t n_tokens, const uint8_t *hbuf) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_EXTEND, &rec);
  if (r != CAPSULE_OK) return r;
  if (!kv || n_tokens == 0 || n_tokens % CAPSULE_CHUNK_TOKENS != 0)
    return CAPSULE_E_INVAL;
  if (rec->flags & CAPS_FLAG_SEALED) return CAPSULE_E_SEALED;
  if (rec->flags & CAPS_FLAG_EVICTED) return CAPSULE_E_EVICTED;

  uint64_t cb = (uint64_t)CAPSULE_CHUNK_TOKENS * rec->bytes_per_token;
  uint32_t nch = n_tokens / CAPSULE_CHUNK_TOKENS;

  if ((r = caps_lock_ex(s)) != CAPSULE_OK) return r;

  int dedup_mode = 1, meta_dirty = 0;
  uint8_t chain[32];
  memcpy(chain, rec->chain_hash, 32);

  if ((r = rec_chunks_ensure(rec, rec->n_chunks + nch)) != CAPSULE_OK) goto out;

  for (uint32_t c = 0; c < nch; c++) {
    const uint8_t *chunk = (const uint8_t *)kv + (uint64_t)c * cb;
    uint8_t h[32], chain2[32];
    memcpy(h, hbuf + (size_t)c * 32, 32);
    caps_chain_step(chain, h, chain2);

    pi_entry_t *hit = dedup_mode ? pi_lookup(s, chain2) : NULL;
    if (hit) {
      /* adopt existing bytes */
      uint64_t heid = hit->eid, hoff = hit->off;
      extent_rec_t *e = &s->exts[heid];
      if (!e->has_owner || e->owner_id != rec->id) {
        e->frozen = 1;         /* now (potentially) shared: freeze appends */
        e->has_owner = 0;
      }
      if (!rec_references(rec, heid)) e->refcount++;
      if ((r = rec_add_loc(rec, heid, hoff, cb)) != CAPSULE_OK) goto out;
      s->dedup_hits++;
      s->dedup_saved_bytes += cb;
      if (s->flags & CAPSULE_STORE_PARANOID) {
        /* keep extend write-dominated on hits too (S1c paranoid mode) */
        if ((r = scratch_append(s, chunk, cb, &meta_dirty)) != CAPSULE_OK) goto out;
      }
    } else {
      dedup_mode = 0;  /* leave dedup mode for the rest of this call */
      /* find or create the open tail extent */
      uint64_t teid = ~0ULL;
      if (rec->n_runs > 0) {
        uint64_t last_eid = rec->runs[rec->n_runs - 1].extent_id;
        extent_rec_t *e = &s->exts[last_eid];
        if (!e->frozen && e->has_owner && e->owner_id == rec->id && !e->evicted)
          teid = last_eid;
      }
      if (teid == ~0ULL) {
        teid = s->next_extent_id;
        if ((r = exts_ensure(s, teid + 1)) != CAPSULE_OK) goto out;
        s->next_extent_id = teid + 1;
        meta_dirty = 1;
        char p[1200];
        caps_extent_path(s, teid, p, sizeof p);
        int fd = open(p, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { r = CAPSULE_E_IO; goto out; }
        extent_rec_t *e = &s->exts[teid];
        e->present = 1; e->evicted = 0; e->garbage = 0; e->frozen = 0;
        e->has_owner = 1; e->owner_id = rec->id;
        e->file_bytes = 0; e->refcount = 0; e->fd_rw = fd;
      }
      extent_rec_t *e = &s->exts[teid];
      int fd = ext_fd(s, teid);
      if (fd < 0) { r = CAPSULE_E_IO; goto out; }
      uint64_t off = e->file_bytes;
      if (write_full(fd, chunk, cb, (off_t)off) != 0) {
        e->frozen = 1;   /* orphan bytes possible; abandon this tail */
        r = CAPSULE_E_IO;
        goto out;
      }
      e->file_bytes += cb;
      if (!rec_references(rec, teid)) e->refcount++;
      if ((r = rec_add_loc(rec, teid, off, cb)) != CAPSULE_OK) goto out;
      if ((r = pi_insert(s, chain2, teid, off, cb)) != CAPSULE_OK) goto out;
    }

    memcpy(rec->chunk_hashes + rec->n_chunks * 32, h, 32);
    rec->n_chunks++;
    memcpy(chain, chain2, 32);
  }

  rec->n_tokens += n_tokens;
  memcpy(rec->chain_hash, chain, 32);
  rec->last_access_us = caps_now_us();
  r = caps_manifest_write(s, rec);
  if (r == CAPSULE_OK && meta_dirty) r = meta_write(s);
out:
  caps_unlock(s);
  return r;
}

/* -------------------------------------------------------------------- seal */

static void freeze_tail(capsule_store_t *s, capsule_rec_t *rec) {
  if (rec->n_runs == 0) return;
  uint64_t eid = rec->runs[rec->n_runs - 1].extent_id;
  extent_rec_t *e = &s->exts[eid];
  if (e->has_owner && e->owner_id == rec->id) { e->frozen = 1; e->has_owner = 0; }
}

static int capsule_seal_inner(capsule_store_t *s, const capsule_cap_t *cap) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_SEAL, &rec);
  if (r != CAPSULE_OK) return r;
  if (rec->flags & CAPS_FLAG_SEALED) return CAPSULE_OK;  /* idempotent */
  if ((r = caps_lock_ex(s)) != CAPSULE_OK) return r;
  rec->flags |= CAPS_FLAG_SEALED;
  freeze_tail(s, rec);
  rec->last_access_us = caps_now_us();
  r = caps_manifest_write(s, rec);
  caps_unlock(s);
  return r;
}

/* -------------------------------------------------------------------- fork */

static int capsule_fork_inner(capsule_store_t *s, const capsule_cap_t *cap, const char *child_path,
                 capsule_cap_t *child_cap_out, uint64_t *child_id_out) {
  if (!s || !cap || !child_cap_out || !child_id_out) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  capsule_rec_t *parent = NULL;
  int r = caps_validate(s, cap, CAP_R_READ, &parent);
  if (r != CAPSULE_OK) return r;
  if (parent->flags & CAPS_FLAG_EVICTED) return CAPSULE_E_EVICTED;
  const char *pp = child_path ? child_path : "";
  if (strlen(pp) >= CAPS_MAX_PATH) return CAPSULE_E_INVAL;

  if ((r = caps_lock_ex(s)) != CAPSULE_OK) return r;

  uint64_t parent_id = parent->id;
  uint64_t id = s->next_capsule_id;
  if ((r = recs_ensure(s, id + 1)) != CAPSULE_OK) { caps_unlock(s); return r; }
  parent = &s->recs[parent_id];  /* recs may have been reallocated */
  if ((r = epoch_ensure(s, id + 1)) != CAPSULE_OK) { caps_unlock(s); return r; }
  s->next_capsule_id = id + 1;
  if ((r = meta_write(s)) != CAPSULE_OK) { s->next_capsule_id = id; caps_unlock(s); return r; }

  freeze_tail(s, parent);

  capsule_rec_t *rec = &s->recs[id];
  memset(rec, 0, sizeof *rec);
  rec->present = 1;
  rec->id = id;
  rec->bytes_per_token = parent->bytes_per_token;
  rec->chunk_tokens = parent->chunk_tokens;
  rec->n_tokens = parent->n_tokens;
  rec->flags = 0;                       /* child starts unsealed, unevicted */
  rec->parent_id = parent_id;
  rec->created_us = rec->last_access_us = caps_now_us();
  memcpy(rec->chain_hash, parent->chain_hash, 32);
  memcpy(rec->initial_chain_hash, parent->initial_chain_hash, 32);
  memcpy(rec->domain_seed, parent->domain_seed, 32);
  memcpy(rec->compatibility_hash, parent->compatibility_hash, 32);
  strncpy(rec->path, pp, CAPS_MAX_PATH - 1);

  if (parent->n_runs) {
    if ((r = rec_runs_ensure(rec, parent->n_runs)) != CAPSULE_OK) goto undo;
    memcpy(rec->runs, parent->runs, (size_t)parent->n_runs * sizeof(mf_run_t));
    rec->n_runs = parent->n_runs;
  }
  if (parent->n_chunks) {
    if ((r = rec_chunks_ensure(rec, parent->n_chunks)) != CAPSULE_OK) goto undo;
    memcpy(rec->chunk_hashes, parent->chunk_hashes, (size_t)parent->n_chunks * 32);
    rec->n_chunks = parent->n_chunks;
  }

  /* refcount++ once per distinct extent (never copies payload bytes: O(metadata)) */
  for (uint32_t i = 0; i < rec->n_runs; i++) {
    uint64_t eid = rec->runs[i].extent_id;
    int counted = 0;
    for (uint32_t j = 0; j < i; j++)
      if (rec->runs[j].extent_id == eid) { counted = 1; break; }
    if (!counted) s->exts[eid].refcount++;
  }

  if ((r = caps_manifest_write(s, rec)) != CAPSULE_OK) {
    for (uint32_t i = 0; i < rec->n_runs; i++) {
      uint64_t eid = rec->runs[i].extent_id;
      int counted = 0;
      for (uint32_t j = 0; j < i; j++)
        if (rec->runs[j].extent_id == eid) { counted = 1; break; }
      if (!counted) s->exts[eid].refcount--;
    }
    goto undo;
  }
  parent->last_access_us = caps_now_us();
  caps_unlock(s);
  *child_id_out = id;
  return caps_mint(s, id, CAP_R_ALL, 0, child_cap_out);

undo:
  free(rec->runs); free(rec->chunk_hashes);
  memset(rec, 0, sizeof *rec);
  caps_unlock(s);
  return r;
}

/* ---------------------------------------------------------- attach/detach */

#define CAPS_MAP_MAGIC 0xA77AC4E5u

static int capsule_attach_inner(capsule_store_t *s, const capsule_cap_t *cap, capsule_map_t *out) {
  if (!s || !cap || !out) return CAPSULE_E_INVAL;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_READ, &rec);
  if (r != CAPSULE_OK) return r;
  if (rec->flags & CAPS_FLAG_EVICTED) return CAPSULE_E_EVICTED;
  if (rec->n_runs > CAPSULE_MAX_EXTENTS) return CAPSULE_E_INVAL;

  memset(out, 0, sizeof *out);
  uint64_t total = 0;
  for (uint32_t i = 0; i < rec->n_runs; i++) {
    mf_run_t *run = &rec->runs[i];
    if (s->exts[run->extent_id].evicted) { r = CAPSULE_E_EVICTED; goto fail; }
    int fd = ext_fd(s, run->extent_id);
    if (fd < 0) { r = CAPSULE_E_IO; goto fail; }
    /* byte_off is a multiple of chunk bytes which is a multiple of the 16 KiB
     * page (enforced by bytes_per_token % 1024 == 0 at create). */
    void *m = mmap(NULL, (size_t)run->byte_len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE, fd, (off_t)run->byte_off);
    if (m == MAP_FAILED) { r = CAPSULE_E_IO; goto fail; }
    out->seg_base[i] = m;
    out->seg_len[i] = run->byte_len;
    out->nsegs = i + 1;
    total += run->byte_len;
  }
  /* Validation linearizes attachment after all mappings are established.
   * A revoke/destroy that overlaps setup therefore makes this call fail and
   * tears the mappings down; revocation after this point governs future calls
   * and does not retroactively unmap an already-returned private view. */
  capsule_rec_t *validated = NULL;
  r = caps_validate(s, cap, CAP_R_READ, &validated);
  if (r != CAPSULE_OK || validated != rec) {
    if (r == CAPSULE_OK) r = CAPSULE_E_REVOKED;
    goto fail;
  }
  out->total_bytes = total;
  out->_internal[0] = rec->id;
  out->_internal[1] = CAPS_MAP_MAGIC;
  rec->live_attaches++;
  rec->last_access_us = caps_now_us();
  return CAPSULE_OK;

fail:
  for (uint32_t i = 0; i < out->nsegs; i++)
    munmap(out->seg_base[i], (size_t)out->seg_len[i]);
  memset(out, 0, sizeof *out);
  return r;
}

static int capsule_attach_compatible_inner(capsule_store_t *s,
        const capsule_cap_t *cap, const uint8_t expected_hash[32],
        capsule_map_t *out) {
  if (!s || !cap || !expected_hash || !out) return CAPSULE_E_INVAL;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_READ, &rec);
  if (r != CAPSULE_OK) return r;
  if (memcmp(rec->compatibility_hash, expected_hash, 32) != 0)
    return CAPSULE_E_INVAL;
  return capsule_attach_inner(s, cap, out);
}

static int capsule_detach_inner(capsule_store_t *s, capsule_map_t *m) {
  if (!m) return CAPSULE_E_INVAL;
  for (uint32_t i = 0; i < m->nsegs && i < CAPSULE_MAX_EXTENTS; i++)
    if (m->seg_base[i]) munmap(m->seg_base[i], (size_t)m->seg_len[i]);
  if (s && m->_internal[1] == CAPS_MAP_MAGIC) {
    uint64_t id = m->_internal[0];
    if (id < s->recs_cap && s->recs[id].present && s->recs[id].live_attaches > 0)
      s->recs[id].live_attaches--;
  }
  memset(m, 0, sizeof *m);
  return CAPSULE_OK;
}

/* -------------------------------------------------------------------- read */

static int read_impl(capsule_store_t *s, capsule_rec_t *rec,
                     uint64_t off, uint64_t len, void *buf) {
  uint64_t size = rec->n_tokens * (uint64_t)rec->bytes_per_token;
  if (off + len < off || off + len > size) return CAPSULE_E_INVAL;
  if (len == 0) return CAPSULE_OK;
  if (!buf) return CAPSULE_E_INVAL;
  if (rec->flags & CAPS_FLAG_EVICTED) return CAPSULE_E_EVICTED;

  char *dst = buf;
  uint64_t pos = 0, want = off, left = len;
  for (uint32_t i = 0; i < rec->n_runs && left > 0; i++) {
    mf_run_t *run = &rec->runs[i];
    uint64_t run_end = pos + run->byte_len;
    if (want < run_end) {
      uint64_t in_off = want - pos;
      uint64_t take = run->byte_len - in_off;
      if (take > left) take = left;
      if (s->exts[run->extent_id].evicted) return CAPSULE_E_EVICTED;
      int fd = ext_fd(s, run->extent_id);
      if (fd < 0) return CAPSULE_E_IO;
      if (read_full(fd, dst, (size_t)take, (off_t)(run->byte_off + in_off)) != 0)
        return CAPSULE_E_IO;
      dst += take;
      want += take;
      left -= take;
    }
    pos = run_end;
  }
  if (left != 0) return CAPSULE_E_IO;
  /* shared (read) lock is held here: use a relaxed atomic store */
  __atomic_store_n(&rec->last_access_us, caps_now_us(), __ATOMIC_RELAXED);
  return CAPSULE_OK;
}

static int capsule_read_inner(capsule_store_t *s, const capsule_cap_t *cap,
                 uint64_t off, uint64_t len, void *buf) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_READ, &rec);  /* full check EVERY call */
  if (r != CAPSULE_OK) return r;
  return read_impl(s, rec, off, len, buf);
}

/* Bench entry point (not in the frozen public ABI): read path without the
 * capability check, to isolate per-access protection cost (E1 / claim C3). */
static int capsule_read_unchecked_inner(capsule_store_t *s, uint64_t capsule_id,
                           uint64_t off, uint64_t len, void *buf) {
  if (!s) return CAPSULE_E_INVAL;
  if (capsule_id >= s->recs_cap || !s->recs[capsule_id].present)
    return CAPSULE_E_NOENT;
  return read_impl(s, &s->recs[capsule_id], off, len, buf);
}

/* ------------------------------------------------------------------ revoke */

static int capsule_revoke_inner(capsule_store_t *s, const capsule_cap_t *cap) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_REVOKE, &rec);
  if (r != CAPSULE_OK) return r;
  r = epoch_advance_durable(s, rec->id);
  if (r != CAPSULE_OK) return r;
  s->revocations++;
  return CAPSULE_OK;
}

/* --------------------------------------------------------------------- pin */

static int capsule_pin_inner(capsule_store_t *s, const capsule_cap_t *cap, uint64_t ttl_ms) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_READ, &rec);
  if (r != CAPSULE_OK) return r;
  if ((r = caps_lock_ex(s)) != CAPSULE_OK) return r;
  rec->pin_deadline_us = ttl_ms ? caps_now_us() + ttl_ms * 1000ULL : 0;
  r = caps_manifest_write(s, rec);
  caps_unlock(s);
  return r;
}

/* ----------------------------------------------------------------- destroy */

static void erase_extent(capsule_store_t *s, uint64_t eid) {
  extent_rec_t *e = &s->exts[eid];
  char p[1200];
  caps_extent_path(s, eid, p, sizeof p);
  if (!e->evicted && e->file_bytes > 0) {
    int fd = ext_fd(s, eid);
    if (fd >= 0) {
      void *m = mmap(NULL, (size_t)e->file_bytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
      if (m != MAP_FAILED) {
        /* volatile stores: cannot be dead-store-eliminated (guaranteed erase) */
        volatile uint64_t *q = (volatile uint64_t *)m;
        uint64_t nq = e->file_bytes / 8;
        for (uint64_t i = 0; i < nq; i++) q[i] = 0;
        volatile unsigned char *b = (volatile unsigned char *)m;
        for (uint64_t i = nq * 8; i < e->file_bytes; i++) b[i] = 0;
        msync(m, (size_t)e->file_bytes, MS_SYNC);
        munmap(m, (size_t)e->file_bytes);
      }
    }
  }
  caps_ext_close_fd(e);
  unlink(p);
  caps_pi_remove_extent(s, eid);
  memset(e, 0, sizeof *e);
  e->fd_rw = -1;
}

static int capsule_destroy_inner(capsule_store_t *s, const capsule_cap_t *cap) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  if (s->flags & CAPSULE_STORE_RDONLY) return CAPSULE_E_PERM;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_REVOKE, &rec);  /* destroy implies revoke */
  if (r != CAPSULE_OK) return r;

  if ((r = caps_lock_ex(s)) != CAPSULE_OK) return r;

  r = epoch_advance_durable(s, rec->id);
  if (r != CAPSULE_OK) { caps_unlock(s); return r; }
  s->revocations++;

  for (uint32_t i = 0; i < rec->n_runs; i++) {
    uint64_t eid = rec->runs[i].extent_id;
    int counted = 0;
    for (uint32_t j = 0; j < i; j++)
      if (rec->runs[j].extent_id == eid) { counted = 1; break; }
    if (counted) continue;
    extent_rec_t *e = &s->exts[eid];
    if (e->refcount <= 1) {
      erase_extent(s, eid);   /* erase-on-last-reference */
    } else {
      e->refcount--;
    }
  }

  char p[1200];
  manifest_path(s, rec->id, p, sizeof p);
  unlink(p);
  free(rec->runs);
  free(rec->chunk_hashes);
  memset(rec, 0, sizeof *rec);
  caps_unlock(s);
  return CAPSULE_OK;
}

/* -------------------------------------------------------------------- stat */

static int capsule_stat_inner(capsule_store_t *s, const capsule_cap_t *cap, capsule_stat_t *out) {
  if (!s || !cap || !out) return CAPSULE_E_INVAL;
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_READ, &rec);
  if (r != CAPSULE_OK) return r;
  memset(out, 0, sizeof *out);
  out->capsule_id = rec->id;
  out->bytes_per_token = rec->bytes_per_token;
  out->chunk_tokens = rec->chunk_tokens;
  out->n_tokens = rec->n_tokens;
  for (uint32_t i = 0; i < rec->n_runs; i++) {
    out->referenced_bytes += rec->runs[i].byte_len;
    if (s->exts[rec->runs[i].extent_id].refcount == 1)
      out->owned_bytes += rec->runs[i].byte_len;  /* runs in sole-reference extents */
  }
  out->n_extents = rec->n_runs;
  out->epoch = (rec->id < s->epoch_capacity)
                 ? __atomic_load_n(&s->epochs[rec->id], __ATOMIC_SEQ_CST) : 0;
  out->flags = rec->flags;
  if (rec->pin_deadline_us && rec->pin_deadline_us > caps_now_us())
    out->flags |= 0x4;  /* PINNED */
  out->live_attaches = rec->live_attaches;
  memcpy(out->chain_hash, rec->chain_hash, 32);
  memcpy(out->compatibility_hash, rec->compatibility_hash, 32);
  out->created_us = rec->created_us;
  out->last_access_us = rec->last_access_us;
  out->pin_deadline_us = rec->pin_deadline_us;
  return CAPSULE_OK;
}


/* ---- thread-safety wrappers ------------------------------------------
 * Every public entry point acquires the store rwlock: writers exclusive,
 * readers shared. The _inner functions are the original bodies, unchanged.
 */

int capsule_store_stats(capsule_store_t *s, capsule_store_stats_t *out) {
  if (!s) return CAPSULE_E_INVAL;
  int _sh = caps_br_rlock(&s->rw);
  int _r = capsule_store_stats_inner(s, out);
  caps_br_runlock(&s->rw, _sh);
  return _r;
}

int capsule_create(capsule_store_t *s, const char *path, const char *dedup_domain,
                   uint32_t bytes_per_token, capsule_cap_t *owner_cap_out,
                   uint64_t *id_out) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_create_inner(s, path, dedup_domain, bytes_per_token, owner_cap_out, id_out);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_create_compatible(capsule_store_t *s, const char *path,
                   const char *dedup_domain, uint32_t bytes_per_token,
                   const uint8_t compatibility_hash[32],
                   capsule_cap_t *owner_cap_out, uint64_t *id_out) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_create_compatible_inner(s, path, dedup_domain,
      bytes_per_token, compatibility_hash, owner_cap_out, id_out);
  caps_br_wunlock(&s->rw);
  return _r;
}

/* capsule_extend does its own locking rather than using the uniform wrapper.
 *
 * SHA-256 over the payload is roughly half the cost of an extend and touches no
 * shared state, so hashing under the store write lock serialized the single
 * most parallelizable step in the system (round-1 E7c: extend throughput flat
 * at ~2.5k/s from 1 to 32 threads). The call is therefore staged:
 *
 *   1. shared lock  -> validate, learn bytes_per_token, release
 *   2. NO lock      -> hash every chunk (the parallel part)
 *   3. exclusive    -> index lookups, extent allocation, payload write, manifest
 *
 * Step 3 re-validates, so revocation, expiry, sealing and eviction landing
 * between steps 1 and 3 are still honoured, and geometry drift is rejected. */
int capsule_extend(capsule_store_t *s, const capsule_cap_t *cap,
                   const void *kv, uint32_t n_tokens) {
  if (!s || !cap) return CAPSULE_E_INVAL;
  if (!kv || n_tokens == 0 || n_tokens % CAPSULE_CHUNK_TOKENS != 0)
    return CAPSULE_E_INVAL;

  int sh = caps_br_rlock(&s->rw);
  capsule_rec_t *rec = NULL;
  int r = caps_validate(s, cap, CAP_R_EXTEND, &rec);
  uint32_t bpt = (r == CAPSULE_OK) ? rec->bytes_per_token : 0;
  caps_br_runlock(&s->rw, sh);
  if (r != CAPSULE_OK) return r;

  uint64_t cb = (uint64_t)CAPSULE_CHUNK_TOKENS * bpt;
  uint32_t nch = n_tokens / CAPSULE_CHUNK_TOKENS;
  uint8_t *hbuf = (uint8_t *)malloc((size_t)nch * 32);
  if (!hbuf) return CAPSULE_E_IO;
  for (uint32_t c = 0; c < nch; c++)
    caps_sha256((const uint8_t *)kv + (uint64_t)c * cb, cb,
                hbuf + (size_t)c * 32);

  caps_br_wlock(&s->rw);
  capsule_rec_t *rec2 = NULL;
  r = caps_validate(s, cap, CAP_R_EXTEND, &rec2);
  if (r == CAPSULE_OK && rec2->bytes_per_token != bpt) r = CAPSULE_E_INVAL;
  if (r == CAPSULE_OK) r = capsule_extend_inner(s, cap, kv, n_tokens, hbuf);
  caps_br_wunlock(&s->rw);
  free(hbuf);
  return r;
}

int capsule_seal(capsule_store_t *s, const capsule_cap_t *cap) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_seal_inner(s, cap);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_fork(capsule_store_t *s, const capsule_cap_t *cap, const char *child_path,
                 capsule_cap_t *child_cap_out, uint64_t *child_id_out) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_fork_inner(s, cap, child_path, child_cap_out, child_id_out);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_attach(capsule_store_t *s, const capsule_cap_t *cap, capsule_map_t *out) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_attach_inner(s, cap, out);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_attach_compatible(capsule_store_t *s, const capsule_cap_t *cap,
                              const uint8_t expected_hash[32],
                              capsule_map_t *out) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_attach_compatible_inner(s, cap, expected_hash, out);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_detach(capsule_store_t *s, capsule_map_t *m) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_detach_inner(s, m);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_read(capsule_store_t *s, const capsule_cap_t *cap,
                 uint64_t off, uint64_t len, void *buf) {
  if (!s) return CAPSULE_E_INVAL;
  int _sh = caps_br_rlock(&s->rw);
  int _r = capsule_read_inner(s, cap, off, len, buf);
  caps_br_runlock(&s->rw, _sh);
  return _r;
}

int capsule_read_unchecked(capsule_store_t *s, uint64_t capsule_id,
                           uint64_t off, uint64_t len, void *buf) {
  if (!s) return CAPSULE_E_INVAL;
  int _sh = caps_br_rlock(&s->rw);
  int _r = capsule_read_unchecked_inner(s, capsule_id, off, len, buf);
  caps_br_runlock(&s->rw, _sh);
  return _r;
}

int capsule_revoke(capsule_store_t *s, const capsule_cap_t *cap) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_revoke_inner(s, cap);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_pin(capsule_store_t *s, const capsule_cap_t *cap, uint64_t ttl_ms) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_pin_inner(s, cap, ttl_ms);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_destroy(capsule_store_t *s, const capsule_cap_t *cap) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_destroy_inner(s, cap);
  caps_br_wunlock(&s->rw);
  return _r;
}

int capsule_stat(capsule_store_t *s, const capsule_cap_t *cap, capsule_stat_t *out) {
  if (!s) return CAPSULE_E_INVAL;
  int _sh = caps_br_rlock(&s->rw);
  int _r = capsule_stat_inner(s, cap, out);
  caps_br_runlock(&s->rw, _sh);
  return _r;
}

int capsule_create_chained(capsule_store_t *s, const char *path,
                           const char *dedup_domain, uint32_t bytes_per_token,
                           const capsule_cap_t *chain_parent,
                           capsule_cap_t *owner_cap_out, uint64_t *id_out) {
  if (!s) return CAPSULE_E_INVAL;
  caps_br_wlock(&s->rw);
  int _r = capsule_create_chained_inner(s, path, dedup_domain, bytes_per_token,
                                        chain_parent, owner_cap_out, id_out);
  caps_br_wunlock(&s->rw);
  return _r;
}
