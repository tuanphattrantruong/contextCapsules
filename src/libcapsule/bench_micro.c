/* bench_micro.c — E1 microbenchmarks (DESIGN.md 6 / claims C1,C2,C3).
 *
 * Usage: bench_micro <store_root> <out.json> [quick|full]
 *
 *   C1  fork latency is flat in capsule size; memcpy copy grows linearly.
 *   C2  CoW attach makes N-way sharing ~1x physical: N children attaching a
 *       sealed capsule add ~0 dirty phys_footprint each; N copiers add full size.
 *   C3  per-op capability validation is sub-us; capsule_read with the check vs the
 *       internal no-check build path isolates the protection cost.
 *
 * Real Llama-3-8B geometry (REAL_BPT = 131072 B/token, 2 MiB chunk). All timings
 * measured with mach_absolute_time (timebase queried at runtime). Buffers are
 * pseudorandom and consumed via a volatile checksum so -O2 cannot elide them.
 * Physical footprint via proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint —
 * counts dirty/anonymous pages only (the CoW-sharing metric, per DESIGN 0).
 */
#include "store_internal.h"
#include "bench_util.h"

#include <fcntl.h>
#include <libproc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define REAL_BPT 131072u
#define CHUNK_BYTES (16ull * REAL_BPT)     /* 2 MiB */

static volatile uint64_t g_sink = 0;       /* global consume sink */

static uint64_t phys_footprint(void) {
  struct rusage_info_v4 ri;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri) == 0)
    return ri.ri_phys_footprint;
  return 0;
}

static void rm_rf(const char *p) {
  char cmd[2048];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", p);
  int rc = system(cmd); (void)rc;
}

/* fresh store subdir under base root */
static capsule_store_t *fresh(const char *base, const char *sub, uint32_t flags) {
  char r[1200];
  snprintf(r, sizeof r, "%s/%s", base, sub);
  rm_rf(r);
  capsule_store_t *s = NULL;
  int rc = capsule_store_open(r, CAPSULE_STORE_CREATE | flags, &s);
  if (rc != CAPSULE_OK) { fprintf(stderr, "open %s = %d\n", r, rc); exit(2); }
  return s;
}

/* -------------------------- size-independent ops -------------------------- */

typedef struct {
  bu_dist_t create_destroy, extend_chunk, seal, derive, revoke, stat, validate_only;
  bu_dist_t read4k_checked, read4k_unchecked;
} scalar_bench_t;

static scalar_bench_t bench_scalars(const char *base, int quick) {
  scalar_bench_t out;
  memset(&out, 0, sizeof out);
  int N_cd     = quick ? 50    : 1000;
  int N_ext    = quick ? 20    : 200;
  int N_seal   = quick ? 50    : 1000;
  int N_derive = quick ? 1000  : 10000;
  int N_revoke = quick ? 100   : 1000;
  int N_stat   = quick ? 1000  : 10000;
  int N_valid  = quick ? 5000  : 100000;
  int N_read   = quick ? 1000  : 10000;
  int warm = 10;

  /* create+destroy (empty capsule; size-independent) */
  {
    capsule_store_t *s = fresh(base, "cd", 0);
    double *v = malloc(sizeof(double) * N_cd);
    for (int i = 0; i < N_cd + warm; i++) {
      capsule_cap_t c; uint64_t id;
      uint64_t t0 = bu_now_ticks();
      capsule_create(s, "/cd", "", REAL_BPT, &c, &id);
      capsule_destroy(s, &c);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
    }
    out.create_destroy = bu_dist(v, N_cd);
    free(v);
    capsule_store_close(s);
  }

  /* extend one 2 MiB chunk (write path: unique content each time) */
  {
    capsule_store_t *s = fresh(base, "ext", 0);
    capsule_cap_t c; uint64_t id;
    capsule_create(s, "/ext", "", REAL_BPT, &c, &id);
    uint8_t *chunk = malloc(CHUNK_BYTES);
    double *v = malloc(sizeof(double) * N_ext);
    for (int i = 0; i < N_ext + warm; i++) {
      bu_fill_random(chunk, CHUNK_BYTES, 0xE00 + (uint64_t)i);
      uint64_t t0 = bu_now_ticks();
      capsule_extend(s, &c, chunk, 16);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
    }
    out.extend_chunk = bu_dist(v, N_ext);
    free(v); free(chunk);
    capsule_store_close(s);
  }

  /* seal: each on its own 1-chunk capsule (unique content so no dedup) */
  {
    capsule_store_t *s = fresh(base, "sl", 0);
    uint8_t *chunk = malloc(CHUNK_BYTES);
    double *v = malloc(sizeof(double) * N_seal);
    for (int i = 0; i < N_seal + warm; i++) {
      capsule_cap_t c; uint64_t id;
      capsule_create(s, "/sl", "", REAL_BPT, &c, &id);
      bu_fill_random(chunk, CHUNK_BYTES, 0x5EA1000 + (uint64_t)i);
      capsule_extend(s, &c, chunk, 16);
      uint64_t t0 = bu_now_ticks();
      capsule_seal(s, &c);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
      capsule_destroy(s, &c);
    }
    out.seal = bu_dist(v, N_seal);
    free(v); free(chunk);
    capsule_store_close(s);
  }

  /* a base capsule for derive/stat/validate/read (single 2 MiB chunk) */
  capsule_store_t *bs = fresh(base, "scal", 0);
  capsule_cap_t base_cap; uint64_t base_id;
  capsule_create(bs, "/base", "", REAL_BPT, &base_cap, &base_id);
  uint8_t *content = malloc(CHUNK_BYTES);
  bu_fill_random(content, CHUNK_BYTES, 0xBA5E);
  capsule_extend(bs, &base_cap, content, 16);
  capsule_seal(bs, &base_cap);

  /* derive */
  {
    double *v = malloc(sizeof(double) * N_derive);
    for (int i = 0; i < N_derive + warm; i++) {
      capsule_cap_t d;
      uint64_t t0 = bu_now_ticks();
      capsule_derive(bs, &base_cap, CAP_R_READ, 0, &d);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
      g_sink ^= d.bytes[0];
    }
    out.derive = bu_dist(v, N_derive);
    free(v);
  }
  /* stat */
  {
    double *v = malloc(sizeof(double) * N_stat);
    capsule_stat_t stt;
    for (int i = 0; i < N_stat + warm; i++) {
      uint64_t t0 = bu_now_ticks();
      capsule_stat(bs, &base_cap, &stt);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
      g_sink ^= stt.n_tokens;
    }
    out.stat = bu_dist(v, N_stat);
    free(v);
  }
  /* validate-only (internal entry point) */
  {
    double *v = malloc(sizeof(double) * N_valid);
    for (int i = 0; i < N_valid + warm; i++) {
      uint64_t t0 = bu_now_ticks();
      int rc = capsule_internal_validate(bs, &base_cap, CAP_R_READ);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
      g_sink ^= (uint64_t)rc;
    }
    out.validate_only = bu_dist(v, N_valid);
    free(v);
  }
  /* read 4KiB with cap check vs internal no-check */
  {
    uint8_t rbuf[4096];
    double *vc = malloc(sizeof(double) * N_read);
    double *vu = malloc(sizeof(double) * N_read);
    for (int i = 0; i < N_read + warm; i++) {
      uint64_t t0 = bu_now_ticks();
      capsule_read(bs, &base_cap, 0, 4096, rbuf);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) vc[i - warm] = bu_ticks_to_ns(t1 - t0);
      g_sink ^= rbuf[0];
      uint64_t t2 = bu_now_ticks();
      capsule_read_unchecked(bs, base_id, 0, 4096, rbuf);
      uint64_t t3 = bu_now_ticks();
      if (i >= warm) vu[i - warm] = bu_ticks_to_ns(t3 - t2);
      g_sink ^= rbuf[1];
    }
    out.read4k_checked = bu_dist(vc, N_read);
    out.read4k_unchecked = bu_dist(vu, N_read);
    free(vc); free(vu);
  }
  free(content);
  capsule_store_close(bs);

  /* revoke: each on its own tiny capsule (size-independent atomic bump) */
  {
    capsule_store_t *s = fresh(base, "rev", 0);
    double *v = malloc(sizeof(double) * N_revoke);
    for (int i = 0; i < N_revoke + warm; i++) {
      capsule_cap_t c; uint64_t id;
      capsule_create(s, "/r", "", REAL_BPT, &c, &id);
      uint64_t t0 = bu_now_ticks();
      capsule_revoke(s, &c);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
    }
    out.revoke = bu_dist(v, N_revoke);
    free(v);
    capsule_store_close(s);
  }
  return out;
}

/* ---------------------------- per-size ops -------------------------------- */

typedef struct {
  int tokens;
  uint64_t modeled_bytes;
  bu_dist_t fork_ns, memcpy_ns, attach_lazy_ns, attach_touch_ns, detach_ns;
} size_bench_t;

static size_bench_t bench_size(const char *base, int tokens, int quick) {
  size_bench_t sb;
  memset(&sb, 0, sizeof sb);
  sb.tokens = tokens;
  sb.modeled_bytes = (uint64_t)tokens * REAL_BPT;
  int nch = tokens / 16;
  int N_fork   = quick ? 50  : 1000;
  int N_copy   = quick ? 5   : (tokens >= 4096 ? 20 : 50);
  int N_attach = quick ? 20  : (tokens >= 4096 ? 50 : 200);
  int N_touch  = quick ? 5   : 20;
  int warm = 10;

  char sub[64];
  snprintf(sub, sizeof sub, "sz%d", tokens);
  capsule_store_t *s = fresh(base, sub, 0);

  /* seed capsule: unique content of `tokens` tokens, stored once, then sealed */
  capsule_cap_t seed; uint64_t sid;
  capsule_create(s, "/seed", "", REAL_BPT, &seed, &sid);
  uint8_t *content = malloc((size_t)nch * CHUNK_BYTES);
  bu_fill_random(content, (size_t)nch * CHUNK_BYTES, 0x51ED + (uint64_t)tokens);
  capsule_extend(s, &seed, content, tokens);
  capsule_seal(s, &seed);

  /* fork latency (C1): O(metadata), expected flat across sizes */
  {
    double *v = malloc(sizeof(double) * N_fork);
    for (int i = 0; i < N_fork + warm; i++) {
      capsule_cap_t ch; uint64_t cid;
      uint64_t t0 = bu_now_ticks();
      capsule_fork(s, &seed, "/seed/c", &ch, &cid);
      uint64_t t1 = bu_now_ticks();
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
      g_sink ^= cid;
    }
    sb.fork_ns = bu_dist(v, N_fork);
    free(v);
  }

  /* memcpy copy baseline (C1): grows linearly with size */
  {
    size_t nbytes = (size_t)nch * CHUNK_BYTES;
    uint8_t *src = malloc(nbytes), *dst = malloc(nbytes);
    bu_fill_random(src, nbytes, 0xC0FFEE + (uint64_t)tokens);
    double *v = malloc(sizeof(double) * N_copy);
    for (int i = 0; i < N_copy + warm; i++) {
      uint64_t t0 = bu_now_ticks();
      memcpy(dst, src, nbytes);
      uint64_t t1 = bu_now_ticks();
      /* consume so the copy is not dead-store-eliminated */
      g_sink ^= bu_checksum(dst, nbytes < 65536 ? nbytes : 65536);
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
    }
    sb.memcpy_ns = bu_dist(v, N_copy);
    free(src); free(dst); free(v);
  }

  /* attach lazy (map only) + detach (unmap) timed separately */
  {
    double *v = malloc(sizeof(double) * N_attach);
    double *vd = malloc(sizeof(double) * N_attach);
    for (int i = 0; i < N_attach + warm; i++) {
      capsule_map_t m;
      uint64_t t0 = bu_now_ticks();
      capsule_attach(s, &seed, &m);
      uint64_t t1 = bu_now_ticks();
      uint64_t t2 = bu_now_ticks();
      capsule_detach(s, &m);
      uint64_t t3 = bu_now_ticks();
      if (i >= warm) {
        v[i - warm] = bu_ticks_to_ns(t1 - t0);
        vd[i - warm] = bu_ticks_to_ns(t3 - t2);
      }
    }
    sb.attach_lazy_ns = bu_dist(v, N_attach);
    sb.detach_ns = bu_dist(vd, N_attach);
    free(v); free(vd);
  }

  /* attach + touch all pages (fault them in; volatile sum consumes) */
  {
    double *v = malloc(sizeof(double) * N_touch);
    for (int i = 0; i < N_touch + warm; i++) {
      capsule_map_t m;
      uint64_t t0 = bu_now_ticks();
      capsule_attach(s, &seed, &m);
      uint64_t acc = 0;
      for (uint32_t sg = 0; sg < m.nsegs; sg++) {
        const volatile uint8_t *p = (const volatile uint8_t *)m.seg_base[sg];
        for (uint64_t off = 0; off < m.seg_len[sg]; off += 16384) acc += p[off];
      }
      uint64_t t1 = bu_now_ticks();
      g_sink ^= acc;
      if (i >= warm) v[i - warm] = bu_ticks_to_ns(t1 - t0);
      capsule_detach(s, &m);
    }
    sb.attach_touch_ns = bu_dist(v, N_touch);
    free(v);
  }

  free(content);
  capsule_store_close(s);
  return sb;
}

/* ------------------------ 8-process CoW sharing (C2) ---------------------- */

#define SHARE_REPS 3   /* DESIGN.md 6: "3 repetitions" of the 8-process study */

typedef struct {
  int n_children;
  int n_reps;
  int tokens;
  uint64_t modeled_bytes_each;
  uint64_t attach_footprint_deltas[SHARE_REPS][16];
  uint64_t copy_footprint_deltas[SHARE_REPS][16];
  uint64_t attach_sum[SHARE_REPS], copy_sum[SHARE_REPS];
} share_bench_t;

/* child: attach + touch all pages, report phys_footprint delta over a pipe */
static void child_attach(capsule_store_t *s, capsule_cap_t *cap, int wfd) {
  uint64_t before = phys_footprint();
  capsule_map_t m;
  if (capsule_attach(s, cap, &m) != CAPSULE_OK) { uint64_t z = 0; ssize_t w = write(wfd, &z, 8); (void)w; _exit(1); }
  volatile uint64_t acc = 0;
  for (uint32_t sg = 0; sg < m.nsegs; sg++) {
    const volatile uint8_t *p = (const volatile uint8_t *)m.seg_base[sg];
    for (uint64_t off = 0; off < m.seg_len[sg]; off += 16384) acc += p[off];
  }
  uint64_t after = phys_footprint();
  uint64_t delta = after > before ? after - before : 0;
  ssize_t w = write(wfd, &delta, 8); (void)w;
  g_sink ^= acc;
  _exit(0);
}

/* child: private malloc + pseudorandom fill (dirty) of the same size */
static void child_copy(size_t nbytes, uint64_t seed, int wfd) {
  uint64_t before = phys_footprint();
  uint8_t *buf = malloc(nbytes);
  bu_fill_random(buf, nbytes, seed);
  volatile uint64_t acc = bu_checksum(buf, nbytes);
  uint64_t after = phys_footprint();
  uint64_t delta = after > before ? after - before : 0;
  ssize_t w = write(wfd, &delta, 8); (void)w;
  g_sink ^= acc;
  free(buf);
  _exit(0);
}

static share_bench_t bench_share(const char *base, int quick) {
  share_bench_t sh;
  memset(&sh, 0, sizeof sh);
  int n = quick ? 3 : 8;
  int tokens = quick ? 256 : 4096;   /* full: 512 MiB */
  size_t nch = tokens / 16;
  size_t nbytes = nch * CHUNK_BYTES;
  sh.n_children = n;
  sh.n_reps = SHARE_REPS;
  sh.tokens = tokens;
  sh.modeled_bytes_each = (uint64_t)tokens * REAL_BPT;

  capsule_store_t *s = fresh(base, "share", 0);
  capsule_cap_t seed; uint64_t sid;
  capsule_create(s, "/share", "", REAL_BPT, &seed, &sid);
  uint8_t *content = malloc(nbytes);
  bu_fill_random(content, nbytes, 0x5417ED);
  capsule_extend(s, &seed, content, tokens);
  capsule_seal(s, &seed);
  free(content);

  for (int rep = 0; rep < SHARE_REPS; rep++) {
    /* attach group */
    for (int i = 0; i < n; i++) {
      int fds[2];
      if (pipe(fds) != 0) { sh.attach_footprint_deltas[rep][i] = 0; continue; }
      pid_t pid = fork();
      if (pid == 0) { close(fds[0]); child_attach(s, &seed, fds[1]); }
      close(fds[1]);
      uint64_t d = 0; ssize_t r = read(fds[0], &d, 8); (void)r;
      close(fds[0]);
      int st; waitpid(pid, &st, 0);
      sh.attach_footprint_deltas[rep][i] = d;
      sh.attach_sum[rep] += d;
    }
    /* copy group */
    for (int i = 0; i < n; i++) {
      int fds[2];
      if (pipe(fds) != 0) { sh.copy_footprint_deltas[rep][i] = 0; continue; }
      pid_t pid = fork();
      if (pid == 0) { close(fds[0]); child_copy(nbytes, 0xC0B1E5 + (uint64_t)(rep * 16 + i), fds[1]); }
      close(fds[1]);
      uint64_t d = 0; ssize_t r = read(fds[0], &d, 8); (void)r;
      close(fds[0]);
      int st; waitpid(pid, &st, 0);
      sh.copy_footprint_deltas[rep][i] = d;
      sh.copy_sum[rep] += d;
    }
  }
  capsule_store_close(s);
  return sh;
}

/* ------------------------------- emit JSON -------------------------------- */

static void emit_size(FILE *f, size_bench_t *sb) {
  fprintf(f, "{\"tokens\": %d, \"modeled_bytes\": %llu, ",
          sb->tokens, (unsigned long long)sb->modeled_bytes);
  bu_json_dist(f, "fork_ns", sb->fork_ns);           fprintf(f, ", ");
  bu_json_dist(f, "memcpy_ns", sb->memcpy_ns);       fprintf(f, ", ");
  bu_json_dist(f, "attach_lazy_ns", sb->attach_lazy_ns); fprintf(f, ", ");
  bu_json_dist(f, "attach_touch_ns", sb->attach_touch_ns); fprintf(f, ", ");
  bu_json_dist(f, "detach_ns", sb->detach_ns);
  fprintf(f, "}");
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <store_root> <out.json> [quick|full]\n", argv[0]);
    return 2;
  }
  const char *base = argv[1];
  const char *out = argv[2];
  int quick = (argc > 3 && strcmp(argv[3], "quick") == 0);
  bu_time_init();

  int sizes_full[] = {16, 64, 256, 1024, 4096};
  int sizes_quick[] = {16, 64, 256};
  int *sizes = quick ? sizes_quick : sizes_full;
  int nsizes = quick ? 3 : 5;

  scalar_bench_t scal = bench_scalars(base, quick);
  size_bench_t *szb = malloc(sizeof(size_bench_t) * nsizes);
  for (int i = 0; i < nsizes; i++) szb[i] = bench_size(base, sizes[i], quick);
  share_bench_t sh = bench_share(base, quick);

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "cannot open %s\n", out); return 2; }
  fprintf(f, "{\n\"meta\": {");
  fprintf(f, "\"experiment\": \"E1_microbench\", ");
  fprintf(f, "\"cmd\": \"bench_micro %s %s %s\", ", base, out, quick ? "quick" : "full");
  fprintf(f, "\"mode\": \"%s\", ", quick ? "quick" : "full");
  bu_emit_meta_fields(f);
  fprintf(f, ", \"kv_geometry\": {\"REAL_BPT\": %u, \"chunk_bytes\": %llu, "
             "\"label\": \"modeled bytes use REAL_BPT=131072 (Llama-3-8B GQA)\"}",
          REAL_BPT, (unsigned long long)CHUNK_BYTES);
  fprintf(f, "},\n");

  fprintf(f, "\"data\": {\n \"measured\": {\n");
  /* scalars */
  fprintf(f, "  \"scalar_ns\": {");
  bu_json_dist(f, "create_destroy", scal.create_destroy); fprintf(f, ", ");
  bu_json_dist(f, "extend_2mib_chunk", scal.extend_chunk); fprintf(f, ", ");
  bu_json_dist(f, "seal", scal.seal); fprintf(f, ", ");
  bu_json_dist(f, "derive", scal.derive); fprintf(f, ", ");
  bu_json_dist(f, "revoke", scal.revoke); fprintf(f, ", ");
  bu_json_dist(f, "stat", scal.stat); fprintf(f, ", ");
  bu_json_dist(f, "validate_only", scal.validate_only); fprintf(f, ", ");
  bu_json_dist(f, "read4k_checked", scal.read4k_checked); fprintf(f, ", ");
  bu_json_dist(f, "read4k_unchecked", scal.read4k_unchecked);
  fprintf(f, "},\n");
  /* per-size */
  fprintf(f, "  \"by_size\": [");
  for (int i = 0; i < nsizes; i++) {
    if (i) fprintf(f, ", ");
    emit_size(f, &szb[i]);
  }
  fprintf(f, "],\n");
  /* sharing (per-rep arrays: SHARE_REPS repetitions x n_children) */
  fprintf(f, "  \"cow_sharing\": {\"n_children\": %d, \"n_reps\": %d, \"tokens\": %d, "
             "\"modeled_bytes_each\": %llu, \"attach_footprint_delta_bytes\": [",
          sh.n_children, sh.n_reps, sh.tokens, (unsigned long long)sh.modeled_bytes_each);
  for (int r = 0; r < sh.n_reps; r++) {
    fprintf(f, "%s[", r ? ", " : "");
    for (int i = 0; i < sh.n_children; i++)
      fprintf(f, "%s%llu", i ? ", " : "", (unsigned long long)sh.attach_footprint_deltas[r][i]);
    fprintf(f, "]");
  }
  fprintf(f, "], \"copy_footprint_delta_bytes\": [");
  for (int r = 0; r < sh.n_reps; r++) {
    fprintf(f, "%s[", r ? ", " : "");
    for (int i = 0; i < sh.n_children; i++)
      fprintf(f, "%s%llu", i ? ", " : "", (unsigned long long)sh.copy_footprint_deltas[r][i]);
    fprintf(f, "]");
  }
  fprintf(f, "], \"attach_sum_bytes\": [");
  for (int r = 0; r < sh.n_reps; r++)
    fprintf(f, "%s%llu", r ? ", " : "", (unsigned long long)sh.attach_sum[r]);
  fprintf(f, "], \"copy_sum_bytes\": [");
  for (int r = 0; r < sh.n_reps; r++)
    fprintf(f, "%s%llu", r ? ", " : "", (unsigned long long)sh.copy_sum[r]);
  fprintf(f, "]}\n");
  fprintf(f, " },\n");
  fprintf(f, " \"modeled\": {\"note\": \"*_ns are measured; modeled_bytes are token*REAL_BPT\"},\n");
  fprintf(f, " \"consume_sink\": %llu\n", (unsigned long long)g_sink);
  fprintf(f, "}\n}\n");
  fclose(f);

  /* clean scratch subdirs */
  const char *subs[] = {"cd","ext","scal","rev","share"};
  for (size_t i = 0; i < sizeof subs/sizeof subs[0]; i++) {
    char r[1200]; snprintf(r, sizeof r, "%s/%s", base, subs[i]); rm_rf(r);
  }
  for (int i = 0; i < nsizes; i++) {
    char r[1200]; snprintf(r, sizeof r, "%s/sz%d", base, sizes[i]); rm_rf(r);
  }
  free(szb);

  fprintf(stderr, "E1 %s: wrote %s\n", quick ? "quick" : "full", out);
  return 0;
}
