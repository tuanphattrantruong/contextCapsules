/* bench_conc.c — E7: what protection costs under concurrency.
 *
 * The 292 ns capability check reported in E1 is a single-threaded number, and
 * the mount point the paper proposes (an engine's admission path) is anything
 * but single-threaded. This driver measures the three concurrent paths that
 * matter, on all core counts up to the machine's:
 *
 *   E7a  validate scaling : N threads doing capsule_stat (validate + manifest
 *                           lookup, no store mutation) on distinct capsules.
 *                           Read-mostly: should scale near-linearly if the
 *                           validation path is genuinely lock-free.
 *   E7b  read scaling     : N threads doing capsule_read (full mediation +
 *                           data copy) on distinct capsules.
 *   E7c  extend contention: N threads doing capsule_extend, which takes the
 *                           store lock. This is the serialization point and
 *                           the honest bad news; measuring it is the point.
 *   E7d  revoke under load: revocation latency observed by readers while N-1
 *                           threads hammer the store.
 *
 * Same methodology contract as bench_micro.c: warmup, pseudorandom buffers,
 * volatile consume sinks, mach_absolute_time, JSON out.
 */

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench_util.h"
#include "capsule.h"

#define MAX_THREADS 64
#define BPT 131072u                 /* Llama-3.1-8B geometry, measured in E5a */
#define SMALL_BPT 16384u            /* keeps the extend study's I/O sane      */

static volatile uint64_t g_sink = 0;

typedef struct {
  int tid;
  int nthreads;
  capsule_store_t *store;
  capsule_cap_t cap;
  uint64_t iters;
  double *samples;                  /* per-op ns, owned by the thread */
  uint64_t ops_done;
  uint64_t errors;
  void *buf;
  size_t buflen;
  atomic_int *go;
  atomic_int *stop;
  uint64_t local_sink;
  /* revoke study */
  atomic_ullong *revoke_tick;
  uint64_t observed_tick;
  int observed;
} targ_t;

static void wait_go(atomic_int *go) {
  while (atomic_load_explicit(go, memory_order_acquire) == 0) {
    /* spin: we are measuring steady-state contention, not wakeup latency */
  }
}

/* ---------------------------------------------------------------- E7a/E7b */

static void *th_stat(void *p) {
  targ_t *a = (targ_t *)p;
  capsule_stat_t st;
  for (uint64_t i = 0; i < 10; i++) capsule_stat(a->store, &a->cap, &st);
  wait_go(a->go);
  for (uint64_t i = 0; i < a->iters; i++) {
    uint64_t t0 = bu_now_ticks();
    int rc = capsule_stat(a->store, &a->cap, &st);
    uint64_t t1 = bu_now_ticks();
    a->samples[i] = bu_ticks_to_ns(t1 - t0);
    if (rc != CAPSULE_OK) a->errors++;
    a->local_sink += st.n_tokens ^ (uint64_t)st.epoch;
    a->ops_done++;
  }
  g_sink += a->local_sink;
  return NULL;
}

static void *th_read(void *p) {
  targ_t *a = (targ_t *)p;
  for (uint64_t i = 0; i < 10; i++)
    capsule_read(a->store, &a->cap, 0, a->buflen, a->buf);
  wait_go(a->go);
  for (uint64_t i = 0; i < a->iters; i++) {
    uint64_t t0 = bu_now_ticks();
    int rc = capsule_read(a->store, &a->cap, 0, a->buflen, a->buf);
    uint64_t t1 = bu_now_ticks();
    a->samples[i] = bu_ticks_to_ns(t1 - t0);
    if (rc != CAPSULE_OK) a->errors++;
    a->local_sink += bu_checksum(a->buf, 64);
    a->ops_done++;
  }
  g_sink += a->local_sink;
  return NULL;
}

static void *th_extend(void *p) {
  targ_t *a = (targ_t *)p;
  wait_go(a->go);
  for (uint64_t i = 0; i < a->iters; i++) {
    /* unique content per (thread, iter) so nothing dedups away */
    bu_fill_random(a->buf, a->buflen,
                   0x1234567ULL + a->tid * 1000003ULL + i);
    uint64_t t0 = bu_now_ticks();
    int rc = capsule_extend(a->store, &a->cap, a->buf,
                            CAPSULE_CHUNK_TOKENS);
    uint64_t t1 = bu_now_ticks();
    a->samples[i] = bu_ticks_to_ns(t1 - t0);
    if (rc != CAPSULE_OK) a->errors++;
    a->ops_done++;
  }
  return NULL;
}

/* ---------------------------------------------------------------- E7d */

static void *th_revoke_reader(void *p) {
  targ_t *a = (targ_t *)p;
  capsule_stat_t st;
  wait_go(a->go);
  for (;;) {
    int rc = capsule_stat(a->store, &a->cap, &st);
    if (rc == CAPSULE_E_REVOKED) {
      a->observed_tick = bu_now_ticks();
      a->observed = 1;
      return NULL;
    }
    if (atomic_load_explicit(a->stop, memory_order_relaxed)) return NULL;
    a->local_sink += (uint64_t)st.epoch;
  }
}

static void *th_noise(void *p) {
  targ_t *a = (targ_t *)p;
  capsule_stat_t st;
  wait_go(a->go);
  while (!atomic_load_explicit(a->stop, memory_order_relaxed)) {
    capsule_stat(a->store, &a->cap, &st);
    a->local_sink += (uint64_t)st.epoch;
    a->ops_done++;
  }
  g_sink += a->local_sink;
  return NULL;
}

/* ---------------------------------------------------------------- driver */

typedef struct {
  int nthreads;
  double agg_p50, agg_mean, agg_p95;
  double throughput_kops;
  uint64_t total_ops;
  uint64_t errors;
  double wall_s;
} scale_row_t;

static scale_row_t run_scale(const char *root, const char *mode, int nthreads,
                             uint64_t iters_per_thread) {
  scale_row_t row;
  memset(&row, 0, sizeof row);
  row.nthreads = nthreads;

  char sroot[1200];
  snprintf(sroot, sizeof sroot, "%s/%s_t%d", root, mode, nthreads);
  char rmcmd[1400];
  snprintf(rmcmd, sizeof rmcmd, "rm -rf '%s'", sroot);
  if (system(rmcmd) != 0) { /* fresh store; ignore rc */ }

  capsule_store_t *s = NULL;
  if (capsule_store_open(sroot, CAPSULE_STORE_CREATE, &s) != CAPSULE_OK) {
    fprintf(stderr, "store_open failed for %s\n", sroot);
    return row;
  }

  int is_extend = (strcmp(mode, "extend") == 0);
  uint32_t bpt = is_extend ? SMALL_BPT : BPT;
  size_t chunk = (size_t)CAPSULE_CHUNK_TOKENS * bpt;

  static targ_t args[MAX_THREADS];
  static pthread_t ths[MAX_THREADS];
  memset(args, 0, sizeof args);

  atomic_int go = 0;
  void *seed_buf = malloc(chunk);
  bu_fill_random(seed_buf, chunk, 0xABCDEF);

  for (int i = 0; i < nthreads; i++) {
    char path[256];
    snprintf(path, sizeof path, "/bench/t%d", i);
    uint64_t id;
    if (capsule_create(s, path, "bench", bpt, &args[i].cap, &id) != CAPSULE_OK) {
      fprintf(stderr, "create failed\n");
      capsule_store_close(s);
      free(seed_buf);
      return row;
    }
    /* give every capsule one chunk so stat/read have something to touch */
    if (!is_extend) {
      bu_fill_random(seed_buf, chunk, 0x100 + i);
      capsule_extend(s, &args[i].cap, seed_buf, CAPSULE_CHUNK_TOKENS);
    }
    args[i].tid = i;
    args[i].nthreads = nthreads;
    args[i].store = s;
    args[i].iters = iters_per_thread;
    args[i].samples = (double *)malloc(sizeof(double) * iters_per_thread);
    args[i].go = &go;
    args[i].buflen = (strcmp(mode, "read") == 0) ? 4096 : chunk;
    args[i].buf = malloc(args[i].buflen);
    bu_fill_random(args[i].buf, args[i].buflen, 0x777 + i);
  }
  free(seed_buf);

  void *(*fn)(void *) = th_stat;
  if (strcmp(mode, "read") == 0) fn = th_read;
  else if (is_extend) fn = th_extend;

  for (int i = 0; i < nthreads; i++)
    pthread_create(&ths[i], NULL, fn, &args[i]);

  uint64_t w0 = bu_now_ticks();
  atomic_store_explicit(&go, 1, memory_order_release);
  for (int i = 0; i < nthreads; i++) pthread_join(ths[i], NULL);
  uint64_t w1 = bu_now_ticks();
  row.wall_s = bu_ticks_to_ns(w1 - w0) / 1e9;

  size_t total = 0;
  for (int i = 0; i < nthreads; i++) total += args[i].ops_done;
  double *all = (double *)malloc(sizeof(double) * (total ? total : 1));
  size_t k = 0;
  for (int i = 0; i < nthreads; i++) {
    for (uint64_t j = 0; j < args[i].ops_done; j++) all[k++] = args[i].samples[j];
    row.errors += args[i].errors;
    free(args[i].samples);
    free(args[i].buf);
  }
  bu_dist_t d = bu_dist(all, k);
  row.agg_p50 = d.p50;
  row.agg_mean = d.mean;
  row.agg_p95 = d.p95;
  row.total_ops = k;
  row.throughput_kops = (row.wall_s > 0) ? (double)k / row.wall_s / 1000.0 : 0.0;
  free(all);
  capsule_store_close(s);
  return row;
}

static void emit_scale(FILE *f, const char *name, const char *root,
                       const char *mode, const int *counts, int ncounts,
                       uint64_t iters) {
  fprintf(f, "\"%s\": [", name);
  for (int i = 0; i < ncounts; i++) {
    scale_row_t r = run_scale(root, mode, counts[i], iters);
    fprintf(f,
            "%s{\"threads\": %d, \"p50_ns\": %.2f, \"mean_ns\": %.2f, "
            "\"p95_ns\": %.2f, \"throughput_kops_s\": %.2f, "
            "\"total_ops\": %llu, \"errors\": %llu, \"wall_s\": %.4f}",
            i ? ", " : "", r.nthreads, r.agg_p50, r.agg_mean, r.agg_p95,
            r.throughput_kops, (unsigned long long)r.total_ops,
            (unsigned long long)r.errors, r.wall_s);
    fflush(f);
    fprintf(stderr, "  %-7s t=%-3d p50=%8.1f ns  thr=%9.1f kops/s\n",
            mode, r.nthreads, r.agg_p50, r.throughput_kops);
  }
  fprintf(f, "]");
}

/* revocation propagation with N-1 noise threads hammering the store */
static void emit_revoke(FILE *f, const char *root, const int *counts,
                        int ncounts, int trials) {
  fprintf(f, "\"e7d_revoke_under_load\": [");
  for (int c = 0; c < ncounts; c++) {
    int n = counts[c];
    double *lat = (double *)malloc(sizeof(double) * trials);
    int got = 0;
    for (int t = 0; t < trials; t++) {
      char sroot[1200];
      snprintf(sroot, sizeof sroot, "%s/revoke_t%d_%d", root, n, t);
      char rmcmd[1400];
      snprintf(rmcmd, sizeof rmcmd, "rm -rf '%s'", sroot);
      if (system(rmcmd) != 0) { }
      capsule_store_t *s = NULL;
      if (capsule_store_open(sroot, CAPSULE_STORE_CREATE, &s) != CAPSULE_OK)
        continue;
      capsule_cap_t cap;
      uint64_t id;
      capsule_create(s, "/rev/target", "bench", SMALL_BPT, &cap, &id);
      void *b = malloc((size_t)CAPSULE_CHUNK_TOKENS * SMALL_BPT);
      bu_fill_random(b, (size_t)CAPSULE_CHUNK_TOKENS * SMALL_BPT, 42);
      capsule_extend(s, &cap, b, CAPSULE_CHUNK_TOKENS);
      free(b);

      atomic_int go = 0, stop = 0;
      static targ_t a[MAX_THREADS];
      static pthread_t th[MAX_THREADS];
      memset(a, 0, sizeof a);
      int nread = 1, nnoise = n - 1;
      for (int i = 0; i < n; i++) {
        a[i].tid = i; a[i].store = s; a[i].cap = cap;
        a[i].go = &go; a[i].stop = &stop;
      }
      pthread_create(&th[0], NULL, th_revoke_reader, &a[0]);
      for (int i = 1; i <= nnoise; i++)
        pthread_create(&th[i], NULL, th_noise, &a[i]);
      atomic_store_explicit(&go, 1, memory_order_release);
      usleep(2000);                      /* let everyone reach steady state */
      uint64_t t0 = bu_now_ticks();
      capsule_revoke(s, &cap);
      pthread_join(th[0], NULL);
      atomic_store_explicit(&stop, 1, memory_order_relaxed);
      for (int i = 1; i <= nnoise; i++) pthread_join(th[i], NULL);
      if (a[0].observed) {
        lat[got++] = bu_ticks_to_ns(a[0].observed_tick - t0);
      }
      (void)nread;
      capsule_store_close(s);
    }
    bu_dist_t d = bu_dist(lat, got);
    fprintf(f,
            "%s{\"threads\": %d, \"trials\": %d, \"p50_ns\": %.1f, "
            "\"p95_ns\": %.1f, \"max_ns\": %.1f, \"mean_ns\": %.1f}",
            c ? ", " : "", n, got, d.p50, d.p95, d.max, d.mean);
    fprintf(stderr, "  revoke t=%-3d p50=%8.1f ns  p95=%9.1f ns\n",
            n, d.p50, d.p95);
    free(lat);
  }
  fprintf(f, "]");
}

int main(int argc, char **argv) {
  bu_time_init();
  const char *root = (argc > 1) ? argv[1] : "/tmp/capsule_conc";
  const char *out = (argc > 2) ? argv[2] : "e7_concurrency.json";
  uint64_t iters = (argc > 3) ? strtoull(argv[3], NULL, 10) : 20000;

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  int counts[16];
  int nc = 0;
  for (int t = 1; t <= ncpu && t <= MAX_THREADS && nc < 16; t *= 2)
    counts[nc++] = t;
  if (nc < 16 && counts[nc - 1] != (int)ncpu && ncpu <= MAX_THREADS)
    counts[nc++] = (int)ncpu;

  char mkcmd[1300];
  snprintf(mkcmd, sizeof mkcmd, "mkdir -p '%s'", root);
  if (system(mkcmd) != 0) { }

  FILE *f = fopen(out, "w");
  if (!f) { perror("fopen"); return 1; }
  fprintf(f, "{\n\"meta\": {\"experiment\": \"E7_concurrency\", ");
  bu_emit_meta_fields(f);
  fprintf(f, ", \"ncpu_online\": %ld, \"iters_per_thread\": %llu, "
             "\"bytes_per_token\": %u, \"extend_bytes_per_token\": %u, "
             "\"note\": \"aggregate percentiles pool every thread's samples\"},\n",
          ncpu, (unsigned long long)iters, BPT, SMALL_BPT);

  fprintf(stderr, "E7 concurrency study, %ld cores\n", ncpu);
  emit_scale(f, "e7a_validate_stat", root, "stat", counts, nc, iters);
  fprintf(f, ",\n");
  emit_scale(f, "e7b_checked_read4k", root, "read", counts, nc, iters / 4);
  fprintf(f, ",\n");
  emit_scale(f, "e7c_extend_contended", root, "extend", counts, nc, 200);
  fprintf(f, ",\n");
  emit_revoke(f, root, counts, nc, 25);
  fprintf(f, ",\n\"consume_sink\": %llu\n}\n", (unsigned long long)g_sink);
  fclose(f);
  fprintf(stderr, "wrote %s\n", out);
  return 0;
}
