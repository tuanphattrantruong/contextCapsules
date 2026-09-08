/* sec_c.c — E3 security case studies in the real library (DESIGN.md 4).
 *
 * Usage: sec_c <s1b|s1c|s2|s3> <store_root> <out.json> [quick]
 *
 *   S1b  constant-time denial: forged caps for an EXISTING id and a NONEXISTENT
 *        id take the same HMAC-first path; their capsule_attach latency
 *        distributions must overlap (else it is a bug, not a result).
 *   S1c  residual dedup-timing channel: within one domain, capsule_extend is
 *        faster on a dedup hit (skips the write) than on a miss — quantified —
 *        and CAPSULE_STORE_PARANOID collapses the two distributions (scratch
 *        write on every hit), at the cost of extra write bandwidth.
 *   S2   revocation propagation: 1 revoker + 4 reader processes sharing the
 *        mmap'd epoch table; latency = reader's first E_REVOKED tick minus the
 *        revoker's t_revoke tick (mach_absolute_time, global monotonic).
 *   S3   seal-then-share integrity: read-only sharee cannot extend (E_PERM),
 *        owner cannot extend a sealed capsule (E_SEALED), and a CoW scribble on
 *        the sharee's private mapping never changes the store's chain hash.
 *
 * All timings measured with mach_absolute_time. JSON written to <out.json>.
 */
#include "store_internal.h"
#include "bench_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define REAL_BPT 131072u
#define CHUNK_BYTES (16ull * REAL_BPT)     /* 2 MiB */
#define SBPT 1024u                          /* scaled for lighter studies */
#define SCB (16ull * SBPT)                  /* 16 KiB chunk */

static volatile uint64_t g_sink = 0;

static void rm_rf(const char *p) {
  char cmd[2048];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", p);
  int rc = system(cmd); (void)rc;
}

static capsule_store_t *fresh(const char *root, uint32_t flags) {
  rm_rf(root);
  capsule_store_t *s = NULL;
  int rc = capsule_store_open(root, CAPSULE_STORE_CREATE | flags, &s);
  if (rc != CAPSULE_OK) { fprintf(stderr, "open %s = %d\n", root, rc); exit(2); }
  return s;
}

static void meta_open(FILE *f, const char *exp, const char *root,
                      const char *out, int quick) {
  fprintf(f, "{\n\"meta\": {\"experiment\": \"%s\", ", exp);
  fprintf(f, "\"cmd\": \"sec_c %s %s %s %s\", ", exp, root, out, quick ? "quick" : "full");
  fprintf(f, "\"mode\": \"%s\", ", quick ? "quick" : "full");
  bu_emit_meta_fields(f);
  fprintf(f, "},\n\"data\": {\n");
}

/* ------------------------------- S1b -------------------------------------- */

static int study_s1b(const char *root, const char *out, int quick) {
  int N = quick ? 1000 : 10000;
  int warm = 10;
  capsule_store_t *s = fresh(root, 0);
  capsule_cap_t owner; uint64_t id;
  capsule_create(s, "/target", "", REAL_BPT, &owner, &id);
  uint8_t *buf = malloc(CHUNK_BYTES);
  bu_fill_random(buf, CHUNK_BYTES, 1);
  capsule_extend(s, &owner, buf, 16);

  /* forged cap for the EXISTING id: correct id field, garbage HMAC */
  capsule_cap_t forg_exist;
  bu_fill_random(forg_exist.bytes, 72, 0xF0F0);
  memcpy(forg_exist.bytes + 8, &id, 8);          /* capsule_id field */
  /* forged cap for a NONEXISTENT id */
  capsule_cap_t forg_none;
  bu_fill_random(forg_none.bytes, 72, 0x0F0F);
  uint64_t huge = 0x7FFFFFFFULL;
  memcpy(forg_none.bytes + 8, &huge, 8);

  double *ve = malloc(sizeof(double) * N);
  double *vn = malloc(sizeof(double) * N);
  capsule_map_t m;
  for (int i = 0; i < N + warm; i++) {
    uint64_t t0 = bu_now_ticks();
    int r1 = capsule_attach(s, &forg_exist, &m);
    uint64_t t1 = bu_now_ticks();
    int r2 = capsule_attach(s, &forg_none, &m);
    uint64_t t2 = bu_now_ticks();
    g_sink ^= (uint64_t)(r1 + r2);
    if (i >= warm) { ve[i - warm] = bu_ticks_to_ns(t1 - t0); vn[i - warm] = bu_ticks_to_ns(t2 - t1); }
  }
  bu_dist_t de = bu_dist(ve, N);
  bu_dist_t dn = bu_dist(vn, N);
  /* overlap heuristic: |mean diff| < one combined std (no per-existence signal) */
  double mdiff = de.mean > dn.mean ? de.mean - dn.mean : dn.mean - de.mean;
  int overlap = mdiff < (de.std + dn.std);

  /* both denials return E_PERM (HMAC-first) — verify functionally too */
  int both_perm = (capsule_attach(s, &forg_exist, &m) == CAPSULE_E_PERM &&
                   capsule_attach(s, &forg_none, &m) == CAPSULE_E_PERM);

  FILE *f = fopen(out, "w");
  meta_open(f, "E3_S1b_constant_time_denial", root, out, quick);
  fprintf(f, " \"measured\": {\"trials\": %d, ", N);
  bu_json_dist(f, "forged_existing_id_ns", de); fprintf(f, ", ");
  bu_json_dist(f, "forged_nonexistent_id_ns", dn);
  fprintf(f, ", \"mean_diff_ns\": %.3f, \"distributions_overlap\": %s, "
             "\"both_return_E_PERM\": %s},\n",
          mdiff, overlap ? "true" : "false", both_perm ? "true" : "false");
  fprintf(f, " \"modeled\": {\"note\": \"latencies measured; no modeled quantities\"},\n");
  fprintf(f, " \"consume_sink\": %llu\n}\n}\n", (unsigned long long)g_sink);
  fclose(f);
  free(ve); free(vn); free(buf);
  capsule_store_close(s);
  rm_rf(root);
  fprintf(stderr, "S1b: overlap=%d both_perm=%d -> %s\n", overlap, both_perm, out);
  return 0;
}

/* ------------------------------- S1c -------------------------------------- */

static void s1c_measure(capsule_store_t *s, int N, int warm,
                        double *miss_out, double *hit_out) {
  /* Real 2 MiB chunks (DESIGN S1c): the extend write dominates, so the dedup
   * hit (which skips the write) is measurably faster — that is the channel. */
  capsule_cap_t a; uint64_t ia;
  capsule_create(s, "/a", "dom", REAL_BPT, &a, &ia);
  uint8_t *chunk = malloc(CHUNK_BYTES);
  /* capsule A stores N unique chunks (miss/write path, timed) */
  for (int i = 0; i < N + warm; i++) {
    bu_fill_random(chunk, CHUNK_BYTES, 0xA5A5 + (uint64_t)i);
    uint64_t t0 = bu_now_ticks();
    capsule_extend(s, &a, chunk, 16);
    uint64_t t1 = bu_now_ticks();
    if (i >= warm) miss_out[i - warm] = bu_ticks_to_ns(t1 - t0);
  }
  /* capsule B extends the SAME content, same domain -> every chunk is a hit */
  capsule_cap_t b; uint64_t ib;
  capsule_create(s, "/b", "dom", REAL_BPT, &b, &ib);
  for (int i = 0; i < N + warm; i++) {
    bu_fill_random(chunk, CHUNK_BYTES, 0xA5A5 + (uint64_t)i);   /* identical bytes */
    uint64_t t0 = bu_now_ticks();
    capsule_extend(s, &b, chunk, 16);
    uint64_t t1 = bu_now_ticks();
    if (i >= warm) hit_out[i - warm] = bu_ticks_to_ns(t1 - t0);
  }
  free(chunk);
}

typedef struct {
  double threshold, accuracy, auc;
  int samples_for_95pct;
} s1c_classifier_t;

/* Train a one-threshold attacker on even samples and test on odd samples.
 * AUC is the probability that a randomly selected miss is slower than a
 * randomly selected hit. samples_for_95pct repeats the same decision on the
 * mean of k observations; -1 means 95% was not reached by k=32. */
static s1c_classifier_t s1c_classify(const double *miss, const double *hit, int n) {
  s1c_classifier_t out = {0};
  out.samples_for_95pct = -1;
  double best = -1.0, best_t = 0.0;
  for (int c = 0; c < n; c++) {
    double t = 0.5 * (miss[c] + hit[c]);
    int ok = 0, total = 0;
    for (int i = 0; i < n; i += 2) {
      ok += miss[i] > t; ok += hit[i] <= t; total += 2;
    }
    double a = total ? (double)ok / total : 0.0;
    if (a > best) { best = a; best_t = t; }
  }
  int ok = 0, total = 0, pairs = 0;
  double ordered = 0.0;
  for (int i = 1; i < n; i += 2) {
    ok += miss[i] > best_t; ok += hit[i] <= best_t; total += 2;
    for (int j = 1; j < n; j += 2) {
      ordered += miss[i] > hit[j];
      ordered += (miss[i] == hit[j]) ? 0.5 : 0.0;
      pairs++;
    }
  }
  out.threshold = best_t;
  out.accuracy = total ? (double)ok / total : 0.0;
  out.auc = pairs ? (double)ordered / pairs : 0.0;
  const int ks[] = {1, 2, 4, 8, 16, 32};
  int nt = n / 2;
  for (size_t q = 0; q < sizeof ks / sizeof ks[0]; q++) {
    int k = ks[q], groups = nt / k, good = 0;
    if (groups < 2) continue;
    for (int g = 0; g < groups; g++) {
      double sm = 0.0, sh = 0.0;
      for (int z = 0; z < k; z++) {
        int idx = 1 + 2 * (g * k + z);
        sm += miss[idx]; sh += hit[idx];
      }
      good += (sm / k) > best_t;
      good += (sh / k) <= best_t;
    }
    if ((double)good / (2 * groups) >= 0.95) {
      out.samples_for_95pct = k;
      break;
    }
  }
  return out;
}

static int study_s1c(const char *root, const char *out, int quick) {
  int N = quick ? 100 : 500;
  int warm = 10;
  double *nm = malloc(sizeof(double) * N), *nh = malloc(sizeof(double) * N);
  double *pm = malloc(sizeof(double) * N), *ph = malloc(sizeof(double) * N);

  char rnorm[1200], rpar[1200];
  snprintf(rnorm, sizeof rnorm, "%s_norm", root);
  snprintf(rpar, sizeof rpar, "%s_par", root);

  capsule_store_t *sn = fresh(rnorm, 0);
  s1c_measure(sn, N, warm, nm, nh);
  uint64_t norm_garbage = capsule_internal_paranoid_garbage_bytes(sn);
  capsule_store_close(sn);

  capsule_store_t *sp = fresh(rpar, CAPSULE_STORE_PARANOID);
  s1c_measure(sp, N, warm, pm, ph);
  uint64_t par_garbage = capsule_internal_paranoid_garbage_bytes(sp);
  capsule_store_close(sp);

  s1c_classifier_t cn = s1c_classify(nm, nh, N);
  s1c_classifier_t cp = s1c_classify(pm, ph, N);
  bu_dist_t dnm = bu_dist(nm, N), dnh = bu_dist(nh, N);
  bu_dist_t dpm = bu_dist(pm, N), dph = bu_dist(ph, N);
  /* normal: hit clearly faster than miss (channel exists) */
  int channel_exists = dnh.p50 < dnm.p50;
  /* paranoid: hit/miss much closer (channel collapsed). Use a ratio test. */
  double norm_gap = dnm.p50 - dnh.p50;
  double par_gap  = dpm.p50 > dph.p50 ? dpm.p50 - dph.p50 : dph.p50 - dpm.p50;
  int channel_collapsed = par_gap < norm_gap * 0.5;

  FILE *f = fopen(out, "w");
  meta_open(f, "E3_S1c_dedup_timing_channel", root, out, quick);
  fprintf(f, " \"measured\": {\"trials_each\": %d, \"chunk_bytes\": %llu,\n", N,
          (unsigned long long)CHUNK_BYTES);
  fprintf(f, "  \"normal\": {");
  bu_json_dist(f, "miss_ns", dnm); fprintf(f, ", ");
  bu_json_dist(f, "hit_ns", dnh);
  fprintf(f, ", \"garbage_bytes\": %llu, \"classifier_accuracy\": %.6f, "
             "\"classifier_auc\": %.6f, \"samples_for_95pct\": ",
          (unsigned long long)norm_garbage, cn.accuracy, cn.auc);
  if (cn.samples_for_95pct < 0) fprintf(f, "null");
  else fprintf(f, "%d", cn.samples_for_95pct);
  fprintf(f, ", \"max_samples_tested\": 32}, \n");
  fprintf(f, "  \"paranoid\": {");
  bu_json_dist(f, "miss_ns", dpm); fprintf(f, ", ");
  bu_json_dist(f, "hit_ns", dph);
  fprintf(f, ", \"garbage_bytes\": %llu, \"classifier_accuracy\": %.6f, "
             "\"classifier_auc\": %.6f, \"samples_for_95pct\": ",
          (unsigned long long)par_garbage, cp.accuracy, cp.auc);
  if (cp.samples_for_95pct < 0) fprintf(f, "null");
  else fprintf(f, "%d", cp.samples_for_95pct);
  fprintf(f, ", \"max_samples_tested\": 32},\n");
  fprintf(f, "  \"normal_hit_miss_gap_ns\": %.3f, \"paranoid_hit_miss_gap_ns\": %.3f, "
             "\"channel_exists_normal\": %s, \"channel_reduced_paranoid\": %s, "
             "\"channel_collapsed_paranoid\": %s},\n",
          norm_gap, par_gap, channel_exists ? "true" : "false",
          channel_collapsed ? "true" : "false",
          channel_collapsed ? "true" : "false");
  fprintf(f, " \"modeled\": {\"note\": \"paranoid cost = extra write bandwidth "
             "(garbage_bytes); latencies measured\"},\n");
  fprintf(f, " \"consume_sink\": %llu\n}\n}\n", (unsigned long long)g_sink);
  fclose(f);
  free(nm); free(nh); free(pm); free(ph);
  rm_rf(rnorm); rm_rf(rpar);
  fprintf(stderr, "S1c: channel_exists=%d collapsed=%d -> %s\n",
          channel_exists, channel_collapsed, out);
  return 0;
}

/* -------------------------------- S2 -------------------------------------- */

#define S2_READERS 4

/* reader child: spin capsule_read until E_REVOKED; report detection tick */
static void s2_reader(capsule_store_t *s, capsule_cap_t *cap, int wfd) {
  uint8_t buf[16];
  uint64_t detect = 0;
  long cap_iters = 200000000L;
  for (long k = 0; k < cap_iters; k++) {
    int rc = capsule_read(s, cap, 0, 16, buf);
    if (rc == CAPSULE_E_REVOKED) { detect = bu_now_ticks(); break; }
    g_sink ^= (uint64_t)buf[0];
  }
  ssize_t w = write(wfd, &detect, 8); (void)w;
  _exit(0);
}

static int study_s2(const char *root, const char *out, int quick) {
  int trials = quick ? 5 : 50;
  int warm = 0;
  (void)warm;
  double *lat[S2_READERS];
  for (int r = 0; r < S2_READERS; r++) lat[r] = malloc(sizeof(double) * trials);
  int valid[S2_READERS];
  for (int r = 0; r < S2_READERS; r++) valid[r] = 0;

  capsule_store_t *s = fresh(root, 0);
  uint8_t *content = malloc(SCB);
  bu_fill_random(content, SCB, 0x5222);

  for (int t = 0; t < trials; t++) {
    capsule_cap_t owner; uint64_t id;
    char path[64]; snprintf(path, sizeof path, "/rev%d", t);
    capsule_create(s, path, "", SBPT, &owner, &id);
    capsule_extend(s, &owner, content, 16);
    capsule_cap_t rcap;
    capsule_derive(s, &owner, CAP_R_READ, 0, &rcap);

    int pfd[S2_READERS][2];
    pid_t pids[S2_READERS];
    for (int r = 0; r < S2_READERS; r++) {
      if (pipe(pfd[r]) != 0) { pids[r] = -1; continue; }
      pid_t pid = fork();
      if (pid == 0) {
        /* close only pipes that already exist (q < r) plus our own read end;
         * pipes for q > r are not created yet — do NOT touch those slots. */
        for (int q = 0; q < r; q++) { close(pfd[q][0]); close(pfd[q][1]); }
        close(pfd[r][0]);
        s2_reader(s, &rcap, pfd[r][1]);
      }
      pids[r] = pid;
      close(pfd[r][1]);
    }
    /* give readers a moment to start spinning, then revoke */
    usleep(1000);
    uint64_t t_revoke = bu_now_ticks();
    capsule_revoke(s, &owner);

    for (int r = 0; r < S2_READERS; r++) {
      if (pids[r] < 0) continue;
      uint64_t detect = 0;
      ssize_t rr = read(pfd[r][0], &detect, 8); (void)rr;
      close(pfd[r][0]);
      int st; waitpid(pids[r], &st, 0);
      if (detect > t_revoke) {
        lat[r][valid[r]++] = bu_ticks_to_ns(detect - t_revoke);
      } else if (detect != 0) {
        lat[r][valid[r]++] = 0.0;   /* detected at/just before revoke tick */
      }
    }
  }
  free(content);
  capsule_store_close(s);
  rm_rf(root);

  FILE *f = fopen(out, "w");
  meta_open(f, "E3_S2_revocation_propagation", root, out, quick);
  fprintf(f, " \"measured\": {\"trials\": %d, \"readers\": %d, \"per_reader\": [",
          trials, S2_READERS);
  for (int r = 0; r < S2_READERS; r++) {
    bu_dist_t d = bu_dist(lat[r], valid[r]);
    if (r) fprintf(f, ", ");
    fprintf(f, "{\"reader\": %d, ", r);
    bu_json_dist(f, "latency_ns", d);
    fprintf(f, "}");
  }
  /* pooled */
  int total = 0;
  for (int r = 0; r < S2_READERS; r++) total += valid[r];
  double *pool = malloc(sizeof(double) * (total ? total : 1));
  int pi = 0;
  for (int r = 0; r < S2_READERS; r++)
    for (int k = 0; k < valid[r]; k++) pool[pi++] = lat[r][k];
  bu_dist_t dp = bu_dist(pool, total);
  fprintf(f, "], ");
  bu_json_dist(f, "pooled_latency_ns", dp);
  fprintf(f, "},\n \"modeled\": {\"note\": \"cross-process latencies measured via "
             "mach_absolute_time global clock\"},\n");
  fprintf(f, " \"consume_sink\": %llu\n}\n}\n", (unsigned long long)g_sink);
  fclose(f);
  for (int r = 0; r < S2_READERS; r++) free(lat[r]);
  free(pool);
  fprintf(stderr, "S2: %d trials x %d readers -> %s\n", trials, S2_READERS, out);
  return 0;
}

/* -------------------------------- S3 -------------------------------------- */

static int study_s3(const char *root, const char *out, int quick) {
  (void)quick;
  capsule_store_t *s = fresh(root, 0);
  const char *domain = "sealshare";
  capsule_cap_t owner; uint64_t id;
  capsule_create(s, "/seal", domain, SBPT, &owner, &id);
  int nch = 4;
  uint8_t *content = malloc((size_t)nch * SCB);
  bu_fill_random(content, (size_t)nch * SCB, 0x533A1);
  capsule_extend(s, &owner, content, nch * 16);
  capsule_seal(s, &owner);

  capsule_stat_t before; capsule_stat(s, &owner, &before);

  /* read-only sharee */
  capsule_cap_t sharee;
  int derive_rc = capsule_derive(s, &owner, CAP_R_READ, 0, &sharee);
  int sharee_extend = capsule_extend(s, &sharee, content, 16);   /* expect E_PERM */
  int owner_extend  = capsule_extend(s, &owner, content, 16);    /* expect E_SEALED */

  /* sharee scribbles over its private (CoW) attach view */
  capsule_map_t m;
  int attach_rc = capsule_attach(s, &sharee, &m);
  if (attach_rc == CAPSULE_OK) {
    for (uint32_t sg = 0; sg < m.nsegs; sg++) {
      volatile uint8_t *p = (volatile uint8_t *)m.seg_base[sg];
      for (uint64_t off = 0; off < m.seg_len[sg]; off += 4096) p[off] ^= 0xFF;
    }
    capsule_detach(s, &m);
  }

  /* re-read via API and recompute the chain hash from scratch */
  uint64_t size = (uint64_t)nch * SCB;
  uint8_t *reread = malloc(size);
  int read_rc = capsule_read(s, &owner, 0, size, reread);
  uint8_t chain[32];
  caps_domain_seed(domain, chain);
  for (int c = 0; c < nch; c++) {
    uint8_t h[32], nx[32];
    caps_sha256(reread + (uint64_t)c * SCB, SCB, h);
    caps_chain_step(chain, h, nx);
    memcpy(chain, nx, 32);
  }
  int chain_matches = (memcmp(chain, before.chain_hash, 32) == 0);

  int pass = (derive_rc == CAPSULE_OK &&
              sharee_extend == CAPSULE_E_PERM &&
              owner_extend == CAPSULE_E_SEALED &&
              read_rc == CAPSULE_OK &&
              chain_matches);

  FILE *f = fopen(out, "w");
  meta_open(f, "E3_S3_seal_integrity", root, out, quick);
  fprintf(f, " \"measured\": {\"pass\": %s, "
             "\"derive_rc\": %d, \"sharee_extend_rc\": %d, \"owner_extend_rc\": %d, "
             "\"attach_rc\": %d, \"read_rc\": %d, \"chain_hash_matches\": %s, "
             "\"sharee_extend_is_PERM\": %s, \"owner_extend_is_SEALED\": %s},\n",
          pass ? "true" : "false", derive_rc, sharee_extend, owner_extend,
          attach_rc, read_rc, chain_matches ? "true" : "false",
          sharee_extend == CAPSULE_E_PERM ? "true" : "false",
          owner_extend == CAPSULE_E_SEALED ? "true" : "false");
  fprintf(f, " \"modeled\": {\"note\": \"pass/fail integrity check; no timings\"}\n");
  fprintf(f, "}\n}\n");
  fclose(f);
  free(content); free(reread);
  capsule_store_close(s);
  rm_rf(root);
  fprintf(stderr, "S3: pass=%d -> %s\n", pass, out);
  return pass ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <s1b|s1c|s2|s3> <store_root> <out.json> [quick]\n", argv[0]);
    return 2;
  }
  const char *study = argv[1];
  const char *root = argv[2];
  const char *out = argv[3];
  int quick = (argc > 4 && strcmp(argv[4], "quick") == 0);
  bu_time_init();

  if (strcmp(study, "s1b") == 0) return study_s1b(root, out, quick);
  if (strcmp(study, "s1c") == 0) return study_s1c(root, out, quick);
  if (strcmp(study, "s2") == 0)  return study_s2(root, out, quick);
  if (strcmp(study, "s3") == 0)  return study_s3(root, out, quick);
  fprintf(stderr, "unknown study %s\n", study);
  return 2;
}
