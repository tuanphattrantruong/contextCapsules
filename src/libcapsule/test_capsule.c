/* test_capsule.c — libcapsule functional/adversarial unit tests (package 1).
 *
 * Run by `make test` with CAPSULE_TEST_ROOT pointing at a scratch directory.
 * Covers the correctness-critical invariants of DESIGN.md:
 *   - create-time geometry validation (bytes_per_token % 1024)
 *   - chain-hash determinism + dedup accounting within a domain
 *   - dedup domain isolation (identical bytes, different domain -> no share)
 *   - fork = O(metadata): child references parent extents, no byte copy
 *   - CoW attach isolation: private scribbles never reach the store
 *   - seal immutability (E_SEALED), capability validation order (HMAC-first),
 *     rights attenuation, expiry, cross-process-safe revocation epoch
 *   - pin/evict interaction (E_EVICTED), destroy-with-erasure (bytes zeroed)
 *   - PARANOID dedup still writes scratch (S1c mechanism present)
 *
 * Exit status: 0 = all pass, 1 = one or more failed.
 */
#include "store_internal.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>

/* ------------------------------------------------------------- test harness */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...)                                                      \
  do {                                                                        \
    if (cond) { g_pass++; }                                                   \
    else { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
  } while (0)

#define CHECK_EQ(a, b, name)                                                  \
  do {                                                                        \
    long long _a = (long long)(a), _b = (long long)(b);                       \
    if (_a == _b) { g_pass++; }                                               \
    else { g_fail++; printf("  FAIL: %s: got %lld want %lld\n", name, _a, _b); } \
  } while (0)

static char g_root[1024];

static void root_for(const char *sub, char *out, size_t n) {
  snprintf(out, n, "%s/%s", g_root, sub);
}

static void rm_rf(const char *p) {
  char cmd[2048];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", p);
  int rc = system(cmd);
  (void)rc;
}

static capsule_store_t *fresh(const char *sub, uint32_t extra_flags) {
  char r[1024];
  root_for(sub, r, sizeof r);
  rm_rf(r);
  capsule_store_t *s = NULL;
  int rc = capsule_store_open(r, CAPSULE_STORE_CREATE | extra_flags, &s);
  if (rc != CAPSULE_OK) {
    printf("  FATAL: capsule_store_open(%s) = %d\n", r, rc);
    exit(2);
  }
  return s;
}

/* deterministic distinct content for a capsule of `nchunks` chunks */
static void fill_content(uint8_t *buf, uint64_t cb, uint32_t nchunks, uint64_t seed) {
  for (uint32_t c = 0; c < nchunks; c++) {
    uint8_t *p = buf + (uint64_t)c * cb;
    uint64_t x = seed * 0x100000001B3ULL + c + 1;
    for (uint64_t i = 0; i < cb; i++) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      p[i] = (uint8_t)x;
    }
  }
}

#define BPT 1024u                 /* scaled: chunk = 16 KiB, %1024 == 0 */
#define CB  (16ull * BPT)         /* chunk bytes = 16384 */

/* ------------------------------------------------------------------- tests */

static void t_create_validation(void) {
  printf("[t_create_validation]\n");
  capsule_store_t *s = fresh("cv", 0);
  capsule_cap_t cap; uint64_t id;
  CHECK_EQ(capsule_create(s, "/a", "", 0, &cap, &id), CAPSULE_E_INVAL, "bpt=0");
  CHECK_EQ(capsule_create(s, "/a", "", 1000, &cap, &id), CAPSULE_E_INVAL, "bpt%1024!=0");
  CHECK_EQ(capsule_create(s, "/a", "", BPT, &cap, &id), CAPSULE_OK, "bpt ok");
  CHECK_EQ(id, 0, "dense id 0");
  capsule_cap_t cap2; uint64_t id2;
  CHECK_EQ(capsule_create(s, "/b", "", BPT, &cap2, &id2), CAPSULE_OK, "create 2");
  CHECK_EQ(id2, 1, "dense id 1");
  capsule_store_close(s);
}

static void t_extend_dedup_domain(void) {
  printf("[t_extend_dedup_domain]\n");
  capsule_store_t *s = fresh("ed", 0);
  capsule_cap_t a, b, c; uint64_t ia, ib, ic;
  capsule_create(s, "/a", "", BPT, &a, &ia);
  capsule_create(s, "/b", "", BPT, &b, &ib);            /* same domain "" */
  capsule_create(s, "/c", "tenant:X", BPT, &c, &ic);    /* different domain */

  uint8_t *buf = malloc(2 * CB);
  fill_content(buf, CB, 2, 0xABCD);

  /* extend validation */
  CHECK_EQ(capsule_extend(s, &a, buf, 8), CAPSULE_E_INVAL, "n_tokens%16!=0");
  CHECK_EQ(capsule_extend(s, &a, buf, 0), CAPSULE_E_INVAL, "n_tokens=0");

  CHECK_EQ(capsule_extend(s, &a, buf, 32), CAPSULE_OK, "extend a 2 chunks");
  capsule_store_stats_t st;
  capsule_store_stats(s, &st);
  CHECK_EQ(st.stored_bytes, 2 * CB, "stored 2 chunks after a");
  CHECK_EQ(st.dedup_hits, 0, "no dedup yet");

  /* b: identical content, same domain -> full dedup, no new stored bytes */
  CHECK_EQ(capsule_extend(s, &b, buf, 32), CAPSULE_OK, "extend b identical");
  capsule_store_stats(s, &st);
  CHECK_EQ(st.dedup_hits, 2, "b dedups 2 chunks");
  CHECK_EQ(st.dedup_saved_bytes, 2 * CB, "b saved 2 chunks");
  CHECK_EQ(st.stored_bytes, 2 * CB, "stored still 2 chunks (b shared)");

  /* chain hash determinism: a and b have identical chain hashes */
  capsule_stat_t sa, sb;
  capsule_stat(s, &a, &sa);
  capsule_stat(s, &b, &sb);
  CHECK(memcmp(sa.chain_hash, sb.chain_hash, 32) == 0, "a/b chain hashes equal");

  /* c: identical bytes but different domain -> NO dedup (domain isolation) */
  CHECK_EQ(capsule_extend(s, &c, buf, 32), CAPSULE_OK, "extend c identical bytes");
  capsule_store_stats(s, &st);
  CHECK_EQ(st.dedup_hits, 2, "c does NOT dedup (different domain)");
  CHECK_EQ(st.stored_bytes, 4 * CB, "stored grows for c (isolated)");
  capsule_stat_t sc;
  capsule_stat(s, &c, &sc);
  CHECK(memcmp(sa.chain_hash, sc.chain_hash, 32) != 0, "c chain differs (domain seed)");

  free(buf);
  capsule_store_close(s);
}

static void t_seal(void) {
  printf("[t_seal]\n");
  capsule_store_t *s = fresh("seal", 0);
  capsule_cap_t a; uint64_t ia;
  capsule_create(s, "/a", "", BPT, &a, &ia);
  uint8_t *buf = malloc(CB);
  fill_content(buf, CB, 1, 7);
  capsule_extend(s, &a, buf, 16);
  CHECK_EQ(capsule_seal(s, &a), CAPSULE_OK, "seal");
  CHECK_EQ(capsule_seal(s, &a), CAPSULE_OK, "seal idempotent");
  CHECK_EQ(capsule_extend(s, &a, buf, 16), CAPSULE_E_SEALED, "extend after seal");
  capsule_stat_t sa; capsule_stat(s, &a, &sa);
  CHECK(sa.flags & 0x1, "SEALED flag set");
  free(buf);
  capsule_store_close(s);
}

static void t_fork_metadata(void) {
  printf("[t_fork_metadata]\n");
  capsule_store_t *s = fresh("fork", 0);
  capsule_cap_t p; uint64_t ip;
  capsule_create(s, "/p", "", BPT, &p, &ip);
  uint8_t *buf = malloc(4 * CB);
  fill_content(buf, CB, 4, 99);
  capsule_extend(s, &p, buf, 64);   /* 4 chunks */

  capsule_store_stats_t before; capsule_store_stats(s, &before);

  capsule_cap_t ch; uint64_t ich;
  CHECK_EQ(capsule_fork(s, &p, "/p/child", &ch, &ich), CAPSULE_OK, "fork");
  CHECK_EQ(ich, ip + 1, "child dense id");

  capsule_store_stats_t after; capsule_store_stats(s, &after);
  CHECK_EQ(after.stored_bytes, before.stored_bytes, "fork copies NO bytes");
  CHECK_EQ(after.n_extents, before.n_extents, "fork adds NO extents");

  capsule_stat_t sp, sc;
  capsule_stat(s, &p, &sp);
  capsule_stat(s, &ch, &sc);
  CHECK_EQ(sc.n_tokens, sp.n_tokens, "child inherits token length");
  CHECK(memcmp(sc.chain_hash, sp.chain_hash, 32) == 0, "child inherits chain hash");
  /* extents now shared: parent's owned_bytes drops to 0 (refcount 2) */
  CHECK_EQ(sp.owned_bytes, 0, "parent no longer sole owner");

  /* extend child with new content: parent unchanged, child grows */
  uint8_t *b2 = malloc(CB);
  fill_content(b2, CB, 1, 12345);
  CHECK_EQ(capsule_extend(s, &ch, b2, 16), CAPSULE_OK, "extend child");
  capsule_stat(s, &p, &sp);
  capsule_stat(s, &ch, &sc);
  CHECK_EQ(sp.n_tokens, 64, "parent length unchanged");
  CHECK_EQ(sc.n_tokens, 80, "child grew");

  free(buf); free(b2);
  capsule_store_close(s);
}

static void t_attach_cow(void) {
  printf("[t_attach_cow]\n");
  capsule_store_t *s = fresh("cow", 0);
  capsule_cap_t a; uint64_t ia;
  capsule_create(s, "/a", "", BPT, &a, &ia);
  uint32_t nch = 3;
  uint8_t *buf = malloc(nch * CB);
  fill_content(buf, CB, nch, 0x5151);
  capsule_extend(s, &a, buf, nch * 16);
  capsule_seal(s, &a);

  capsule_map_t m;
  CHECK_EQ(capsule_attach(s, &a, &m), CAPSULE_OK, "attach");
  CHECK_EQ(m.total_bytes, nch * CB, "map total bytes");
  CHECK(m.nsegs >= 1, "at least one segment");

  /* read first bytes via the mapping and verify against source */
  int first_ok = (memcmp(m.seg_base[0], buf, 64) == 0);
  CHECK(first_ok, "mapped bytes match source");

  /* scribble on the PRIVATE (CoW) mapping */
  ((volatile uint8_t *)m.seg_base[0])[0] ^= 0xFF;
  ((volatile uint8_t *)m.seg_base[0])[1] ^= 0xFF;
  CHECK_EQ(capsule_detach(s, &m), CAPSULE_OK, "detach");

  /* re-read through the API: store must be untouched (CoW isolation) */
  uint8_t got[64];
  CHECK_EQ(capsule_read(s, &a, 0, 64, got), CAPSULE_OK, "read");
  CHECK(memcmp(got, buf, 64) == 0, "store bytes unchanged after CoW scribble");

  /* read bounds */
  uint64_t size = (uint64_t)nch * CB;
  CHECK_EQ(capsule_read(s, &a, size - 16, 32, got), CAPSULE_E_INVAL, "read past end");

  free(buf);
  capsule_store_close(s);
}

static void t_capability_enforcement(void) {
  printf("[t_capability_enforcement]\n");
  capsule_store_t *s = fresh("cap", 0);
  capsule_cap_t owner; uint64_t id;
  capsule_create(s, "/a", "", BPT, &owner, &id);
  uint8_t *buf = malloc(CB);
  fill_content(buf, CB, 1, 3);
  capsule_extend(s, &owner, buf, 16);

  /* forged cap: flip a byte inside the HMAC region -> E_PERM (HMAC-first) */
  capsule_cap_t forged = owner;
  forged.bytes[45] ^= 0x01;    /* within hmac[32] (offset 40..71) */
  CHECK_EQ(capsule_read(s, &forged, 0, 16, buf), CAPSULE_E_PERM, "forged hmac -> PERM");

  /* forged cap for a NONEXISTENT id: same first-step failure (constant path) */
  capsule_cap_t nofake;
  memset(&nofake, 0xAB, sizeof nofake);   /* random-ish, id far out of range */
  CHECK_EQ(capsule_read(s, &nofake, 0, 16, buf), CAPSULE_E_PERM,
           "forged nonexistent -> PERM (no existence signal)");

  /* derive a READ-only cap (owner holds SHARE) */
  capsule_cap_t ro;
  CHECK_EQ(capsule_derive(s, &owner, CAP_R_READ, 0, &ro), CAPSULE_OK, "derive READ");
  CHECK_EQ(capsule_read(s, &ro, 0, 16, buf), CAPSULE_OK, "read with RO cap");
  CHECK_EQ(capsule_extend(s, &ro, buf, 16), CAPSULE_E_PERM, "extend with RO cap -> PERM");
  CHECK_EQ(capsule_seal(s, &ro), CAPSULE_E_PERM, "seal with RO cap -> PERM");

  /* attenuation: cannot derive MORE rights than the parent holds */
  CHECK_EQ(capsule_derive(s, &ro, CAP_R_ALL, 0, &forged), CAPSULE_E_PERM,
           "cannot escalate rights via derive");
  /* ro lacks SHARE, so even a subset derive from it is refused (needs SHARE) */
  CHECK_EQ(capsule_derive(s, &ro, CAP_R_READ, 0, &forged), CAPSULE_E_PERM,
           "derive requires SHARE on parent");

  free(buf);
  capsule_store_close(s);
}

static void t_expiry(void) {
  printf("[t_expiry]\n");
  capsule_store_t *s = fresh("exp", 0);
  capsule_cap_t owner; uint64_t id;
  capsule_create(s, "/a", "", BPT, &owner, &id);
  uint8_t *buf = malloc(CB);
  fill_content(buf, CB, 1, 1);
  capsule_extend(s, &owner, buf, 16);

  capsule_cap_t shortlived;
  CHECK_EQ(capsule_derive(s, &owner, CAP_R_READ, 1 /*ms*/, &shortlived),
           CAPSULE_OK, "derive 1ms ttl");
  usleep(6000);   /* 6 ms > 1 ms */
  CHECK_EQ(capsule_read(s, &shortlived, 0, 16, buf), CAPSULE_E_EXPIRED,
           "expired cap -> EXPIRED");
  /* owner cap (no expiry) still works */
  CHECK_EQ(capsule_read(s, &owner, 0, 16, buf), CAPSULE_OK, "owner unaffected");
  free(buf);
  capsule_store_close(s);
}

static void t_revocation(void) {
  printf("[t_revocation]\n");
  capsule_store_t *s = fresh("rev", 0);
  char root[1200]; root_for("rev", root, sizeof root);
  capsule_cap_t owner; uint64_t id;
  capsule_create(s, "/a", "", BPT, &owner, &id);
  uint8_t *buf = malloc(CB);
  fill_content(buf, CB, 1, 2);
  capsule_extend(s, &owner, buf, 16);
  capsule_cap_t ro;
  capsule_derive(s, &owner, CAP_R_READ, 0, &ro);

  CHECK_EQ(capsule_read(s, &ro, 0, 16, buf), CAPSULE_OK, "read before revoke");
  CHECK_EQ(capsule_revoke(s, &owner), CAPSULE_OK, "revoke");
  /* every outstanding cap dies at next validation (epoch bump) */
  CHECK_EQ(capsule_read(s, &ro, 0, 16, buf), CAPSULE_E_REVOKED, "derived cap revoked");
  CHECK_EQ(capsule_read(s, &owner, 0, 16, buf), CAPSULE_E_REVOKED, "owner cap revoked");
  capsule_store_close(s);

  /* A successful revoke is a persistence boundary, not just a live mmap bump. */
  s = NULL;
  CHECK_EQ(capsule_store_open(root, 0, &s), CAPSULE_OK, "reopen after revoke");
  if (s) {
    CHECK_EQ(capsule_read(s, &ro, 0, 16, buf), CAPSULE_E_REVOKED,
             "revocation survives close/reopen");
    capsule_store_close(s);
  }
  free(buf);
}

static void t_revocation_crash_boundary(void) {
  printf("[t_revocation_crash_boundary]\n");
  capsule_store_t *s = fresh("revcrash", 0);
  char root[1200]; root_for("revcrash", root, sizeof root);
  capsule_cap_t owner, ro; uint64_t id;
  CHECK_EQ(capsule_create(s, "/a", "", BPT, &owner, &id), CAPSULE_OK,
           "create for crash-boundary revoke");
  uint8_t buf[CB]; fill_content(buf, CB, 1, 77);
  CHECK_EQ(capsule_extend(s, &owner, buf, 16), CAPSULE_OK, "extend");
  CHECK_EQ(capsule_derive(s, &owner, CAP_R_READ, 0, &ro), CAPSULE_OK,
           "derive stale cap");
  pid_t p = fork();
  if (p == 0) {
    int rc = capsule_revoke(s, &owner);
    _exit(rc == CAPSULE_OK ? 0 : 3); /* no close: emulate process loss */
  }
  int st = 0;
  CHECK(p > 0 && waitpid(p, &st, 0) == p && WIFEXITED(st) &&
        WEXITSTATUS(st) == 0, "child revoke returned before abrupt exit");
  capsule_store_close(s);
  s = NULL;
  CHECK_EQ(capsule_store_open(root, 0, &s), CAPSULE_OK,
           "reopen after revoker process loss");
  if (s) {
    CHECK_EQ(capsule_read(s, &ro, 0, 16, buf), CAPSULE_E_REVOKED,
             "stale cap denied after abrupt revoker exit");
    capsule_store_close(s);
  }
}

static void t_pin_evict(void) {
  printf("[t_pin_evict]\n");
  capsule_store_t *s = fresh("evict", 0);
  capsule_cap_t p, u; uint64_t ip, iu;
  capsule_create(s, "/pinned", "dp", BPT, &p, &ip);
  capsule_create(s, "/unpinned", "du", BPT, &u, &iu);
  uint8_t *bp = malloc(2 * CB), *bu = malloc(2 * CB);
  fill_content(bp, CB, 2, 111);
  fill_content(bu, CB, 2, 222);
  capsule_extend(s, &p, bp, 32);
  capsule_extend(s, &u, bu, 32);
  capsule_seal(s, &p);
  capsule_seal(s, &u);

  CHECK_EQ(capsule_pin(s, &p, 60000), CAPSULE_OK, "pin p 60s");

  /* target = 2 chunks: forces one capsule's extent out; pinned p is protected */
  int rc = capsule_evict(s, 2 * CB, 0 /*LRU*/);
  CHECK(rc == CAPSULE_OK || rc == CAPSULE_E_BUDGET, "evict returns ok/budget");

  uint8_t got[16];
  CHECK_EQ(capsule_read(s, &p, 0, 16, got), CAPSULE_OK, "pinned survives");
  CHECK_EQ(capsule_read(s, &u, 0, 16, got), CAPSULE_E_EVICTED, "unpinned evicted");
  capsule_stat_t su; capsule_stat(s, &u, &su);
  CHECK(su.flags & 0x2, "EVICTED flag on u");
  free(bp); free(bu);
  capsule_store_close(s);
}

static void t_destroy_erasure(void) {
  printf("[t_destroy_erasure]\n");
  /* dedicated fresh store so exactly one extent file exists */
  capsule_store_t *s = fresh("erase", 0);
  capsule_cap_t d; uint64_t id;
  capsule_create(s, "/d", "solo", BPT, &d, &id);
  uint8_t *buf = malloc(2 * CB);
  fill_content(buf, CB, 2, 0xDEAD);
  capsule_extend(s, &d, buf, 32);

  /* locate the single extent file, open a read fd BEFORE destroy */
  char extdir[1200];
  snprintf(extdir, sizeof extdir, "%s/erase/extents", g_root);
  DIR *dd = opendir(extdir);
  char extpath[1400] = {0};
  if (dd) {
    struct dirent *de;
    while ((de = readdir(dd))) {
      if (strstr(de->d_name, ".ext")) {
        snprintf(extpath, sizeof extpath, "%s/%s", extdir, de->d_name);
        break;
      }
    }
    closedir(dd);
  }
  CHECK(extpath[0] != 0, "found extent file");
  int fd = open(extpath, O_RDONLY);
  CHECK(fd >= 0, "opened extent fd pre-destroy");

  CHECK_EQ(capsule_destroy(s, &d), CAPSULE_OK, "destroy");

  /* file is unlinked */
  CHECK(access(extpath, F_OK) != 0, "extent file unlinked after destroy");

  /* retained fd sees ZEROED bytes (erase memset before unlink) */
  if (fd >= 0) {
    uint8_t chk[4096];
    ssize_t r = pread(fd, chk, sizeof chk, 0);
    int all_zero = (r > 0);
    for (ssize_t i = 0; i < r; i++) if (chk[i] != 0) { all_zero = 0; break; }
    CHECK(all_zero, "erased bytes are zero on retained fd");
    close(fd);
  }

  /* post-destroy cap is dead (epoch bumped) */
  uint8_t got[16];
  int rc = capsule_read(s, &d, 0, 16, got);
  CHECK(rc == CAPSULE_E_REVOKED || rc == CAPSULE_E_NOENT, "cap dead after destroy");

  free(buf);
  capsule_store_close(s);
}

static void t_paranoid_scratch(void) {
  printf("[t_paranoid_scratch]\n");
  capsule_store_t *s = fresh("para", CAPSULE_STORE_PARANOID);
  capsule_cap_t a, b; uint64_t ia, ib;
  capsule_create(s, "/a", "", BPT, &a, &ia);
  capsule_create(s, "/b", "", BPT, &b, &ib);
  uint8_t *buf = malloc(2 * CB);
  fill_content(buf, CB, 2, 0x1234);
  capsule_extend(s, &a, buf, 32);
  uint64_t g0 = capsule_internal_paranoid_garbage_bytes(s);
  capsule_extend(s, &b, buf, 32);   /* dedup hit -> paranoid writes scratch */
  uint64_t g1 = capsule_internal_paranoid_garbage_bytes(s);
  capsule_store_stats_t st; capsule_store_stats(s, &st);
  CHECK_EQ(st.dedup_hits, 2, "paranoid still dedups (return code identical)");
  CHECK(g1 > g0, "paranoid mode writes scratch bytes on dedup hit");
  free(buf);
  capsule_store_close(s);
}

static void t_persistence_reopen(void) {
  printf("[t_persistence_reopen]\n");
  char r[1024]; root_for("persist", r, sizeof r);
  rm_rf(r);
  capsule_cap_t a; uint64_t id;
  uint8_t chain_before[32];
  {
    capsule_store_t *s = NULL;
    capsule_store_open(r, CAPSULE_STORE_CREATE, &s);
    capsule_create(s, "/a", "dom", BPT, &a, &id);
    uint8_t *buf = malloc(3 * CB);
    fill_content(buf, CB, 3, 55);
    capsule_extend(s, &a, buf, 48);
    capsule_seal(s, &a);
    capsule_stat_t sa; capsule_stat(s, &a, &sa);
    memcpy(chain_before, sa.chain_hash, 32);
    free(buf);
    capsule_store_close(s);
  }
  /* reopen: manifests + prefix index rebuilt from disk; content name stable */
  {
    capsule_store_t *s = NULL;
    int rc = capsule_store_open(r, 0, &s);
    CHECK_EQ(rc, CAPSULE_OK, "reopen existing store");
    /* re-mint an owner cap is not public; but a fresh capsule with identical
     * content+domain must dedup fully against the reopened index */
    capsule_cap_t b; uint64_t ib;
    CHECK_EQ(capsule_create(s, "/b", "dom", BPT, &b, &ib), CAPSULE_OK, "create b");
    uint8_t *buf = malloc(3 * CB);
    fill_content(buf, CB, 3, 55);
    capsule_extend(s, &b, buf, 48);
    capsule_stat_t sb; capsule_stat(s, &b, &sb);
    CHECK(memcmp(sb.chain_hash, chain_before, 32) == 0,
          "reopened content name matches");
    capsule_store_stats_t st; capsule_store_stats(s, &st);
    CHECK_EQ(st.dedup_hits, 3, "b fully dedups against rebuilt index");
    free(buf);
    capsule_store_close(s);
  }
}


/* ABI v2: capsule_create_chained. A capsule seeded from a parent's chain must
 * (a) dedup against another capsule seeded from the SAME parent and domain,
 * (b) NOT dedup against one seeded from a DIFFERENT parent, even though the
 *     content bytes and the domain are identical, and
 * (c) NOT dedup across domains, as always. This is the property that makes
 *     position-chained private naming sound for dense-transformer KV. */
static void t_create_chained(void) {
  printf("[t_create_chained]\n");
  char r[1024]; root_for("chained", r, sizeof r);
  rm_rf(r);
  capsule_store_t *s = NULL;
  CHECK_EQ(capsule_store_open(r, CAPSULE_STORE_CREATE, &s), CAPSULE_OK,
           "open chained store");

  /* two DIFFERENT public prefixes */
  capsule_cap_t p1, p2; uint64_t i1, i2;
  capsule_create(s, "/pub1", "", BPT, &p1, &i1);
  capsule_create(s, "/pub2", "", BPT, &p2, &i2);
  uint8_t *pa = malloc(CB), *pb = malloc(CB);
  fill_content(pa, CB, 1, 11);
  fill_content(pb, CB, 1, 22);
  capsule_extend(s, &p1, pa, 16);
  capsule_extend(s, &p2, pb, 16);

  capsule_store_stats_t st0; capsule_store_stats(s, &st0);

  /* identical private content, chained from p1 twice and from p2 once */
  uint8_t *priv = malloc(2 * CB);
  fill_content(priv, CB, 2, 77);

  capsule_cap_t a, b, c; uint64_t ia, ib, ic;
  CHECK_EQ(capsule_create_chained(s, "/a", "tenant:1", BPT, &p1, &a, &ia),
           CAPSULE_OK, "create_chained a from pub1");
  capsule_extend(s, &a, priv, 32);
  capsule_store_stats_t st1; capsule_store_stats(s, &st1);
  CHECK_EQ(st1.dedup_hits - st0.dedup_hits, 0, "first chained store is a miss");

  CHECK_EQ(capsule_create_chained(s, "/b", "tenant:1", BPT, &p1, &b, &ib),
           CAPSULE_OK, "create_chained b from pub1");
  capsule_extend(s, &b, priv, 32);
  capsule_store_stats_t st2; capsule_store_stats(s, &st2);
  CHECK_EQ(st2.dedup_hits - st1.dedup_hits, 2,
           "same parent + same domain + same bytes dedups");

  CHECK_EQ(capsule_create_chained(s, "/c", "tenant:1", BPT, &p2, &c, &ic),
           CAPSULE_OK, "create_chained c from pub2");
  capsule_extend(s, &c, priv, 32);
  capsule_store_stats_t st3; capsule_store_stats(s, &st3);
  CHECK_EQ(st3.dedup_hits - st2.dedup_hits, 0,
           "different parent prefix does NOT dedup (position chaining works)");

  /* same parent, different domain: still no dedup */
  capsule_cap_t d; uint64_t idd;
  CHECK_EQ(capsule_create_chained(s, "/d", "tenant:2", BPT, &p1, &d, &idd),
           CAPSULE_OK, "create_chained d, other domain");
  capsule_extend(s, &d, priv, 32);
  capsule_store_stats_t st4; capsule_store_stats(s, &st4);
  CHECK_EQ(st4.dedup_hits - st3.dedup_hits, 0,
           "cross-domain still never dedups");

  /* chains of a and b agree; a and c differ */
  capsule_stat_t sa, sb, sc;
  capsule_stat(s, &a, &sa); capsule_stat(s, &b, &sb); capsule_stat(s, &c, &sc);
  CHECK(memcmp(sa.chain_hash, sb.chain_hash, 32) == 0,
        "same-parent chains match");
  CHECK(memcmp(sa.chain_hash, sc.chain_hash, 32) != 0,
        "different-parent chains differ");

  /* a forged/absent parent capability must be refused */
  capsule_cap_t bogus; memset(&bogus, 0xAB, sizeof bogus);
  capsule_cap_t out; uint64_t oid;
  CHECK_EQ(capsule_create_chained(s, "/e", "tenant:1", BPT, &bogus, &out, &oid),
           CAPSULE_E_PERM, "forged chain parent refused");

  free(pa); free(pb); free(priv);
  capsule_store_close(s);
}

/* ABI v3 compatibility binding: equal bytes in the same tenant domain must not
 * deduplicate across incompatible runtime descriptors, and attach must reject
 * a caller that presents the wrong descriptor before exposing any mapping. */
static void t_compatibility_binding(void) {
  printf("[t_compatibility_binding]\n");
  char r[1024]; root_for("compat", r, sizeof r);
  rm_rf(r);
  capsule_store_t *s = NULL;
  CHECK_EQ(capsule_store_open(r, CAPSULE_STORE_CREATE, &s), CAPSULE_OK,
           "open compatibility store");

  uint8_t h1[32], h2[32];
  memset(h1, 0x11, sizeof h1); memset(h2, 0x22, sizeof h2);
  capsule_cap_t a, b; uint64_t ia, ib;
  CHECK_EQ(capsule_create_compatible(s, "/a", "tenant:1", BPT, h1, &a, &ia),
           CAPSULE_OK, "create compatible a");
  CHECK_EQ(capsule_create_compatible(s, "/b", "tenant:1", BPT, h2, &b, &ib),
           CAPSULE_OK, "create compatible b");
  uint8_t *buf = malloc(CB); fill_content(buf, CB, 1, 91);
  capsule_store_stats_t st0, st1, st2;
  capsule_store_stats(s, &st0);
  capsule_extend(s, &a, buf, 16); capsule_store_stats(s, &st1);
  capsule_extend(s, &b, buf, 16); capsule_store_stats(s, &st2);
  CHECK_EQ(st2.dedup_hits - st1.dedup_hits, 0,
           "incompatible descriptors never dedup");

  capsule_map_t m;
  CHECK_EQ(capsule_attach_compatible(s, &a, h2, &m), CAPSULE_E_INVAL,
           "wrong compatibility descriptor denied");
  CHECK_EQ(capsule_attach_compatible(s, &a, h1, &m), CAPSULE_OK,
           "matching compatibility descriptor attaches");
  capsule_detach(s, &m);
  capsule_stat_t sa; capsule_stat(s, &a, &sa);
  CHECK(memcmp(sa.compatibility_hash, h1, 32) == 0,
        "stat reports bound compatibility hash");
  free(buf); capsule_store_close(s);
}

int main(void) {
  const char *env = getenv("CAPSULE_TEST_ROOT");
  if (env && env[0]) {
    strncpy(g_root, env, sizeof g_root - 1);
  } else {
    snprintf(g_root, sizeof g_root, "./tmp_store_c_test");
  }
  /* ensure the parent exists */
  {
    char cmd[2048];
    snprintf(cmd, sizeof cmd, "mkdir -p '%s'", g_root);
    int rc = system(cmd); (void)rc;
  }
  printf("libcapsule unit tests (root=%s)\n", g_root);

  t_create_validation();
  t_extend_dedup_domain();
  t_seal();
  t_fork_metadata();
  t_attach_cow();
  t_capability_enforcement();
  t_expiry();
  t_revocation();
  t_revocation_crash_boundary();
  t_pin_evict();
  t_destroy_erasure();
  t_paranoid_scratch();
  t_persistence_reopen();
  t_create_chained();
  t_compatibility_binding();

  /* clean scratch subdirectories we created */
  const char *subs[] = {"cv","ed","seal","fork","cow","cap","exp","rev","revcrash",
                        "evict","erase","para","persist","chained","compat"};
  for (size_t i = 0; i < sizeof subs / sizeof subs[0]; i++) {
    char r[1200]; root_for(subs[i], r, sizeof r); rm_rf(r);
  }

  printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
