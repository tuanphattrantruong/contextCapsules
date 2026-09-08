/* llamacap.c — E10: capsules mounted under a real inference engine.
 *
 * Round-1 review objection R1: E5 drove a real model *through* the capsule
 * interface, but capsules were never mounted *inside* an engine, so nothing
 * measured the abstraction at the point a serving stack would actually use it.
 *
 * This program mounts libcapsule under llama.cpp's own per-sequence KV state
 * API — the exact serialization path llama-server uses for slot save/restore:
 *
 *     llama_state_seq_get_size(ctx, seq)      how big is this sequence's KV
 *     llama_state_seq_get_data(ctx, buf, n, seq)   engine -> bytes
 *     llama_state_seq_set_data(ctx, buf, n, seq)   bytes -> engine
 *
 * The capsule store replaces the engine's flat, unnamed, unprotected slot
 * cache: a sequence's KV becomes a named, capability-protected, sealed capsule
 * that a *different process* can attach and resume from.
 *
 * Modes
 *   save    : prefill a prompt in the engine, store its KV as a capsule, print
 *             the capability; also decode N tokens for a reference transcript.
 *   restore : in a FRESH process, attach by capability alone, push the bytes
 *             back into the engine, and decode N tokens. Compare.
 *   bench   : in one process, alternate engine-prefill vs capsule-restore over
 *             a range of prompt lengths and emit JSON.
 *   export/import: a narrow sidecar boundary used by E12.  The engine exports
 *             or imports its opaque sequence-state blob; an authenticated
 *             service adapter owns the capability and persistence protocol.
 *
 * Everything measured here is engine-side: the timers wrap llama.cpp calls.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include "llama.h"
#include "capsule.h"
#include "../libcapsule/bench_util.h"

#define MAX_PROMPT_TOK 16384

static void die(const char *m) { fprintf(stderr, "fatal: %s\n", m); exit(1); }

/* libcapsule requires bytes_per_token % 1024 == 0 and extends in whole
 * 16-token chunks. The engine's blob is one opaque per-sequence buffer, so we
 * present it to the store as a synthetic geometry: pad to a multiple of
 * (16 * BPT) and record the true length alongside. */
#define CAP_BPT 1024u
#define CAP_CHUNK_BYTES ((size_t)CAPSULE_CHUNK_TOKENS * CAP_BPT)

static size_t pad_to_chunk(size_t n) {
    return ((n + CAP_CHUNK_BYTES - 1) / CAP_CHUNK_BYTES) * CAP_CHUNK_BYTES;
}

/* The engine's per-sequence blob has no self-describing length, and
 * llama_state_seq_get_size() on a *fresh* context reports the size of an EMPTY
 * sequence -- so a restoring process cannot recover the true length from the
 * engine. The capsule therefore carries a small header ahead of the payload:
 * a named object should describe itself. */
#define LC_MAGIC 0x4C43415053554C45ULL   /* "LCAPSULE" */
typedef struct { uint64_t magic; uint64_t blob_bytes; uint64_t n_prompt;
                 uint64_t _pad; } lc_hdr_t;

typedef struct {
    struct llama_model   *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
} eng_t;

static struct llama_model *model_open(const char *gguf, int ngl) {
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    struct llama_model *m = llama_model_load_from_file(gguf, mp);
    if (!m) die("model load failed");
    return m;
}

/* A context sized to the prompt: llama.cpp's compute buffers scale with n_ctx,
 * so measuring a 512-token prefill inside an 8192-token context would inflate
 * it. Each length gets its own right-sized context. */
static eng_t ctx_open(struct llama_model *model, int n_ctx) {
    eng_t e;
    e.model = model;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx;
    cp.n_batch = n_ctx;
    cp.no_perf = true;
    e.ctx = llama_init_from_model(e.model, cp);
    if (!e.ctx) die("context init failed");
    e.vocab = llama_model_get_vocab(e.model);
    return e;
}

static eng_t eng_open(const char *gguf, int n_ctx, int ngl) {
    return ctx_open(model_open(gguf, ngl), n_ctx);
}

static void eng_close(eng_t *e) {
    llama_free(e->ctx);
    llama_model_free(e->model);
    llama_backend_free();
}

/* Deterministic filler prompt, grown until it tokenizes to at least n tokens. */
static int build_prompt(eng_t *e, int n, llama_token *out) {
    static char buf[1 << 22];
    const char *w[] = {"the","system","memory","context","cache","token","model",
                       "tenant","capsule","inference","process","kernel",
                       "protection","lifecycle","revocation","naming","domain",
                       "prefix","attention","transformer","serving","agent"};
    size_t want = (size_t)n * 8 + 512;
    for (int attempt = 0; attempt < 8; attempt++) {
        if (want > sizeof(buf) - 32) want = sizeof(buf) - 32;
        size_t off = 0;
        unsigned x = 12345;
        while (off < want) {
            x = x * 1103515245u + 12345u;
            const char *s = w[(x >> 16) % (sizeof(w) / sizeof(w[0]))];
            int k = snprintf(buf + off, sizeof(buf) - off, "%s ", s);
            if (k <= 0) break;
            off += (size_t)k;
        }
        int got = llama_tokenize(e->vocab, buf, (int32_t)off, out,
                                 MAX_PROMPT_TOK, true, false);
        if (got >= n) return n;
        if (want >= sizeof(buf) - 32) break;
        want *= 2;
    }
    die("prompt too short");
    return 0;
}

/* Prefill `n` tokens into sequence `seq`. Returns wall ns. */
static double eng_prefill(eng_t *e, const llama_token *toks, int n, int seq) {
    llama_memory_seq_rm(llama_get_memory(e->ctx), seq, -1, -1);
    struct llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq;
        b.logits[i] = (i == n - 1);
    }
    b.n_tokens = n;
    uint64_t t0 = bu_now_ticks();
    int rc = llama_decode(e->ctx, b);
    uint64_t t1 = bu_now_ticks();
    llama_batch_free(b);
    if (rc != 0) die("llama_decode failed");
    return bu_ticks_to_ns(t1 - t0);
}

/* Resuming from a restored KV cache leaves the engine with no logits for the
 * last prompt position (logits are not part of the sequence state). A serving
 * engine handles a prefix-cache hit the same way: it keeps positions 0..n-2 and
 * recomputes the final token. This does that, and its cost is counted inside
 * the restore time we report. */
static void eng_refresh_logits(eng_t *e, int seq, int pos, llama_token tok) {
    llama_memory_seq_rm(llama_get_memory(e->ctx), seq, pos, -1);
    struct llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1;
    b.seq_id[0][0] = seq; b.logits[0] = 1; b.n_tokens = 1;
    if (llama_decode(e->ctx, b) != 0) die("logits refresh failed");
    llama_batch_free(b);
}

/* Greedy-decode `n_new` tokens from sequence `seq` starting at position `pos`. */
static void eng_decode(eng_t *e, int seq, int pos, int n_new,
                       llama_token *out) {
    for (int i = 0; i < n_new; i++) {
        const float *lg = llama_get_logits_ith(e->ctx, -1);
        int nv = llama_vocab_n_tokens(e->vocab);
        int best = 0;
        float bv = lg[0];
        for (int v = 1; v < nv; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
        out[i] = best;
        struct llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = best; b.pos[0] = pos + i; b.n_seq_id[0] = 1;
        b.seq_id[0][0] = seq; b.logits[0] = 1; b.n_tokens = 1;
        if (llama_decode(e->ctx, b) != 0) die("decode step failed");
        llama_batch_free(b);
    }
}

/* ---- capsule <-> engine ------------------------------------------------- */

static void cap_put(capsule_store_t *s, const char *path, const char *domain,
                    const uint8_t *blob, size_t nbytes, int n_prompt,
                    capsule_cap_t *cap_out, uint64_t *id_out) {
    if (capsule_create(s, path, domain, CAP_BPT, cap_out, id_out) != CAPSULE_OK)
        die("capsule_create");
    size_t padded = pad_to_chunk(sizeof(lc_hdr_t) + nbytes);
    uint8_t *tmp = calloc(1, padded);
    if (!tmp) die("oom");
    lc_hdr_t h = { LC_MAGIC, (uint64_t)nbytes, (uint64_t)n_prompt, 0 };
    memcpy(tmp, &h, sizeof h);
    memcpy(tmp + sizeof(lc_hdr_t), blob, nbytes);
    size_t done = 0;
    while (done < padded) {
        size_t take = padded - done;
        if (take > 64 * CAP_CHUNK_BYTES) take = 64 * CAP_CHUNK_BYTES;
        uint32_t ntok = (uint32_t)(take / CAP_BPT);
        if (capsule_extend(s, cap_out, tmp + done, ntok) != CAPSULE_OK)
            die("capsule_extend");
        done += take;
    }
    free(tmp);
    if (capsule_seal(s, cap_out) != CAPSULE_OK) die("capsule_seal");
}

static void hex_of(const capsule_cap_t *c, char *out) {
    for (int i = 0; i < 72; i++) sprintf(out + 2 * i, "%02x", c->bytes[i]);
    out[144] = 0;
}

static void cap_of_hex(const char *h, capsule_cap_t *c) {
    for (int i = 0; i < 72; i++) {
        unsigned v; sscanf(h + 2 * i, "%2x", &v); c->bytes[i] = (uint8_t)v;
    }
}

static void blob_write(const char *path, const lc_hdr_t *h,
                       const uint8_t *blob) {
    FILE *f = fopen(path, "wb");
    if (!f) die("open blob output");
    if (fwrite(h, sizeof *h, 1, f) != 1 ||
        fwrite(blob, 1, (size_t)h->blob_bytes, f) != (size_t)h->blob_bytes ||
        fclose(f) != 0)
        die("write blob output");
}

static uint8_t *blob_read(const char *path, lc_hdr_t *h) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open blob input");
    if (fread(h, sizeof *h, 1, f) != 1 || h->magic != LC_MAGIC ||
        h->blob_bytes == 0 || h->blob_bytes > (1ull << 34))
        die("invalid engine blob");
    uint8_t *blob = malloc((size_t)h->blob_bytes);
    if (!blob || fread(blob, 1, (size_t)h->blob_bytes, f) !=
                     (size_t)h->blob_bytes)
        die("short engine blob");
    fclose(f);
    return blob;
}

/* E12 sidecar producer: this process has engine memory but no capsule store,
 * HMAC key, manifest, epoch table, or service namespace. */
static int mode_export(int argc, char **argv) {
    /* export <gguf> <blobfile> <n_prompt> <n_new> <ngl> */
    const char *gguf = argv[2], *blobfile = argv[3];
    int n_prompt = atoi(argv[4]), n_new = atoi(argv[5]), ngl = atoi(argv[6]);
    eng_t e = eng_open(gguf, n_prompt + n_new + 64, ngl);
    static llama_token toks[MAX_PROMPT_TOK], ref[4096];
    build_prompt(&e, n_prompt, toks);
    double pre_ns = eng_prefill(&e, toks, n_prompt, 0);
    size_t blob_n = llama_state_seq_get_size(e.ctx, 0);
    uint8_t *blob = malloc(blob_n);
    uint64_t t0 = bu_now_ticks();
    size_t got = llama_state_seq_get_data(e.ctx, blob, blob_n, 0);
    uint64_t t1 = bu_now_ticks();
    if (!got) die("state_seq_get_data failed");
    lc_hdr_t h = {LC_MAGIC, (uint64_t)got, (uint64_t)n_prompt, 0};
    blob_write(blobfile, &h, blob);
    eng_decode(&e, 0, n_prompt, n_new, ref);
    printf("{\"role\":\"service_export\",\"n_prompt\":%d,"
           "\"n_new\":%d,\"engine_blob_bytes\":%zu,\"prefill_ms\":%.3f,"
           "\"state_get_ms\":%.3f,\"ref_tokens\":[",
           n_prompt, n_new, got, pre_ns / 1e6,
           bu_ticks_to_ns(t1 - t0) / 1e6);
    for (int i = 0; i < n_new; i++) printf("%s%d", i ? "," : "", ref[i]);
    printf("]}\n");
    free(blob);
    eng_close(&e);
    return 0;
}

/* E12 sidecar consumer: a fresh engine process receives only bytes released
 * by the authenticated adapter after capability validation. */
static int mode_import(int argc, char **argv) {
    /* import <gguf> <blobfile> <n_new> <ngl> */
    const char *gguf = argv[2], *blobfile = argv[3];
    int n_new = atoi(argv[4]), ngl = atoi(argv[5]);
    lc_hdr_t h;
    uint8_t *blob = blob_read(blobfile, &h);
    int n_prompt = (int)h.n_prompt;
    eng_t e = eng_open(gguf, n_prompt + n_new + 64, ngl);
    static llama_token toks[MAX_PROMPT_TOK], got[4096];
    build_prompt(&e, n_prompt, toks);
    uint64_t t0 = bu_now_ticks();
    size_t used = llama_state_seq_set_data(e.ctx, blob, (size_t)h.blob_bytes, 0);
    if (!used) die("state_seq_set_data failed");
    eng_refresh_logits(&e, 0, n_prompt - 1, toks[n_prompt - 1]);
    uint64_t t1 = bu_now_ticks();
    eng_decode(&e, 0, n_prompt, n_new, got);
    printf("{\"role\":\"service_import\",\"n_prompt\":%d,"
           "\"engine_blob_bytes\":%llu,\"state_set_ms\":%.3f,"
           "\"tokens\":[",
           n_prompt, (unsigned long long)h.blob_bytes,
           bu_ticks_to_ns(t1 - t0) / 1e6);
    for (int i = 0; i < n_new; i++) printf("%s%d", i ? "," : "", got[i]);
    printf("]}\n");
    free(blob);
    eng_close(&e);
    return 0;
}

/* ---- modes -------------------------------------------------------------- */

static int mode_save(int argc, char **argv) {
    /* save <gguf> <store> <n_prompt> <n_new> <ngl> */
    const char *gguf = argv[2], *store_root = argv[3];
    int n_prompt = atoi(argv[4]), n_new = atoi(argv[5]), ngl = atoi(argv[6]);

    eng_t e = eng_open(gguf, n_prompt + n_new + 64, ngl);
    static llama_token toks[MAX_PROMPT_TOK];
    build_prompt(&e, n_prompt, toks);

    double pre_ns = eng_prefill(&e, toks, n_prompt, 0);

    size_t blob_n = llama_state_seq_get_size(e.ctx, 0);
    uint8_t *blob = malloc(blob_n);
    uint64_t t0 = bu_now_ticks();
    size_t got = llama_state_seq_get_data(e.ctx, blob, blob_n, 0);
    uint64_t t1 = bu_now_ticks();
    if (got == 0) die("state_seq_get_data failed");

    capsule_store_t *s = NULL;
    if (capsule_store_open(store_root, CAPSULE_STORE_CREATE, &s) != CAPSULE_OK)
        die("store_open");
    capsule_cap_t cap; uint64_t cid;
    uint64_t t2 = bu_now_ticks();
    cap_put(s, "/engine/seq0", "tenant:1", blob, got, n_prompt, &cap, &cid);
    uint64_t t3 = bu_now_ticks();

    capsule_stat_t stt;
    capsule_stat(s, &cap, &stt);
    char chain[65];
    for (int i = 0; i < 32; i++) sprintf(chain + 2 * i, "%02x", stt.chain_hash[i]);
    chain[64] = 0;

    static llama_token ref[4096];
    eng_decode(&e, 0, n_prompt, n_new, ref);

    char caphex[145]; hex_of(&cap, caphex);
    printf("{\"role\":\"save\",\"n_prompt\":%d,\"n_new\":%d,"
           "\"engine_blob_bytes\":%zu,\"prefill_ms\":%.3f,"
           "\"state_get_ms\":%.3f,\"capsule_put_ms\":%.3f,"
           "\"capsule_id\":%llu,\"chain\":\"%s\",\"cap\":\"%s\","
           "\"ref_tokens\":[",
           n_prompt, n_new, got, pre_ns / 1e6,
           bu_ticks_to_ns(t1 - t0) / 1e6, bu_ticks_to_ns(t3 - t2) / 1e6,
           (unsigned long long)cid, chain, caphex);
    for (int i = 0; i < n_new; i++) printf("%s%d", i ? "," : "", ref[i]);
    printf("]}\n");

    capsule_store_close(s);
    eng_close(&e);
    return 0;
}

static int mode_restore(int argc, char **argv) {
    /* restore <gguf> <store> <n_prompt> <n_new> <ngl> <caphex> */
    const char *gguf = argv[2], *store_root = argv[3];
    int n_prompt = atoi(argv[4]), n_new = atoi(argv[5]), ngl = atoi(argv[6]);
    const char *caphex = argv[7];

    eng_t e = eng_open(gguf, n_prompt + n_new + 64, ngl);
    static llama_token toks[MAX_PROMPT_TOK];
    build_prompt(&e, n_prompt, toks);

    capsule_store_t *s = NULL;
    if (capsule_store_open(store_root, 0, &s) != CAPSULE_OK) die("store_open");
    capsule_cap_t cap; cap_of_hex(caphex, &cap);

    capsule_stat_t stt;
    if (capsule_stat(s, &cap, &stt) != CAPSULE_OK) die("capsule_stat denied");
    char chain[65];
    for (int i = 0; i < 32; i++) sprintf(chain + 2 * i, "%02x", stt.chain_hash[i]);
    chain[64] = 0;

    /* attach: CoW map of the sealed extents, then hand the bytes to the engine */
    capsule_map_t m;
    uint64_t t0 = bu_now_ticks();
    if (capsule_attach(s, &cap, &m) != CAPSULE_OK) die("capsule_attach denied");
    uint64_t t1 = bu_now_ticks();

    /* flatten the mapped extents, then read the self-describing header */
    size_t mapped = 0;
    for (uint32_t i = 0; i < m.nsegs; i++) mapped += m.seg_len[i];
    uint8_t *flat = malloc(mapped);
    size_t off = 0;
    for (uint32_t i = 0; i < m.nsegs; i++) {
        memcpy(flat + off, m.seg_base[i], m.seg_len[i]);
        off += m.seg_len[i];
    }
    lc_hdr_t h;
    memcpy(&h, flat, sizeof h);
    if (h.magic != LC_MAGIC) die("not an engine-state capsule");
    size_t blob_n = (size_t)h.blob_bytes;
    uint8_t *blob = flat + sizeof(lc_hdr_t);
    uint64_t t2 = bu_now_ticks();
    size_t used = llama_state_seq_set_data(e.ctx, blob, blob_n, 0);
    if (used == 0) die("state_seq_set_data failed");
    eng_refresh_logits(&e, 0, n_prompt - 1, toks[n_prompt - 1]);
    uint64_t t3 = bu_now_ticks();
    capsule_detach(s, &m);

    static llama_token got[4096];
    eng_decode(&e, 0, n_prompt, n_new, got);

    /* lifecycle: revoke here must deny this process's next validated call */
    int revoked = 0;
    if (capsule_revoke(s, &cap) == CAPSULE_OK) {
        capsule_stat_t d;
        revoked = (capsule_stat(s, &cap, &d) == CAPSULE_E_REVOKED);
    }

    printf("{\"role\":\"restore\",\"attach_ms\":%.3f,\"copy_ms\":%.3f,"
           "\"state_set_ms\":%.3f,\"restore_total_ms\":%.3f,"
           "\"chain\":\"%s\",\"revoke_denies\":%s,\"tokens\":[",
           bu_ticks_to_ns(t1 - t0) / 1e6, bu_ticks_to_ns(t2 - t1) / 1e6,
           bu_ticks_to_ns(t3 - t2) / 1e6, bu_ticks_to_ns(t3 - t0) / 1e6,
           chain, revoked ? "true" : "false");
    for (int i = 0; i < n_new; i++) printf("%s%d", i ? "," : "", got[i]);
    printf("]}\n");

    capsule_store_close(s);
    eng_close(&e);
    return 0;
}

static int mode_bench(int argc, char **argv) {
    /* bench <gguf> <store> <ngl> <reps> <len1,len2,...> */
    const char *gguf = argv[2], *store_root = argv[3];
    int ngl = atoi(argv[4]), reps = atoi(argv[5]);
    char lens[256]; snprintf(lens, sizeof lens, "%s", argv[6]);

    int L[16], nL = 0, maxlen = 0;
    for (char *t = strtok(lens, ","); t && nL < 16; t = strtok(NULL, ",")) {
        L[nL] = atoi(t); if (L[nL] > maxlen) maxlen = L[nL]; nL++;
    }

    struct llama_model *model = model_open(gguf, ngl);
    static llama_token toks[MAX_PROMPT_TOK];

    printf("{\"experiment\":\"E10_llamacpp_engine\",");
    bu_emit_meta_fields(stdout);
    printf(",\"gguf\":\"%s\",\"n_gpu_layers\":%d,\"reps\":%d,\"rows\":[", gguf, ngl, reps);

    for (int i = 0; i < nL; i++) {
        int n = L[i];
        eng_t e = ctx_open(model, n + 128);
        build_prompt(&e, n, toks);
        eng_prefill(&e, toks, n, 0);            /* warm */

        double *pre = malloc(sizeof(double) * reps);
        for (int r = 0; r < reps; r++) pre[r] = eng_prefill(&e, toks, n, 0);

        size_t blob_n = llama_state_seq_get_size(e.ctx, 0);
        uint8_t *blob = malloc(blob_n);
        size_t got = llama_state_seq_get_data(e.ctx, blob, blob_n, 0);

        char sroot[1100]; snprintf(sroot, sizeof sroot, "%s/n%d", store_root, n);
        char rm[1300]; snprintf(rm, sizeof rm, "rm -rf '%s'", sroot);
        if (system(rm) != 0) { }
        char mk[1300]; snprintf(mk, sizeof mk, "mkdir -p '%s'", store_root);
        if (system(mk) != 0) { }
        capsule_store_t *s = NULL;
        if (capsule_store_open(sroot, CAPSULE_STORE_CREATE, &s) != CAPSULE_OK)
            die("store_open");
        capsule_cap_t cap; uint64_t cid;
        uint64_t p0 = bu_now_ticks();
        cap_put(s, "/engine/seq0", "tenant:1", blob, got, n, &cap, &cid);
        uint64_t p1 = bu_now_ticks();
        size_t need = sizeof(lc_hdr_t) + got;
        uint8_t *hdrbuf = malloc(need);

        double *res = malloc(sizeof(double) * reps);
        int exact = 1;
        for (int r = 0; r < reps; r++) {
            llama_memory_seq_rm(llama_get_memory(e.ctx), 0, -1, -1);
            capsule_map_t m;
            uint64_t t0 = bu_now_ticks();
            if (capsule_attach(s, &cap, &m) != CAPSULE_OK) die("attach");
            size_t off = 0;
            for (uint32_t k = 0; k < m.nsegs && off < need; k++) {
                size_t take = m.seg_len[k];
                if (off + take > need) take = need - off;
                memcpy(hdrbuf + off, m.seg_base[k], take);
                off += take;
            }
            if (llama_state_seq_set_data(e.ctx, hdrbuf + sizeof(lc_hdr_t),
                                         got, 0) == 0) exact = 0;
            eng_refresh_logits(&e, 0, n - 1, toks[n - 1]);
            uint64_t t1 = bu_now_ticks();
            capsule_detach(s, &m);
            res[r] = bu_ticks_to_ns(t1 - t0);
        }

        bu_dist_t dp = bu_dist(pre, reps), dr = bu_dist(res, reps);
        printf("%s{\"n_tokens\":%d,\"engine_blob_bytes\":%zu,"
               "\"prefill_ms_p50\":%.3f,\"restore_ms_p50\":%.3f,"
               "\"speedup_p50\":%.2f,\"capsule_put_ms\":%.3f,"
               "\"state_set_ok\":%s}",
               i ? "," : "", n, got, dp.p50 / 1e6, dr.p50 / 1e6,
               dp.p50 / dr.p50, bu_ticks_to_ns(p1 - p0) / 1e6,
               exact ? "true" : "false");
        fflush(stdout);
        fprintf(stderr, "  n=%-6d engine prefill %8.1f ms  capsule restore %7.1f ms"
                        "  speedup %5.1fx  blob %.1f MiB\n",
                n, dp.p50 / 1e6, dr.p50 / 1e6, dp.p50 / dr.p50,
                got / 1048576.0);
        free(pre); free(res); free(blob); free(hdrbuf);
        capsule_store_close(s);
        llama_free(e.ctx);
    }
    printf("]}\n");
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

/* E13: request-level repeated-prefix loop.  Each measured request produces its
 * first generated token.  "prefill" recomputes the prefix; "native" restores
 * from an in-process byte buffer; "session" reads the same opaque llama.cpp
 * sequence-state blob from a plain file; "capsule" validates, maps and flattens
 * a sealed capsule before the identical engine restore.  The loop is
 * intentionally single-stream: it measures the serialization boundary, not a
 * production scheduler or paged GPU block manager. */
static int mode_requests(int argc, char **argv) {
    (void)argc;
    /* requests <gguf> <store> <ngl> <n_prompt> <reps> */
    const char *gguf = argv[2], *store_root = argv[3];
    int ngl = atoi(argv[4]), n = atoi(argv[5]), reps = atoi(argv[6]);
    if (n <= 0 || n >= MAX_PROMPT_TOK || reps < 2) die("bad request arguments");

    eng_t e = eng_open(gguf, n + 128, ngl);
    static llama_token toks[MAX_PROMPT_TOK];
    build_prompt(&e, n, toks);
    eng_prefill(&e, toks, n, 0);
    size_t blob_n = llama_state_seq_get_size(e.ctx, 0);
    uint8_t *blob = malloc(blob_n);
    size_t got = llama_state_seq_get_data(e.ctx, blob, blob_n, 0);
    if (!got) die("state_seq_get_data failed");

    capsule_store_t *s = NULL;
    if (capsule_store_open(store_root, CAPSULE_STORE_CREATE, &s) != CAPSULE_OK)
        die("store_open");
    capsule_cap_t cap; uint64_t cid;
    cap_put(s, "/engine/request-prefix", "tenant:1", blob, got, n, &cap, &cid);
    size_t need = sizeof(lc_hdr_t) + got;
    uint8_t *flat = malloc(need);
    uint8_t *session_blob = malloc(got);
    char session_path[1400];
    snprintf(session_path, sizeof session_path, "%s/native-session.bin", store_root);
    lc_hdr_t session_hdr = {LC_MAGIC, got, (uint64_t)n, 0};
    blob_write(session_path, &session_hdr, blob);

    double *prefill = malloc(sizeof(double) * (size_t)reps);
    double *native = malloc(sizeof(double) * (size_t)reps);
    double *session = malloc(sizeof(double) * (size_t)reps);
    double *capsule = malloc(sizeof(double) * (size_t)reps);
    int exact = 1;
    for (int r = 0; r < reps; r++) {
        llama_token a, b, c, d;
        uint64_t t0 = bu_now_ticks();
        eng_prefill(&e, toks, n, 0);
        eng_decode(&e, 0, n, 1, &a);
        uint64_t t1 = bu_now_ticks();
        prefill[r] = bu_ticks_to_ns(t1 - t0);

        llama_memory_seq_rm(llama_get_memory(e.ctx), 0, -1, -1);
        t0 = bu_now_ticks();
        if (!llama_state_seq_set_data(e.ctx, blob, got, 0)) die("native state set");
        eng_refresh_logits(&e, 0, n - 1, toks[n - 1]);
        eng_decode(&e, 0, n, 1, &b);
        t1 = bu_now_ticks();
        native[r] = bu_ticks_to_ns(t1 - t0);

        llama_memory_seq_rm(llama_get_memory(e.ctx), 0, -1, -1);
        t0 = bu_now_ticks();
        FILE *sf = fopen(session_path, "rb");
        lc_hdr_t sh;
        if (!sf || fread(&sh, sizeof sh, 1, sf) != 1 ||
            sh.magic != LC_MAGIC || sh.blob_bytes != got ||
            fread(session_blob, 1, got, sf) != got || fclose(sf) != 0)
            die("native session read");
        if (!llama_state_seq_set_data(e.ctx, session_blob, got, 0))
            die("native session state set");
        eng_refresh_logits(&e, 0, n - 1, toks[n - 1]);
        eng_decode(&e, 0, n, 1, &d);
        t1 = bu_now_ticks();
        session[r] = bu_ticks_to_ns(t1 - t0);

        llama_memory_seq_rm(llama_get_memory(e.ctx), 0, -1, -1);
        capsule_map_t m;
        t0 = bu_now_ticks();
        if (capsule_attach(s, &cap, &m) != CAPSULE_OK) die("request attach");
        size_t off = 0;
        for (uint32_t k = 0; k < m.nsegs && off < need; k++) {
            size_t take = m.seg_len[k];
            if (off + take > need) take = need - off;
            memcpy(flat + off, m.seg_base[k], take); off += take;
        }
        if (!llama_state_seq_set_data(e.ctx, flat + sizeof(lc_hdr_t), got, 0))
            die("capsule state set");
        eng_refresh_logits(&e, 0, n - 1, toks[n - 1]);
        eng_decode(&e, 0, n, 1, &c);
        capsule_detach(s, &m);
        t1 = bu_now_ticks();
        capsule[r] = bu_ticks_to_ns(t1 - t0);
        if (a != b || a != c || a != d) exact = 0;
    }

    bu_dist_t dp = bu_dist(prefill, reps), dn = bu_dist(native, reps);
    bu_dist_t ds = bu_dist(session, reps), dc = bu_dist(capsule, reps);
    printf("{\"experiment\":\"E13_request_level\",");
    bu_emit_meta_fields(stdout);
    printf(",\"model\":\"%s\",\"n_prompt\":%d,\"reps\":%d,"
           "\"engine_blob_bytes\":%zu,\"single_stream\":true,"
           "\"first_token_identical\":%s,\"prefill_ttft_ms_p50\":%.3f,"
           "\"prefill_ttft_ms_p95\":%.3f,\"native_ttft_ms_p50\":%.3f,"
           "\"native_ttft_ms_p95\":%.3f,\"session_ttft_ms_p50\":%.3f,"
           "\"session_ttft_ms_p95\":%.3f,\"capsule_ttft_ms_p50\":%.3f,"
           "\"capsule_ttft_ms_p95\":%.3f,\"capsule_over_native_p50\":%.3f,"
           "\"capsule_over_session_p50\":%.3f,"
           "\"capsule_speedup_over_prefill_p50\":%.3f,"
           "\"session_baseline\":\"plain-file llama.cpp sequence-state save/restore on the same hardware\","
           "\"scope\":\"single-stream repeated-prefix request loop; no concurrent scheduler or paged GPU-block claim\"}\n",
           gguf, n, reps, got, exact ? "true" : "false",
           dp.p50 / 1e6, dp.p95 / 1e6, dn.p50 / 1e6, dn.p95 / 1e6,
           ds.p50 / 1e6, ds.p95 / 1e6, dc.p50 / 1e6, dc.p95 / 1e6,
           dc.p50 / dn.p50, dc.p50 / ds.p50, dp.p50 / dc.p50);
    free(prefill); free(native); free(session); free(capsule);
    free(session_blob); free(flat); free(blob);
    capsule_store_close(s); eng_close(&e);
    return exact ? 0 : 1;
}

typedef struct {
    pthread_mutex_t mu; pthread_cond_t cv;
    int arrived, total, generation;
} req_barrier_t;

static void req_barrier_init(req_barrier_t *b, int total) {
    pthread_mutex_init(&b->mu, NULL); pthread_cond_init(&b->cv, NULL);
    b->arrived = 0; b->total = total; b->generation = 0;
}
static void req_barrier_wait(req_barrier_t *b) {
    pthread_mutex_lock(&b->mu); int g = b->generation;
    if (++b->arrived == b->total) {
        b->arrived = 0; b->generation++; pthread_cond_broadcast(&b->cv);
    } else while (g == b->generation) pthread_cond_wait(&b->cv, &b->mu);
    pthread_mutex_unlock(&b->mu);
}

typedef struct {
    eng_t *eng; const llama_token *toks; int n, reps, policy;
    const uint8_t *blob; size_t blob_n;
    capsule_store_t *store; const capsule_cap_t *cap;
    req_barrier_t *bar; double *lat; double elapsed_ns;
    llama_token first_token;
} mt_req_t;

static void *mt_req_worker(void *arg) {
    mt_req_t *w = arg;
    size_t need = sizeof(lc_hdr_t) + w->blob_n;
    uint8_t *flat = w->policy == 2 ? malloc(need) : NULL;
    req_barrier_wait(w->bar);
    uint64_t all0 = bu_now_ticks();
    for (int r = 0; r < w->reps; r++) {
        uint64_t t0 = bu_now_ticks(); llama_token tok;
        if (w->policy == 0) {
            eng_prefill(w->eng, w->toks, w->n, 0);
        } else {
            llama_memory_seq_rm(llama_get_memory(w->eng->ctx), 0, -1, -1);
            const uint8_t *state = w->blob;
            capsule_map_t m;
            if (w->policy == 2) {
                if (capsule_attach(w->store, w->cap, &m) != CAPSULE_OK)
                    die("concurrent attach");
                size_t off = 0;
                for (uint32_t k = 0; k < m.nsegs && off < need; k++) {
                    size_t take = m.seg_len[k];
                    if (off + take > need) take = need - off;
                    memcpy(flat + off, m.seg_base[k], take); off += take;
                }
                state = flat + sizeof(lc_hdr_t);
            }
            if (!llama_state_seq_set_data(w->eng->ctx, state, w->blob_n, 0))
                die("concurrent state set");
            eng_refresh_logits(w->eng, 0, w->n - 1, w->toks[w->n - 1]);
            if (w->policy == 2) capsule_detach(w->store, &m);
        }
        eng_decode(w->eng, 0, w->n, 1, &tok);
        uint64_t t1 = bu_now_ticks();
        if (r == 0) w->first_token = tok;
        w->lat[r] = bu_ticks_to_ns(t1 - t0);
    }
    w->elapsed_ns = bu_ticks_to_ns(bu_now_ticks() - all0);
    free(flat); return NULL;
}

/* E14: four independent llama.cpp contexts share a model and concurrently
 * issue repeated-prefix requests.  This is still not continuous batching,
 * but it exposes contention at the capsule attachment boundary. */
static int mode_requests_mt(int argc, char **argv) {
    (void)argc;
    /* requests-mt <gguf> <store> <ngl> <n_prompt> <streams> <reps> */
    const char *gguf = argv[2], *store_root = argv[3];
    int ngl = atoi(argv[4]), n = atoi(argv[5]);
    int streams = atoi(argv[6]), reps = atoi(argv[7]);
    if (streams < 2 || streams > 16 || reps < 2) die("bad mt arguments");
    struct llama_model *model = model_open(gguf, ngl);
    eng_t *engs = calloc((size_t)streams, sizeof *engs);
    llama_token **toks = calloc((size_t)streams, sizeof *toks);
    for (int i = 0; i < streams; i++) {
        engs[i] = ctx_open(model, n + 128);
        toks[i] = malloc(sizeof(llama_token) * MAX_PROMPT_TOK);
        build_prompt(&engs[i], n, toks[i]); eng_prefill(&engs[i], toks[i], n, 0);
    }
    size_t blob_n = llama_state_seq_get_size(engs[0].ctx, 0);
    uint8_t *blob = malloc(blob_n);
    if (!llama_state_seq_get_data(engs[0].ctx, blob, blob_n, 0)) die("mt get state");
    capsule_store_t *s = NULL;
    if (capsule_store_open(store_root, CAPSULE_STORE_CREATE, &s) != CAPSULE_OK)
        die("mt store open");
    capsule_cap_t cap; uint64_t cid;
    cap_put(s, "/engine/concurrent-prefix", "tenant:1", blob, blob_n, n, &cap, &cid);

    const char *names[] = {"prefill", "native", "capsule"};
    bu_dist_t dist[3]; double throughput[3]; llama_token reference = -1;
    int exact = 1;
    for (int policy = 0; policy < 3; policy++) {
        req_barrier_t bar; req_barrier_init(&bar, streams);
        mt_req_t *ws = calloc((size_t)streams, sizeof *ws);
        pthread_t *ths = calloc((size_t)streams, sizeof *ths);
        double *all = malloc(sizeof(double) * (size_t)streams * (size_t)reps);
        for (int i = 0; i < streams; i++) {
            ws[i] = (mt_req_t){&engs[i], toks[i], n, reps, policy, blob,
                blob_n, s, &cap, &bar, all + (size_t)i * reps, 0, 0};
            pthread_create(&ths[i], NULL, mt_req_worker, &ws[i]);
        }
        double max_elapsed = 0;
        for (int i = 0; i < streams; i++) {
            pthread_join(ths[i], NULL);
            if (ws[i].elapsed_ns > max_elapsed) max_elapsed = ws[i].elapsed_ns;
            if (policy == 0 && i == 0) reference = ws[i].first_token;
            if (ws[i].first_token != reference) exact = 0;
        }
        dist[policy] = bu_dist(all, (size_t)streams * reps);
        throughput[policy] = (double)(streams * reps) * 1e9 / max_elapsed;
        free(all); free(ws); free(ths);
    }
    printf("{\"experiment\":\"E14_concurrent_requests\",");
    bu_emit_meta_fields(stdout);
    printf(",\"model\":\"%s\",\"streams\":%d,\"reps_per_stream\":%d,"
           "\"n_prompt\":%d,\"engine_blob_bytes\":%zu,"
           "\"first_token_identical\":%s,\"arms\":[", gguf, streams,
           reps, n, blob_n, exact ? "true" : "false");
    for (int p = 0; p < 3; p++)
        printf("%s{\"name\":\"%s\",\"ttft_ms_p50\":%.3f,"
               "\"ttft_ms_p95\":%.3f,\"throughput_req_s\":%.3f}",
               p ? "," : "", names[p], dist[p].p50 / 1e6,
               dist[p].p95 / 1e6, throughput[p]);
    printf("],\"scope\":\"shared model with independent contexts; no continuous batching or paged GPU block manager\"}\n");
    for (int i = 0; i < streams; i++) { llama_free(engs[i].ctx); free(toks[i]); }
    free(engs); free(toks); free(blob); capsule_store_close(s);
    llama_model_free(model); llama_backend_free(); return exact ? 0 : 1;
}

int main(int argc, char **argv) {
    bu_time_init();
    llama_log_set(NULL, NULL);   /* silence engine chatter on stderr */
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  llamacap save    <gguf> <store> <n_prompt> <n_new> <ngl>\n"
            "  llamacap restore <gguf> <store> <n_prompt> <n_new> <ngl> <caphex>\n"
            "  llamacap bench   <gguf> <store> <ngl> <reps> <len1,len2,...>\n"
            "  llamacap requests <gguf> <store> <ngl> <n_prompt> <reps>\n"
            "  llamacap requests-mt <gguf> <store> <ngl> <n_prompt> <streams> <reps>\n"
            "  llamacap export  <gguf> <blobfile> <n_prompt> <n_new> <ngl>\n"
            "  llamacap import  <gguf> <blobfile> <n_new> <ngl>\n");
        return 2;
    }
    if (!strcmp(argv[1], "save"))    return mode_save(argc, argv);
    if (!strcmp(argv[1], "restore")) return mode_restore(argc, argv);
    if (!strcmp(argv[1], "bench"))   return mode_bench(argc, argv);
    if (!strcmp(argv[1], "requests")) return mode_requests(argc, argv);
    if (!strcmp(argv[1], "requests-mt")) return mode_requests_mt(argc, argv);
    if (!strcmp(argv[1], "export"))  return mode_export(argc, argv);
    if (!strcmp(argv[1], "import"))  return mode_import(argc, argv);
    die("unknown mode");
    return 2;
}
