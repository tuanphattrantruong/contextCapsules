/* hash.c — SHA-256 / HMAC-SHA-256 wrappers over CommonCrypto (in libSystem,
 * no extra link flags), plus the capsule chain rule (DESIGN.md §2.5):
 *   chain_0    = domain_seed = SHA256(domain string; NULL treated as "")
 *   chain_{i+1} = SHA256(chain_i || chunk_hash_{i+1})
 */
#include "store_internal.h"

#include <string.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonHMAC.h>

_Static_assert(sizeof(caps_sha256_ctx_t) >= sizeof(CC_SHA256_CTX),
               "opaque ctx too small for CC_SHA256_CTX");

void caps_sha256(const void *data, size_t n, uint8_t out[32]) {
  /* CC_LONG is 32-bit; every input here is <= 512 MiB (chunks are <= 2 MiB). */
  CC_SHA256(data, (CC_LONG)n, out);
}

void caps_sha256_init(caps_sha256_ctx_t *c) {
  CC_SHA256_Init((CC_SHA256_CTX *)c->opaque);
}

void caps_sha256_update(caps_sha256_ctx_t *c, const void *data, size_t n) {
  CC_SHA256_Update((CC_SHA256_CTX *)c->opaque, data, (CC_LONG)n);
}

void caps_sha256_final(caps_sha256_ctx_t *c, uint8_t out[32]) {
  CC_SHA256_Final(out, (CC_SHA256_CTX *)c->opaque);
}

void caps_hmac_sha256(const uint8_t key[32], const void *data, size_t n, uint8_t out[32]) {
  CCHmac(kCCHmacAlgSHA256, key, 32, data, n, out);
}

#pragma clang diagnostic pop

void caps_chain_step(const uint8_t chain[32], const uint8_t chunk_hash[32], uint8_t out[32]) {
  caps_sha256_ctx_t c;
  caps_sha256_init(&c);
  caps_sha256_update(&c, chain, 32);
  caps_sha256_update(&c, chunk_hash, 32);
  caps_sha256_final(&c, out);
}

void caps_domain_seed(const char *domain, uint8_t out[32]) {
  const char *d = domain ? domain : "";
  caps_sha256(d, strlen(d), out);
}
