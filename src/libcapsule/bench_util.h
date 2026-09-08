/* bench_util.h — header-only helpers shared by the package-1 driver programs
 * (test_capsule.c, bench_micro.c, sec_c.c). NOT part of the frozen public ABI.
 *
 * Provides: mach_absolute_time timing (timebase queried at runtime), a small
 * JSON emitter, `date -u` / `uname -a` capture via popen (literal output, per the
 * honesty contract), pseudorandom buffer fill (never memset patterns — DESIGN
 * benchmark hazard note), and simple distribution stats. Everything is static so
 * each driver gets its own copy; no extra objects to link.
 */
#ifndef CAPS_BENCH_UTIL_H
#define CAPS_BENCH_UTIL_H

#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- timing ---- */

static __attribute__((unused)) mach_timebase_info_data_t g_tb;

static __attribute__((unused)) void bu_time_init(void) {
  mach_timebase_info(&g_tb);
}

static inline uint64_t bu_now_ticks(void) { return mach_absolute_time(); }

/* convert a tick delta to nanoseconds using the runtime timebase (125/3 here) */
static inline double bu_ticks_to_ns(uint64_t ticks) {
  return (double)ticks * (double)g_tb.numer / (double)g_tb.denom;
}

/* ---- distribution stats over an array of ns samples ---- */

typedef struct {
  size_t n;
  double mean, std, p50, p95, min, max;
} bu_dist_t;

static __attribute__((unused)) int bu_cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* Computes stats; sorts `v` in place. */
static __attribute__((unused)) bu_dist_t bu_dist(double *v, size_t n) {
  bu_dist_t d;
  memset(&d, 0, sizeof d);
  d.n = n;
  if (n == 0) return d;
  qsort(v, n, sizeof(double), bu_cmp_double);
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) sum += v[i];
  d.mean = sum / (double)n;
  double var = 0.0;
  for (size_t i = 0; i < n; i++) {
    double dv = v[i] - d.mean;
    var += dv * dv;
  }
  d.std = (n > 1) ? sqrt(var / (double)n) : 0.0;   /* population std */
  d.min = v[0];
  d.max = v[n - 1];
  d.p50 = v[n / 2];
  d.p95 = v[(size_t)(0.95 * (double)n) < n ? (size_t)(0.95 * (double)n) : n - 1];
  return d;
}

/* ---- JSON emit (to a FILE*) ---- */

static __attribute__((unused)) void bu_json_dist(FILE *f, const char *key, bu_dist_t d) {
  fprintf(f,
          "\"%s\": {\"n\": %zu, \"mean\": %.3f, \"std\": %.3f, \"p50\": %.3f, "
          "\"p95\": %.3f, \"min\": %.3f, \"max\": %.3f}",
          key, d.n, d.mean, d.std, d.p50, d.p95, d.min, d.max);
}

/* Capture literal stdout of a shell command (e.g. "date -u"), trimmed of the
 * trailing newline, into a caller buffer. Returns buf. On failure records an
 * explicit marker rather than fabricating output. */
static __attribute__((unused)) const char *bu_capture(const char *cmd, char *buf, size_t n) {
  buf[0] = '\0';
  FILE *p = popen(cmd, "r");
  if (!p) {
    snprintf(buf, n, "<%s failed to run>", cmd);
    return buf;
  }
  size_t got = fread(buf, 1, n - 1, p);
  buf[got] = '\0';
  pclose(p);
  /* trim trailing newlines/spaces */
  while (got > 0 && (buf[got - 1] == '\n' || buf[got - 1] == '\r' ||
                     buf[got - 1] == ' ')) {
    buf[--got] = '\0';
  }
  return buf;
}

/* Emit a JSON string value with minimal escaping (\\ and " and control). */
static __attribute__((unused)) void bu_json_str(FILE *f, const char *s) {
  fputc('"', f);
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    unsigned char c = *p;
    if (c == '"' || c == '\\') {
      fputc('\\', f);
      fputc(c, f);
    } else if (c == '\n') {
      fputs("\\n", f);
    } else if (c == '\t') {
      fputs("\\t", f);
    } else if (c < 0x20) {
      fprintf(f, "\\u%04x", c);
    } else {
      fputc(c, f);
    }
  }
  fputc('"', f);
}

/* Emit the shared meta block. Caller opens the object and supplies experiment,
 * cmd, iterations note; this writes date_u/uname_a/arch/timer. */
static __attribute__((unused)) void bu_emit_meta_fields(FILE *f) {
  char dbuf[256], ubuf[512];
  bu_capture("date -u", dbuf, sizeof dbuf);
  bu_capture("uname -a", ubuf, sizeof ubuf);
  fprintf(f, "\"date_u\": ");
  bu_json_str(f, dbuf);
  fprintf(f, ", \"uname_a\": ");
  bu_json_str(f, ubuf);
#if defined(__aarch64__) || defined(__arm64__)
  const char *arch = "arm64";
#elif defined(__x86_64__)
  const char *arch = "x86_64";
#else
  const char *arch = "unknown";
#endif
  fprintf(f, ", \"arch\": ");
  bu_json_str(f, arch);
  fprintf(f, ", \"timer\": \"mach_absolute_time\", \"timebase_numer\": %u, "
             "\"timebase_denom\": %u",
          g_tb.numer, g_tb.denom);
}

/* ---- pseudorandom fill (xorshift; NOT memset patterns; DESIGN hazard note) ---- */

static __attribute__((unused)) void bu_fill_random(void *dst, size_t n, uint64_t seed) {
  uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
  uint8_t *p = (uint8_t *)dst;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    memcpy(p + i, &x, 8);
  }
  for (; i < n; i++) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    p[i] = (uint8_t)x;
  }
}

/* volatile checksum so -O2 cannot dead-store-eliminate a consumed buffer */
static __attribute__((unused)) uint64_t bu_checksum(const void *src, size_t n) {
  const volatile uint8_t *p = (const volatile uint8_t *)src;
  uint64_t h = 1469598103934665603ULL;   /* FNV-ish */
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

#endif /* CAPS_BENCH_UTIL_H */
