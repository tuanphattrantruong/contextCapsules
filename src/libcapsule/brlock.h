/* brlock.h — sharded reader / exclusive writer lock for the capsule store.
 *
 * Why not pthread_rwlock_t: every reader increments one shared reader counter,
 * so on a read-mostly workload -- which is exactly what capability validation
 * is -- all cores contend on a single cache line. Measured on a 32-core M3
 * Ultra, capsule_stat throughput under pthread_rwlock peaked at 4 threads and
 * then fell by 5x (E7).
 *
 * This is the classic big-reader lock: readers touch only their own padded
 * shard, writers set a flag and drain every shard. Readers are therefore
 * cache-line private in the common case; writers pay O(shards).
 *
 * Correctness sketch. A reader publishes its presence (fetch_add on its shard)
 * and then re-checks the writer flag; if a writer is in flight it withdraws and
 * waits. A writer sets the flag and then waits for every shard to reach zero.
 * Because both orders are seq_cst, a reader that observes flag==0 after its
 * increment cannot be missed by a writer that observes all-shards-zero after
 * setting the flag: one of the two must see the other. Writers are serialized
 * among themselves by a mutex, so the lock is exclusive for them.
 */
#ifndef CAPS_BRLOCK_H
#define CAPS_BRLOCK_H

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#define CAPS_BR_SHARDS 64
#define CAPS_BR_LINE 128

typedef struct {
  _Atomic int count;
  char pad[CAPS_BR_LINE - sizeof(_Atomic int)];
} caps_br_shard_t;

typedef struct {
  caps_br_shard_t shard[CAPS_BR_SHARDS];
  _Atomic int writer;
  pthread_mutex_t wmutex;
  int nshards;                 /* live shard count; 1 == single-counter mode */
} caps_brlock_t;

/* Number of live shards. CAPSULE_BR_SHARDS=1 collapses every reader onto one
 * cache line, which is precisely how a pthread_rwlock_t behaves; it exists so
 * the scalability ablation in E7 is reproducible from a single build. */
static inline int caps_br_nshards_env(void) {
  const char *e = getenv("CAPSULE_BR_SHARDS");
  if (!e) return CAPS_BR_SHARDS;
  int v = atoi(e);
  if (v < 1) v = 1;
  if (v > CAPS_BR_SHARDS) v = CAPS_BR_SHARDS;
  return v;
}

static inline void caps_br_init(caps_brlock_t *l) {
  for (int i = 0; i < CAPS_BR_SHARDS; i++)
    atomic_init(&l->shard[i].count, 0);
  atomic_init(&l->writer, 0);
  l->nshards = caps_br_nshards_env();
  pthread_mutex_init(&l->wmutex, NULL);
}

static inline void caps_br_destroy(caps_brlock_t *l) {
  pthread_mutex_destroy(&l->wmutex);
}

/* Stable per-thread shard index; assigned on first use, never reused-hot. */
static inline int caps_br_myshard(int nshards) {
  static _Atomic unsigned next = 0;
  static _Thread_local int mine = -1;
  if (nshards <= 1) return 0;
  if (mine < 0)
    mine = (int)(atomic_fetch_add_explicit(&next, 1u, memory_order_relaxed)
                 % CAPS_BR_SHARDS);
  return mine % nshards;
}

static inline int caps_br_rlock(caps_brlock_t *l) {
  int i = caps_br_myshard(l->nshards);
  for (;;) {
    atomic_fetch_add_explicit(&l->shard[i].count, 1, memory_order_seq_cst);
    if (atomic_load_explicit(&l->writer, memory_order_seq_cst) == 0) return i;
    atomic_fetch_sub_explicit(&l->shard[i].count, 1, memory_order_seq_cst);
    while (atomic_load_explicit(&l->writer, memory_order_acquire) != 0)
      sched_yield();
  }
}

static inline void caps_br_runlock(caps_brlock_t *l, int i) {
  atomic_fetch_sub_explicit(&l->shard[i].count, 1, memory_order_seq_cst);
}

static inline void caps_br_wlock(caps_brlock_t *l) {
  pthread_mutex_lock(&l->wmutex);
  atomic_store_explicit(&l->writer, 1, memory_order_seq_cst);
  for (int i = 0; i < CAPS_BR_SHARDS; i++)
    while (atomic_load_explicit(&l->shard[i].count, memory_order_seq_cst) != 0)
      sched_yield();
}

static inline void caps_br_wunlock(caps_brlock_t *l) {
  atomic_store_explicit(&l->writer, 0, memory_order_seq_cst);
  pthread_mutex_unlock(&l->wmutex);
}

#endif /* CAPS_BRLOCK_H */
