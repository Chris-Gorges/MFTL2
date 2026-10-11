/**
 * @file coupling_memory.c
 * @ingroup mftl_coupling
 * @brief Per-rank module allocation accounting and hard capacity budget.
 */
#include "coupling_internal.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* A header preserves malloc's alignment for every C type. No allocation map,
 * per-object walk or caller-maintained capacity estimate is needed. */
typedef union
{
  max_align_t alignment;
  size_t bytes;
} memory_header;

static _Atomic size_t live_bytes, peak_bytes, live_blocks, memory_limit;

static void reserve_bytes(size_t bytes, const char *what)
{
  size_t old = atomic_load(&live_bytes);
  const size_t limit = atomic_load(&memory_limit);
  do
  {
    if (bytes > SIZE_MAX - old || (limit && (old > limit || bytes > limit - old)))
      MFTLCplMemoryAbort(what, bytes, old, limit);
  } while (!atomic_compare_exchange_weak(&live_bytes, &old, old + bytes));
  const size_t now = old + bytes;
  size_t peak = atomic_load(&peak_bytes);
  while (peak < now && !atomic_compare_exchange_weak(&peak_bytes, &peak, now)) { }
}

static size_t block_bytes(size_t bytes, const char *what)
{
  const size_t payload = bytes ? bytes : 1;
  if (payload > SIZE_MAX - sizeof(memory_header))
    MFTLCplMemoryAbort(what, bytes, atomic_load(&live_bytes), atomic_load(&memory_limit));
  return sizeof(memory_header) + payload;
}

void *MFTLCplAlloc(size_t bytes, const char *what)
{
  const size_t total = block_bytes(bytes, what);
  reserve_bytes(total, what);
  memory_header *h = malloc(total);
  if (!h) MFTLCplMemoryAbort(what, total, atomic_load(&live_bytes), atomic_load(&memory_limit));
  h->bytes = total;
  atomic_fetch_add(&live_blocks, 1);
  return h + 1;
}

void *MFTLCplCalloc(size_t count, size_t size, const char *what)
{
  if (size && count > SIZE_MAX / size)
    MFTLCplMemoryAbort(what, SIZE_MAX, atomic_load(&live_bytes), atomic_load(&memory_limit));
  const size_t bytes = count * size;
  void *p = MFTLCplAlloc(bytes, what);
  memset(p, 0, bytes);
  return p;
}

void MFTLCplFree(void *ptr)
{
  if (!ptr) return;
  memory_header *h = (memory_header *) ptr - 1;
  const size_t bytes = h->bytes;
  free(h);
  atomic_fetch_sub(&live_bytes, bytes);
  atomic_fetch_sub(&live_blocks, 1);
}

void *MFTLCplRealloc(void *ptr, size_t bytes, const char *what)
{
  if (!ptr) return MFTLCplAlloc(bytes, what);
  if (!bytes)
  {
    MFTLCplFree(ptr);
    return NULL;
  }
  memory_header *h = (memory_header *) ptr - 1;
  const size_t old = h->bytes, total = block_bytes(bytes, what);
  if (total > old) reserve_bytes(total - old, what);
  memory_header *next = realloc(h, total);
  if (!next) MFTLCplMemoryAbort(what, total, atomic_load(&live_bytes), atomic_load(&memory_limit));
  next->bytes = total;
  if (old > total) atomic_fetch_sub(&live_bytes, old - total);
  return next + 1;
}

int MFTLCouplerSetMemoryLimit(size_t bytes)
{
  if (bytes && bytes < atomic_load(&live_bytes)) return MFTL_ERROR;
  atomic_store(&memory_limit, bytes);
  return MFTL_SUCCESS;
}

void MFTLCouplerGetMemoryStats(MFTL_CouplingMemoryStats *stats)
{
  if (!stats) return;
  stats->current_bytes = atomic_load(&live_bytes);
  stats->peak_bytes = atomic_load(&peak_bytes);
  stats->limit_bytes = atomic_load(&memory_limit);
  stats->live_allocations = atomic_load(&live_blocks);
}

void MFTLCouplerResetMemoryPeak(void) { atomic_store(&peak_bytes, atomic_load(&live_bytes)); }
