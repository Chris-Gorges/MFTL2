/**
 * @file coupling_index.c
 * @ingroup mftl_coupling
 * @brief Compact bounds over spatially ordered tiles, with stack-free traversal.
 */
#include "coupling_internal.h"
#include "mftl_memory.h"
#include <float.h>
#include <stdlib.h>

void MFTLCplBoxTreeBuild(cpl_box_tree *t, const double *data, int64_t n, int group, bool points)
{
  t->group = group;
  t->count = (n + group - 1) / group;
  t->base = 1;
  while (t->base < t->count) t->base *= 2;
  t->box = MFTLCplAlloc(12 * (size_t) t->base * sizeof(double), "coupling spatial index");
  for (int64_t i = t->base; i < 2 * t->base; ++i)
  {
    double *b = &t->box[6 * i];
    for (int k = 0; k < 3; ++k) b[k] = DBL_MAX, b[k + 3] = -DBL_MAX;
    const int64_t start = (i - t->base) * group, end = MFTL_MIN(start + group, n);
    for (int64_t j = start; j < end; ++j)
      for (int k = 0; k < 3; ++k)
      {
        const double lo = data[(points ? 3 : 6) * j + k];
        const double hi = data[(points ? 3 : 6) * j + k + (points ? 0 : 3)];
        b[k] = fmin(b[k], lo);
        b[k + 3] = fmax(b[k + 3], hi);
      }
  }
  for (int64_t i = t->base - 1; i > 0; --i)
    for (int k = 0; k < 3; ++k)
    {
      t->box[6 * i + k] = fmin(t->box[12 * i + k], t->box[12 * i + 6 + k]);
      t->box[6 * i + 3 + k] = fmax(t->box[12 * i + 3 + k], t->box[12 * i + 9 + k]);
    }
}

static int64_t next_subtree(int64_t node)
{
  while (node > 1 && (node & 1)) node /= 2;
  return node > 1 ? node + 1 : 0;
}

int64_t MFTLCplBoxTreeNext(const cpl_box_tree *t, const double lo[3], const double hi[3], int64_t *cursor)
{
  while (*cursor)
  {
    const int64_t node = *cursor;
    const double *b = &t->box[6 * node];
    if (cpl_box_overlap(b, b + 3, lo, hi))
    {
      if (node < t->base)
      {
        *cursor = 2 * node;
        continue;
      }
      *cursor = next_subtree(node);
      return node - t->base;
    }
    *cursor = next_subtree(node);
  }
  return -1;
}
