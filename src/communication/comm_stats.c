/**
 * @file comm_stats.c
 * @author Christian Gorges
 * @date 2026-08-25
 * @ingroup mftl_communication
 *
 * @brief The one accumulator behind #MFTL_RedistributeStats.
 *
 * There is exactly one, it is rank-local, and it is written from every
 * translation unit of this module through the CS_BEGIN/CS_END macros in
 * comm_internal.h. Nothing here takes a collective: a reduction inside the
 * measurement would synchronise the phases it is trying to time, and the whole
 * point of these counters is that they are cheap enough to leave on.
 *
 * A singleton rather than a `MFTL_RedistributeStats *` parameter, which is what
 * the remesher does. The remesher's call sites are a handful; the
 * redistribution's are some sixty across tests and examples, and threading a
 * diagnostic through four public signatures to reach them is a worse trade than
 * one file-scope object in a library that is process-per-rank by construction.
 */

#include <string.h>

#include "comm_internal.h"

/** @brief The accumulator. Zero-initialised at load, so a caller that never
 *         resets still reads a valid (whole-run) total. */
MFTL_RedistributeStats MFTLCommStats;

void MFTLRedistributeStatsReset(void) { memset(&MFTLCommStats, 0, sizeof(MFTLCommStats)); }

void MFTLRedistributeGetStats(MFTL_RedistributeStats *out)
{
  MFTL_ASSERT(out != MFTL_NULL);
  *out = MFTLCommStats;
}
