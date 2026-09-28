/**
 * @file comm_stats.c
 * @author Christian Gorges
 * @date 2026-08-25
 * @ingroup mftl_communication
 *
 * @brief The rank-local accumulator behind #MFTL_RedistributeStats.
 *
 * A file-scope singleton written through the CS_* macros of comm_internal.h
 * (the library is process-per-rank, so one accumulator per process suffices).
 * It takes no collective, so leaving it on never synchronises what it measures.
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
