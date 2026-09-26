/**
 * @file mftl_core.h
 * @author Christian Gorges
 * @date 2026-08-14
 *
 * @brief Convenience umbrella header.
 *
 * Includes every public MFTL header so that application code can get the whole
 * library with a single include. Library sources should NOT use this -- each .c
 * file includes its own module header first, plus exactly the module headers it
 * actually needs. That keeps the dependency graph visible and acyclic:
 *
 *   layer 0   mftl_constants.h  mftl_memory.h  mftl_vectoroperations.h
 *             mftl_arrayfunctions.h
 *   layer 1   mftl_types.h
 *   layer 2   mftl_topology.h   mftl_geometry.h
 *   layer 3   mftl_object.h     mftl_ensemble.h   mftl_meshgen.h
 *             mftl_communication.h                mftl_io.h
 *             mftl_remeshing.h                    mftl_bvh.h
 *   layer 4   mftl_core.h  (this file)
 *
 * A header may only include headers from a strictly lower layer.
 */

#ifndef MFTL_CORE_H
#define MFTL_CORE_H

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_vectoroperations.h"

#include "mftl_types.h"

#include "mftl_geometry.h"
#include "mftl_topology.h"

#include "mftl_bvh.h"
#include "mftl_communication.h"
#include "mftl_ensemble.h"
#include "mftl_io.h"
#include "mftl_meshgen.h"
#include "mftl_object.h"
#include "mftl_remeshing.h"
#include "mftl_topology_change.h"

#endif  // MFTL_CORE_H
