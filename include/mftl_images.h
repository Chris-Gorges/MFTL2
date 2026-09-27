/*
 * MFTL2 periodic — image counters, homological winding, canonicalisation, cached per-axis wrap
 * flags and the body reference point (plans/16 §3.1–§3.5).
 *
 * Every vertex stores its wrapped position pos (in [min, max) on periodic axes) and integer
 * image counters img; X = pos + img ⊙ L is its continuous coordinate. Helpers for the
 * displacement rule and edge lattice shifts are in mftl_geometry.h (MFTLWrapPosImg,
 * MFTLImgFollow, MFTLWrapContinuous, MFTLVertexRelative, MFTLEdgeShift).
 */
#ifndef MFTL_IMAGES_H
#define MFTL_IMAGES_H

#include "mftl_core.h"

/**
 * @brief Canonicalises the image counters of a body (collective, §3.3).
 *
 * Gathers the edge graph with its lattice shifts to rank 0, builds a deterministic BFS spanning
 * tree per connected component (root = smallest vertex gid), sets img' = img − φ (tree edges
 * get shift 0, positions unchanged, the root keeps its counter), records the cycle residuals
 * (winding lattice) and the cached per-axis wrap flags, and applies the new counters to every
 * stored copy. Duplicate authoritative copies of a vertex must agree on img (collective error
 * otherwise). No-op (flags 0) on a non-periodic domain. Idempotent.
 */
int MFTLCanonicaliseImages(MFTL_Object *obj);

/** @brief As MFTLCanonicaliseImages, but every component's root counter is brought to 0
 *  (whole-body lattice re-gauge; keeps counters bounded). */
int MFTLObjectRegaugeImages(MFTL_Object *obj);

/** @brief Collective: canonicalises the body if its winding state is unknown on any rank
 *  (never canonicalised, or invalidated by MFTLBuildTopologyFromTriples, MFTLObjectClear or a
 *  staged MFTLObjectAppendTriangleCopy). Every wrap-sensitive collective operation (integrals,
 *  point queries, phase sets, remesh volume targets, transforms) calls it first, so an unknown
 *  winding state can never be mistaken for COMPACT. No communication on non-periodic domains. */
int MFTLObjectEnsureWinding(MFTL_Object *obj);

/** @brief Cached wrap flags; MFTL_ERROR (flags 0) before the first canonicalisation. */
int MFTLObjectWrapFlags(const MFTL_Object *obj, int wrap[3]);

/** @brief 1 if the body is KNOWN to be compact along every periodic axis (always 1
 *  non-periodic; 0 while the winding state is unknown — call MFTLObjectEnsureWinding). */
int MFTLObjectIsCompact(const MFTL_Object *obj);

/** @brief 1 if the body encloses a volume by itself (closed and compact): the class guard of
 *  the remesh drivers' volume target / recovery (plans/16 §9, 16B). */
int MFTLObjectHasVolume(const MFTL_Object *obj);

/** @brief Collective debug check: no cut edge along an axis whose wrap flag is clear. */
int MFTLCheckWrapFlags(MFTL_Object *obj, int verbose);

/** @brief Collective: number of cut half-edges (s_k != 0) per axis over authoritative triangles. */
int MFTLCountCutEdges(MFTL_Object *obj, int count[3]);

/** @brief Collective body reference point (§3.5): the lowest-gid authoritative vertex.
 *  Returns 0 for an empty body. */
int MFTLObjectReference(const MFTL_Object *obj, double pos_ref[3], int img_ref[3]);

#endif
