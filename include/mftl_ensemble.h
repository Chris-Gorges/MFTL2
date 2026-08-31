/**
 * @file mftl_ensemble.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief The top-level library handle: the MPI/physical world plus its objects.
 *
 * The ensemble owns exactly one MFTL_Domain (MPI communicators, process grid,
 * global bounds, periodicity) and a list of MFTL_Object*. Objects borrow the
 * domain; they never own MPI state themselves.
 *
 * ## WHERE A GEOMETRY FUNCTION LIVES, AND WHY IT IS NOT ARBITRARY
 *
 * Geometry appears in two headers and the split is by SCOPE, not by entity type:
 *
 *   mftl_geometry.h / src/geometry/     ONE object, LOCAL, no MPI.
 *                                       MFTLComputeTriangleGeometry
 *                                       MFTLComputeVertexNormals
 *
 *   mftl_ensemble.h / src/ensemble/     EVERY object, COLLECTIVE.
 *                                       MFTLEnsembleUpdate
 *                                       MFTLEnsembleUpdateVertexNormals
 *
 * An MFTLEnsemble* function is a loop over the object list plus exactly the
 * collective its result needs, and nothing else -- the arithmetic belongs to
 * src/geometry/. The naming carries the same information: MFTLCompute* is local
 * and per object, MFTLEnsembleUpdate* is collective and ensemble-wide, so the
 * call site says which one it is without looking anything up.
 *
 * There is ONE deliberate exception, and it is a performance decision rather
 * than an oversight: MFTLEnsembleUpdate's kernel is a file-static function in
 * src/ensemble/ensemble_update.c rather than in src/geometry/, because it
 * carries nine live reduction accumulators. Moving it across a translation-unit
 * boundary would force them into an address-taken struct, which
 * docs/performance-analysis.md §6 measures at 15 % -- so the kernel stays next
 * to the reduction it feeds. Tidiness lost to a number.
 */

#ifndef MFTL_ENSEMBLE_H
#define MFTL_ENSEMBLE_H

#include "mftl_types.h"

/**
 * @brief Creates an ensemble: the global domain and the MPI topology.
 *
 * 1. Duplicates @p comm, so the library owns a communicator no user code can
 *    interfere with.
 * 2. Validates the process grid against the communicator size.
 * 3. Builds the Cartesian communicator, the rank lookup table, the grid factors
 *    and the (up to) 26-neighbour distributed graph.
 * 4. Allocates the object list.
 *
 * @note Collective: every rank of @p comm must call it.
 *
 * @param comm The base communicator (usually MPI_COMM_WORLD).
 * @param cuts Process grid {nx, ny, nz}; the product must equal the comm size.
 * @param global_min Minimum physical coordinates.
 * @param global_max Maximum physical coordinates.
 * @param is_periodic Per-axis periodicity. MFTL_NULL means non-periodic.
 * @param initial_object_capacity Initial size of the object list; it grows on demand.
 * @return The new ensemble, or MFTL_NULL on failure.
 */
MFTL_Ensemble *MFTLEnsembleCreate(MPI_Comm comm, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3], int initial_object_capacity);

/**
 * @brief Adds an object and links it to the ensemble's domain.
 *
 * After this call `object->domain` points at the ensemble's domain and the
 * ensemble owns the object: MFTLEnsembleDestroy() will destroy it.
 *
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments.
 *
 * @param ensemble The ensemble, which takes OWNERSHIP of @p object.
 * @param object   The mesh. Its `domain` pointer is set here.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 */
int MFTLEnsembleAddObject(MFTL_Ensemble *ensemble, MFTL_Object *object);

/**
 * @brief Destroys an ensemble, its domain and every object it owns. Safe on NULL.
 *
 * @param p_ens Address of the ensemble pointer. Set to #MFTL_NULL on return.
 */
void MFTLEnsembleDestroy(MFTL_Ensemble **p_ens);

/**
 * @brief Sums the owned entity counts of every object across all ranks.
 *
 * One MPI_Allreduce covers the whole ensemble, so the message count is O(1) in
 * the number of objects and the volume is 2 ints per object. With ownership as
 * an array prefix the local half is two array reads rather than a scan over
 * every entity testing a flag.
 *
 * @note Collective. Assumes every rank holds the same object list in the same
 *       order, which is what the object `global_id` is for.
 *
 * @param ensemble The ensemble. Every object's `global_n_*` and the ensemble's
 *                 `total_global_triangles` are refreshed.
 */
void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble *ensemble);

/**
 * @brief The per-timestep workhorse: all triangle geometry AND the quality
 *        statistics, in one streaming pass.
 *
 * For every locally owned triangle of every object it writes `tris.area`,
 * `tris.normal` and `tris.center`, and accumulates from the same loads:
 *
 *   min/max/avg triangle area
 *   min/max/avg mean-ratio quality   q = 4*sqrt(3)*A / (a^2+b^2+c^2), 1 = equilateral
 *   min/max/avg squared edge length
 *
 * It is cheaper than separate geometry and quality passes because the three
 * gathered positions and their minimum-image edges serve both outputs. Compact
 * `tris.verts` is streamed; the edge statistics come from vectors already in
 * registers instead of a second pass.
 *
 * @note Requires current derived connectivity -- call
 *       MFTLObjectBuildDerived() after a topology change. Vertex motion needs no
 *       refresh. Asserted in Debug.
 * @note GHOST triangles are not touched. Anything preceding a redistribution
 *       must additionally refresh the ghosts' centres with
 *       MFTLComputeTriangleGeometry(obj, obj->tris.n).
 * @note Collective: exactly three reductions regardless of how many objects the
 *       ensemble holds, and they are issued together as non-blocking collectives
 *       so their latencies overlap instead of adding up.
 *
 * @param ensemble The ensemble. Triangle geometry and quality statistics are
 *                 written for every object.
 */
void MFTLEnsembleUpdate(MFTL_Ensemble *ensemble);

/**
 * @brief The per-vertex counterpart: exact vertex normals everywhere, in one call.
 *
 * For every object it runs MFTLComputeVertexNormals(), then performs ONE ghost
 * exchange for the whole ensemble. After it, `verts.normal` is exact for every
 * vertex in every pool -- owned and ghost alike -- so a consumer does not have to
 * know where the ownership prefix ends.
 *
 * ── Why the collective is an exchange and not a reduction ──────────────────
 *
 * MFTLEnsembleUpdate reduces, because a minimum over one rank's partition is not
 * the mesh's minimum. Nothing of that kind applies here: a vertex normal is a
 * function of its one-ring, and a vertex-complete halo puts every incident
 * triangle on the rank that owns the vertex. So the owned normals are already
 * exact and no reduction could improve them.
 *
 * What is NOT already right is the ghost half. MFTLComputeVertexNormals
 * accumulates over all locally held triangles, and a ghost vertex's far-side
 * faces are not here -- it ends up holding a partial sum that is correctly
 * normalised and quietly wrong. That is a trap, because a partial sum looks
 * exactly like a real normal, and closing it is the entire reason this function
 * exists. Use MFTLComputeVertexNormals() directly only if you are certain you
 * will read owned vertices and nothing else.
 *
 * @note Requires current derived connectivity. After a redistribution it is
 *       already current; vertex motion does not invalidate it.
 * @note Collective: exactly three neighbourhood collectives for the whole
 *       ensemble, sized by the ghost count, none of them scaling with nproc.
 * @return MFTL_SUCCESS, or MFTL_ERROR if the exchange could not serve every
 *         ghost (the same value on every rank is NOT guaranteed -- it reports
 *         this rank's outcome).
 *
 * @param ensemble The ensemble.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the ghost exchange failed.
 */
int MFTLEnsembleUpdateVertexNormals(MFTL_Ensemble *ensemble);

/* ═══════════════════════════════════════════════════════════════════════════
 * Decomposition metrics
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The narrowest rank box over the three axes: \f$\min_k L_k / \mathrm{cuts}_k\f$.
 *
 * The length scale every parallel scheme with a boundary band is measured
 * against -- the distributed remesher's `h > 4*delta` precondition and its
 * dispersed/spanning classification are both statements about this number.
 *
 * Axes with fewer than two cuts are skipped rather than counted as zero: an
 * effectively 2D decomposition has no interior plane on the flat axis, so that
 * axis constrains nothing. Returns 0 only when NO axis is cut, i.e. nproc == 1.
 *
 * @param domain The MPI/physical world.
 * @return The narrowest cut rank-box width, or 0 when no axis is cut.
 *
 * @rank_local Every rank computes the same value from the same fields.
 */
double MFTLDomainGetHMin(const MFTL_Domain *domain);

#endif  // MFTL_ENSEMBLE_H
