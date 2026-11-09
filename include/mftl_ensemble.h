/**
 * @file mftl_ensemble.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief The library handle: the MPI/physical domain plus the bodies this rank
 *        holds, the per-body collectives, and the collective geometry passes.
 *
 * An #MFTL_Ensemble owns one #MFTL_Domain and a gid-sorted list of the
 * #MFTL_Object s this rank holds entities of. Geometry is split by SCOPE:
 * `MFTLCompute*` (mftl_geometry.h) is one object, local, no MPI;
 * `MFTLEnsembleUpdate*` (here) loops over the objects and adds exactly the
 * collective the result needs -- a REDUCTION for the triangle quality
 * statistics, an EXCHANGE for vertex normals. The one exception, for speed:
 * MFTLEnsembleUpdate()'s kernel is file-static in src/ensemble/ensemble_update.c
 * because its nine reduction accumulators must stay in registers (an
 * address-taken struct across a translation unit measured 15 % slower,
 * docs/analysis.md §1.6). Theory: src/ensemble/README.md.
 */

#ifndef MFTL_ENSEMBLE_H
#define MFTL_ENSEMBLE_H

#include "mftl_geometry.h"
#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Creates an ensemble: the global domain and its MPI topology.
 *
 * Duplicates @p comm (the library communicates only on its own duplicate),
 * validates `cuts[0]*cuts[1]*cuts[2] == size(comm)` and the bounds, then builds
 * the Cartesian communicator (no rank reordering), the `rank_map` lookup table,
 * the grid factors and the distributed graph over the up to 26 distinct
 * neighbouring cells (periodic wrap-around included).
 *
 * @param comm                    Base communicator, usually `MPI_COMM_WORLD`.
 * @param cuts                    Process grid {nx, ny, nz}; product = communicator size.
 * @param global_min              Lower corner of the domain box.
 * @param global_max              Upper corner; `global_max >= global_min` per axis.
 * @param is_periodic             Per-axis periodicity, or #MFTL_NULL for none.
 * @param initial_object_capacity Initial length of the object list (grows on demand).
 * @return The new ensemble, or #MFTL_NULL on invalid input or MPI failure.
 *
 * @collective
 */
MFTL_Ensemble *MFTLEnsembleCreate(MPI_Comm comm, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3], int initial_object_capacity);

/**
 * @brief Adds an object, links it to the domain, and gives it its body id.
 *
 * The ensemble takes ownership (MFTLEnsembleDestroy() destroys the object).
 *
 * @warning `object->global_id` is OVERWRITTEN with `ens->n_bodies_global++`, so
 *          body ids are dense in add order. Every rank must add the same bodies
 *          in the same order, so that one id names one body everywhere.
 *
 * @param ensemble The ensemble, which takes ownership of @p object.
 * @param object   The mesh; its `domain` and `global_id` are set.
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
 * @brief MFTLObjectMarkChanged() on every body this rank holds.
 *
 * What a solver calls after its advection loop has written `verts.pos`, so that
 * anything holding a description of the old geometry -- a coupling binding --
 * reports itself stale rather than being applied to the new one.
 *
 * @param ens The ensemble; #MFTL_NULL is a no-op.
 *
 * @rank_local
 */
void MFTLEnsembleMarkChanged(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * The local object directory
 *
 * `local_objects` holds the bodies THIS RANK has entities of, sorted by
 * ascending global id (strictly). Hence: gid -> slot is a binary search (the id
 * space is global and dense, the holding sparse, so the only array alternative
 * would be O(bodies in the ensemble)); local iteration visits bodies in global
 * order; and two sorted lists join with a merge. Insert and erase are ordered
 * memmoves, never swap-with-tail.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Slot of body @p body_gid in the local list, by binary search.
 * @param ens      The ensemble.
 * @param body_gid The body id.
 * @return The slot, or #MFTL_ERROR if this rank holds nothing of that body.
 */
int MFTLEnsembleFindLocalSlot(const MFTL_Ensemble *ens, int body_gid);

/**
 * @brief The local object for body @p body_gid, or #MFTL_NULL if absent.
 * @param ens      The ensemble.
 * @param body_gid The body id.
 * @return The object, or #MFTL_NULL.
 */
MFTL_Object *MFTLEnsembleFindLocalObject(const MFTL_Ensemble *ens, int body_gid);

/**
 * @brief Inserts @p object into the local list, keeping it sorted by body id.
 *
 * Takes ownership. Unlike MFTLEnsembleAddObject() it does NOT assign an id; the
 * object must already carry its own (a body arriving in a redistribution).
 *
 * @param ens    The ensemble, which takes ownership of @p object.
 * @param object The mesh; its `global_id` must not already be present.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 *
 * @rank_local
 */
int MFTLEnsembleInsertLocalObject(MFTL_Ensemble *ens, MFTL_Object *object);

/**
 * @brief Removes local slot @p slot, keeping the list sorted. Does NOT destroy
 *        the object; the caller owns it afterwards.
 *
 * @param ens  The ensemble.
 * @param slot Index into `local_objects`; out of range is a no-op.
 *
 * @rank_local
 */
void MFTLEnsembleEraseLocalSlot(MFTL_Ensemble *ens, int slot);

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-body reductions over the HOLDERS of each body
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The operator applied to one slot of a per-body reduction.
 *
 * @see MFTLEnsembleReduceBodies()
 */
typedef enum MFTL_BodyReduceOp
{
  MFTL_BODY_REDUCE_MIN = 0,  ///< Smallest contribution. Identity \f$+\infty\f$.
  MFTL_BODY_REDUCE_MAX = 1,  ///< Largest contribution. Identity \f$-\infty\f$.
  MFTL_BODY_REDUCE_SUM = 2  ///< Sum of contributions. Identity 0.
} MFTL_BodyReduceOp;

/**
 * @brief Which set of ranks a per-body reduction is taken over.
 *
 * @warning Must be the SAME on every rank: it selects between a neighbourhood
 *          and a `comm_world` collective, and disagreement deadlocks.
 *
 * There is no value that forces the neighbourhood route: whether it is exact is
 * only known after the exchange, so it is always certified.
 *
 * @see MFTLEnsembleReduceBodies()
 */
typedef enum MFTL_BodyReduceScope
{
  /**
   * Neighbourhood route when certified exact, global route otherwise. The
   * default. The certificate is re-derived on every call, never cached.
   */
  MFTL_BODY_REDUCE_SCOPE_AUTO = 0,

  /**
   * Always the `comm_world` reduction: exact for any ensemble, but
   * `O(bodies in the ensemble)` per rank in payload and staging memory. Worth
   * passing only for a knowingly domain-spanning ensemble, where AUTO would pay
   * for an exchange it then discards.
   */
  MFTL_BODY_REDUCE_SCOPE_GLOBAL = 1
} MFTL_BodyReduceScope;

/**
 * @brief The pre-exchange half of the neighbourhood-route certificate, for this
 *        rank's pools as they are.
 *
 * Statements 1 and 2 of the argument on MFTLEnsembleReduceBodies(), plus an
 * early refusal of two visible holders two cells apart. Reads only the ghost
 * suffix of each pool (every entity in `[0, n_local)` is owned here). Necessary,
 * not sufficient: for a connected body it proves the holders connected on the
 * graph; the post-exchange certificate proves them a clique.
 *
 * @warning A caller that is about to rewrite `owner_proc` (redistribution) must
 *          take this first and pass it to MFTLEnsembleReduceBodiesPrescanned().
 *
 * @param ens The ensemble.
 * @return `true` if every local body owns an entity here and every ghost owner
 *         is a graph neighbour, with no two visible holders two cells apart.
 *
 * @rank_local
 */
bool MFTLEnsembleBodyHoldersFitTheGraph(const MFTL_Ensemble *ens);

/**
 * @brief Reduces per-body values over the ranks that HOLD each body.
 *
 * The shape of every per-body collective in the library (quality statistics,
 * global counts, the redistribution skip vote, body radii, remesher
 * measurements): each rank contributes a row per body it holds, and every
 * holder receives the combination over all holders. This is the only place the
 * communicator for it is chosen.
 *
 * **Indexing by local slot.** @p values is `[n_local_objects][n_per_body]`,
 * row `o` for `local_objects[o]`, reduced in place. A body-id-indexed buffer
 * (the shape `MPI_Allreduce` needs) would cost every rank
 * `O(bodies in the ensemble)` to learn about the few it holds.
 *
 * **Routes.** *Holders:* one `MPI_Neighbor_allgather` of counts and one
 * `MPI_Neighbor_allgatherv` of `(body id, row)` records on `comm_graph`, then a
 * local fold -- `O(local bodies x degree)`, independent of the process and body
 * counts, exact iff every two holders of every body are graph neighbours.
 * *Global:* `MPI_Allreduce` over a body-id-indexed staging buffer, one call per
 * distinct operator. AUTO exchanges the records, certifies them, agrees the
 * verdict with one `MPI_LAND` of an int, and on refusal discards them and takes
 * the global route.
 *
 * **Certificate.** Let \f$H(X)\f$ be the holders of body \f$X\f$; each holder
 * \f$R\f$ checks
 *
 * 1. \f$R\f$ owns at least one entity of \f$X\f$;
 * 2. every rank owning a ghost of \f$X\f$ in \f$R\f$'s pool is a graph neighbour;
 * 3. after the exchange, \f$R\f$ and every neighbour that published a record
 *    for \f$X\f$ span at most one cell per axis.
 *
 * *1 and 2 make \f$H(X)\f$ connected on the graph*: redistribution sends each
 * triangle to the owner of each corner and edge neighbour, so the owners of two
 * incident entities hold each other's entity as a ghost and, by 2, are graph
 * neighbours; the surface is connected, and by 1 every holder is an owner.
 * *3 makes a connected \f$H(X)\f$ a clique*: two holders \f$a, c\f$ at graph
 * distance 2 would share a holder neighbour \f$m\f$, both would be in \f$m\f$'s
 * record set, so by 3 their offsets from \f$m\f$ differ by at most one per axis
 * and \f$c\f$ is in \f$a\f$'s 26-stencil -- a contradiction. Checking 3 on the
 * halo-visible holders alone (before the exchange) is NOT sufficient: a body
 * bent around a grid corner passes on every rank while its two ends never meet
 * (`test_a_bent_body_is_certified_after_the_exchange`). An empty holder fails 1
 * and vetoes (the state of a generated ensemble before its first
 * redistribution). One conservative case: a periodic axis with 3 cuts, where a
 * body occupying all three cells is refused.
 *
 * @warning Assumes each body is ONE CONNECTED surface. Nothing in MFTL produces
 *          another kind (topology changes give every component its own id); a
 *          caller assembling a disconnected body must pass
 *          #MFTL_BODY_REDUCE_SCOPE_GLOBAL.
 *
 * **Bitwise identity.** Contributions are folded in ascending SOURCE RANK
 * order, this rank spliced in at its position, so every holder of a body
 * obtains bit-identical results (a solver branching on them never splits the
 * ranks). A self-loop in `comm_graph` (grids with a single cut on a periodic
 * axis) is dropped, or a SUM would count this rank twice.
 *
 * @param ens The ensemble.
 * @param[in,out] values `[n_local_objects * n_per_body]`, row-major by local
 *                slot: this rank's contribution in, the reduction out. May be
 *                #MFTL_NULL only when this rank holds no bodies.
 * @param n_per_body Values per body, identical on every rank. 0 returns
 *                   immediately on every rank, entering no collective.
 * @param ops `[n_per_body]` operators, identical on every rank. Mixing them is
 *            free on the holder route.
 * @param scope Route selection; identical on every rank.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble, a negative
 *         @p n_per_body or NULL @p ops.
 *
 * @collective Ranks holding no bodies must call it too.
 *             With one process no MPI call is made.
 */
int MFTLEnsembleReduceBodies(MFTL_Ensemble *ens, double *values, int n_per_body, const MFTL_BodyReduceOp *ops, MFTL_BodyReduceScope scope);

/**
 * @brief MFTLEnsembleReduceBodies() with #MFTL_BODY_REDUCE_SCOPE_AUTO, taking
 *        the pre-exchange verdict from the caller.
 *
 * For redistribution, which rewrites `owner_proc` with the DESTINATION owner
 * while classifying, so the pools no longer describe the partition the vote has
 * to reach. It takes MFTLEnsembleBodyHoldersFitTheGraph() beforehand; the
 * post-exchange certificate reads only the records and remains valid.
 *
 * @param ens The ensemble.
 * @param[in,out] values As for MFTLEnsembleReduceBodies().
 * @param n_per_body As for MFTLEnsembleReduceBodies().
 * @param ops As for MFTLEnsembleReduceBodies().
 * @param holders_fit This rank's pre-exchange verdict (local; need not agree
 *                    across ranks -- it is agreed inside).
 * @return As for MFTLEnsembleReduceBodies().
 *
 * @collective
 */
int MFTLEnsembleReduceBodiesPrescanned(MFTL_Ensemble *ens, double *values, int n_per_body, const MFTL_BodyReduceOp *ops, bool holders_fit);

/**
 * @brief Per-body EXCLUSIVE prefix sum over the ranks that hold each body.
 *
 * Handing out global ids after a distributed remeshing phase needs a SCAN, not a
 * reduction: two holders that both created vertices of one body need disjoint
 * id ranges, so each needs the count created by the holders BEFORE it. "Before"
 * is ascending rank -- the order the reduction folds in -- so the holders'
 * ranges are disjoint and tile `[0, total)` exactly
 * (`test_the_scan_tiles_every_body_exactly_once`). Same records, same
 * certificate and same fallback (`MPI_Exscan` + `MPI_Allreduce` over a
 * body-id-indexed buffer) as MFTLEnsembleReduceBodies().
 *
 * @param ens The ensemble.
 * @param[in,out] values `[n_local_objects * n_per_body]` by local slot: this
 *                rank's contribution in; out, the sum over the holders of that
 *                body with a SMALLER rank. May be #MFTL_NULL only when this rank
 *                holds no bodies.
 * @param[out] total `[n_local_objects * n_per_body]` sum over all holders, or
 *             #MFTL_NULL. Costs no extra collective.
 * @param n_per_body Values per body, identical on every rank; 0 returns
 *                   immediately.
 * @param scope Route selection, identical on every rank.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble or a negative
 *         @p n_per_body.
 *
 * @collective Ranks holding no bodies must call it too.
 */
int MFTLEnsembleExscanBodies(MFTL_Ensemble *ens, double *values, double *total, int n_per_body, MFTL_BodyReduceScope scope);

/**
 * @struct MFTL_BodyReduceStats
 * @brief Counters of the per-body reductions and scans, by route taken.
 *
 * The route is invisible in the results (both are bit-identical by
 * construction), so these counters are the only way to see a reduction pinned
 * to the `O(bodies)` global route. `n_global` should stay 0 for a droplet
 * ensemble whose bodies are smaller than a rank box. `n_bytes` counts payload
 * sent plus received on either route, including a discarded neighbourhood
 * exchange; the one-int agreement is not counted.
 */
typedef struct MFTL_BodyReduceStats
{
  long n_holders;  ///< Reductions that stayed on the 26-neighbour graph.
  long n_global;  ///< Reductions that fell back to comm_world.
  long n_bytes;  ///< Payload bytes moved by either route, sent plus received.
} MFTL_BodyReduceStats;

/**
 * @brief Copies the per-body reduction counters out.
 *
 * @param[out] out Receives this rank's counters. Must not be #MFTL_NULL.
 * @rank_local
 */
void MFTLEnsembleGetBodyReduceStats(MFTL_BodyReduceStats *out);

/**
 * @brief Zeroes the per-body reduction counters (e.g. after a benchmark's warm-up).
 *
 * @rank_local
 */
void MFTLEnsembleResetBodyReduceStats(void);

/**
 * @struct MFTL_PeriodicExtentReport
 * @brief What MFTLEnsembleCheckPeriodicExtents() found. Every field is GLOBAL,
 *        identical on every rank.
 */
typedef struct MFTL_PeriodicExtentReport
{
  int n_violations;  ///< Bodies breaking the periodic extent rule, each counted once.
  int n_covering;  ///< Bodies covering at least one periodic axis (wrapping it, or at least 3/4 of it wide), each counted once.
  int first_body_gid;  ///< Smallest offending body id, -1 if none.
  int first_axis;  ///< Its first offending axis, -1 if none.
  double first_extent;  ///< Its measured extent along that axis, 0 if none.
} MFTL_PeriodicExtentReport;

/**
 * @brief Checks THE PERIODIC EXTENT RULE: on every periodic axis, every body
 *        either wraps the axis or is narrower than half the period.
 *
 * Front tracking on a periodic domain lifts a body into one image by the
 * nearest image about its centre, and the library measures that centre in
 * four frames (MFTLPeriodicExtentResolve()). Every body narrower than
 * \f$3L/4\f$ is measured exactly, so the band \f$[L/2, 3L/4)\f$ is where a body
 * that grows past half the period is SEEN doing so; a wider compact body is
 * indistinguishable from a wrapping one and is not reported. Call this at an
 * interval in which no body can grow by a quarter period -- at output steps, or
 * after a topology change, which can merge two bodies into one wide one.
 *
 * Nothing in the library is wrong for a body in the band, which is why this is
 * a check the application calls rather than a refusal inside a module: the
 * remesher's operators are local, topology change and the per-body integrals
 * measure exactly up to \f$3L/4\f$ and treat anything wider as covering. The rule
 * is the supported envelope -- physically, a body wider than \f$L/2\f$ is nearer
 * its own periodic image than its own far side -- and this is its safety check.
 *
 * Cost: one MFTLEnsembleReduceBodies() of #MFTL_PERIODIC_EXTENT_SLOTS + 1 MIN
 * values per held body, and one `MPI_Allreduce` of two integers; two more of one
 * double each when a body breaks the rule. A body is counted once, by its
 * lowest-numbered holder. On a violation rank 0 names the first offending body,
 * its axis and its extent on stderr.
 *
 * @param ens          The ensemble.
 * @param[out] report  What was found, or #MFTL_NULL.
 * @return #MFTL_SUCCESS when every body obeys the rule (also on a domain with no
 *         periodic axis, where no MPI call is made), #MFTL_ERROR on every rank
 *         when one does not, or on a NULL ensemble.
 *
 * @collective Ranks holding no bodies must call it too.
 */
int MFTLEnsembleCheckPeriodicExtents(MFTL_Ensemble *ens, MFTL_PeriodicExtentReport *report);

/**
 * @brief Whether a per-body integral could be computed.
 */
typedef enum MFTL_IntegralStatus
{
  MFTL_INTEGRAL_DEFINED = 0,  ///< Computed; exact for the mesh.
  MFTL_INTEGRAL_UNDEFINED = 1,  ///< No such quantity: a rim off the walls, a centroid along an axis the body wraps, a zero volume.
  MFTL_INTEGRAL_UNSUPPORTED = 2  ///< Defined physically, but no wall to measure from: rims on both walls of every non-periodic axis, a sheet across a fully periodic box.
} MFTL_IntegralStatus;

/**
 * @struct MFTL_BodyIntegrals
 * @brief Area, volume and centroid of one body. Identical, to the bit, on every
 *        rank that holds the body.
 *
 * The VOLUME is that of the region the surface bounds: alone for a closed body
 * that encloses one, or together with the domain walls otherwise -- the walls
 * its rims lie on and, when it touches neither wall of `axis`, the wall at
 * `reference` (an interface spanning the box: the region below it). It is
 * SIGNED, positive when the normals point away from the region (outward for a
 * closed body). The CENTROID is that region's, wrapped into the box on a
 * periodic axis.
 */
typedef struct MFTL_BodyIntegrals
{
  double area;  ///< Surface area; always defined.
  double volume;  ///< Signed volume of the region (see above).
  double centroid[3];  ///< Centroid of the region.
  MFTL_IntegralStatus volume_status;  ///< Whether `volume` is defined.
  MFTL_IntegralStatus centroid_status[3];  ///< Per component; UNDEFINED along an axis the body wraps.
  int axis;  ///< The flux axis measured along, -1 if none.
  double reference;  ///< On `axis`: the reference wall, or the body's centre for a closed body.
  bool closed;  ///< The body has no rim.
  int lowest_holder;  ///< Lowest rank holding the body: sum a per-body value where this equals your rank to count it once.
} MFTL_BodyIntegrals;

/**
 * @brief Area, volume and centroid of every body this rank holds, for closed
 *        bodies and for open ones closed by the domain walls.
 *
 * Every quantity is the flux of a polynomial field through the surface, summed
 * over owned triangles and reduced over the body's holders; the divergence
 * theorem makes it a volume integral, exact for the mesh. Which region, and so
 * which field, is decided per body from one first reduction:
 *
 * | the body | measured | along |
 * |---|---|---|
 * | closed, area vector zero (it encloses a region) | that region | any axis it does not cover, about its centre |
 * | open, every rim on a wall | closed by the walls | an axis with rims on exactly one wall, from that wall; else the first non-periodic axis whose walls carry none, from its lower wall; UNSUPPORTED if every non-periodic axis has rims on both |
 * | closed, area vector non-zero (a sheet across periodic axes) | between sheet and wall | the axis the area vector points along, from its lower wall; UNSUPPORTED if that axis is periodic |
 * | open, a rim off the walls | -- | UNDEFINED |
 *
 * The rule for open bodies is the one every open-surface example hand-coded:
 * Rayleigh-Taylor and the travelling waves measure along z from the floor, the
 * jet along x from the nozzle wall; a drop hanging from the ceiling is measured
 * from the ceiling. A centroid component along an axis the body wraps (covers,
 * see MFTLPeriodicExtentResolve()) is UNDEFINED. Compositions -- the liquid
 * around bubbles in a periodic box, \f$V_\mathrm{cell} - \sum_b V_b\f$; the gas
 * above an interface, \f$V_\mathrm{box} - V\f$ -- are the caller's arithmetic.
 *
 * Cost: two MFTLEnsembleReduceBodies() calls, of 37 and 4 values per held body,
 * and two streaming passes over the owned triangles.
 *
 * @param ens      The ensemble.
 * @param[out] out `[n_local_objects]`, by LOCAL SLOT. May be #MFTL_NULL when
 *                 this rank holds no body.
 * @return #MFTL_SUCCESS (the per-body verdicts are in the statuses), or
 *         #MFTL_ERROR on a NULL ensemble or a NULL @p out on a rank holding
 *         bodies.
 *
 * @collective Ranks holding no bodies must call it too.
 */
int MFTLEnsembleBodyIntegrals(MFTL_Ensemble *ens, MFTL_BodyIntegrals *out);

/**
 * @brief Refreshes every local object's `global_n_triangles` /
 *        `global_n_vertices` and the ensemble's `total_global_triangles`.
 *
 * Two SUM slots per body through MFTLEnsembleReduceBodies() (the local halves
 * are just `n_local`, thanks to the ownership prefix), plus one scalar
 * `MPI_Allreduce` for the ensemble total. A body's counts land only on the
 * ranks holding it.
 *
 * @param ensemble The ensemble.
 *
 * @collective Ranks holding no bodies must call it too.
 */
void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble *ensemble);

/**
 * @brief Triangle geometry of every OWNED triangle plus per-body quality
 *        statistics, in one streaming pass per object and one reduction.
 *
 * Writes `tris.area`, `tris.normal`, `tris.center` for `[0, tris.n_local)` of
 * every local object and fills each object's `quality` with the body-wide
 * minimum, maximum and mean of the triangle area, the mean-ratio quality
 * \f$q = 4\sqrt3 A/(a^2+b^2+c^2)\f$ and the squared edge length (means over
 * owned triangles, resp. their half-edges, divided by global counts). The
 * statistics come from the edges already in registers. Eleven values per body
 * go through ONE MFTLEnsembleReduceBodies() call, so the payload scales with the
 * bodies a rank holds, not with the ensemble. Statistics are PER BODY because
 * that is what decides whether a particular body needs remeshing.
 *
 * @note Requires current derived connectivity (asserted); vertex motion does
 *       not invalidate it. GHOST triangles are not touched -- before a
 *       redistribution call MFTLComputeTriangleGeometry(obj, obj->tris.n).
 * @note The statistics cost +90 % over geometry alone (a second divide), so on
 *       a non-periodic domain this is slower than MFTLComputeTriangleGeometry();
 *       call it for the statistics (docs/analysis.md §1.6).
 *
 * @param ensemble The ensemble.
 *
 * @collective Ranks holding no bodies must call it too.
 */
void MFTLEnsembleUpdate(MFTL_Ensemble *ensemble);

/**
 * @brief For every local object: rebuild derived connectivity if stale, then
 *        MFTLComputeTriangleGeometry(obj, obj->tris.n) -- ghosts included.
 *
 * Brings the ensemble into the state a migration routes by (ghosts are routed by
 * their cached centre), and restores the triangle geometry a redistribution
 * leaves unspecified. Despite the `MFTLEnsemble` prefix it makes no collective
 * call.
 *
 * @param ensemble The ensemble.
 *
 * @rank_local
 */
void MFTLEnsembleComputeTriangleGeometry(MFTL_Ensemble *ensemble);

/**
 * @brief Exact vertex normals for every vertex, owned and ghost, in every pool.
 *
 * MFTLComputeVertexNormals() on every object -- exact on owned vertices thanks
 * to the vertex-complete halo, a partial sum on ghosts -- followed by ONE ghost
 * exchange for the whole ensemble (MFTLUpdateGhostVertexNormals()) that
 * overwrites every ghost normal with its owner's value. An EXCHANGE, not a
 * reduction: nothing needs combining, only the ghost half is missing.
 *
 * @note Requires current derived connectivity. Three neighbourhood collectives
 *       (counts, requests, replies), 8 B request + 32 B reply per ghost vertex.
 *
 * @param ensemble  The ensemble.
 * @param weighting #MFTL_NORMAL_WEIGHT_AREA (volume gradient; advection and
 *                  smoothing) or #MFTL_NORMAL_WEIGHT_ANGLE (pseudo-normal for
 *                  signed distances). Should be the same on every rank for a
 *                  consistent field; the exchange does not read it.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if this rank could not refresh every
 *         ghost (a LOCAL outcome, not agreed across ranks).
 *
 * @collective Ranks holding no bodies must call it too.
 */
int MFTLEnsembleUpdateVertexNormals(MFTL_Ensemble *ensemble, MFTL_NormalWeighting weighting);

/* ═══════════════════════════════════════════════════════════════════════════
 * Decomposition metrics
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The narrowest rank box over the CUT axes: \f$\min_{k:\,\mathrm{cuts}_k\ge2} L_k / \mathrm{cuts}_k\f$.
 *
 * The length the distributed remesher's band condition and its
 * dispersed/spanning classification are measured against. Axes with a single
 * cut have no interior plane and are skipped.
 *
 * @param domain The MPI/physical world.
 * @return The narrowest cut rank-box width; 0 when no axis is cut (one process).
 *
 * @rank_local Identical on every rank.
 */
double MFTLDomainGetHMin(const MFTL_Domain *domain);

/**
 * @brief Slot of rank @p r in `domain->neighbors`, by linear scan (at most 26).
 *
 * On grids with a single cut on a periodic axis `comm_graph` contains a
 * self-loop, so asking for this rank itself is legitimate.
 *
 * @param domain The MPI/physical world.
 * @param r      A rank.
 * @return The first slot holding @p r, or #MFTL_ERROR if @p r is not a graph
 *         neighbour.
 *
 * @rank_local
 */
static inline int MFTLDomainNeighborIndex(const MFTL_Domain *domain, int r)
{
  for (int i = 0; i < domain->num_neighbors; ++i)
  {
    if (domain->neighbors[i] == r) return i;
  }
  return MFTL_ERROR;
}

/**
 * @brief Collectively register a double field before adding the first body.
 * @param ens Ensemble; every rank calls in the same order.
 * @param name Unique nonempty field name, identical on every rank.
 * @param entity Vertex or triangle storage.
 * @param ncomp Positive component count, identical on every rank.
 * @param mode Lifetime/transfer policy. Conservative density currently requires triangles.
 * @return Dense field id, or MFTL_ERROR on every rank for invalid/mismatched registration.
 * Registration does not update densities during application-driven physical motion.
 */
int MFTLEnsembleRegisterField(MFTL_Ensemble *ens, const char *name, MFTL_FieldEntity entity, int ncomp, MFTL_FieldMode mode);

/**
 * @brief Look up a field locally by name.
 * @param ens Ensemble holding the registry.
 * @param name Registered name.
 * @return Field id, or MFTL_ERROR if absent.
 */
int MFTLEnsembleFindField(const MFTL_Ensemble *ens, const char *name);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_ENSEMBLE_H
