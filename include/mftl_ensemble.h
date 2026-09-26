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

#include "mftl_geometry.h"
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
 * @brief Adds an object, links it to the ensemble's domain and gives it its body id.
 *
 * After this call `object->domain` points at the ensemble's domain and the
 * ensemble owns the object: MFTLEnsembleDestroy() will destroy it.
 *
 * @warning **`object->global_id` is OVERWRITTEN** with the next dense body id,
 *          `ens->n_bodies_global++`. Whatever MFTLObjectCreate() was given is
 *          provisional. Body ids are the identity bodies are known by across
 *          ranks and they must be dense, so the ensemble is the one place that
 *          hands them out -- exactly as vertex and triangle ids are handed out
 *          by whatever creates the entity, never by the caller.
 *
 * @note Every rank must add the same bodies in the same order. That was already
 *       required (bodies were identified by list position on the wire) and is
 *       now what makes the ids agree.
 *
 * @param ensemble The ensemble, which takes OWNERSHIP of @p object.
 * @param object   The mesh. Its `domain` and `global_id` are set here.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 */
int MFTLEnsembleAddObject(MFTL_Ensemble *ensemble, MFTL_Object *object);

/**
 * @brief Destroys an ensemble, its domain and every object it owns. Safe on NULL.
 *
 * @param p_ens Address of the ensemble pointer. Set to #MFTL_NULL on return.
 */
void MFTLEnsembleDestroy(MFTL_Ensemble **p_ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * The local object directory
 *
 * #MFTL_Ensemble::local_objects holds the bodies THIS RANK has entities of, and
 * it is SORTED BY ASCENDING #MFTL_Object::global_id. Three things depend on
 * that, which is why it is an invariant rather than a convenience:
 *
 *   1. `gid -> slot` is a binary search -- an array, not a hash. The library has
 *      no hash maps and asks any new lookup to say why it cannot be an array;
 *      the answer here is that the id space is dense and GLOBAL while the
 *      holding is sparse, so the only array form is O(bodies in the ensemble).
 *
 *   2. Iterating locally visits bodies in ascending gid, i.e. in the global
 *      order restricted to this rank. Anything whose determinism rests on
 *      "bodies in order" keeps working on a rank holding only some of them.
 *
 *   3. Arrival runs are already grouped, so they join the directory with a merge
 *      rather than a search per record.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Bodies this rank holds entities of. NOT the size of the ensemble.
 * @param ens The ensemble.
 * @return #MFTL_Ensemble::n_local_objects.
 */
static inline int MFTLEnsembleLocalBodyCount(const MFTL_Ensemble *ens) { return (ens != MFTL_NULL) ? ens->n_local_objects : 0; }

/**
 * @brief The size of the body-id space: every id ever handed out.
 *
 * Equal to the number of bodies until a topology change retires one -- ids are
 * never reused, so after MFTLEnsembleReconstructBodies() some ids below it name
 * no body anywhere. The number every rank agrees about, and the one a collective
 * may be sized or guarded by. MFTLEnsembleLocalBodyCount() is not: returning early on it is a
 * rank-local decision about a collective call, which is a deadlock.
 *
 * @param ens The ensemble.
 * @return #MFTL_Ensemble::n_bodies_global.
 */
static inline int MFTLEnsembleGlobalBodyCount(const MFTL_Ensemble *ens) { return (ens != MFTL_NULL) ? ens->n_bodies_global : 0; }

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
 * Takes ownership. Unlike MFTLEnsembleAddObject() it does NOT hand out an id --
 * the object must already carry the one it is known by, which is the case for a
 * body arriving in a redistribution.
 *
 * @rank_local No MPI. A body arriving on one rank is not an event on any other.
 *
 * @param ens    The ensemble, which takes OWNERSHIP of @p object.
 * @param object The mesh. Its `global_id` must not already be present.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 */
int MFTLEnsembleInsertLocalObject(MFTL_Ensemble *ens, MFTL_Object *object);

/**
 * @brief Removes local slot @p slot, keeping the list sorted. Does NOT destroy it.
 *
 * The caller owns the object afterwards and is responsible for destroying it.
 * Ordered erase, not swap-with-tail: the sort is the directory's invariant.
 *
 * @rank_local No MPI.
 *
 * @param ens  The ensemble.
 * @param slot Index into `local_objects`. Out-of-range is a no-op.
 */
void MFTLEnsembleEraseLocalSlot(MFTL_Ensemble *ens, int slot);

/**
 * @brief A `[n_bodies_global]` view in which slot `g` is body `g`, or a shared
 *        EMPTY object if this rank holds nothing of it.
 *
 * For a CALLER that genuinely has to enumerate every body in the ensemble. Do not
 * use it for anything on the per-timestep path: it is `O(bodies in the ensemble)`
 * per rank by definition, which is the cost the sparse local list exists to
 * avoid.
 *
 * @note **The library itself has no caller left.** Contact detection and the
 *       spatial halo were the two, on the premise that their work list had to be
 *       identical on every rank; the premise was false, and
 *       MFTLExtendSpatialHaloNearBodies() derives the work list from the body
 *       extents it already exchanges instead.
 *
 * @warning **READ-ONLY.** Every absent slot points at the SAME placeholder, so a
 *          write through one lands on all of them. MFTLEnsembleReleaseDenseView()
 *          asserts in Debug that the placeholder is untouched. A module that
 *          needs to write per body must iterate `local_objects` instead, or key
 *          its own array by `global_id`.
 *
 * @note The view is invalidated by anything that changes the local object list --
 *       a redistribution, in particular. Acquire it after, not across.
 *
 * @rank_local No MPI. The ensemble owns the storage and reuses it.
 *
 * @param ens The ensemble.
 * @return The view, or #MFTL_NULL on a NULL ensemble.
 */
MFTL_Object *const *MFTLEnsembleAcquireDenseView(MFTL_Ensemble *ens);

/**
 * @brief Ends a dense-view borrow and checks the placeholder was not written.
 *
 * Frees nothing -- the storage is reused. It exists so the read-only contract is
 * verified rather than trusted, which is why it should be called even though
 * skipping it leaks nothing.
 *
 * @param ens The ensemble.
 */
void MFTLEnsembleReleaseDenseView(MFTL_Ensemble *ens);

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
 * @warning Whichever value is passed, it must be the SAME on every rank: it
 *          selects between a neighbourhood collective and a `comm_world` one,
 *          so two ranks disagreeing is a deadlock with no diagnostic. Passing it
 *          per call rather than setting it as library state is what makes that
 *          checkable by reading the call site.
 *
 * @see MFTLEnsembleReduceBodies()
 */
typedef enum MFTL_BodyReduceScope
{
  /**
   * Take the neighbourhood route when it is provably exact, the global one
   * otherwise. The value to pass unless there is a reason not to.
   *
   * It runs MFTLEnsembleBodyHoldersFitTheGraph() and agrees the answer with one
   * `MPI_Allreduce` of a single int. The verdict is re-derived on every call and
   * never cached: it is invalidated by any holder-set change anywhere in the
   * job, and a stale one is a silently wrong reduction rather than a crash.
   */
  MFTL_BODY_REDUCE_SCOPE_AUTO = 0,

  /**
   * Take the neighbourhood route, unchecked.
   *
   * @warning NOT verified at run time, in Debug either. On an ensemble where
   *          some body has a holder outside another holder's 26-neighbourhood
   *          this computes a wrong per-body answer in silence -- that rank's
   *          contribution is simply absent. Pass it only after establishing the
   *          property, which is what MFTLEnsembleBodyHoldersFitTheGraph() is
   *          for.
   *
   * It exists for the caller that cannot let #MFTL_BODY_REDUCE_SCOPE_AUTO run
   * the test at call time. Redistribution is that caller: it rewrites every
   * entity's `owner_proc` with the DESTINATION owner while classifying, so by
   * the time its skip vote is reduced the pools describe the partition that is
   * about to exist rather than the one the vote has to reach. It therefore takes
   * the test before classifying, agrees it, and passes the answer here.
   */
  MFTL_BODY_REDUCE_SCOPE_HOLDERS = 1,

  /**
   * Take the `comm_world` reduction. Correct for any ensemble, and
   * `O(bodies in the ensemble)` per rank in both payload and staging memory.
   *
   * Worth passing on a knowingly spanning ensemble: #MFTL_BODY_REDUCE_SCOPE_AUTO
   * would pay the ghost-suffix scan and the extra one-int collective every call
   * only to arrive here anyway.
   */
  MFTL_BODY_REDUCE_SCOPE_GLOBAL = 2
} MFTL_BodyReduceScope;

/**
 * @brief Whether every body this rank holds has all its holders on the graph.
 *
 * The rank-local half of the #MFTL_BODY_REDUCE_SCOPE_AUTO test, and the whole of
 * it once every rank agrees -- the property is a conjunction over ranks, so an
 * `MPI_LAND` of this is exactly "the neighbourhood route is exact".
 *
 * It reads only the GHOST SUFFIX of each pool: everything in `[0, n_local)` is
 * owned here by the ownership-prefix invariant, so the halo is what names the
 * other holders. A body this rank holds whole has no ghost suffix and costs
 * nothing at all.
 *
 * @warning It describes the pools AS THEY ARE. A caller that is about to rewrite
 *          `owner_proc` -- redistribution, while it classifies -- must take the
 *          answer first; afterwards the pools describe a different partition.
 *
 * @param ens The ensemble.
 * @return #MFTL_TRUE if no body of this rank has a holder outside this rank's
 *         26-neighbourhood, or one two cells away from another of its holders.
 *
 * @rank_local Each rank answers about its own bodies. Combine with `MPI_LAND`.
 */
bool MFTLEnsembleBodyHoldersFitTheGraph(const MFTL_Ensemble *ens);

/**
 * @brief Reduces per-body values over the ranks that HOLD each body.
 *
 * The library's per-body collectives -- quality statistics, global entity
 * counts, the redistribution skip vote -- all have the same shape: every rank
 * contributes something about the bodies it holds, and every holder of a body
 * wants the combination over the other holders. This is that operation, and it
 * is the one place the choice of communicator is made.
 *
 * ## THE INDEXING IS BY LOCAL SLOT, AND THAT IS THE WHOLE POINT
 *
 * @p values is `[ens->n_local_objects][n_per_body]`, row `o` belonging to
 * `ens->local_objects[o]`, reduced IN PLACE. A buffer indexed by body id would
 * be the obvious shape -- an `MPI_Allreduce` aligns element `k` of every rank's
 * buffer, so the id is the only index two ranks holding different bodies agree
 * about -- and it is exactly the shape that does not scale: at a million bodies
 * every rank allocates, fills with identities and reduces a million slots to
 * learn about the hundred it holds. Making the CALLER's buffer local is what
 * removes that, and it is why the identity fill lives inside this function
 * (only the global route needs one) rather than at every call site.
 *
 * ## THE TWO ROUTES
 *
 * **Holders.** One `MPI_Neighbor_allgather` of a record count and one
 * `MPI_Neighbor_allgatherv` of `(body id, payload)` records over
 * `MFTL_Domain::comm_graph`; each rank combines the records whose body it holds
 * into its own contribution. Cost `O(local bodies * degree)` -- constant in the
 * process count AND in the ensemble's body count. Exact if and only if every
 * holder of every body is inside every other holder's 26-neighbourhood.
 *
 * **Global.** The `comm_world` `MPI_Allreduce` over a body-id-indexed staging
 * buffer, one call per distinct operator (so at most three). Always exact,
 * `O(bodies in the ensemble)` per rank.
 *
 * ## WHEN THE NEIGHBOURHOOD IS EXACT, AND HOW THAT IS ESTABLISHED
 *
 * A rank holds an entity of a body only if it OWNS one: redistribution sends a
 * triangle to the owner of each of its three corners and each of its three edge
 * neighbours, so a rank holding nothing but ghosts of a body cannot exist. The
 * holders of a body are therefore the cells its entities occupy, and the
 * question is whether that cell set fits inside one 26-neighbourhood, i.e. has
 * per-axis extent at most one cell.
 *
 * That is decided WITHOUT geometry and without a body-global measurement, from
 * three local statements about the set \f$V_R(X)\f$ of holders rank \f$R\f$ can
 * see for body \f$X\f$ -- itself, plus the `owner_proc` of every ghost of
 * \f$X\f$ in its pool:
 *
 *   1. \f$R\f$ owns at least one entity of \f$X\f$;
 *   2. every member of \f$V_R(X)\f$ is a graph neighbour of \f$R\f$;
 *   3. the per-axis span of their cell offsets from \f$R\f$ is at most 1.
 *
 * Statement 1 is what makes \f$V\f$ complete: a rank holding an EMPTY object
 * owns nothing, appears in no other holder's halo, and could therefore neither be
 * discovered nor served. It cannot be seen from outside, so such a rank vetoes.
 *
 * Given it, all three hold everywhere iff every body's holder set has Chebyshev
 * diameter at most one. The forward direction is immediate. For the converse,
 * take holders \f$c, c'\f$ of \f$X\f$ two cells or more apart; the surface is
 * connected, so there is a chain of holders from \f$c\f$ to \f$c'\f$ in which
 * consecutive members share an entity and therefore see each other. If any step
 * of that chain is itself two cells or more, statement 2 fails at its start. If
 * every step is one cell, the chain covers a distance of at least two in unit
 * steps, so some member has chain neighbours on opposite sides of it along one
 * axis -- both in its own \f$V\f$, spanning two cells -- and statement 3 fails
 * there.
 *
 * @warning The converse rests on ONE assumption, and it is the assumption the
 *          whole library already makes about an #MFTL_Object: that a body is one
 *          CONNECTED surface. Two components of one body could sit on ranks that
 *          neither see each other nor have anything between them, and no local
 *          test could tell. Nothing in MFTL produces such a body -- generation
 *          emits a closed sphere, remeshing preserves connectivity, and the fan
 *          walks, the mean edge length and the halo rule all read one surface --
 *          but a caller that assembles one from an index list owes
 *          #MFTL_BODY_REDUCE_SCOPE_GLOBAL.
 *
 * The test is deliberately conservative in one case: three or more cells on a
 * periodic axis are all mutual neighbours when `cuts[k] == 3`, and offsets
 * \f$\{-1, 0, +1\}\f$ report a span of 2. A body occupying three of three cells
 * spans the domain, which is the regime the global route is for anyway.
 *
 * ## THE ANSWER IS BITWISE IDENTICAL ON EVERY HOLDER, AND THAT IS NOT FREE
 *
 * Contributions are combined in ascending SOURCE RANK order, with this rank
 * spliced in at its own position, because floating-point summation is not
 * associative and `MFTL_Domain::neighbors` is in an MPI-chosen order that
 * differs from rank to rank. Combining in receive order would leave two holders
 * of one body with `quality` values differing in the last ulp -- and a solver
 * that branches on such a value to decide whether to remesh would then have two
 * ranks entering different collectives. `MPI_Allreduce` gives the same guarantee
 * for free; here it has to be arranged.
 *
 * @note Collective over `comm_world`: every rank must call it, including ranks
 *       holding no bodies at all. `n_per_body` and `ops` must be identical on
 *       every rank.
 *
 * @param ens The ensemble.
 * @param[in,out] values `[n_local_objects * n_per_body]`, row-major by local
 *                slot. Read as this rank's contribution, overwritten with the
 *                reduction over the body's holders. May be #MFTL_NULL only when
 *                this rank holds no bodies.
 * @param n_per_body Values per body. 0 makes the call a no-op that still enters
 *                   every collective.
 * @param ops `[n_per_body]` operators, one per slot. Mixing them costs nothing
 *            on the holder route -- it combines in local memory -- and at most
 *            one `MPI_Allreduce` per distinct operator on the global one.
 * @param scope Which route to take. Pass #MFTL_BODY_REDUCE_SCOPE_AUTO unless the
 *              route has already been established and agreed; it must be the
 *              same value on every rank.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble or a negative
 *         @p n_per_body.
 */
int MFTLEnsembleReduceBodies(MFTL_Ensemble *ens, double *values, int n_per_body, const MFTL_BodyReduceOp *ops, MFTL_BodyReduceScope scope);

/**
 * @brief Per-body EXCLUSIVE prefix sum over the ranks that hold each body.
 *
 * ## WHY A REDUCTION IS NOT ENOUGH
 *
 * Handing out global ids after a distributed remeshing phase is the case this
 * exists for. Two ranks that both split an edge of the same body each create a
 * vertex, and the two need DIFFERENT ids -- so a holder needs to know how many
 * entities the holders BEFORE it created, which is a scan and not a combination.
 * `MPI_Exscan` gives it over `comm_world` on a buffer indexed by body id;
 * MFTLEnsembleReduceBodies() gives a combination over a body's holders. This is
 * the missing corner: a scan over a body's holders, indexed by LOCAL SLOT.
 *
 * "Before" means ASCENDING RANK, the same total order the reduction folds in,
 * so the ranges the holders receive are disjoint and cover `[0, total)` exactly.
 *
 * @note Collective over `comm_world`: every rank must call it, including ranks
 *       holding no bodies at all. @p n_per_body and @p scope must be identical
 *       on every rank.
 *
 * @param ens The ensemble.
 * @param[in,out] values `[n_local_objects * n_per_body]`, row-major by local
 *                slot. Read as this rank's contribution, overwritten with the
 *                sum over the holders of that body with a SMALLER RANK. May be
 *                #MFTL_NULL only when this rank holds no bodies.
 * @param[out] total `[n_local_objects * n_per_body]` sum over every holder,
 *             including this rank, or #MFTL_NULL if it is not wanted. It comes
 *             out of the same records, so asking for it costs no collective.
 * @param n_per_body Values per body. 0 makes the call a no-op that still enters
 *                   every collective.
 * @param scope Which route to take; see #MFTL_BodyReduceScope. The exactness
 *              argument is identical to MFTLEnsembleReduceBodies()'.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble or a negative
 *         @p n_per_body.
 */
int MFTLEnsembleExscanBodies(MFTL_Ensemble *ens, double *values, double *total, int n_per_body, MFTL_BodyReduceScope scope);

/**
 * @struct MFTL_BodyReduceStats
 * @brief What MFTLEnsembleReduceBodies() did, split by the route it took.
 *
 * `n_global` is the number to watch in a weak-scaling sweep: it counts the
 * per-body reductions that were `O(bodies in the ensemble)` per rank, and it
 * should stay at zero for a droplet ensemble on a decomposition coarser than
 * the droplets. Non-zero without a domain-spanning body means some body's
 * holders no longer fit inside one 26-neighbourhood -- usually because the rank
 * boxes have become narrower than the bodies, which is the same condition the
 * distributed remesher dispatches on.
 *
 * `n_bytes` counts payload bytes on whichever route was taken, sent plus
 * received, so the two are directly comparable. The one-int route agreement is
 * not counted: it is latency, not volume.
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
 * @brief Zeroes the per-body reduction counters.
 *
 * Call it after a benchmark's warm-up, so the initial scatter does not land in
 * the steady-state figures.
 *
 * @rank_local
 */
void MFTLEnsembleResetBodyReduceStats(void);

/**
 * @brief Sums the owned entity counts of every object across the ranks holding it.
 *
 * Two sums per body through MFTLEnsembleReduceBodies(), so the buffer is what
 * THIS rank holds rather than the id space, plus ONE scalar `MPI_Allreduce` for
 * the ensemble total. With ownership as an array prefix the local half is two
 * array reads per body rather than a scan over every entity testing a flag.
 *
 * @note Collective: every rank must call it, including one holding no bodies.
 * @note A body's counts land only on the ranks that HOLD it; a rank that holds
 *       nothing of a body has nowhere to put them and does not need them.
 *
 * @param ensemble The ensemble. Every local object's `global_n_*` and the
 *                 ensemble's `total_global_triangles` are refreshed.
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
 * @note Collective: ONE call to MFTLEnsembleReduceBodies() for the whole
 *       ensemble, whatever the body count. Every rank must reach it, including
 *       one holding no bodies.
 *
 * ── THE STATISTICS ARE PER BODY, AND THEY HAVE TO BE ──────────────────────
 *
 * Each object's own MFTL_Object::quality is filled, which costs eleven doubles
 * per body. An ensemble-wide aggregate would be eleven doubles full stop and
 * would answer a different, less useful question: per-body quality is what a
 * solver reads to decide whether THIS droplet needs remeshing, and one
 * degenerating interface among a thousand healthy ones moves an ensemble minimum
 * without saying which body moved it.
 *
 * The payload therefore grows with the body count on purpose. What does NOT grow
 * with it is the set of ranks each body's statistics are reduced over: only the
 * holders of a body have anything to contribute, so the reduction goes through
 * MFTLEnsembleReduceBodies() and both the buffer and the message are sized by
 * what this rank holds. The eleven slots carry three different operators, which
 * costs nothing there -- a neighbourhood exchange combines in local memory, so
 * mixing min, max and sum is one message rather than three.
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
 * @param weighting #MFTL_NORMAL_WEIGHT_AREA (the volume gradient, what advection
 *                  and smoothing want) or #MFTL_NORMAL_WEIGHT_ANGLE (the
 *                  pseudo-normal a signed distance needs). Must be the same on
 *                  every rank only in the sense that the result is then the same
 *                  normal field everywhere; the exchange itself does not read it.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the ghost exchange failed.
 */
int MFTLEnsembleUpdateVertexNormals(MFTL_Ensemble *ensemble, MFTL_NormalWeighting weighting);

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
