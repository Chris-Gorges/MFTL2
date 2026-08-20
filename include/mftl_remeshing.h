/**
 * @file mftl_remeshing.h
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief Isotropic surface remeshing: split, collapse, flip, smooth.
 *
 * Layer 3. The OPERATORS and the per-object driver are serial and contain no MPI
 * -- they work on one rank's complete mesh, and that is deliberate rather than
 * provisional. The distributed driver (MFTLRemeshEnsembleParallel) is a
 * SCHEDULE around them: it decides who may touch what, and calls exactly the
 * same kernel. See "Parallel remeshing" at the bottom of this comment.
 *
 * ## THE ALGORITHM
 *
 * Botsch & Kobbelt's incremental isotropic remesher. One iteration is four
 * sweeps against a target edge length L:
 *
 *   1. SPLIT    every edge longer than 4/3 L
 *   2. COLLAPSE every edge shorter than 4/5 L
 *   3. FLIP     every edge whose flip lowers the total squared valence deviation
 *   4. SMOOTH   tangential Laplacian relaxation
 *
 * The 4/5 and 4/3 bounds are not tuning: they are the widest pair for which a
 * split cannot immediately produce a collapsible edge and vice versa, which is
 * what stops the two sweeps oscillating.
 *
 * ## WHAT IMPLICIT INDEXING DOES TO A TOPOLOGICAL OPERATOR
 *
 * Half-edge 3t+k IS corner k of triangle t (mftl_topology.h). An operator
 * therefore cannot allocate a half-edge, or move one between triangles, or patch
 * a `next` pointer -- none of those are representable. What it does instead:
 *
 *   REWRITE WHOLE TRIANGLES, THEN RE-LINK THE PATCH'S OUTER TWINS.
 *
 * A triangle is written by storing three TARGET vertices (`he[3t+k].vertex_idx`
 * is the target of corner k, so the triple (v0,v1,v2) is stored as v1,v2,v0),
 * and the patch is closed by re-pointing the twins of the half-edges on its rim.
 * Three consequences worth stating, because they are what make this version's
 * operators shorter than the previous version's rather than longer:
 *
 *   - FLIP allocates nothing at all. Both triangles are rewritten in place and
 *     four rim twins are re-pointed. There is no `next` cycle to repair, which
 *     was most of the previous version's flip.
 *   - SPLIT appends one triangle per side (two for an interior edge, one at a
 *     mesh boundary) and its three half-edges come with it, at the right indices,
 *     by construction.
 *   - COLLAPSE cannot free anything, because a triangle's half-edges are its
 *     storage. It marks the two collapsed triangles and the retired vertex
 *     MFTL_FLAG_DEAD, and a compaction pass removes them later.
 *
 * A half-edge's global id is 3 * (triangle global id) + k, so it is a function of
 * the SLOT, never of the logical edge. Rewriting a triangle leaves the ids alone
 * and the invariant holds for free (inv_halfedge_gids_are_implicit).
 *
 * What does NOT come for free is the flag word: when triangle t is rewritten, the
 * logical edge that used to sit in slot 1 may end up in slot 2, and MFTL_FLAG_
 * FEATURE / BOUNDARY have to travel with the EDGE, not with the slot. Every
 * operator below therefore captures the rim flags before it writes anything and
 * places them explicitly. A feature curve that develops a gap after a split is
 * the symptom of getting this wrong.
 *
 * ## DELETION IS COMPACTION, AND IT HAPPENS INSIDE THE ITERATION
 *
 * The previous version marked entities dead and compacted once, at the very end
 * of the whole remeshing call. That is cheaper by one pass and wrong by one
 * class of bug: between the first collapse and the final compaction, every
 * consumer that walks the pools rather than the topology sees the dead entities.
 * The previous version's smoother skipped them with a per-vertex flag test, and
 * its feature detector read `triangles[t].normal` of dead triangles.
 *
 * Here MFTLRemeshCompactObject() runs at the end of each iteration's collapse
 * sweep, before anything geometric. What that buys is not tidiness, it is reuse:
 * after it the object is an ordinary valid mesh, so the smoother calls
 * MFTLObjectRefreshFrames() and MFTLComputeVertexNormals() -- the library's own
 * measured kernels, with exact area-weighted normals -- instead of the previous
 * version's open-coded per-vertex cross-product loop.
 *
 * ## GLOBAL IDS AND WHY THEY ARE RENUMBERED
 *
 * Splits hand out fresh ids and collapses retire them, so a remeshed mesh's ids
 * are unique but no longer dense. Density is not merely cosmetic: redistribution
 * and the ghost sync size their arrival lookup tables from `next_*_gid`
 * (src/communication/redistribute.c, src/communication/ghost_sync.c), so holes
 * accumulated over a few hundred timesteps become an unboundedly growing table
 * for a mesh of constant size.
 *
 * MFTLRemeshRenumberGlobalIds() therefore re-densifies at the end of a pass, and
 * MFTL_RemeshParameters.renumber_global_ids (default on) controls it. It is
 * correct EXACTLY BECAUSE THIS IS SERIAL: the renumbering is a local decision,
 * and it is only sound when this rank holds the entire object. Both the
 * renumbering and MFTLRemeshObject() assert that -- no ghosts, and the object's
 * owned counts equal its global counts. The distributed driver re-densifies with
 * an MPI_Exscan instead, and pays one ghost exchange for it.
 *
 * ## PARALLEL REMESHING: MOVE THE PLANES, OR MOVE THE BODIES
 *
 * An edge whose two endpoints are owned by different ranks cannot be collapsed
 * by either of them. The answer is NOT a protocol -- a lock/ask/agree round trip
 * inside a remesher's innermost loop makes the RESULT depend on message arrival
 * order, so the same mesh remeshes differently twice in a row.
 *
 * There are two answers instead, they are opposites, and which one applies is
 * decided by ONE ratio -- the body radius against the rank box width:
 *
 *   SPANNING (R >> h)   the body cannot be made local to anybody, so MOVE THE
 *                       CUT PLANES: run the serial kernel on the INTERIOR of
 *                       each rank's share, refuse anything whose footprint
 *                       reaches the partition boundary, then move the planes by
 *                       half a box and run it again.
 *
 *   DISPERSED (R << h)  the body FITS in a rank box, so MOVE THE BODY: migrate
 *                       all of it to the rank owning its centroid and the
 *                       untouched serial kernel runs on 100 % of it. No frozen
 *                       entities, no phases, exactly two migrations.
 *
 * MFTLRemeshEnsemble picks between them (MFTLRemeshChooseStrategy) and the
 * choice is COLLECTIVE -- it comes out of a reduction, because two ranks
 * entering different drivers is an immediate deadlock.
 *
 * Nothing is negotiated in either, because nothing overlaps. Four consequences
 * worth knowing before reading any of the code:
 *
 *   THE OPERATORS ARE UNCHANGED. MFTLSplitEdge, MFTLCollapseEdge, MFTLFlipEdge
 *   and their approval predicates contain no MPI and no ownership test. The
 *   parallel driver gates them from OUTSIDE, on MFTL_FLAG_FROZEN, and on a
 *   serial object no such flag is ever set -- so nproc == 1 goes down exactly
 *   the code path it did before, to the bit.
 *
 *   "FROZEN" IS DERIVED, NOT COMMUNICATED. An owned entity is frozen exactly
 *   when some other rank holds a copy of it, and this rank can evaluate that
 *   alone: it is the same rule the redistribution used to decide what to send.
 *   MFTLRemeshMarkFrozen() is two streaming reads and six integer compares per
 *   triangle -- no distance test, no tolerance, no message.
 *
 *   THE PRECONDITION IS GEOMETRIC AND IT IS CHECKED. With delta the half-width
 *   of the frozen band and h the rank box, an entity frozen under one placement
 *   is interior under the half-shifted one iff h > 4 delta. Violate it and some
 *   entities are never offered to anyone; MFTL_RemeshStats.converged says so,
 *   and the driver warns rather than aborting -- the failure is a mesh-quality
 *   shortfall, not a corruption.
 *
 *   ALL THE MPI IS IN ONE FILE. src/remeshing/remesh_comm.c holds every
 *   collective and every point-to-point the remesher makes; the operators, the
 *   smoother, the compaction and both drivers contain none, and a CMake source
 *   check (test_remeshing_mpi_is_confined) fails the build if that changes. The
 *   rule is there because the alternative was tried: the shift reached
 *   redistribution by growing MFTLGetOwnerProc, which cost 1.70x on the
 *   production ownership loop to serve a routine that runs a few times per
 *   hundred timesteps.
 *
 * The full argument -- why no ghost position sync is needed inside a phase, why
 * the dilation ring is optional, and why the ids are renumbered the way they are
 * -- is in src/remeshing/remesh_parallel.c, and the dispersed counterpart is in
 * src/remeshing/remesh_dispersed.c.
 *
 * WHAT IS STILL NOT HERE: nothing for the LOAD-BALANCE half of the N_bodies << P
 * case. The dispersed strategy sends each body to one rank, so ranks holding no
 * body do nothing, and there is no work for them to do -- that is a property of
 * the problem, not a gap in the code. What WAS missing, the PRECONDITION half
 * (rank boxes too small for the frozen band, which made the shift schedule
 * degrade silently), is now `MFTL_RemeshParameters.parallel_coarsen`. It merges
 * rank boxes per axis through the owner map that already exists, so it needed no
 * second communicator -- see MFTLRemeshCoarseOwnerProc in
 * src/remeshing/remesh_comm.h.
 */

#ifndef MFTL_REMESHING_H
#define MFTL_REMESHING_H

#include "mftl_constants.h"
#include "mftl_topology.h"
#include "mftl_types.h"

// -----------------------------------------------------------------------------
// 0. THE PHASE SCHEDULE  (src/remeshing/remesh_parallel.c)
// -----------------------------------------------------------------------------

/** @brief Distinct placements of the cut planes: {0, h/2}^3. */
#define MFTL_REMESH_NUM_PHASES 8

/**
 * @brief Edge lengths spanned by the frozen band, either side of a cut plane.
 *
 * The band is "one triangle to reach the plane, one more to reach a triangle
 * that reaches it, and one more because a vertex is frozen when ANY incident
 * triangle is". Three is what that adds up to, and it is measured against
 * `max_edge_length` rather than the target, because the target is what the mesh
 * converges TO and the maximum is what it may legally contain on the way.
 *
 * It appears here rather than as a literal because it is the constant in the
 * h > 4*delta precondition -- the one number that decides how many ranks a mesh
 * of a given resolution may be remeshed on.
 */
#define MFTL_REMESH_BAND_EDGES 3.0

/**
 * @brief The eight placements, in units of h_k/2 per axis.
 *
 * Ordered by how much they are worth: the first two cover every entity that is
 * near the planes of ONE axis, which is all but O((delta/h)^2) of the mesh. The
 * next two complete the even-parity orthogonal array and cover every TUBE (near
 * two axes with mixed parity); the last four cover every CORNER. The driver
 * normally exits during the first few.
 */
extern const double MFTL_REMESH_PHASES[MFTL_REMESH_NUM_PHASES][3];

/**
 * @brief The two distributed strategies, and how to pick between them.
 *
 * They are not variations on a theme -- they solve the same problem from
 * opposite ends, and which one is possible is decided by ONE ratio, the body
 * radius against the rank box width:
 *
 *   SPANNING   R >> h. The body cannot be made local to anybody, so MOVE THE
 *              CUT PLANES: run the gated serial kernel on each rank's interior,
 *              then shift the planes by half a box so that what was on the band
 *              is in somebody's middle. Cost is a migration per phase, and the
 *              phase count is what CLAUDE.md deferred issue 11 is about.
 *
 *   DISPERSED  R << h. The body FITS in a rank box, so MOVE THE BODY: migrate
 *              every entity of it to the rank owning its centroid, and the
 *              untouched serial kernel then runs on 100 % of it with no frozen
 *              entities, no phases, and no per-phase id collective -- one rank
 *              owning a whole body makes that body's ids unique for free. Two
 *              migrations for the entire pass, whatever the mesh looks like.
 *
 * The dispersed path is strictly better where it applies, and it applies to the
 * regime a real emulsion or spray is in (thousands of droplets, each far smaller
 * than a rank box). Its weakness is LOAD BALANCE: bodies go where their centroids
 * are, so ranks holding no droplet do nothing. Coarsening does not help that --
 * there is no work to give those ranks -- and it is not what
 * `parallel_coarsen` is for; that addresses the SPANNING path's precondition,
 * which is a different failure with a different symptom.
 */
typedef enum
{
  /** Measure the bodies and choose. The default; see MFTLRemeshChooseStrategy. */
  MFTL_REMESH_PARALLEL_AUTO = 0,
  /** Force the shift/freeze schedule: move the CUT PLANES over \f$\{0, h/2\}^3\f$. */
  MFTL_REMESH_PARALLEL_SPANNING,
  /** Force body centering: move the BODIES onto the rank owning their centroid. */
  MFTL_REMESH_PARALLEL_DISPERSED
} MFTL_RemeshParallelStrategy;

// -----------------------------------------------------------------------------
// 1. PARAMETERS
// -----------------------------------------------------------------------------

/** @brief How the min/max edge-length band is derived from the target length. */
typedef enum
{
  MFTL_REMESH_ISOTROPIC,  ///< Botsch & Kobbelt: [4/5 L, 4/3 L].
  MFTL_REMESH_CUSTOM  ///< The caller sets min_edge_length / max_edge_length itself.
} MFTL_RemeshType;

/** @brief Which neighbourhood the smoother's target point is built from. */
typedef enum
{
  /**
   * Umbrella operator: the unweighted mean of the one-ring VERTICES. Cheaper and
   * the classical choice, but it equalises edge lengths rather than areas.
   */
  MFTL_SMOOTH_VERTEX_BASED,

  /**
   * AREA-WEIGHTED mean of the incident TRIANGLE centroids,
   *
   *     q = SUM(A_i * c_i) / SUM(A_i)
   *
   * which is the form Botsch & Kobbelt specify. The area weight is not a detail:
   * it is the entire mechanism by which the smoother equalises triangle SIZE, by
   * pulling vertices out of densely triangulated regions towards sparse ones. An
   * unweighted mean of centroids -- which is what the previous version computed
   * under this name -- is very nearly another umbrella operator, and a remesher
   * using it loses most of its isotropy.
   */
  MFTL_SMOOTH_TRIANGLE_BASED
} MFTL_SmoothMode;

/**
 * @struct MFTL_RemeshParameters
 * @brief Everything MFTLRemeshObject() needs. Fill it with MFTLInitRemeshParams.
 */
typedef struct
{
  MFTL_RemeshType type;  ///< How min/max are derived from `target_length`.
  MFTL_SmoothMode smooth_mode;  ///< Which neighbourhood the smoother's target point is built from.

  /** Outer iterations. Each is one split/collapse/flip/smooth round. */
  int iterations;

  /** Stop early once an iteration performs fewer than this many operations. */
  int min_convergence_ops;

  // --- Length targets -------------------------------------------------------
  /**
   * The edge length \f$L\f$ the mesh converges to. Everything else scales off it.
   *
   * MFTLMeanEdgeLength() (or its collective form) is the usual source: remeshing
   * to the mesh's own current mean maintains the resolution rather than changing
   * it.
   */
  double target_length;
  double min_edge_length;  ///< Collapse below this. \f$\tfrac45 L\f$ under #MFTL_REMESH_ISOTROPIC.
  double max_edge_length;  ///< Split above this. \f$\tfrac43 L\f$ under #MFTL_REMESH_ISOTROPIC.
  double min_edge_length_sq;  ///< Precomputed; every length test is on squares.
  double max_edge_length_sq;  ///< Precomputed. See MFTLRemeshParamsFinalise().

  // --- Features -------------------------------------------------------------
  /**
   * Dihedral angle in DEGREES above which an edge is a feature.
   *
   * >= 180 DISABLES feature detection entirely, and that is a legitimate
   * production setting rather than a debug switch: on a closed smooth interface
   * -- a droplet, a bubble, the Leveque test sphere -- there are no features to
   * preserve, and every edge the detector flags is a false positive that locks a
   * vertex the smoother needed to move. See docs/remeshing-quality-analysis.md.
   *
   * A mesh boundary is ALWAYS treated as a feature, threshold or not, because it
   * is a feature of the surface rather than of its curvature.
   */
  double feature_angle_threshold;

  /**
   * Turn angle in DEGREES above which a vertex WITH two feature edges is a
   * corner rather than a slidable ridge point.
   *
   * A separate question from feature_angle_threshold, and separately settable
   * because it applies where that one does not: a mesh boundary is a feature
   * curve at any dihedral threshold, so the corners of an open patch need this
   * test even with angle detection disabled entirely. Without it a square patch's
   * four corners have exactly two rim edges each, classify as ridge points, and
   * slide along the diagonal between their neighbours -- off the patch.
   *
   * The default is 45 degrees regardless of feature_angle_threshold.
   */
  double feature_corner_angle;

  // --- Smoothing ------------------------------------------------------------
  int smooth_iterations;  ///< Relaxation sweeps per outer iteration.
  double smooth_relaxation;  ///< Step scale in [0, 1]. 1.0 is the full Jacobi step.

  /**
   * Cap on one smoothing step, as a fraction of the vertex's shortest incident
   * edge. The tangential projection makes a fold-over unlikely rather than
   * impossible -- the projection is against the vertex normal, which is itself
   * meaningless where the one-ring is already degenerate. This clamp costs one
   * comparison against a length the smoother has already computed.
   */
  double smooth_max_step_fraction;

  // --- Safety ---------------------------------------------------------------
  /**
   * Minimum cos(angle) between a triangle's normal before and after an operation
   * for that operation to be allowed. Guards collapse and flip against creating
   * inverted or near-degenerate triangles.
   *
   * This has no counterpart in the previous version, and its absence there is
   * the head of a feedback loop rather than a missing nicety: an inverted
   * triangle presents a ~180 degree dihedral angle, the feature detector reads
   * that as a sharp edge, the smoother locks every vertex of it, and the
   * inversion becomes permanent. Measured in
   * docs/remeshing-quality-analysis.md.
   */
  double max_normal_deviation_cos;

  /** Re-densify global ids after the pass. Serial only -- see the file header. */
  bool renumber_global_ids;

  // --- Parallel (ignored when nproc == 1) -----------------------------------
  /**
   * Placements of the cut planes to try before giving up, counted over the
   * 8-entry MFTL_REMESH_PHASES table and wrapping when larger.
   *
   * ══ THIS IS THE ONE PARAMETER A SOLVER MUST SET, AND THE DEFAULT IS NOT IT ══
   *
   * The default (32) means RUN TO A FIXED POINT: keep placing planes until no
   * rank anywhere wants to perform an operation. That is right for a ONE-SHOT
   * remesh -- bringing a mesh to a target length it is far from, which is what
   * the test suite does -- and it is measurably WRONG inside a per-timestep
   * loop, in both directions at once.
   *
   * Measured on the LeVeque deformation, 150 remeshed timesteps, sphere in a
   * time-reversed vortex (examples/leveque_remeshing_parallel.c). The budget is
   * the only thing varied:
   *
   *          np = 1 (phase driver forced)        np = 4
   *   budget  phases  V/V0    min q   ms/step    V/V0    min q   ms/step
   *        1    1.0   0.751   0.876     7.1        --      --       --
   *        2    2.0   0.769   0.856    20.4      0.745   0.897    12.9
   *        4    3.7   0.733   0.836    37.8      0.771   0.879    16.7
   *        8    5.7   0.683   0.459    52.7      0.670   0.796    35.4
   *       32   13.3   0.682   0.278    99.0      0.639   0.621    84.7
   *
   *   serial reference (MFTLRemeshObject): V/V0 0.751, min q 0.876, 7.1 ms/step
   *
   * Two things to read off it. Volume conservation and triangle quality are
   * NON-MONOTONE and get WORSE as the budget grows -- re-converging an
   * already-good mesh to a fixed point every timestep is not free polish, it is
   * repeated split/collapse churn that erodes an under-resolved surface. And the
   * cost is linear in the phase count, because every phase is a migration.
   *
   * At budget 4 the distributed answer at np = 4 matches or beats the serial one
   * on every correctness measure (V/V0 0.771 vs 0.751, min q 0.879 vs 0.876,
   * 0.05 % of edges out of band vs 0.09 %). At the default it is worse on all of
   * them and five times slower.
   *
   * ── So: ────────────────────────────────────────────────────────────────────
   *
   *   PER-TIMESTEP LOOP   set 2..8. Four covers slab and tube conflicts (the
   *                       even-parity orthogonal array) and was the best measured
   *                       compromise; eight is a full covering set of the table
   *                       and is what to use if the process grid is large enough
   *                       for corner conflicts to matter. `converged` will often
   *                       be false and that is expected -- the mesh is being
   *                       maintained, not converged.
   *
   *   ONE-SHOT REMESH     leave it at 32 or raise it, and check `converged`.
   *                       A phase converges the INTERIOR of each rank's share;
   *                       the band becomes interior in a later phase and is
   *                       converged there -- but converging it creates new work
   *                       on ITS OWN band, so a mesh far from its target needs
   *                       several sweeps of the table. Measured on a sphere going
   *                       1280 -> 3830 triangles: 4 phases at np = 1, 6 at np = 2,
   *                       19 at np = 4, 17 at np = 8.
   *
   * The default stays at 32 because the one-shot case FAILS LOUDLY when the
   * budget is too small (`converged` comes back false) and the per-timestep case
   * fails quietly when it is too large. A default that is wrong in the loud
   * direction is the safer of the two.
   */
  int parallel_phase_budget;

  /**
   * Extra triangle rings added to the frozen band beyond the exact set.
   *
   * ZERO IS CORRECT and is the default. The classification is not an
   * approximation of a distance -- it is the exact statement "some other rank
   * holds a copy of this", and MFTLRemeshMarkFrozen() derives it from the same
   * rule the redistribution sent by. Every operation the gate then admits has
   * its whole modified set owned exclusively by this rank AND its whole read
   * neighbourhood present locally, so there is nothing for a margin to protect.
   *
   * It is settable anyway, for two honest reasons: it widens delta, which is a
   * lever on the h > 4*delta precondition worth being able to test against; and
   * if the exactness argument in remesh_parallel.c ever turns out to have a
   * hole, this is the knob that isolates it. Each ring costs coverage -- more
   * entities frozen per phase, so more phases -- and buys nothing otherwise.
   */
  int freeze_dilation_rings;

  /**
   * Which distributed strategy to use. MFTL_REMESH_PARALLEL_AUTO (the default)
   * measures the bodies and decides; see MFTLRemeshChooseStrategy.
   */
  MFTL_RemeshParallelStrategy parallel_strategy;

  /**
   * AUTO's threshold: a body larger than this fraction of the narrowest rank box
   * makes the whole ensemble SPANNING.
   *
   * 0.4 rather than 0.5, and the margin is not arbitrary. Object-centering is a
   * 26-neighbour exchange exactly while every triangle of a body lies within one
   * rank box of the body's centroid; at R = 0.5*h a body centred on a corner
   * reaches two boxes away on the diagonal and the redistribution falls back to
   * the GLOBAL communicator -- correct, but an all-to-all where the design
   * promised a stencil. The gap between 0.4 and 0.5 is what absorbs one
   * remeshing pass's worth of growth before the next measurement.
   */
  double dispersed_radius_fraction;

  /**
   * SPANNING only: merge rank boxes into groups so that `h > 4*delta` can hold.
   *
   *   0  never (the behaviour before this existed, and still the default)
   *   1  automatic -- coarsen exactly as far as the MEASURED band requires
   *  >1  automatic, but capped at this many ranks per group
   *
   * ══ THIS IS A CORRECTNESS ESCAPE HATCH, NOT A PERFORMANCE FEATURE ═══════
   *
   * Raise the rank count on a fixed domain and the rank box shrinks until it is
   * no longer wider than four frozen bands. Past that point the shift schedule
   * DEGRADES SILENTLY: a lattice of bands along the cut planes that no placement
   * ever offers to anyone, each of them a perfectly valid piece of mesh that no
   * structural check can see. `converged` and `n_edges_hidden_long` are the only
   * symptoms.
   *
   * Coarsening merges `group[k]` boxes per axis into one, which multiplies the
   * box width by the same factor and makes the precondition hold again. The
   * price is exactly what it sounds like: `1/(g0*g1*g2)` of the ranks do ALL the
   * sweep work, and the remesh is 96-99 % of a timestep in the measured case.
   * Two cheaper answers exist and should be preferred whenever they are
   * available -- remesh to a LONGER TARGET LENGTH, or use the DISPERSED strategy
   * if the bodies fit in a rank box (it has no band and so no precondition).
   *
   * The factors are per axis (MFTLRemeshCoarseningFactors), because the
   * precondition is per axis: a domain thin in x only costs 2x parallelism
   * rather than 8x.
   *
   * Default 0 so that nothing about an existing run changes. Set it to 1 when
   * `converged` comes back false and rank 0 has printed the band warning.
   */
  int parallel_coarsen;
} MFTL_RemeshParameters;

/**
 * @struct MFTL_RemeshStats
 * @brief What one MFTLRemeshObject() call actually did. Purely diagnostic.
 *
 * It exists because the interesting failure mode of a remesher is not a crash,
 * it is a pass that reports success having refused most of its own operations.
 * The rejection counters are what distinguish "converged" from "wedged".
 */
typedef struct
{
  /**
   * ── WHAT IS RANK-LOCAL AND WHAT IS GLOBAL ────────────────────────────────
   *
   * After a DISTRIBUTED pass the counters below split into two kinds and the
   * difference is not visible from their names, so it is stated here:
   *
   *   RANK-LOCAL   every operation count and every rejection count, plus the
   *                vertex classifications and the timings. They describe what
   *                THIS rank did. Printing `n_split` from rank 0 at np = 8 shows
   *                roughly an eighth of the mesh's splits.
   *   GLOBAL       converged, n_edges_hidden*, n_phases, n_migrations*,
   *                n_coarse_groups, n_bodies_not_centred, band_*, and
   *                n_ops_busiest_rank.
   *
   * They are not all reduced because they are not all reducible: summing
   * `n_feature_edges` would double-count every edge in a halo, and `iterations`
   * is a maximum rather than a sum. A caller wanting a global operation count
   * should reduce the ones it cares about itself -- or read
   * `n_ops_busiest_rank`, which is the number that bounds the phase.
   */
  int iterations;  ///< Outer iterations actually run.

  int n_split;  ///< Edges split. RANK-LOCAL after a distributed pass.
  int n_collapse;  ///< Edges collapsed. RANK-LOCAL.
  int n_flip;  ///< Edges flipped. RANK-LOCAL.

  int n_collapse_rejected_link;  ///< Link condition failed (would pinch the surface).
  int n_collapse_rejected_fold;  ///< Would have inverted a triangle.
  int n_collapse_rejected_feature;  ///< Blocked by a feature or boundary rule.
  int n_flip_rejected_fold;  ///< Would have inverted a triangle or made a non-convex flip.

  int n_feature_edges;  ///< Undirected feature edges after the last detection.
  int n_corner_vertices;  ///< Locked outright by the smoother.
  int n_ridge_vertices;  ///< Smoothed along a feature curve only.
  int n_free_vertices;  ///< Fully tangentially smoothed.

  int n_vertices_removed;  ///< By collapse.
  int n_orphans_removed;  ///< Live vertices left with no incident triangle.

  // --- Parallel (zero / false after a serial pass) ---------------------------
  int n_phases;  ///< Cut-plane placements actually used.

  /**
   * Redistributions actually PERFORMED. Not `n_phases + 1` any more: a placement
   * that every entity already satisfies is detected (one int through an LOR) and
   * skipped, and phase 0 of the schedule is exactly that case -- it asks for the
   * canonical placement, which is what the caller handed in. So the usual figure
   * is `n_phases`, and `n_phases + 1 - n_migrations` is `n_migrations_skipped`.
   */
  int n_migrations;

  /** Placements that turned out to be the identity. See `n_migrations`. */
  int n_migrations_skipped;

  /**
   * How many of MFTL_REMESH_PHASES are DIFFERENT on this process grid: 2, 4 or 8.
   *
   * An axis with fewer than two (coarse) cells has no interior plane to move, so
   * two table entries differing only on such an axis are the same placement. On
   * a 2x1x1 grid the eight entries are two placements, on 2x2x1 four.
   *
   * It is reported because it is the size of the covering set, and therefore how
   * many consecutive silent placements the driver needs before it may certify.
   * It is also a blunt statement about the grid: a slab decomposition gives the
   * shift schedule two chances at every entity, a cubic one gives eight.
   */
  int n_distinct_placements;

  /**
   * GLOBAL: operations performed by the rank that performed the most.
   *
   * The phase loop is barrier-synchronised, so this -- not the sum -- is what
   * bounds a phase's sweep time, and comparing it against the mean is how load
   * imbalance becomes visible. Dispersed driver only so far, where it is exact
   * for a different reason: one rank owns a whole body, so no two ranks work on
   * the same body and a maximum is the busiest rank's true total.
   */
  int n_ops_busiest_rank;

  /**
   * Ranks per coarse group, i.e. `g0*g1*g2`. 1 means no coarsening happened,
   * which is the normal case and the default.
   *
   * Greater than 1 means the pass could not have offered every entity to some
   * rank at the fine decomposition, and merged boxes to fix it -- so this many
   * ranks shared one box and only one of them did any sweeping. It is the field
   * that says "this run is not scaling and here is by how much".
   */
  int n_coarse_groups;

  /**
   * THE ONE FIELD TO CHECK after a parallel pass. True means the driver reached
   * a state in which no rank anywhere wanted to perform an operation, so the
   * mesh is as converged as the serial remesher would have made it.
   *
   * False means the phase budget ran out first, and the mesh may carry a lattice
   * of un-remeshed bands along the cut planes -- which no structural check can
   * see, because every one of those bands is a perfectly valid piece of mesh.
   * There are exactly two causes: `parallel_phase_budget` too small for how far
   * the mesh was from its target, or h <= 4*delta, which the driver also warns
   * about on rank 0.
   *
   * SCOPE, because it is narrower than it reads. The certificate is measured
   * BEFORE the sweeps of the phase that certifies, and MFTLRemeshIterate smooths
   * after them -- so true means "no rank had topological work pending at the
   * last measurement", not "the mesh handed back has every edge inside the
   * band". Smoothing moves vertices and can carry an edge out of the band after
   * the fact. That is why the tests bound the out-of-band fraction rather than
   * asserting it is zero.
   */
  bool converged;

  /**
   * GLOBAL number of out-of-band edges that the LAST placement of the cut planes
   * hid from every rank at once -- a residual, not a running total.
   *
   * Diagnostic rather than a verdict, and the difference matters: a converged
   * pass can legitimately report a non-zero value. An edge that no operator
   * would fix anyway (a short edge the link condition may not collapse) still
   * counts as hidden whenever it happens to sit on a band, and asking whether it
   * WOULD have been fixed needs a one-ring this rank may not have. Use
   * `converged` for the verdict and this for the size of the affected region.
   */
  int n_edges_hidden;

  /**
   * `n_edges_hidden` split by the OPERATOR that would fix the edge, and the two
   * halves are worth very different amounts.
   *
   * `n_edges_hidden_long` is EXACT. A long edge offered to a rank is split --
   * MFTLSplitApproved has no merit refusal, only DEAD/VISITED/canonicity -- so a
   * non-zero value means some placement genuinely failed to reach work, never
   * that an operator looked and declined. It is the half `converged` is allowed
   * to depend on, and the streak exit will not fire while it is non-zero.
   *
   * `n_edges_hidden_short` carries every false alarm. A collapse is refused on
   * its merits by the link condition, the wing-valence floor, the stretch bound,
   * and by the DIRECTIONAL feature/boundary/corner rules -- the collapse sweep
   * only ever offers the canonical half-edge, so a short edge collapsible in the
   * other direction only is collapsed by nobody and stays out of band forever.
   * Expect it to be small and non-zero on a converged mesh.
   */
  int n_edges_hidden_long;  ///< Out-of-band LONG edges with a frozen endpoint. GLOBAL.
  int n_edges_hidden_short;  ///< Out-of-band SHORT edges with a frozen endpoint. GLOBAL.

  /**
   * The frozen band as it actually occurred, and a measured upper bound on it.
   * SPANNING only; zero after a dispersed or serial pass, which have no band.
   *
   * `band_measured` is the largest distance from a cut plane at which any owned
   * entity was frozen, in the FIRST phase -- where the mesh is coarsest and the
   * band therefore widest. Compare it with `MFTLDomainGetHMin(domain) / 4`: the
   * shift schedule can offer every entity to somebody exactly when the band is
   * the smaller of the two.
   *
   * `band_bound` is MFTL_REMESH_BAND_EDGES times the longest owned edge, i.e.
   * the same conservative rule the library used to apply to `max_edge_length`,
   * but measured. Expect it to be roughly THREE TIMES `band_measured`: the reach
   * argument counts three edges and the observed reach is about one. That gap is
   * the reason both are reported rather than one -- the bound is what a
   * worst-case decision (the coarsening factor) has to use, and the measurement
   * is what a diagnostic should quote.
   */
  double band_measured;  ///< Largest observed distance from a cut plane at which an owned entity was frozen.
  double band_bound;  ///< #MFTL_REMESH_BAND_EDGES times the longest owned edge. Roughly 3x `band_measured`.

  /** Which distributed path actually ran. AUTO is never reported -- it resolves. */
  MFTL_RemeshParallelStrategy strategy;

  /**
   * DISPERSED only: bodies that did not become wholly local after centering, so
   * the pass fell back to the spanning driver for the whole ensemble.
   *
   * Non-zero means the strategy choice was wrong for this configuration -- a
   * body grew past `dispersed_radius_fraction * h_min` since it was measured, or
   * the caller forced DISPERSED on a mesh that spans ranks. The result is
   * correct either way; the cost is one wasted migration.
   */
  int n_bodies_not_centred;

  // --- Profile (parallel drivers only) ---------------------------------------
  //
  // WHY THESE EXIST. Everything anyone wants to know about the cost of a
  // distributed pass is a RATIO of these -- is the pass bound by moving the mesh
  // or by remeshing it? -- and until they existed the question could only be
  // answered by timing the whole call, which is what
  // docs/parallel-remeshing-analysis.md section 7 records as the reason the
  // work-list decision could not be made. They are also the only way to see the
  // O(n) floor: `t_geometry` is a full triangle-geometry pass per phase over
  // owned AND ghost triangles, and it does not get cheaper when less is moved.
  //
  // Seconds, and REDUCED AS A MAXIMUM OVER RANKS rather than summed. The phase
  // loop is barrier-synchronised by the id collective, so the critical path is
  // the slowest rank; a mean would hide exactly the load imbalance that the
  // dispersed strategy and the coarsening path are judged on. One Allreduce of
  // six doubles per PASS, against several migrations -- below the noise floor.

  double t_total;  ///< The whole call.
  double t_migrate;  ///< Migrations, including the closing MFTLRedistribute.
  double t_geometry;  ///< The pre-migration geometry passes. Invariant to how much moves.
  double t_freeze;  ///< Freeze marking, band dilation, and the hidden-edge scan.
  double t_sweep;  ///< MFTLRemeshIterate, i.e. the actual remeshing.
  double t_reduce;  ///< The id and convergence collectives.

  /**
   * Entities that LEFT this rank, summed over every migration of the pass.
   *
   * Rank-local on purpose: it answers "how much did I ship", which is what pairs
   * with `t_migrate` on the same rank. A shifted placement moves every entity
   * whose coordinate falls between the old plane and the new one, so this is a
   * large fraction of the mesh and not a band -- measuring it is how that claim
   * stops being an assumption.
   */
  int n_entities_migrated;
} MFTL_RemeshStats;

/**
 * @brief Fills @p params with standard values for the given target edge length.
 *
 * The remaining fields are given defaults that are safe rather than aggressive;
 * a caller wanting Botsch & Kobbelt exactly sets smooth_iterations to 1 and
 * relaxation to 1.0.
 *
 * @param[out] params                  Fully initialised parameter block.
 * @param      type                    #MFTL_REMESH_ISOTROPIC derives the band as
 *                                     \f$[\tfrac45 L, \tfrac43 L]\f$;
 *                                     #MFTL_REMESH_CUSTOM leaves min/max to the
 *                                     caller, who must then call
 *                                     MFTLRemeshParamsFinalise().
 * @param      target_length           The edge length \f$L\f$ the mesh should
 *                                     converge to.
 * @param      iterations              Outer split/collapse/flip/smooth rounds.
 * @param      smooth_mode             Which neighbourhood the smoother's target
 *                                     point is built from.
 * @param      feature_angle_threshold Degrees. \f$\ge 180\f$ DISABLES feature
 *                                     detection, which is the correct production
 *                                     setting for any smooth closed interface.
 */
static inline void MFTLInitRemeshParams(MFTL_RemeshParameters *params, MFTL_RemeshType type, double target_length, int iterations, MFTL_SmoothMode smooth_mode,
                                        double feature_angle_threshold)
{
  params->type = type;
  params->target_length = target_length;
  params->iterations = iterations;
  params->smooth_mode = smooth_mode;
  params->feature_angle_threshold = feature_angle_threshold;

  switch (type)
  {
    case MFTL_REMESH_ISOTROPIC:
      params->min_edge_length = (4.0 / 5.0) * target_length;
      params->max_edge_length = (4.0 / 3.0) * target_length;
      break;

    case MFTL_REMESH_CUSTOM:
    default:
      /* The caller overwrites these; equal bounds mean "do nothing" rather than
       * an unbounded split/collapse loop if they forget. */
      params->min_edge_length = target_length;
      params->max_edge_length = target_length;
      break;
  }

  params->min_edge_length_sq = params->min_edge_length * params->min_edge_length;
  params->max_edge_length_sq = params->max_edge_length * params->max_edge_length;

  params->feature_corner_angle = 45.0;
  params->min_convergence_ops = 0;
  params->smooth_iterations = 1;
  params->smooth_relaxation = 1.0;
  params->smooth_max_step_fraction = 0.4;
  params->max_normal_deviation_cos = 0.2;  /* about 78 degrees */
  params->renumber_global_ids = MFTL_TRUE;

  params->parallel_phase_budget = 4 * MFTL_REMESH_NUM_PHASES;
  params->freeze_dilation_rings = 0;  /* the classification is exact; see the field */
  params->parallel_strategy = MFTL_REMESH_PARALLEL_AUTO;
  params->dispersed_radius_fraction = 0.4;
  params->parallel_coarsen = 0; /* correctness escape hatch, off by default; see the field */
}

/**
 * @brief Body measurements taken in this process so far, and a reset.
 *
 * A DIAGNOSTIC AND A TEST HOOK, not a statistic. MFTLRemeshMeasureBodies is the
 * only O(num_objects) collective in the module -- two Allreduces, the first
 * carrying 7 doubles per body -- so how many times a pass takes it is the number
 * that decides the module's cost at ten thousand droplets. It is one per pass,
 * and it is one only because MFTLRemeshEnsemble threads the measurement from the
 * strategy choice into the dispersed driver rather than letting each take its
 * own. That arrangement is easy to undo by accident and impossible to see in a
 * profile, so a test asserts the count.
 *
 * @return Body measurements taken by this process since the last reset.
 *
 * @rank_local A plain per-process counter, neither collective nor reduced.
 */
int MFTLRemeshMeasurementCount(void);

/**
 * @brief Zeroes the counter MFTLRemeshMeasurementCount() reports.
 * @rank_local
 */
void MFTLRemeshResetMeasurementCount(void);

/**
 * @brief Recomputes the squared length band after min/max were set by hand.
 *
 * Every length test in the module works on squares, so the two squared bounds
 * are the ones the sweeps actually read. Call this after setting
 * `min_edge_length` / `max_edge_length` directly under #MFTL_REMESH_CUSTOM.
 *
 * @param params Parameter block; `min_edge_length_sq` and `max_edge_length_sq`
 *               are overwritten.
 */
static inline void MFTLRemeshParamsFinalise(MFTL_RemeshParameters *params)
{
  params->min_edge_length_sq = params->min_edge_length * params->min_edge_length;
  params->max_edge_length_sq = params->max_edge_length * params->max_edge_length;
}

// -----------------------------------------------------------------------------
// 2. TOPOLOGICAL OPERATORS  (src/remeshing/remesh_ops.c)
// -----------------------------------------------------------------------------
//
// Each operator has an APPROVAL predicate and an EXECUTION function, and they
// are separate on purpose: the predicate is the entire correctness argument, the
// execution assumes it has already been made. A distributed driver will need to
// evaluate the predicates on one rank and replay the executions on another, and
// that is only possible if they do not share state.
//
// EVERY EXECUTION FUNCTION MAY REALLOCATE THE POOLS. Any raw pointer into
// obj->verts.* or obj->tris.* held across one of these calls is dangling.
// Hold indices.

/**
 * @brief Is half-edge @p h a legal and wanted split candidate?
 *
 * Wanted: longer than `params->max_edge_length`. Legal: alive, not already
 * touched this sweep, and CANONICAL -- an undirected edge is offered to the
 * operators once, through the half-edge of smaller index, because splitting one
 * direction splits both.
 *
 * @param obj    The mesh.
 * @param h      Half-edge index.
 * @param params Supplies `max_edge_length_sq`.
 * @return `true` if the split sweep should perform this split.
 */
bool MFTLSplitApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params);

/**
 * @brief Splits the edge of half-edge @p h at its midpoint.
 *
 *        C                    C
 *       / \                  /|\
 *      /   \                / | \
 *     A-----B     ->       A--M--B
 *      \   /                \ | /
 *       \ /                  \|/
 *        D                    D
 *
 * Rewrites the incident triangle(s) and appends one new triangle per side. The
 * midpoint is computed through the minimum-image convention and wrapped, so it
 * is the true midpoint even for an edge crossing a periodic face.
 *
 * @param obj The mesh. **May reallocate the pools.**
 * @param h   Half-edge index, normally one MFTLSplitApproved() accepted.
 * @return The pool index of the new vertex M, or #MFTL_ERROR if @p h is not a
 *         live half-edge.
 */
int MFTLSplitEdge(MFTL_Object *obj, int h);

/**
 * @brief Is collapsing half-edge @p h (origin into target) legal and wanted?
 *
 * Wanted: shorter than params->min_edge_length. Legal is the long half, and each
 * clause below removes a failure that is silent rather than loud:
 *
 *   LINK CONDITION      the endpoints must share exactly two neighbours (one at
 *                       a mesh boundary edge). Violating it welds two sheets of
 *                       the surface together into a non-manifold vertex, and
 *                       every subsequent one-ring walk from it is wrong.
 *   VALENCE             both wing vertices must keep valence >= 3, and the
 *                       merged vertex must stay inside MFTL_MAX_VALENCE.
 *   STRETCH             no edge of the merged vertex may exceed
 *                       max_edge_length, or the next sweep splits it straight
 *                       back and the two sweeps oscillate forever.
 *   FOLD-OVER           no surviving triangle's normal may turn by more than
 *                       params->max_normal_deviation_cos.
 *   FEATURES            a feature vertex may not collapse into a non-feature
 *                       one (it drags the ridge off its curve), a corner may not
 *                       collapse at all, and a boundary vertex may not collapse
 *                       into an interior one.
 *
 * @param      obj           The mesh.
 * @param      h             Half-edge index. Its ORIGIN is the vertex that would
 *                           be retired.
 * @param      params        Supplies the length band, the fold-over threshold
 *                           and the feature settings.
 * @param[out] reject_reason May be #MFTL_NULL. Receives one of the
 *                           `MFTL_REMESH_REJECT_*` codes, for the statistics.
 * @return `true` if the collapse is both legal and wanted.
 */
bool MFTLCollapseApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, int *reject_reason);

/**
 * @name Rejection codes
 *
 * Reported through the `reject_reason` out-parameter of MFTLCollapseApproved()
 * and MFTLFlipApproved(). They are tallied into #MFTL_RemeshStats because the
 * interesting failure mode of a remesher is not a crash -- it is a pass that
 * reports success having refused most of its own operations.
 * @{
 */
/** @brief Not rejected. */
#define MFTL_REMESH_REJECT_NONE 0
/** @brief Link condition failed: the collapse would weld two sheets into a non-manifold vertex. */
#define MFTL_REMESH_REJECT_LINK 1
/** @brief The operation would invert a triangle, i.e. flip its normal past `max_normal_deviation_cos`. */
#define MFTL_REMESH_REJECT_FOLD 2
/** @brief Blocked by a feature, boundary or corner rule. */
#define MFTL_REMESH_REJECT_FEATURE 3
/** @brief Refused for any other reason -- dead entity, wrong length band, already visited. */
#define MFTL_REMESH_REJECT_OTHER 4
/** @} */

/**
 * @brief Collapses the edge of half-edge @p h, retiring its ORIGIN vertex.
 *
 * The surviving vertex is the target, moved to the position
 * MFTLCollapseApproved() evaluated: the midpoint, unless exactly one endpoint is
 * a feature or boundary vertex, in which case that endpoint's position is kept
 * so the curve is not pulled off itself.
 *
 * The two incident triangles and the origin vertex are flagged #MFTL_FLAG_DEAD.
 * Pool indices do NOT change here -- MFTLRemeshCompactObject() removes them.
 * A collapse cannot free anything, because a triangle's half-edges ARE its
 * storage.
 *
 * @param obj The mesh.
 * @param h   Half-edge index, normally one MFTLCollapseApproved() accepted.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if @p h was not collapsible.
 */
int MFTLCollapseEdge(MFTL_Object *obj, int h);

/**
 * @brief Would flipping half-edge @p h lower the total squared valence deviation?
 *
 * The target valence is 6 in the interior and 4 on a mesh boundary, and the flip
 * is approved only on a STRICT decrease -- an equality would let two flips
 * alternate forever. Additionally rejected when the two opposite vertices are
 * already adjacent (the flip would create a duplicate edge), and when it would
 * turn a triangle normal by more than params->max_normal_deviation_cos, which is
 * what a flip across a non-convex quad does.
 *
 * \f[
 *   E = \sum_{v \in \{a,b,c,d\}} \big(\deg(v) - \deg^{*}(v)\big)^{2},
 *   \qquad \deg^{*} = \begin{cases} 6 & \text{interior} \\ 4 & \text{boundary} \end{cases}
 * \f]
 *
 * @param      obj           The mesh.
 * @param      h             Half-edge index.
 * @param      params        Supplies `max_normal_deviation_cos`.
 * @param[out] reject_reason May be #MFTL_NULL. #MFTL_REMESH_REJECT_FOLD when the
 *                           valence criterion WANTED the flip and the geometry
 *                           refused it, which is the only rejection worth
 *                           counting -- an edge the valence criterion declines is
 *                           the criterion working, not a failure.
 * @return `true` if the flip strictly lowers \f$E\f$ and is geometrically legal.
 */
bool MFTLFlipApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, int *reject_reason);

/**
 * @brief Flips the edge of half-edge @p h to the other diagonal of its quad.
 *
 * Allocation-free: both triangles are rewritten in place and four rim twins are
 * re-pointed. Triangle and half-edge global ids are untouched, because a
 * half-edge's id is a function of its SLOT, never of the logical edge.
 *
 * @param obj The mesh.
 * @param h   Half-edge index, normally one MFTLFlipApproved() accepted.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR at a boundary edge (which has no quad).
 */
int MFTLFlipEdge(MFTL_Object *obj, int h);

// -----------------------------------------------------------------------------
// 3. COMPACTION AND IDENTITY  (src/remeshing/remesh_compact.c)
// -----------------------------------------------------------------------------

/**
 * @brief Removes every MFTL_FLAG_DEAD entity and every orphaned vertex.
 *
 * Both pools are compacted in place with a forward scan, so the new index of an
 * entity is never greater than its old one and the moves are memmove-safe.
 * `he.vertex_idx`, `he.twin_idx` and `rec.out_he_idx` are remapped; half-edges
 * move as TRIPLES with their triangle, which is what preserves both the implicit
 * indexing invariant and the implicit half-edge global ids.
 *
 * `out_he_idx` is not remapped but RE-DERIVED, in one pass over the surviving
 * half-edges -- the same rule MFTLBuildTopologyFromIndexList uses. A local
 * repair after each collapse would be O(1) instead of O(H), and would be a
 * separate correctness argument for every operator; this is one argument for all
 * of them.
 *
 * Rebuilds the derived arrays, so the object is fully valid on return. The
 * cached geometry (`tris.area/normal/center`, `verts.normal/...`) is permuted
 * along rather than discarded, so it stays valid if it was valid.
 *
 * @param      obj   The mesh, compacted in place.
 * @param[out] stats May be #MFTL_NULL. `n_vertices_removed` /
 *                   `n_orphans_removed` are accumulated into it.
 * @return The number of entities removed (vertices plus triangles).
 * @rank_local
 */
int MFTLRemeshCompactObject(MFTL_Object *obj, MFTL_RemeshStats *stats);

/**
 * @brief Makes the global ids of both classes dense in [0, n) again.
 *
 * SERIAL ONLY, and asserted: correct exactly when this rank holds the entire
 * object, because it is a local decision about a mesh-wide namespace. See the
 * file header for why it has to happen at all.
 *
 * @param obj The mesh. Every `global_id` and all three `next_*_gid` counters are
 *            rewritten; half-edge ids follow from \f$3\,t_\mathrm{gid} + k\f$.
 *
 * @rank_local
 */
void MFTLRemeshRenumberGlobalIds(MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 4. FEATURES AND SMOOTHING  (src/remeshing/remesh_smooth.c)
// -----------------------------------------------------------------------------

/**
 * @brief Marks sharp edges MFTL_FLAG_FEATURE and classifies every vertex.
 *
 * Two outputs, from two linear passes and no fan walks at all:
 *
 *   HALF-EDGES  FEATURE set on both directions of every edge whose dihedral
 *               angle exceeds @p angle_degrees, and on every mesh-boundary
 *               half-edge unconditionally. Cleared elsewhere, so a fold that
 *               has since been smoothed out stops being a feature.
 *   VERTICES    FEATURE for exactly two incident feature edges that continue
 *               roughly straight through the vertex (a ridge point), CORNER for
 *               one, for three or more, or for two that turn by more than
 *               @p corner_angle_degrees, neither for zero.
 *
 * The turn test is what makes an open patch's four corners corners: each has
 * exactly two rim edges, so a count alone classifies them as ridge points free to
 * slide along the diagonal between their neighbours -- which takes them straight
 * off the patch. It is a separate threshold because it applies even when dihedral
 * detection is off; see MFTL_RemeshParameters.feature_corner_angle.
 *
 * Doing the vertex classification HERE rather than in the smoother is what keeps
 * the smoothing sweeps free of fan walks: the smoother runs several times per
 * remeshing iteration and would otherwise re-derive the same counts each time.
 *
 * The count is over UNDIRECTED edges -- each is visited once, through the
 * half-edge of smaller index -- which is also what makes the boundary case come
 * out right. A fan walk over outgoing half-edges misses the closing edge of an
 * open fan entirely (mftl_topology.h says so about
 * MFTLGetVertexOneRingHalfedges), so a boundary vertex counted that way is short
 * by one and lands in the wrong class.
 *
 * Recomputes `tris.normal` itself via MFTLComputeTriangleGeometry rather than
 * trusting whatever is in the pool. A feature detector reading stale normals
 * silently classifies the mesh as it was one topological edit ago, and after a
 * split the new triangles' normals have never been written at all.
 *
 * @param      obj                    The mesh. Entity flags are rewritten and
 *                                    `tris.normal` is refreshed.
 * @param      angle_degrees          Dihedral angle in DEGREES. \f$\ge 180\f$
 *                                    disables detection: everything is cleared
 *                                    and only mesh boundaries stay features.
 * @param      corner_angle_degrees   Turn angle above which a vertex with two
 *                                    feature edges is a CORNER rather than a
 *                                    slidable ridge point.
 * @param[out] stats                  May be #MFTL_NULL. Receives the edge and
 *                                    vertex-class counts.
 * @return The number of undirected feature edges.
 * @rank_local
 */
int MFTLDetectFeatures(MFTL_Object *obj, double angle_degrees, double corner_angle_degrees, MFTL_RemeshStats *stats);

/**
 * @brief Tangential Laplacian smoothing, @p iterations Jacobi sweeps.
 *
 * Each vertex is treated according to the class MFTLDetectFeatures() gave it,
 * and the three cases differ in the SUBSPACE the displacement is projected onto
 * -- 0D, 1D and 2D respectively:
 *
 *   CORNER   locked. Zero degrees of freedom.
 *
 *   RIDGE    the Laplacian displacement towards the midpoint of the two feature
 *            neighbours, projected onto the ridge TANGENT. This redistributes
 *            the vertex along its feature curve without moving the curve.
 *
 *            The projection is the whole point. Moving to the midpoint itself --
 *            which is what the previous version did, at full strength and
 *            ignoring its own relaxation factor -- is a chord cut: it pulls the
 *            vertex off the curve towards the straight line between its
 *            neighbours, so a repeatedly smoothed ridge shrinks. Over the tens
 *            of sweeps a remeshing pass performs, that is a large systematic
 *            error, and it is applied precisely where the geometry was supposed
 *            to be PRESERVED.
 *
 *   FREE     the displacement towards the area-weighted centroid of the incident
 *            triangles (or the umbrella centroid, per params->smooth_mode),
 *            projected onto the TANGENT PLANE by removing its component along
 *            the exact area-weighted vertex normal.
 *
 * The normal comes from MFTLComputeVertexNormals(), refreshed each sweep, which
 * is both more accurate and faster than accumulating a cross product per vertex
 * over its one-ring: it streams `tris.edge` with no fan walk and no dependent
 * loads (src/geometry/vertex_geometry.c measures the two at 3.4x).
 *
 * Every difference is a minimum-image vector and every result is wrapped, so the
 * smoother is correct across a periodic seam.
 *
 * @param obj        The mesh. Positions are updated in place.
 * @param params      Supplies `smooth_mode`, `smooth_relaxation` and
 *                    `smooth_max_step_fraction`.
 * @param iterations  Sweeps to perform. Each is a full Jacobi update: all new
 *                    positions are computed from the old ones, then applied.
 *
 * @note Requires #MFTL_DERIVED_TOPOLOGY. Leaves the frames current.
 * @rank_local
 */
void MFTLSmoothTangential(MFTL_Object *obj, const MFTL_RemeshParameters *params, int iterations);

// -----------------------------------------------------------------------------
// 5. DRIVERS  (src/remeshing/remesh_driver.c)
// -----------------------------------------------------------------------------

/**
 * @brief Remeshes one object to params->target_length. The serial entry point.
 *
 * Asserts the object holds no ghosts: an edge with an endpoint on another rank
 * cannot be collapsed without agreement, so a "serial" remesh of a partitioned
 * object would silently produce a different mesh on each rank.
 *
 * On return the object has valid topology, both derived bits set, and current
 * `tris.area/normal/center`. Vertex normals are current as a side effect of the
 * last smoothing sweep.
 *
 * @param      obj    The mesh, remeshed in place.
 * @param      params The parameter block. See MFTLInitRemeshParams().
 * @param[out] stats  May be #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the mesh failed its integrity check.
 * @rank_local
 */
int MFTLRemeshObject(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

/**
 * @brief Remeshes every object of an ensemble. The entry point to use.
 *
 * Dispatches on the rank count, and the two paths are genuinely different
 * algorithms rather than one with a fast case:
 *
 *   nproc == 1   MFTLRemeshObject per object. No MPI is called at all, and the
 *                result is bit-identical to what this function produced before
 *                the distributed driver existed.
 *   nproc >  1   MFTLRemeshChooseStrategy decides between
 *                MFTLRemeshEnsembleDispersed (bodies smaller than a rank box:
 *                move the bodies) and MFTLRemeshEnsembleParallel (bodies larger:
 *                move the cut planes). `params->parallel_strategy` overrides the
 *                choice; MFTL_RemeshStats.strategy reports what actually ran.
 *
 * @param      ens    The ensemble, remeshed in place.
 * @param      params The parameter block.
 * @param[out] stats  May be #MFTL_NULL. Receives the SUM over all objects; in
 *                    the parallel case the counts are this rank's, except
 *                    `converged` and `n_edges_hidden*`, which are global.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if any object failed.
 *
 * @collective For `nproc > 1`, including ranks holding no triangles of any
 *             object -- they still take part in every migration and every
 *             reduction.
 *
 * @warning **Both drivers must be reached by every rank.** The choice is made
 *          from REDUCED data and never from anything local; two ranks entering
 *          different drivers is an immediate deadlock with no diagnostic.
 */
int MFTLRemeshEnsemble(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

// -----------------------------------------------------------------------------
// 6. PARALLEL DRIVER  (src/remeshing/remesh_parallel.c)
// -----------------------------------------------------------------------------

/**
 * @brief Remeshes a distributed ensemble by shifting the cut planes.
 *
 * Reached through MFTLRemeshEnsemble; exposed separately because a caller may
 * want to run the phase schedule at nproc == 1 to compare it against the serial
 * answer, which is what the test suite does.
 *
 * ── The loop ───────────────────────────────────────────────────────────────
 *
 *   per phase:  place the planes  ->  migrate  ->  classify frozen  ->
 *               run the serial kernel on the interior  ->  one Exscan and one
 *               Allreduce (new ids, and the verdict)
 *   at the end: canonical planes  ->  re-densify ids  ->  migrate  ->  geometry
 *
 * TWO COLLECTIVES PER PHASE plus one migration, and neither scales with the mesh
 * size. There is deliberately NO ghost position sync in the loop: every ghost in
 * a rank's pool is frozen on its owner, so the copies are still exact when the
 * sweeps finish. That is a theorem about the classification, not an
 * approximation -- the argument is in src/remeshing/remesh_parallel.c and it is
 * what one otherwise-unavoidable neighbour exchange per phase rests on.
 *
 * ── Preconditions ──────────────────────────────────────────────────────────
 *
 *   - MFTLRegisterMpiTypes() has been called (the driver migrates).
 *   - Every rank holds the same object list in the same order, and agrees on
 *     each object's next_*_gid. Both are checked in Debug.
 *   - h_min > 4 * delta, delta ~ (3 + freeze_dilation_rings) * max_edge_length.
 *     Warned about once rather than enforced; see MFTL_RemeshStats.converged.
 *
 * ── Postconditions ─────────────────────────────────────────────────────────
 *
 *   - Canonical decomposition restored: the mesh is partitioned exactly as
 *     MFTLRedistribute would have partitioned it, because the closing migration
 *     IS MFTLRedistribute.
 *   - Global ids dense over the whole mesh again, and consistent between every
 *     ghost and its owner.
 *   - Valid topology, both derived bits, current `tris.area/normal/center`.
 *     `verts.normal` is NOT current -- the closing migration discards it; call
 *     MFTLEnsembleUpdateVertexNormals if you need it.
 *   - MFTL_FLAG_FROZEN cleared everywhere.
 *
 * @param      ens    The ensemble, remeshed in place.
 * @param      params The parameter block. `parallel_phase_budget` and
 *                    `parallel_coarsen` are read here and nowhere else.
 * @param[out] stats  May be #MFTL_NULL. The phase, migration and band fields are
 *                    GLOBAL; the operation counts are rank-local.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if a migration or the id exchange failed.
 *
 * @collective
 */
int MFTLRemeshEnsembleParallel(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

// -----------------------------------------------------------------------------
// 6b. DISPERSED DRIVER  (src/remeshing/remesh_dispersed.c)
// -----------------------------------------------------------------------------

/**
 * @brief Remeshes a distributed ensemble by moving the BODIES, not the planes.
 *
 * The strategy for an ensemble of objects each smaller than a rank box -- an
 * emulsion, a spray, a bubbly flow -- and it is not a variant of the shift
 * schedule but its opposite. Every entity of a body is migrated to the rank
 * owning the body's global centroid, so the body arrives complete on one rank
 * with no ghosts; the UNMODIFIED serial kernel then runs on 100 % of it.
 *
 * ── What that removes ──────────────────────────────────────────────────────
 *
 * Three things the spanning driver needs, and none of them for free:
 *
 *   the frozen classification   there is nothing replicated, so nothing to
 *                               freeze and no operation refused. The gates in
 *                               the sweeps are the no-ops they already are in
 *                               serial.
 *   the phase schedule          one migration out, one back, whatever the mesh
 *                               looks like -- against 4-17 measured for the
 *                               shift schedule on the same sphere.
 *   the per-phase id collective ONE rank owns a whole body, so the operators'
 *                               local ids are already globally unique for it.
 *                               The only agreement needed is a single
 *                               Allreduce(MAX) of the id counters at the end,
 *                               so that every rank sizes its arrival tables the
 *                               same way.
 *
 * ── What it costs ──────────────────────────────────────────────────────────
 *
 * LOAD BALANCE. Bodies go where their centroids are, so a rank holding no
 * droplet does no remeshing work, and a rank holding six does six bodies' worth.
 * For N_bodies >> P that averages out; for N_bodies < P it does not, and nothing
 * fixes it, because there is no remeshing work for a rank that holds no body.
 * `parallel_coarsen` is NOT the answer here -- it addresses the spanning path's
 * band precondition, not this.
 *
 * ── The fallback ───────────────────────────────────────────────────────────
 *
 * If any body fails to become wholly local -- it grew past a rank box since it
 * was measured, or the caller forced this strategy on a spanning mesh -- the
 * whole ensemble is handed to MFTLRemeshEnsembleParallel BEFORE anything is
 * remeshed, so there is never a half-remeshed intermediate state. The decision
 * is one Allreduce of one int. `MFTL_RemeshStats.n_bodies_not_centred` reports
 * it; the answer is correct either way, the cost is one wasted migration.
 *
 * Preconditions and postconditions are exactly MFTLRemeshEnsembleParallel's,
 * minus the h > 4*delta band condition, which this strategy does not have --
 * that is the point of it.
 *
 * @param      ens    The ensemble, remeshed in place.
 * @param      params The parameter block. `dispersed_radius_fraction` is read
 *                    here and in MFTLRemeshChooseStrategy().
 * @param[out] stats  May be #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if a migration failed.
 *
 * @collective
 */
int MFTLRemeshEnsembleDispersed(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

/**
 * @brief Resolves MFTL_REMESH_PARALLEL_AUTO against the ensemble as it stands.
 *
 * The whole decision is one ratio: the largest body radius anywhere against the
 * narrowest rank box. Above `params->dispersed_radius_fraction` the bodies
 * cannot be made local to anybody and the planes have to move; below it, moving
 * the bodies is strictly cheaper.
 *
 * A MIXED ensemble resolves to SPANNING, because the test is on the largest
 * body. That is the safe direction and it is also the correct one: an ensemble
 * is the unit of parallelisation policy, so one jet among a thousand droplets
 * means the ensemble is a jet. Splitting it is the caller's decision, and the
 * library will not make it silently.
 *
 * @note COLLECTIVE -- it measures the bodies, which needs two Allreduces
 *       (MFTLRemeshMeasureBodies). Returns SPANNING unchanged at nproc == 1,
 *       where the question is meaningless.
 * @param ens    The ensemble whose bodies are measured.
 * @param params May be #MFTL_NULL, in which case the default fraction is used.
 * @return #MFTL_REMESH_PARALLEL_SPANNING or #MFTL_REMESH_PARALLEL_DISPERSED.
 *         Never #MFTL_REMESH_PARALLEL_AUTO -- that is what it resolves.
 */
MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategy(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params);

/**
 * @brief The GLOBAL centroid and radius of every body, in two collectives.
 *
 * What the strategy choice is made from, published because a caller has the same
 * question: the plan's ensemble-rebalancing scheme (§6) is exactly "measure the
 * bodies, then decide", and deciding is the caller's business.
 *
 * Both must be global. A rank's share of a body has a centroid somewhere inside
 * its own box, so a per-rank answer would centre nothing and measure nothing.
 *
 * ── Periodicity is not a detail here ───────────────────────────────────────
 *
 * The arithmetic mean of wrapped coordinates puts a body straddling a seam in
 * the MIDDLE OF THE DOMAIN -- the one place it certainly is not -- and the
 * radius derived from it then comes out as half the box. Both are silent. So a
 * periodic axis is averaged as an ANGLE: x is mapped to
 * theta = 2*pi*(x - min)/L, the area-weighted mean of (cos, sin) is reduced, and
 * atan2 brings it back. That is the circular mean, exact for any placement
 * including the seam.
 *
 * The radius is the minimum-image distance to the furthest CORNER, not to the
 * furthest centre: a triangle sticking out is part of the body, and the
 * difference is one edge length -- the same order as the band widths this is
 * compared against.
 *
 * @param      ens      The ensemble.
 * @param[out] centroid [3*num_objects], may be #MFTL_NULL.
 * @param radius   [num_objects] out, may be MFTL_NULL. Skipping it skips the
 *                 second collective.
 * @note Collective: one MPI_Allreduce of 8*num_objects doubles, then one of
 *       num_objects. Reads `tris.area` and `tris.center`, so the caller owes a
 *       current geometry pass.
 */
void MFTLRemeshMeasureBodies(MFTL_Ensemble *ens, double *centroid, double *radius);

/**
 * @brief How many flow ranks would have to be grouped into one remeshing rank
 *        for the shift schedule's `h > 4*delta` precondition to hold.
 *
 * The plan's Regime C number (§4.4), and the actionable half of the warning the
 * driver prints when the precondition fails: "your boxes are too small" is not
 * useful on its own, "group 8 of them" is.
 *
 *   G > (4*delta / h_min)^3,  rounded up to a perfect cube
 *
 * A perfect cube so that grouped boxes stay roughly cubic; cubing because the
 * constraint is on ONE axis of a 3D grid, so the linear factor has to be applied
 * three times to keep the shape.
 *
 * ── What this is, now that acting on it is possible ────────────────────────
 *
 * A DIAGNOSTIC, and deliberately still one. The library CAN act on the answer --
 * set `MFTL_RemeshParameters.parallel_coarsen` and the driver merges rank boxes
 * itself -- but it acts on MFTLRemeshCoarseningFactors, not on this, and the
 * difference is a factor of nine in wasted parallelism:
 *
 *   this           one number, cubed, from the 3*max_edge_length ESTIMATE
 *   the driver     per axis, from the MEASURED band
 *
 * The estimate overstates the band by about three (MFTL_RemeshStats.band_bound
 * against band_measured), and cubing applies a one-axis constraint to all three.
 * Both are the safe direction and both cost ranks. This function is kept
 * unchanged because it is what the warning message quotes -- a single "group N
 * of them" is what a human reads at 3 a.m. -- and because changing a published
 * return value to mean something else is worse than having two functions.
 *
 * What the plan proposed and what was NOT built: MPI_Comm_split, a gather onto a
 * group master, remesh on the coarse grid, scatter back. It was not needed. A
 * coarsened decomposition is an OWNER FUNCTION, and redistribution has taken an
 * arbitrary owner map since the dispersed strategy landed, so the whole path is
 * one inline function (MFTLRemeshCoarseOwnerProc) rather than a second
 * communication subsystem.
 *
 * @param domain The MPI/physical world.
 * @param params Supplies the target length the band is derived from.
 * @rank_local Every rank computes the same value from replicated data.
 * @return 1 when the precondition already holds, else the smallest perfect cube
 *         that would satisfy it.
 */
int MFTLRemeshCoarseningFactor(const MFTL_Domain *domain, const MFTL_RemeshParameters *params);

/**
 * @brief The per-axis coarsening the shift schedule needs, from a MEASURED band.
 *
 * `g[k] = ceil(4*delta / h_k)` for every axis that is actually cut, and 1
 * everywhere else. The scalar MFTLRemeshCoarseningFactor is the product, kept
 * because it is what a diagnostic message quotes; THIS is what the driver acts
 * on, and the difference is not cosmetic. The precondition is per axis --
 * MFTLDomainGetHMin takes a minimum precisely because one thin axis is enough to
 * break it -- so cubing the worst axis' factor throws away parallelism on the
 * two axes that were fine. A domain thin in x only costs 2x, not 8x.
 *
 * @param      domain    The MPI/physical world.
 * @param      params    Supplies `parallel_coarsen`, which caps the result.
 * @param      band      The frozen band half-width. Pass
 *                       `MFTL_RemeshStats::band_bound` from a previous pass for
 *                       a measured answer, or 0 to fall back to the
 *                       \f$\texttt{MFTL\_REMESH\_BAND\_EDGES} \cdot
 *                       \texttt{max\_edge\_length}\f$ estimate -- which
 *                       overstates the real band by about a factor of three and
 *                       will therefore coarsen further than necessary.
 * @param[out] out_group [3] receives the per-axis factors, each \f$\ge 1\f$.
 *
 * A pure function of replicated data (`domain`, `params`, and a reduced `band`),
 * which is what lets every rank compute the same groups without agreeing about
 * them -- two ranks with different groups is an immediate deadlock.
 */
void MFTLRemeshCoarseningFactors(const MFTL_Domain *domain, const MFTL_RemeshParameters *params, double band, int out_group[3]);

/**
 * @brief Marks every entity this rank may not modify. LOCAL, no MPI.
 *
 * Sets MFTL_FLAG_FROZEN on every ghost, on every owned triangle of which a
 * neighbour holds a copy, and on every vertex of such a triangle; clears it
 * everywhere else. On a ghost-free object it clears the flag and returns, which
 * is why the gated sweeps behave identically in serial.
 *
 * "A neighbour holds a copy" is not estimated. It is re-derived from the rule
 * redistribution sent by -- a triangle travels to the owner of each of its three
 * corners and each of its three edge neighbours -- so the test is
 * `corner >= verts.n_local || neighbour >= tris.n_local`, evaluated over
 * `tris.verts` and `tris.neighbors`.
 *
 * @param obj The mesh. #MFTL_FLAG_FROZEN is set or cleared on every entity.
 *
 * @note Requires #MFTL_DERIVED_TOPOLOGY. A redistribution leaves it current.
 * @rank_local Despite describing a property of the PARTITION -- it is a
 *             statement about THIS rank's halo, so a received copy of it would
 *             be another rank's answer to a different question.
 */
void MFTLRemeshMarkFrozen(MFTL_Object *obj);

/**
 * @brief Grows the frozen band by @p rings triangle rings. LOCAL, no MPI.
 *
 * Not needed for correctness -- see MFTL_RemeshParameters.freeze_dilation_rings
 * for why the classification is already exact. Exposed so that the exactness
 * claim can be tested against a conservative alternative rather than only
 * argued.
 *
 * One ring is "freeze every triangle with a frozen corner, then freeze every
 * corner of a frozen triangle". The two halves read and write different flag
 * arrays, so neither propagates within itself and a pass is exactly one ring
 * regardless of pool order -- which is what a same-array vertex-to-vertex sweep
 * cannot promise.
 *
 * @param obj   The mesh.
 * @param rings Rings to add. 0 is a no-op.
 *
 * @rank_local
 */
void MFTLDilateFrozenBand(MFTL_Object *obj, int rings);

/**
 * @brief Mean edge length over the whole mesh -- the usual source of a target.
 *
 * Averages over UNDIRECTED edges, so an interior edge counts once. Averaging
 * over half-edges instead gives the same number on a closed mesh and a slightly
 * wrong one on an open surface, where boundary edges would be under-weighted by
 * a factor of two.
 *
 * ── DO NOT USE THIS TO DERIVE A TARGET ON A DISTRIBUTED MESH ──────────────
 *
 * It averages over THIS RANK'S POOL, so on a partitioned object every rank gets
 * a different answer and a rank holding no part of the body gets ZERO. Feeding
 * that to MFTLInitRemeshParams gives that rank a length band of [0, 0], under
 * which every edge it owns is too long, and the pass refines without bound until
 * the job is killed. Nothing reports it, because a per-rank length band is not
 * something the algorithm ever communicates.
 *
 * The trap is easy to walk into because the generators are COLLECTIVE and
 * already distribute -- so `MFTLGenerateSphere(obj, ...); MFTLMeanEdgeLength(obj)`
 * looks like it measures the sphere and measures one rank's share of it. Use
 * MFTLEnsembleMeanEdgeLength(). In Debug the drivers additionally check that all
 * ranks were handed the same band (MFTLRemeshParamsAgree).
 *
 * @param obj The mesh.
 * @return The arithmetic mean of the OWNED half-edge lengths, or 0 for an empty
 *         mesh.
 * @rank_local
 */
double MFTLMeanEdgeLength(const MFTL_Object *obj);

/**
 * @brief Mean edge length over the WHOLE distributed mesh: the collective form.
 *
 * What a caller deriving a remeshing target from the mesh itself actually wants,
 * and the reason MFTLMeanEdgeLength is not it -- see the warning there.
 *
 * Sums lengths and counts over every body and reduces both, so the result is the
 * edge-weighted mean over the entire ensemble and is bit-identical on every
 * rank. Each undirected edge is counted exactly once mesh-wide: an edge cut by a
 * partition is attributed to the owner of the incident triangle of smaller
 * global id, because "my triangle whose twin is a ghost" would count every cut
 * edge twice -- by an amount that grows with the rank count.
 *
 * @param ens The ensemble.
 * @return The mean edge length over every object's owned half-edges, on every
 *         rank.
 *
 * @collective One `MPI_Allreduce` of two doubles. Returns 0 for an empty
 *       ensemble, which is the only case a caller must still handle.
 */
double MFTLEnsembleMeanEdgeLength(MFTL_Ensemble *ens);

#endif  // MFTL_REMESHING_H
