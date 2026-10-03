/**
 * @file mftl_remeshing.h
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief Isotropic surface remeshing -- split, collapse, flip, tangential
 *        smoothing -- serial and distributed.
 *
 * Layer 3. The operators, the smoother and the per-object driver are serial and
 * contain no MPI; the two distributed drivers are schedules around the same
 * kernel. Every MPI call of the module is in src/remeshing/remesh_comm.c
 * (enforced by `test_remeshing_mpi_is_confined`).
 *
 * **Algorithm** (Botsch & Kobbelt, SGP 2004). One iteration against a target
 * edge length \f$L\f$:
 *
 * 1. detect features (MFTLDetectFeatures());
 * 2. SPLIT every edge longer than \f$\tfrac43 L\f$ at its midpoint;
 * 3. COLLAPSE every edge shorter than \f$\tfrac45 L\f$, unless a resulting edge
 *    would be longer than \f$\tfrac43 L\f$;
 * 4. compact (MFTLRemeshCompactObject());
 * 5. FLIP every edge whose flip strictly lowers the squared valence deviation;
 * 6. detect features again and smooth tangentially (MFTLSmoothTangential()).
 *
 * A split of an edge longer than \f$\tfrac43 L\f$ gives halves longer than
 * \f$\tfrac23 L\f$, i.e. no further from \f$L\f$ than the original edge; the
 * collapse's stretch clause keeps split and collapse from undoing each other.
 *
 * **Operators under implicit indexing.** Half-edge 3t+k is corner k of triangle
 * t, so an operator rewrites whole triangles and re-links the twins on the rim of
 * its patch: a flip allocates nothing, a split appends one triangle (with its
 * three half-edges) per side, and a collapse flags two triangles and one vertex
 * #MFTL_FLAG_DEAD for the compaction to remove. Edge flags (FEATURE, BOUNDARY)
 * are carried to the logical edge's new slot explicitly. A half-edge's global
 * id, \f$3\,t_\mathrm{gid}+k\f$, depends on the slot only and stays valid.
 *
 * **Global ids** stay unique but become sparse (splits create, collapses retire).
 * Redistribution and the ghost sync size lookup tables from `next_*_gid`, so
 * every pass ends dense again: MFTLRemeshRenumberGlobalIds() on a whole object,
 * per-body prefix sums over each body's holders in the spanning driver.
 *
 * **Distributed remeshing.** An edge whose endpoints two ranks own can be
 * collapsed by neither, and a lock/ask protocol inside the sweeps would make the
 * result depend on message timing. Two exact strategies need no negotiation;
 * AUTO chooses collectively from one ratio, the largest body radius R against
 * the narrowest rank box h:
 *
 * - **SPANNING** (MFTLRemeshEnsembleParallel()): the kernel runs only on
 *   entities no other rank holds a copy of (the others are #MFTL_FLAG_FROZEN,
 *   derived locally by MFTLRemeshMarkFrozen()), and the cut planes are shifted
 *   by half a box over the placements \f$\{0,h/2\}^3\f$ so that every entity
 *   is interior in some placement. Requires \f$h > 4\delta\f$, \f$\delta\f$
 *   the half-width of the frozen band.
 * - **DISPERSED** (MFTLRemeshEnsembleDispersed(), \f$R \le 0.4\,h\f$ by
 *   default): each body is migrated whole onto one rank (its centroid's owner,
 *   load-balanced) and remeshed serially; two migrations per pass, nothing
 *   frozen.
 *
 * On a ghost-free object nothing is frozen, so the serial path is unchanged to
 * the bit. Theory, call graphs and measurements: src/remeshing/README.md and
 * docs/analysis.md §4-5.
 */

#ifndef MFTL_REMESHING_H
#define MFTL_REMESHING_H

#include "mftl_constants.h"
#include "mftl_topology.h"
#include "mftl_types.h"

// -----------------------------------------------------------------------------
// 0. THE PHASE SCHEDULE  (src/remeshing/remesh_parallel.c)
// -----------------------------------------------------------------------------

/** @brief Number of cut-plane placements, \f$\{0, h/2\}^3\f$. */
#define MFTL_REMESH_NUM_PHASES 8

/**
 * @brief Conservative reach of the frozen band either side of a cut plane, in
 *        units of `max_edge_length`.
 *
 * A triangle is replicated when a corner or an edge neighbour is a ghost (one
 * triangle to the plane, one more through the neighbour), and a vertex is frozen
 * when any incident triangle is (one more): three edges. The band actually
 * measured is about \f$0.84\f$ `max_edge_length`
 * (MFTL_RemeshStats::band_measured), so this bound is used only where a worst
 * case is required.
 */
#define MFTL_REMESH_BAND_EDGES 3.0

/**
 * @brief The eight placements, in units of \f$h_k/2\f$ per axis.
 *
 * Ordered so that every prefix is as useful as possible: rows 0-1 ((0,0,0),
 * (1,1,1)) make every entity near the planes of ONE axis interior at least once
 * (all but \f$O((\delta/h)^2)\f$ of the mesh); rows 0 and 2-4 form the
 * even-parity set {000, 011, 101, 110}, which covers every entity near the
 * planes of TWO axes; an entity near planes of all three axes needs its unique
 * complementary placement, so all eight rows are required. Placements differing
 * only on an axis with fewer than two (coarse) cells coincide, and the driver
 * de-duplicates them.
 */
extern const double MFTL_REMESH_PHASES[MFTL_REMESH_NUM_PHASES][3];

/** @brief The distributed remeshing strategy (see the file header). */
typedef enum
{
  /** Measure the bodies and choose (default); see MFTLRemeshChooseStrategy(). */
  MFTL_REMESH_PARALLEL_AUTO = 0,
  /** Shift the cut planes over \f$\{0, h/2\}^3\f$, freezing replicated entities. */
  MFTL_REMESH_PARALLEL_SPANNING,
  /** Move every body whole onto one rank. */
  MFTL_REMESH_PARALLEL_DISPERSED
} MFTL_RemeshParallelStrategy;

/**
 * @brief Why the spanning driver merged rank boxes; a bitmask, since triggers
 *        can fire together.
 *
 * BAND is required for correctness (otherwise bands along the cut planes are
 * never remeshed). FROZEN and GATHER are efficiency trades, for which a longer
 * target length or the dispersed strategy may be the cheaper remedy.
 */
typedef enum
{
  /** No trigger fired; the pass was not coarsened. */
  MFTL_REMESH_COARSEN_NONE = 0,
  /** Correctness: the measured band violates \f$h > 4\delta\f$ on some cut axis. */
  MFTL_REMESH_COARSEN_BAND = 1,
  /** Efficiency: the frozen share exceeded MFTL_RemeshParameters::coarsen_frozen_fraction. */
  MFTL_REMESH_COARSEN_FROZEN = 2,
  /** Size: the ensemble has fewer triangles than MFTL_RemeshParameters::serial_gather_max_triangles. */
  MFTL_REMESH_COARSEN_GATHER = 4
} MFTL_RemeshCoarsenTrigger;

// -----------------------------------------------------------------------------
// 1. PARAMETERS
// -----------------------------------------------------------------------------

/** @brief How the min/max edge-length band is derived from the target length. */
typedef enum
{
  MFTL_REMESH_ISOTROPIC,  ///< Band \f$[\tfrac45 L, \tfrac43 L]\f$ (Botsch & Kobbelt).
  MFTL_REMESH_CUSTOM  ///< The caller sets `min_edge_length` / `max_edge_length`, then calls MFTLRemeshParamsFinalise().
} MFTL_RemeshType;

/** @brief Which neighbourhood the smoother's target point is built from. */
typedef enum
{
  /** Umbrella operator: unweighted mean of the one-ring vertices; equalises edge lengths. */
  MFTL_SMOOTH_VERTEX_BASED,

  /**
   * Area-weighted mean of the incident triangle centroids,
   * \f$\mathbf q = \sum_i A_i \mathbf c_i / \sum_i A_i\f$ (Botsch & Kobbelt). The
   * weight is what equalises triangle AREAS; without it the operator is nearly
   * the umbrella one (docs/analysis.md §4: 6.0 % volume loss unweighted against
   * 3.5 % weighted).
   */
  MFTL_SMOOTH_TRIANGLE_BASED
} MFTL_SmoothMode;

/**
 * @brief Moves a position onto the constraint of an open surface's rim.
 *
 * Supplied by the application through MFTL_RemeshParameters::rim_project when
 * its rim lies on something the rim's own vertices describe only approximately
 * -- a nozzle circle (which the rim curve already reproduces), a curved support.
 * Without one the remesher already places rim vertices on the curve through the
 * rim's own vertices, and on any planar domain wall to the bit; the projector
 * removes what interpolating the rim's own vertices cannot know.
 *
 * Contract: a PURE function of its arguments and the same function on every
 * rank. Owner and ghost copies are then projected to the same bits with no
 * message, which is what lets the library apply it wherever it places a rim
 * vertex.
 *
 * @param dom      The domain.
 * @param[in,out] pos A position on or near the rim, wrapped into the domain;
 *                 moved onto the constraint in place.
 * @param user     MFTL_RemeshParameters::rim_project_user.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR to refuse: the split or collapse that
 *         wanted the position is not made, the smoothing move not taken.
 */
typedef int (*MFTL_RimProjector)(const MFTL_Domain *dom, double pos[3], void *user);

/**
 * @struct MFTL_RemeshParameters
 * @brief Parameters of a remeshing pass; initialise with MFTLInitRemeshParams().
 *
 * A distributed pass needs the same block, bit for bit, on every rank: some
 * fields decide the sequence of collectives, the rest what happens to entities
 * two ranks share. The distributed entry points check that with one Allreduce
 * and refuse on every rank otherwise (see MFTLRemeshEnsemble()). A new field
 * must be added to `MFTL_REMESH_AGREED_FIELDS` in src/remeshing/remesh_comm.c,
 * or the ranks may disagree about it unchecked.
 *
 * Tagged, so that mftl_topology_change.h can forward-declare it.
 */
typedef struct MFTL_RemeshParameters
{
  MFTL_RemeshType type;  ///< How min/max are derived from `target_length`.
  MFTL_SmoothMode smooth_mode;  ///< Which neighbourhood the smoother's target point is built from.

  /** Outer iterations (detect/split/collapse/flip/smooth rounds). */
  int iterations;

  /** Stop after an iteration that performs at most this many operations (default 0). */
  int min_convergence_ops;

  // --- Length targets -------------------------------------------------------
  /**
   * Target edge length \f$L\f$. To keep the current resolution use
   * MFTLEnsembleMeanEdgeLength() (collective), not the rank-local
   * MFTLMeanEdgeLength().
   */
  double target_length;
  double min_edge_length;  ///< Collapse below this. \f$\tfrac45 L\f$ under #MFTL_REMESH_ISOTROPIC.
  double max_edge_length;  ///< Split above this. \f$\tfrac43 L\f$ under #MFTL_REMESH_ISOTROPIC.
  double min_edge_length_sq;  ///< Squared bound; every length test compares squares.
  double max_edge_length_sq;  ///< Squared bound; see MFTLRemeshParamsFinalise().

  // --- Features -------------------------------------------------------------
  /**
   * Angle between the two face normals of an edge, in degrees, above which the
   * edge is a feature. \f$\ge 180\f$ disables detection -- the right setting for
   * a smooth closed interface, where every detected feature is a false positive
   * that locks a vertex the smoother must move (docs/analysis.md §4). Mesh
   * boundary edges are always features.
   */
  double feature_angle_threshold;

  /**
   * Turn angle in degrees above which a vertex with exactly two feature edges is
   * a CORNER (locked) rather than a ridge point (slides along the curve). Also
   * applies to boundary curves with angle detection off, so the corners of an
   * open patch stay put. Default 45.
   */
  double feature_corner_angle;

  // --- Smoothing ------------------------------------------------------------
  int smooth_iterations;  ///< Relaxation sweeps per outer iteration.
  double smooth_relaxation;  ///< Step scale in [0, 1]; 1 is the full Jacobi step.

  /**
   * Cap on one smoothing step as a fraction of the vertex's shortest incident
   * edge (default 0.4). Guards against fold-overs where the vertex normal is
   * ill-defined.
   */
  double smooth_max_step_fraction;

  // --- Safety ---------------------------------------------------------------
  /**
   * A collapse or flip is refused if a surviving triangle's normal would turn by
   * more than \f$\arccos\f$ of this (default 0.2, about 78 degrees), or if a
   * triangle would become degenerate. Without it an inverted triangle reads as a
   * sharp edge to the feature detector, which then locks its vertices for good
   * (docs/analysis.md §4).
   */
  double max_normal_deviation_cos;

  /** Re-densify global ids at the end of MFTLRemeshObject() (default true). The distributed drivers always do. */
  bool renumber_global_ids;

  // --- Parallel (ignored when nproc == 1) -----------------------------------
  /**
   * SPANNING only: maximum cut-plane placements per pass, cycling through
   * #MFTL_REMESH_PHASES. **A per-timestep solver loop must set it.**
   *
   * The default, 32, runs to a fixed point: right for a one-shot remesh far
   * from its target (check `converged`), wrong for maintaining a mesh every
   * step, where re-converging an already good mesh is split/collapse churn that
   * erodes it, at one migration per placement. LeVeque deformation, 150
   * remeshed steps at np = 4 (examples/leveque_remeshing_parallel.c,
   * docs/analysis.md §5.1):
   *
   * | budget | V/V0 | min q | ms/step |
   * |---:|---:|---:|---:|
   * | 2 | 0.745 | 0.897 | 12.9 |
   * | 4 | 0.771 | 0.879 | 16.7 |
   * | 8 | 0.670 | 0.796 | 35.4 |
   * | 32 | 0.639 | 0.621 | 84.7 |
   *
   * (serial MFTLRemeshObject(): 0.751, 0.876, 7.1 ms/step). Use 2-8 in a loop
   * (4 was the best measured compromise; 5 distinct placements cover every
   * entity near one or two axes' planes, 8 every entity; `converged == false`
   * is then normal).
   * The default stays at 32 because a too-small budget fails loudly in the
   * one-shot case, while a too-large one fails quietly in the loop case.
   */
  int parallel_phase_budget;

  /**
   * Extra triangle rings added to the frozen band. 0 (default) is correct: the
   * frozen set is exactly the replicated set, so every admitted operation's
   * modified set belongs to this rank alone and its neighbourhood is complete.
   * A positive value widens \f$\delta\f$ for testing that claim; each ring
   * costs coverage.
   */
  int freeze_dilation_rings;

  /** Distributed strategy (default #MFTL_REMESH_PARALLEL_AUTO). */
  MFTL_RemeshParallelStrategy parallel_strategy;

  /**
   * AUTO chooses DISPERSED iff the largest body radius is at most this fraction
   * of the narrowest rank box (default 0.4). At 0.5 a body centred on a box
   * corner reaches two boxes away diagonally, and the centering migration
   * leaves the 26-neighbourhood; the margin also absorbs one pass of growth.
   */
  double dispersed_radius_fraction;

  /**
   * SPANNING only: fallback that merges rank boxes into groups when a trigger
   * fires (#MFTL_RemeshCoarsenTrigger). 0 disarmed; 1 armed (default); > 1 armed
   * with at most this many ranks per group.
   *
   * Armed is not active: on a healthy partition the factors are {1,1,1} and the
   * pass is bit-identical to a disarmed one
   * (`test_the_fallback_stays_off_on_a_healthy_partition`). Arming costs one
   * pre-pass (geometry refresh, freeze classification, two small reductions;
   * about 0.5 % of a timestep). The BAND trigger fires when the measured band
   * violates \f$h_k > 4\delta\f$ on a cut axis -- past that point bands along
   * the cut planes are never offered to any rank, silently. Factors are per axis
   * (MFTLRemeshCoarseningFactors()). Coarsening leaves \f$1/(g_0g_1g_2)\f$ of
   * the ranks doing all the sweeping: an escape hatch for correctness, not a
   * speed-up.
   */
  int parallel_coarsen;

  /**
   * SPANNING only: coarsen when more than this share of owned triangles is
   * frozen at the start of a pass. \f$\ge 1\f$ disables (default).
   *
   * Disabled because of measurements (docs/analysis.md §5.6): the frozen share of
   * an r = 0.7 sphere on \f$[-1,1]^3\f$ ranges from 9 % (np = 2) to 37 %
   * (np = 27, refine 3). Firing merges boxes, costing a factor \f$g_0g_1g_2\f$
   * in sweeping ranks for about \f$(1-f_\text{new})/(1-f_\text{old})\f$ more
   * useful work per migration; at 36.6 % it made an unconverged pass converge
   * for twice the wall clock. Useful window roughly 30-60 %. The model
   * \f$1-(1-2\delta/h)^3\f$ over-predicts the share by about two, so set this
   * from the measured MFTL_RemeshStats::frozen_fraction, never from geometry.
   */
  double coarsen_frozen_fraction;

  /**
   * SPANNING only: remesh on ONE rank when the whole ensemble has fewer
   * triangles than this. 0 disables (default).
   *
   * The top rung of coarsening (group = cuts): nothing is replicated or frozen,
   * one placement, exactly two migrations. A count rather than a ratio, because
   * that rank then holds the whole ensemble. Measured crossover about 50-100
   * triangles per rank (docs/analysis.md §5.6), i.e. about 50 x nproc, capped by
   * one rank's memory. It gives up the extra quality of the
   * distributed pass (min q 0.85 against 0.72 serially at 18 300 triangles).
   */
  int serial_gather_max_triangles;

  /**
   * DISPERSED only: a rank may carry up to this multiple of the mean body weight
   * (global triangle count) before bodies are reassigned to neighbouring ranks
   * (default 1.25; \f$\le 0\f$ disables). The cap is
   * \f$\max(\text{mean}\cdot\text{slack}, \text{heaviest body})\f$ -- a body
   * is indivisible.
   *
   * Each body is decided by ONE rank, the owner of its reduced centroid
   * ("home"), and the verdict is delivered one hop along `comm_graph`.
   * Reassignments go to neighbouring cells only and are negotiated in a bounded
   * number of ask/grant rounds that cannot push any rank past its cap; every
   * payload is O(local bodies x graph degree). Clustered bodies: imbalance
   * 2.00 / 4.00 / 8.00 / 27.00 -> 1.25 at np = 2 / 4 / 8 / 27 (docs/analysis.md
   * §5.7). Nothing helps with fewer bodies than ranks
   * (MFTL_RemeshStats::n_ranks_idle). Precondition, tested locally: home is a
   * graph neighbour of every holder; otherwise the pass uses plain centroid
   * ownership and reports MFTL_RemeshStats::n_bodies_unreachable.
   */
  double dispersed_balance_slack;

  /**
   * Optional constraint of an open surface's rim (default #MFTL_NULL: the curve
   * alone).
   *
   * Without it a rim vertex is placed on the rim CURVE: a split of a rim edge,
   * a collapse of one, and the smoother's move of a rim vertex all take the
   * average of the two circular arcs over the edge through the neighbouring
   * rim vertices -- exact on a circle at any spacing, the chord on a straight
   * rim, never past the edge's ends -- break it at corners (#MFTL_FLAG_CORNER,
   * or a turn sharper than `feature_corner_angle`), and keep every planar
   * domain wall both ends lie on to the bit. A rim split whose curve point
   * would turn a new triangle over falls back to the chord midpoint. With it, each such
   * position is then passed through the projector BEFORE the approval clauses
   * see it, so they judge where the vertex will actually be. A topology change
   * whose MFTL_TopologyOptions::remesh carries it applies it to every rim vertex
   * it reconstructs, inside the volume correction.
   *
   * Agreed across ranks as PRESENT or ABSENT (a function pointer has no value
   * to compare); that it is the same function everywhere is the caller's
   * contract, see #MFTL_RimProjector.
   */
  MFTL_RimProjector rim_project;

  /** Passed to `rim_project` unchanged; not compared across ranks. */
  void *rim_project_user;
} MFTL_RemeshParameters;

/**
 * @struct MFTL_RemeshStats
 * @brief Diagnostics of one pass.
 *
 * A remesher typically fails not by crashing but by reporting success after
 * refusing most of its work; the rejection counters and `converged` tell the
 * two apart. After a distributed pass the operation and rejection counts, the
 * vertex classes, `iterations` and `n_entities_migrated` are RANK-LOCAL; every
 * other field is GLOBAL (the timings are maxima over ranks). Reduce the local
 * counts yourself if needed.
 */
typedef struct
{
  int iterations;  ///< Outer iterations run (maximum over objects).

  int n_split;  ///< Edges split.
  int n_collapse;  ///< Edges collapsed.
  int n_flip;  ///< Edges flipped.

  int n_collapse_rejected_link;  ///< Wanted collapses refused by the link condition or a valence bound.
  int n_collapse_rejected_fold;  ///< Wanted collapses refused by the fold-over test.
  int n_collapse_rejected_feature;  ///< Wanted collapses refused by a feature, corner or boundary rule.
  int n_flip_rejected_fold;  ///< Flips the valence criterion wanted but the fold-over test refused.

  int n_feature_edges;  ///< Undirected feature edges after the last detection.
  int n_corner_vertices;  ///< Corner vertices (locked by the smoother).
  int n_ridge_vertices;  ///< Ridge vertices (smoothed along their feature curve).
  int n_free_vertices;  ///< Free vertices (smoothed in the tangent plane).

  int n_vertices_removed;  ///< Vertices removed by collapses.
  int n_orphans_removed;  ///< Live vertices left without an incident triangle, removed by compaction.

  // --- Parallel (zero / false after a serial pass) ---------------------------
  int n_phases;  ///< Cut-plane placements used (SPANNING).

  /**
   * Redistributions performed. A placement identical to the current partition
   * (always the case for phase 0) is detected by one logical-OR reduction and
   * skipped.
   */
  int n_migrations;

  /** Placements skipped because they were the identity. */
  int n_migrations_skipped;

  /**
   * Distinct placements of #MFTL_REMESH_PHASES on this grid: 2, 4 or 8. An axis
   * with fewer than two (coarse) cells has no interior plane, so entries
   * differing only there coincide (2 on a 2x1x1 grid, 4 on 2x2x1). It is the
   * length of the silent streak that certifies convergence.
   */
  int n_distinct_placements;

  /**
   * Operations performed by the busiest rank (DISPERSED). A phase is bounded by
   * this, not by the sum; compare with the mean to see load imbalance.
   */
  int n_ops_busiest_rank;

  /**
   * Ranks per coarse group, \f$g_0g_1g_2\f$: 1 without coarsening, `nproc` for
   * the single-rank gather. Only one rank per group sweeps.
   */
  int n_coarse_groups;

  /**
   * Bitmask of #MFTL_RemeshCoarsenTrigger explaining `n_coarse_groups`;
   * #MFTL_REMESH_COARSEN_NONE when that is 1.
   */
  int coarsen_trigger;

  /**
   * Share of owned triangles frozen at the start of the pass, on the canonical
   * partition, in [0, 1]. Measured whenever the coarsening pre-pass runs,
   * otherwise 0 (always 0 for dispersed and serial passes). A frozen entity is
   * migrated and then not touched, so this is the share of every migration that
   * buys nothing -- the most useful single number about a spanning pass.
   */
  double frozen_fraction;

  /**
   * The verdict of a distributed pass. True when a phase saw no operation
   * performed anywhere and no out-of-band edge hidden by the freeze, or a full
   * streak of `n_distinct_placements` silent placements with no hidden LONG
   * edge. False means the budget ran out first and bands along the cut planes
   * may be un-remeshed -- invisible to any structural check. Causes: too small a
   * `parallel_phase_budget`, or \f$h \le 4\delta\f$ (warned on rank 0).
   *
   * Measured before the sweeps of the certifying phase; the smoothing after
   * them may still move a few edges out of the band. True for serial passes.
   */
  bool converged;

  /**
   * Out-of-band edges the LAST placement hid from every rank (a residual, not a
   * total). A diagnostic of the affected region; `converged` is the verdict.
   */
  int n_edges_hidden;

  /*
   * `n_edges_hidden` by the operator that would fix the edge. LONG is exact: an
   * offered long edge is always split (MFTLSplitApproved() has no merit test),
   * so non-zero means work no placement reached. SHORT includes false alarms: a
   * collapse can be refused on merit in both directions (link condition,
   * valence, stretch, feature rules), so a small non-zero value is normal on a
   * converged mesh.
   */
  int n_edges_hidden_long;  ///< Hidden edges longer than `max_edge_length`; exact (an offered long edge is always split).
  int n_edges_hidden_short;  ///< Hidden edges shorter than `min_edge_length`; includes collapses refused on merit.

  /*
   * SPANNING only. `band_measured`: largest distance from a cut plane (of the
   * current placement) at which an owned entity was frozen, maximised over the
   * phases; the schedule can cover every entity only if it is below h_min / 4.
   * `band_bound`: #MFTL_REMESH_BAND_EDGES times the longest owned edge, about
   * three times `band_measured`, for decisions that need a worst case.
   */
  double band_measured;  ///< SPANNING: largest distance from a cut plane at which an owned entity was frozen.
  double band_bound;  ///< SPANNING: #MFTL_REMESH_BAND_EDGES times the longest owned edge (about 3x `band_measured`).

  /** The distributed strategy that ran (never AUTO). */
  MFTL_RemeshParallelStrategy strategy;

  /**
   * DISPERSED: bodies the balancer moved off their centroid's owner. An
   * improving `load_imbalance` with this at zero means nothing needed
   * balancing.
   */
  int n_bodies_reassigned;

  /**
   * DISPERSED: ranks left without any body after balancing. Non-zero with at
   * least as many bodies as ranks is a balancer shortfall; with fewer bodies
   * than ranks it is unavoidable -- a body is the smallest unit of work.
   */
  int n_ranks_idle;

  /**
   * DISPERSED: heaviest rank's body weight over the mean, after balancing
   * (\f$\ge 1\f$). Weights are global triangle counts, a proxy for sweep time.
   * It stays above `dispersed_balance_slack` only when one body exceeds the cap
   * or there are fewer bodies than ranks.
   */
  double load_imbalance;

  /**
   * DISPERSED: bodies too large for the admission test (radius against the
   * possibly coarsened rank box). Non-zero sends the whole ensemble to the
   * spanning driver before anything is modified -- the result is correct, the
   * body measurement was wasted. Arises when the strategy was forced or a body
   * grew since it was measured.
   */
  int n_bodies_not_centred;

  /**
   * DISPERSED: (holder, body) pairs whose home rank is not a graph neighbour of
   * the holder, counted once per holder. Non-zero disables balancing for the
   * whole pass (plain centroid ownership, correct but unbalanced). Typically
   * reached only when `parallel_coarsen` relaxed the admission test to a
   * coarsened box, so that a body legitimately spans several cells.
   */
  int n_bodies_unreachable;

  // --- Profile (distributed drivers only): seconds, maximum over ranks --------
  // The critical path of a synchronised phase loop is the slowest rank; a mean
  // would hide exactly the imbalance these are read for.

  double t_total;  ///< The whole call.
  double t_migrate;  ///< Migrations, including the closing MFTLRedistribute.
  double t_geometry;  ///< The pre-migration geometry passes. Invariant to how much moves.
  double t_freeze;  ///< Freeze marking, band dilation, and the hidden-edge scan.
  double t_sweep;  ///< MFTLRemeshIterate, i.e. the actual remeshing.
  double t_reduce;  ///< The id and convergence collectives.

  /**
   * RANK-LOCAL: owned entities this rank sent away, summed over the pass's
   * migrations. A shifted placement moves a large share of the mesh (87.5 % for
   * (0,0,0) -> (1,1,1)), not only the band.
   */
  int n_entities_migrated;
} MFTL_RemeshStats;

/**
 * @brief Fills @p params with defaults for target edge length @p target_length.
 *
 * Defaults: one smoothing sweep at relaxation 1 (Botsch & Kobbelt), step cap
 * 0.4, `max_normal_deviation_cos` 0.2, corner angle 45 degrees, ids renumbered,
 * phase budget 32 (set 2-8 in a timestep loop), AUTO strategy with radius
 * fraction 0.4, coarsening armed with the FROZEN and GATHER triggers off,
 * balancer slack 1.25.
 *
 * @param[out] params                  Fully initialised parameter block.
 * @param      type                    #MFTL_REMESH_ISOTROPIC sets the band
 *                                     \f$[\tfrac45 L, \tfrac43 L]\f$;
 *                                     #MFTL_REMESH_CUSTOM sets both bounds to
 *                                     \f$L\f$ as placeholders for the caller to
 *                                     overwrite before calling
 *                                     MFTLRemeshParamsFinalise().
 * @param      target_length           Target edge length \f$L\f$.
 * @param      iterations              Outer iterations.
 * @param      smooth_mode             Smoother target point.
 * @param      feature_angle_threshold Degrees; \f$\ge 180\f$ disables feature
 *                                     detection (right for smooth closed
 *                                     interfaces).
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
      /* Placeholders for the caller to overwrite (then MFTLRemeshParamsFinalise). */
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

  /* ARMED, not active. {1,1,1} on a healthy partition, and that path is
   * bit-for-bit the uncoarsened one -- see the field, and
   * test_the_fallback_stays_off_on_a_healthy_partition. */
  params->parallel_coarsen = 1;
  params->coarsen_frozen_fraction = 1.0; /* disabled: measured, and it did not pay. See the field. */
  params->serial_gather_max_triangles = 0; /* off: the crossover is unmeasured */
  params->dispersed_balance_slack = 1.25;
  params->rim_project = MFTL_NULL;
  params->rim_project_user = MFTL_NULL;
}

/**
 * @brief Rim placements in this process since the last reset that fell back to
 *        the chord because a rim neighbour was not found locally.
 *
 * Every operation on a rim runs on entities whose fans are complete, so this
 * should read 0; it exists so a test can say so.
 *
 * @return Fallbacks since the last reset.
 * @rank_local A plain per-process counter.
 */
long MFTLRemeshRimFallbacks(void);

/**
 * @brief Zeroes the counter MFTLRemeshRimFallbacks() reports.
 * @rank_local
 */
void MFTLRemeshResetRimFallbacks(void);

/**
 * @brief Number of MFTLRemeshMeasureBodies() calls in this process since the
 *        last reset.
 *
 * A test hook: a pass must measure the bodies once, which MFTLRemeshEnsemble()
 * achieves by handing the strategy choice's measurement to the dispersed
 * driver.
 *
 * @return Body measurements since the last reset.
 * @rank_local A plain per-process counter.
 */
int MFTLRemeshMeasurementCount(void);

/**
 * @brief Zeroes the counter MFTLRemeshMeasurementCount() reports.
 * @rank_local
 */
void MFTLRemeshResetMeasurementCount(void);

/**
 * @brief Doubles this rank has staged for the dispersed balancer's exchanges
 *        since the last reset.
 *
 * A test hook for a scaling property no mesh digest can see: a rank holding no
 * body must stage the same amount however large the ensemble is
 * (`test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold`).
 *
 * @return Doubles staged since the last reset.
 * @rank_local A plain per-process counter.
 */
long MFTLRemeshBalancerPayload(void);

/**
 * @brief Zeroes the counter MFTLRemeshBalancerPayload() reports.
 * @rank_local
 */
void MFTLRemeshResetBalancerPayload(void);

/**
 * @brief Recomputes the squared bounds after `min_edge_length` /
 *        `max_edge_length` were set by hand (#MFTL_REMESH_CUSTOM).
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
// Each operator is an approval predicate plus an execution function. The
// predicates carry the whole correctness argument; the executions assume it.
// MFTLSplitEdge() appends and may reallocate the pools: hold indices, not
// pointers, across it.

/**
 * @brief Is half-edge @p h a legal and wanted split candidate?
 *
 * Wanted: longer than `max_edge_length`. Legal: alive, not touched earlier in
 * this sweep, and the canonical half of its edge (the smaller half-edge index,
 * or the only one at a boundary), so every undirected edge is offered once.
 *
 * @param obj    The mesh.
 * @param h      Half-edge index.
 * @param params Supplies `max_edge_length_sq`.
 * @return `true` if the split sweep should split this edge.
 */
bool MFTLSplitApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params);

/**
 * @brief Splits the edge of half-edge @p h at its midpoint -- on the rim curve
 *        for a rim edge.
 *
 *        C                    C
 *       / \                  /|\
 *      /   \                / | \
 *     A-----B     ->       A--M--B
 *      \   /                \ | /
 *       \ /                  \|/
 *        D                    D
 *
 * Rewrites the incident triangle(s) and appends one triangle per side. M is the
 * minimum-image midpoint, wrapped into the domain -- or, for an edge of an open
 * surface's RIM, the point of the rim curve over it (see
 * MFTL_RemeshParameters::rim_project), then the projector if one is set. M
 * inherits the edge's FEATURE/BOUNDARY classification, as do both halves. M and
 * the halves are marked visited, so the same sweep neither re-splits nor
 * collapses them.
 *
 * @param obj    The mesh. **May reallocate the pools.**
 * @param h      Half-edge index, normally one MFTLSplitApproved() accepted.
 * @param params Supplies the corner angle and the rim projector of a rim split;
 *               #MFTL_NULL for 45 degrees and none.
 * @return Pool index of M, or #MFTL_ERROR if @p h is not a live half-edge or
 *         the projector refused (nothing is modified then).
 */
int MFTLSplitEdge(MFTL_Object *obj, int h, const MFTL_RemeshParameters *params);

/**
 * @brief Is collapsing half-edge @p h (its origin into its target) legal and
 *        wanted?
 *
 * Wanted: shorter than `min_edge_length`. Each clause prevents a result that is
 * a valid data structure but not a valid surface, and none of them fails loudly:
 *
 * - **Features:** a corner is never retired; a feature (boundary) vertex may
 *   only collapse into a feature (boundary) vertex; two feature (boundary)
 *   vertices joined by a non-feature (interior) edge may not collapse -- that
 *   edge is a chord, and collapsing it pinches the curve.
 * - **Link condition:** the endpoints share exactly two neighbours (one at a
 *   boundary edge); otherwise two sheets are welded into a non-manifold vertex.
 * - **Valence:** the merged vertex has valence in [3, #MFTL_MAX_VALENCE], and
 *   each wing vertex has at least 4 neighbours (a tetrahedron passes the link
 *   condition and would dissolve).
 * - **Stretch:** no edge of the merged vertex exceeds `max_edge_length`, or
 *   split and collapse oscillate.
 * - **Fold-over:** no surviving triangle's normal turns by more than
 *   `max_normal_deviation_cos` or becomes degenerate (the test is relative to
 *   the triangle's own edges, so it is independent of the length unit).
 *
 * Truncated one-ring walks (valence beyond the scratch size) are refused.
 *
 * @param      obj           The mesh.
 * @param      h             Half-edge index; its ORIGIN would be retired.
 * @param      params        Supplies the length band, the fold-over threshold
 *                           and the feature settings.
 * @param[out] reject_reason May be #MFTL_NULL. Receives a
 *                           `MFTL_REMESH_REJECT_*` code.
 * @return `true` if the collapse is legal and wanted.
 */
bool MFTLCollapseApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, int *reject_reason);

/**
 * @name Rejection codes
 *
 * Reported through `reject_reason` of MFTLCollapseApproved() and
 * MFTLFlipApproved() and tallied into #MFTL_RemeshStats.
 * @{
 */
/** @brief Not rejected. */
#define MFTL_REMESH_REJECT_NONE 0
/** @brief Link condition or valence bound failed. */
#define MFTL_REMESH_REJECT_LINK 1
/** @brief A triangle would turn past `max_normal_deviation_cos` or degenerate. */
#define MFTL_REMESH_REJECT_FOLD 2
/** @brief Blocked by a feature, boundary or corner rule. */
#define MFTL_REMESH_REJECT_FEATURE 3
/** @brief Any other reason: not wanted, dead or visited entity, stretch bound, truncated one-ring. */
#define MFTL_REMESH_REJECT_OTHER 4
/** @} */

/**
 * @brief Collapses the edge of half-edge @p h, retiring its ORIGIN vertex.
 *
 * The target survives and moves to the position MFTLCollapseApproved()
 * evaluated with the same @p params: its own position if it is a corner or if
 * it alone lies on a feature curve or boundary; the rim curve's point (then the
 * projector) for an edge of an open surface's rim; otherwise the minimum-image
 * midpoint. The one or two incident triangles and the origin are flagged
 * #MFTL_FLAG_DEAD; pool indices do not change until MFTLRemeshCompactObject().
 *
 * @param obj    The mesh. Allocates nothing.
 * @param h      Half-edge index, normally one MFTLCollapseApproved() accepted.
 * @param params As passed to MFTLCollapseApproved(), or #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if @p h was not collapsible or the
 *         projector refused.
 */
int MFTLCollapseEdge(MFTL_Object *obj, int h, const MFTL_RemeshParameters *params);

/**
 * @brief Would flipping half-edge @p h strictly lower the squared valence
 *        deviation of the four quad vertices?
 *
 * \f[
 *   E = \sum_{v \in \{a,b,c,d\}} \big(\deg(v) - \deg^{*}(v)\big)^{2},
 *   \qquad \deg^{*} = \begin{cases} 6 & \text{interior} \\ 4 & \text{boundary} \end{cases}
 * \f]
 *
 * (\f$\deg\f$ counts neighbours). Only a STRICT decrease is accepted, or two
 * flips could alternate forever. Refused for boundary and feature edges, when
 * an endpoint has three neighbours or fewer, when the opposite vertices are
 * already adjacent (a duplicate edge), when a valence would exceed
 * #MFTL_MAX_VALENCE, and when either new triangle would turn past
 * `max_normal_deviation_cos` against the quad's normal (a non-convex quad).
 *
 * @param      obj           The mesh.
 * @param      h             Half-edge index.
 * @param      params        Supplies `max_normal_deviation_cos`.
 * @param[out] reject_reason May be #MFTL_NULL. #MFTL_REMESH_REJECT_FOLD only when
 *                           the valence criterion wanted the flip and the
 *                           geometry refused it.
 * @return `true` if the flip strictly lowers \f$E\f$ and is geometrically legal.
 */
bool MFTLFlipApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, int *reject_reason);

/**
 * @brief Flips the edge of half-edge @p h to the other diagonal of its quad.
 *
 * Both triangles are rewritten in place and four rim twins re-pointed; nothing
 * is allocated and no global id changes. The new diagonal is marked visited for
 * the rest of the sweep.
 *
 * @param obj The mesh.
 * @param h   Half-edge index, normally one MFTLFlipApproved() accepted.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR at a boundary edge.
 */
int MFTLFlipEdge(MFTL_Object *obj, int h);

// -----------------------------------------------------------------------------
// 3. COMPACTION AND IDENTITY  (src/remeshing/remesh_compact.c)
// -----------------------------------------------------------------------------

/**
 * @brief Removes every #MFTL_FLAG_DEAD entity and every orphaned vertex, and
 *        restores the ownership prefix.
 *
 * A forward compaction of both pools (half-edges move as triples with their
 * triangle, which preserves implicit indexing and the implicit half-edge ids)
 * followed, on a partitioned object, by a swap of the two trailing blocks: the
 * sweeps append past the ghosts, so the survivors come out as
 * [owned | ghosts | created] and the swap makes the owned entities a prefix
 * again. `he.vertex_idx` and `he.twin_idx` are remapped, `out_he_idx` is
 * re-derived from the surviving half-edges, the derived arrays are rebuilt, and
 * the cached geometry is permuted along with its entities.
 *
 * @param      obj   The mesh, compacted in place.
 * @param[out] stats May be #MFTL_NULL. `n_vertices_removed` and
 *                   `n_orphans_removed` are accumulated into it.
 * @return The number of entities removed (vertices plus triangles).
 * @rank_local
 */
int MFTLRemeshCompactObject(MFTL_Object *obj, MFTL_RemeshStats *stats);

/**
 * @brief Renumbers vertex and triangle global ids densely as \f$[0, n)\f$ in
 *        slot order and stamps the partition dirty.
 *
 * Asserts that the object holds no ghosts: the renumbering is a local decision
 * about a mesh-wide namespace, correct only when this rank holds the whole
 * object.
 *
 * @param obj The mesh. Every `global_id` and all three `next_*_gid` counters are
 *            rewritten; half-edge ids follow as \f$3\,t_\mathrm{gid} + k\f$.
 * @rank_local
 */
void MFTLRemeshRenumberGlobalIds(MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 4. FEATURES AND SMOOTHING  (src/remeshing/remesh_smooth.c)
// -----------------------------------------------------------------------------

/**
 * @brief Flags feature edges and classifies every vertex as free, ridge or
 *        corner.
 *
 * Recomputes `tris.normal` first (after a split, new triangles have none). Then,
 * in linear passes over undirected edges (no fan walks, which would miss the
 * closing edge of an open fan):
 *
 * - **edges:** #MFTL_FLAG_FEATURE on both halves of every edge whose face
 *   normals differ by more than @p angle_degrees, and on every mesh-boundary
 *   edge; cleared elsewhere, so a smoothed-out fold stops being a feature. A
 *   missing twin counts as a boundary only on an owned triangle -- on a ghost it
 *   is the rim of the halo, and #MFTL_FLAG_BOUNDARY decides;
 * - **vertices:** #MFTL_FLAG_FEATURE (ridge point) for exactly two feature edges
 *   continuing within @p corner_angle_degrees of straight; #MFTL_FLAG_CORNER for
 *   one, three or more, or two that turn more sharply; neither for none.
 *
 * @param      obj                  The mesh. Flags rewritten, `tris.normal`
 *                                  refreshed.
 * @param      angle_degrees        Normal-angle threshold in degrees;
 *                                  \f$\ge 180\f$ disables it, leaving only
 *                                  boundaries as features.
 * @param      corner_angle_degrees Turn angle above which a two-edge vertex is a
 *                                  corner.
 * @param[out] stats                May be #MFTL_NULL. Receives the edge and
 *                                  vertex-class counts.
 * @return The number of undirected feature edges.
 * @rank_local
 */
int MFTLDetectFeatures(MFTL_Object *obj, double angle_degrees, double corner_angle_degrees, MFTL_RemeshStats *stats);

/**
 * @brief Tangential smoothing: @p iterations Jacobi sweeps in which each vertex
 *        moves towards its neighbourhood centre, PROJECTED onto the subspace its
 *        feature class allows.
 *
 * - **Corner:** locked (0D).
 * - **Ridge:** the pull towards the midpoint of its two feature neighbours,
 *   projected onto the ridge tangent (1D). Moving to the midpoint itself cuts
 *   the chord and erodes the ridge.
 * - **Free:** the pull towards the target point of `smooth_mode`, minus its
 *   component along the area-weighted vertex normal (2D). Without the
 *   projection, Laplacian smoothing shrinks the surface.
 *
 * Vertex normals come from MFTLComputeVertexNormals(), refreshed every sweep; a
 * vertex whose normal is zero (degenerate one-ring) stays put. Steps are scaled
 * by `smooth_relaxation` and capped at `smooth_max_step_fraction` of the
 * shortest incident edge. All differences are minimum-image vectors and
 * positions are wrapped, so the smoother is seam-safe. Frozen vertices and
 * ghosts are not moved but are read, which makes a distributed sweep exact
 * without communication. Jacobi updates make the result independent of pool
 * order.
 *
 * @param obj        The mesh; positions updated in place.
 * @param params     Supplies `smooth_mode`, `smooth_relaxation` and
 *                   `smooth_max_step_fraction`.
 * @param iterations Sweeps to perform.
 *
 * @note Requires #MFTL_DERIVED_TOPOLOGY and a current feature classification.
 * @rank_local
 */
void MFTLSmoothTangential(MFTL_Object *obj, const MFTL_RemeshParameters *params, int iterations);

// -----------------------------------------------------------------------------
// 5. DRIVERS  (src/remeshing/remesh_driver.c)
// -----------------------------------------------------------------------------

/**
 * @brief Remeshes one whole object towards `params->target_length`: the serial
 *        entry point.
 *
 * Asserts that the object holds no ghosts -- a partitioned object needs
 * MFTLRemeshEnsemble(). Zeroes @p stats, runs up to `params->iterations`
 * iterations (see the file header), renumbers ids if requested, and returns an
 * object with valid topology, current derived arrays and current
 * `tris.area/normal/center`. `verts.normal` holds the normals of the last
 * smoothing sweep, not of the final positions; call MFTLComputeVertexNormals()
 * if needed.
 *
 * @param      obj    The mesh, remeshed in place.
 * @param      params The parameter block (MFTLInitRemeshParams()).
 * @param[out] stats  May be #MFTL_NULL.
 * @return #MFTL_SUCCESS.
 * @rank_local
 */
int MFTLRemeshObject(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

/**
 * @brief Remeshes every object of an ensemble: the entry point to use.
 *
 *     nproc == 1 -> MFTLRemeshObject() per object (no MPI; the serial answer)
 *     SPANNING   -> MFTLRemeshEnsembleParallel()
 *     DISPERSED  -> MFTLRemeshEnsembleDispersed()
 *     AUTO       -> measure the bodies once (MFTLRemeshMeasureBodies()), then
 *                   r_max > dispersed_radius_fraction * box ? SPANNING : DISPERSED
 *
 * The AUTO measurement is handed to the dispersed driver rather than retaken.
 * MFTL_RemeshStats::strategy reports what ran. Call graphs: @ref rem_dispatch,
 * @ref rem_spanning_flow, @ref rem_dispersed_flow.
 *
 * **Parameter check.** At `nproc > 1` every entry point agrees the whole
 * parameter block with one `MPI_Allreduce`, taken at the path's first
 * collective (after the local work before it, so it adds a latency, not a
 * synchronisation point). A band that is invalid on ANY rank
 * (\f$\ell_{\max} \le 0\f$, \f$\ell_{\min} < 0\f$ or
 * \f$\ell_{\min} > \ell_{\max}\f$), or a field that differs between ranks,
 * makes EVERY rank return #MFTL_ERROR with the mesh untouched; rank 0 names the
 * cause. A per-rank refusal would leave the other ranks in the next collective.
 * At `nproc == 1` only the band is tested.
 *
 * @param      ens    The ensemble, remeshed in place.
 * @param      params The parameter block, identical on every rank (checked).
 * @param[out] stats  May be #MFTL_NULL. Sums over the objects; see
 *                    #MFTL_RemeshStats for which fields are rank-local.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on every rank if the parameters were
 *         refused, or if any object or migration failed.
 *
 * @collective For `nproc > 1`, including ranks holding nothing. The strategy
 *             is decided from reduced data only; two ranks in different
 *             drivers would deadlock.
 */
int MFTLRemeshEnsemble(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

// -----------------------------------------------------------------------------
// 6. PARALLEL DRIVER  (src/remeshing/remesh_parallel.c)
// -----------------------------------------------------------------------------

/**
 * @brief Remeshes a distributed ensemble by shifting the cut planes (SPANNING).
 *
 * Normally reached through MFTLRemeshEnsemble(); public so the schedule can be
 * run at `nproc == 1` against the serial answer.
 *
 *     [optional pre-pass: measure band and frozen share, choose coarsening]
 *     per placement: migrate (skipped if identity) -> MFTLRemeshMarkFrozen
 *                    -> count hidden edges -> MFTLRemeshIterate on the interior
 *                    -> per-body id prefix sums over the holders
 *                    -> 3-int Allreduce (operations, hidden long, hidden short)
 *                    -> converged?
 *     at the end:    re-densify ids (ghost vertex ids exchanged)
 *                    -> MFTLRedistribute to the canonical partition -> geometry
 *
 * No ghost position sync is needed inside the loop: every ghost in a rank's
 * pool is frozen on its owner, so the copies are still exact after the sweeps.
 *
 * Preconditions: MFTLRegisterMpiTypes() called; the same bodies on every rank
 * in the same order with agreeing `next_*_gid` (checked in Debug); the same
 * parameter block on every rank with a valid band (checked, see
 * MFTLRemeshEnsemble()); \f$h > 4\delta\f$ (warned about once, see
 * MFTL_RemeshStats::converged).
 *
 * Postconditions: the canonical partition (the closing migration IS
 * MFTLRedistribute()); dense global ids, consistent between ghosts and owners;
 * valid topology, current derived arrays and `tris.area/normal/center`;
 * `verts.normal` NOT current (use MFTLEnsembleUpdateVertexNormals());
 * #MFTL_FLAG_FROZEN cleared.
 *
 * @param      ens    The ensemble, remeshed in place.
 * @param      params The parameter block.
 * @param[out] stats  May be #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on every rank if the parameters were
 *         refused, or if a migration or the id exchange failed.
 * @collective
 */
int MFTLRemeshEnsembleParallel(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

// -----------------------------------------------------------------------------
// 6b. DISPERSED DRIVER  (src/remeshing/remesh_dispersed.c)
// -----------------------------------------------------------------------------

/**
 * @brief Remeshes a distributed ensemble by moving the BODIES (DISPERSED).
 *
 * For bodies smaller than a rank box (emulsions, sprays, bubbly flows). Every
 * entity of a body is migrated to one rank -- its centroid's owner, adjusted by
 * the load balancer (`dispersed_balance_slack`) -- so the body arrives whole
 * with no ghosts and MFTLRemeshObject() runs on it unchanged. No freezing, no
 * phases, exactly two migrations (out, and the closing MFTLRedistribute()); ids
 * need only a per-body MAX of the id counters over the holders.
 *
 * Admission is tested BEFORE anything moves, on the body radius (a
 * one-destination migration would make any body local, even one spanning the
 * domain, and serialise it onto one rank): if any body is too large, the whole
 * ensemble goes to MFTLRemeshEnsembleParallel() untouched
 * (MFTL_RemeshStats::n_bodies_not_centred). Pre- and postconditions are those
 * of MFTLRemeshEnsembleParallel() without \f$h > 4\delta\f$.
 *
 * @param      ens    The ensemble, remeshed in place.
 * @param      params The parameter block.
 * @param[out] stats  May be #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on every rank if the parameters were
 *         refused, or if a migration failed.
 * @collective
 */
int MFTLRemeshEnsembleDispersed(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats);

/**
 * @brief Resolves #MFTL_REMESH_PARALLEL_AUTO for the ensemble as it stands.
 *
 * DISPERSED iff the largest body radius is at most `dispersed_radius_fraction`
 * times the narrowest rank box -- or, with `parallel_coarsen` armed and at
 * least as many bodies as ranks, the coarsened box (never the whole grid). The
 * test is on the LARGEST body, so a mixed ensemble resolves to SPANNING.
 *
 * @param ens    The ensemble whose bodies are measured.
 * @param params May be #MFTL_NULL for the default fraction.
 * @return #MFTL_REMESH_PARALLEL_SPANNING or #MFTL_REMESH_PARALLEL_DISPERSED;
 *         SPANNING at `nproc == 1`, where the question does not arise.
 * @collective Measures the bodies (MFTLRemeshMeasureBodies()).
 */
MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategy(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params);

/**
 * @brief Global centroid, radius and weight of every body this rank holds.
 *
 * Each quantity is reduced over the body's HOLDERS
 * (MFTLEnsembleReduceBodies()), so it describes the whole body; rows are
 * indexed by LOCAL SLOT (row `o` is `ens->local_objects[o]`).
 *
 * - **Centroid:** area-weighted mean of the owned triangle centres. A periodic
 *   axis is averaged as an angle (circular mean: map
 *   \f$\theta = 2\pi(x-x_{\min})/L\f$, average \f$(\cos\theta,\sin\theta)\f$
 *   with the area weights, return via `atan2`), because the arithmetic mean of a
 *   seam-straddling body lies in the middle of the domain.
 * - **Radius:** largest minimum-image distance from the centroid to a triangle
 *   CORNER.
 * - **Weight:** global owned-triangle count, a proxy for sweep time; it rides
 *   the centroid reduction.
 * - **Lowest holder:** the lowest rank holding the body, which designates one
 *   holder to count a body in sums over ranks.
 *
 * @param      ens           The ensemble.
 * @param[out] centroid      `[3 * n_local_objects]`, or #MFTL_NULL.
 * @param[out] radius        `[n_local_objects]`, or #MFTL_NULL.
 * @param[out] weight        `[n_local_objects]` (as double), or #MFTL_NULL.
 * @param[out] r_max         The largest radius over the WHOLE ensemble,
 *                           bit-identical on every rank, or #MFTL_NULL.
 * @param[out] lowest_holder `[n_local_objects]`, or #MFTL_NULL.
 *
 * @collective One per-body reduction of 9 values, then one of 1 value when
 *             @p radius or @p r_max is requested, plus one scalar `MPI_MAX` for
 *             @p r_max. Which are taken depends only on which pointers are
 *             null, so that must agree on every rank. Reads `tris.area` and
 *             `tris.center`: call MFTLComputeTriangleGeometry() first.
 */
void MFTLRemeshMeasureBodies(MFTL_Ensemble *ens, double *centroid, double *radius, double *weight, double *r_max, double *lowest_holder);

/**
 * @brief Per-axis coarsening factors that restore \f$g_k h_k > 4\delta\f$.
 *
 * \f$g_k = \lceil 4\delta/h_k \rceil\f$ on every axis with at least two cells,
 * 1 elsewhere, clamped to `cuts[k]`. Per axis because the precondition is: a
 * domain thin in x only loses a factor 2, not 8. A coarsened decomposition is
 * only an owner function (#MFTL_OwnerMap), so no second communicator exists.
 * The `parallel_coarsen` cap is applied by the driver, not here.
 *
 * @param      domain    The MPI/physical world.
 * @param      params    Supplies `max_edge_length` and `freeze_dilation_rings`
 *                       for the estimate used when @p band is 0.
 * @param      band      The band half-width \f$\delta\f$, e.g. a measured
 *                       MFTL_RemeshStats::band_bound; 0 falls back to
 *                       \f$(\texttt{MFTL\_REMESH\_BAND\_EDGES} +
 *                       \texttt{rings})\cdot\texttt{max\_edge\_length}\f$, which
 *                       overstates the real band about threefold.
 * @param[out] out_group [3] per-axis factors, each \f$\ge 1\f$.
 * @rank_local A pure function of replicated data, so all ranks agree without
 *             communicating.
 */
void MFTLRemeshCoarseningFactors(const MFTL_Domain *domain, const MFTL_RemeshParameters *params, double band, int out_group[3]);

/**
 * @brief Flags #MFTL_FLAG_FROZEN on every entity this rank may not modify.
 *
 * An owned entity is frozen exactly when another rank holds a copy of it.
 * Redistribution sends a triangle to the owner of each of its corners and edge
 * neighbours, so an owned triangle is replicated iff
 * `corner >= verts.n_local || neighbour >= tris.n_local` -- read from
 * `tris.verts` and `tris.neighbors`, no distance, no message. Frozen are: every
 * ghost, every replicated owned triangle, and every vertex of a frozen
 * triangle; the flag is cleared elsewhere. Hence an unfrozen vertex has only
 * owned, unreplicated incident triangles, and (with the vertex-complete halo)
 * its whole fan is local. On a ghost-free object nothing is frozen. Stamps the
 * partition dirty, since flags travel with kept ghosts.
 *
 * @param obj The mesh.
 * @note Requires #MFTL_DERIVED_TOPOLOGY (current after a redistribution).
 * @rank_local It describes this rank's halo; no other rank's answer applies.
 */
void MFTLRemeshMarkFrozen(MFTL_Object *obj);

/**
 * @brief Grows the frozen band by @p rings triangle rings.
 *
 * Not needed for correctness (see MFTL_RemeshParameters::freeze_dilation_rings);
 * exposed to test that claim. One ring freezes every triangle with a frozen
 * corner, then every corner of a frozen triangle; the two halves read and write
 * different flag arrays, so a pass is exactly one ring whatever the pool order.
 *
 * @param obj   The mesh.
 * @param rings Rings to add; 0 is a no-op.
 * @rank_local
 */
void MFTLDilateFrozenBand(MFTL_Object *obj, int rings);

/**
 * @brief Mean length of the undirected edges in this rank's pool.
 *
 * Every undirected edge counts once (a boundary edge is not under-weighted).
 * RANK-LOCAL: on a partitioned object it covers this rank's owned and ghost
 * triangles only, and a rank holding nothing of the body gets 0 -- a length band
 * of [0, 0], under which the pass refines without bound. Use
 * MFTLEnsembleMeanEdgeLength() on distributed meshes. The drivers refuse, on
 * every rank, a band that is invalid on any rank or a parameter block that
 * differs between ranks.
 *
 * @param obj The mesh.
 * @return The mean edge length, or 0 for an empty pool.
 * @rank_local
 */
double MFTLMeanEdgeLength(const MFTL_Object *obj);

/**
 * @brief Mean edge length over every body of the distributed ensemble.
 *
 * Each undirected edge is counted exactly once mesh-wide: an edge between two
 * owned triangles by its smaller half-edge, an edge cut by the partition by the
 * owner of the incident triangle with the smaller global id (counting both
 * sides would bias the mean by an amount growing with the rank count). The
 * result is identical on every rank.
 *
 * @param ens The ensemble.
 * @return The mean edge length, or 0 for an empty ensemble.
 * @collective One `MPI_Allreduce` of two doubles.
 */
double MFTLEnsembleMeanEdgeLength(MFTL_Ensemble *ens);

#endif  // MFTL_REMESHING_H
