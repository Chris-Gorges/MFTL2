/**
 * @file mftl_topology_change.h
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief Topology changes -- breakup and coalescence -- by distributed
 *        signed-distance rasterisation and marching cubes, on the partition as
 *        it stands.
 *
 * ## WHAT HAPPENS
 *
 *   1. DETECT. Pairs of triangles with opposing normals closer than a threshold,
 *      within one body and between two (MFTLEnsembleIntersectObjects with
 *      self-proximity). A pair whose gap lies INSIDE the liquid is a thread or
 *      sheet about to BREAK; one whose gap lies outside is a film about to
 *      RUPTURE and let two surfaces COALESCE.
 *   2. AGREE. Every rank learns every event, and the bodies they touch are
 *      grouped into CLUSTERS by union-find: bodies joined by a coalescence event
 *      are reconstructed together.
 *   3. RECONSTRUCT each cluster's bodies WHOLE from their signed distance field
 *      on a global, wall-aligned lattice, with a smooth local bias at every
 *      event that opens the thread or closes the film -- so the outcome is
 *      decided by the event, not by whether a lattice node happens to fall in a
 *      gap. Marching cubes then triangulates the zero level set.
 *   4. SEPARATE the result into connected components, one body each.
 *   5. REDISTRIBUTE to the canonical partition, and optionally remesh.
 *
 * ## NOTHING MIGRATES
 *
 * Every step runs on the partition the caller has. Each rank rasterises its
 * OWNED triangles onto a window around its own box and the neighbours merge the
 * overlap by a commutative rule, so every rank ends up with the exact field on
 * its box. Each rank then runs marching cubes on its own cells plus one layer of
 * its neighbours', which builds the ghost halo directly. A surface vertex is
 * named by the lattice EDGE it lies on and produced from bit-identical values on
 * every rank that produces it, so the ranks' pieces fit together with no
 * stitching and no negotiation. Components are labelled by propagating the
 * smallest triangle key across ghost triangles until nothing changes.
 *
 * The reconstructed surface is therefore independent of the process count up to
 * the last bit of the pseudo-normal sums (see test_topology_change.c).
 *
 * ## WHAT IT COSTS, AND WHAT IT CHANGES
 *
 * A reconstructed body is re-triangulated at the lattice spacing EVERYWHERE, not
 * only at the event: vertices are new, ids are new, and its surface moves by
 * the marching cubes error, O(dx^2) times the curvature, with a local volume
 * change of order bias x area at each event. Bodies not in a cluster are not
 * touched at all. A rim on a domain wall is reproduced ON the wall, as the
 * surface's intersection with the wall plane at lattice resolution; a rim
 * anywhere else cannot be represented by a level set, so a cluster containing
 * one is refused rather than corrupted.
 *
 * Collective throughout: every rank calls with the same options.
 */

#ifndef MFTL_TOPOLOGY_CHANGE_H
#define MFTL_TOPOLOGY_CHANGE_H

#include "mftl_types.h"

struct MFTL_RemeshParameters;

/** @brief What kind of thin feature an event is. */
typedef enum MFTL_TopologyEventKind
{
  /** The gap between the two triangles is INSIDE the body: a thread or a sheet
   *  of liquid thinner than `delta_breakup`. It is cut open. */
  MFTL_TOPOLOGY_BREAKUP = 0,
  /** The gap is OUTSIDE: a film of the surrounding phase thinner than
   *  `delta_coalescence`, between two bodies or two parts of one. It is closed. */
  MFTL_TOPOLOGY_COALESCENCE = 1
} MFTL_TopologyEventKind;

/**
 * @struct MFTL_TopologyEvent
 * @brief One detected thin feature: where, which bodies, which kind.
 */
typedef struct MFTL_TopologyEvent
{
  int body_a;  ///< Body id of the reporting side.
  int body_b;  ///< Body id of the other side; equal to `body_a` within one body.
  int kind;  ///< An #MFTL_TopologyEventKind.
  double point[3];  ///< Midpoint of the gap, wrapped into the domain.
  double gap;  ///< Distance between the two triangles.
} MFTL_TopologyEvent;

/**
 * @struct MFTL_TopologyOptions
 * @brief Everything a topology change needs besides the ensemble.
 *
 * Start from MFTLTopologyDefaultOptions() and set the three lengths: a
 * zero-initialised struct has zero thresholds and detects nothing.
 */
typedef struct MFTL_TopologyOptions
{
  double delta_breakup;  ///< A liquid thread or sheet thinner than this breaks. Absolute length.
  double delta_coalescence;  ///< A film between surfaces thinner than this ruptures. Absolute length.
  double normal_tol_deg;  ///< Two triangles oppose each other when n_A . n_B <= -cos(this). Default 60.

  /**
   * @brief Target lattice spacing for the reconstruction. Absolute length.
   *
   * The spacing actually used divides every axis into a whole number of cells
   * per rank -- see MFTLTopologyLattice() -- so it may differ slightly per axis.
   * Around the target edge length of the remesher that follows is right: finer
   * wastes time on a surface the remesher coarsens anyway, coarser loses
   * features the detection did not flag.
   */
  double dx;

  /**
   * @brief Bias magnitude, as a multiple of half the threshold of the event.
   *
   * A breakup event adds `+bias_factor * delta_breakup / 2` to the field around
   * its gap, which opens any thread of radius below that; a coalescence event
   * subtracts `bias_factor * delta_coalescence / 2`, which closes any film
   * thinner than twice that. Above 1 so that a feature just at the threshold is
   * decided with margin rather than by rounding. Default 1.5.
   */
  double bias_factor;

  /**
   * @brief Radius of the bias bump around each event point, in units of
   *        max(delta, dx). Default 1.5.
   *
   * The bump has full weight out to half its radius, i.e. 0.75 delta by
   * default -- beyond the surface of any thread thinner than delta, whose event
   * points lie on its axis -- and falls smoothly to zero at the radius. Larger
   * spends the bias on the bodies either side of the feature. Measured on the
   * pinched dumbbell of test_topology_change.c (bulbs of radius 0.45, lattice
   * 0.067), volume lost per bulb: 2.9 % at 1.5 against 4.0 % at 2 with
   * delta_breakup = 0.12, and 6.7 % against 11 % with delta_breakup = 0.2 --
   * which is also why delta should be set from the feature size the flow can
   * resolve, not generously.
   */
  double bias_radius_factor;

  /** @brief Marching cubes snap distance, relative to dx. See MFTLExtractLevelSet(). Default 0.02. */
  double mc_snap;

  /**
   * @brief Closed components with fewer triangles than this are deleted.
   *
   * A satellite smaller than a few lattice cells cannot be represented and
   * would be collapsed into a degenerate shape by the remesher. The count is
   * taken TWICE: on the marching-cubes surface, and again after the `remesh`
   * pass when one runs -- a satellite that clears the floor as a few dozen MC
   * slivers can come out of a remesh at a comparable target edge length as a
   * tetrahedron, which the collapse approval then refuses to dissolve. A body
   * deleted at either point gives its volume to its cluster's surviving bodies
   * through `conserve_volume`. 0 keeps everything. An OPEN component is never
   * deleted. Default 16.
   */
  int min_body_triangles;

  /**
   * @brief Restore each cluster's enclosed volume after the reconstruction.
   *
   * Default ON. Marching cubes cuts chords across a curved surface, so every
   * reconstruction loses O(dx^2 * curvature) of volume per unit area -- on the
   * liquid jet about one percent of the jet PER reconstruction, measured as 60
   * of a final 81 volume units over a run of 45 reconstructions without it. The
   * liquid a cluster encloses is therefore measured before and after, and the
   * difference is put back by a uniform offset along the area-weighted vertex
   * normals (a rim vertex on a wall moves only within the wall), in three
   * Newton steps on the exact first variation of the volume -- 1e-9 relative
   * from a 15 % deficit, measured. It runs AFTER the optional `remesh`, which
   * moves volume too. It also puts back what the bias moved, which is what a solver
   * wants: a thread that pinches off does not destroy liquid, and a film that
   * ruptures does not create any.
   *
   * The volume of an open surface is measured together with the walls its rims
   * lie on, as the flux of (x_k - x_k,min) e_k along an axis whose far wall
   * carries no rim. A cluster for which no such axis exists -- rims on both
   * walls of every non-periodic axis -- is left uncorrected.
   */
  bool conserve_volume;

  /**
   * @brief Remesh the ensemble after a reconstruction, with these parameters.
   *
   * #MFTL_NULL for none. Marching cubes output has slivers and edges from a few
   * percent of dx up to sqrt(2) dx, and a solver almost always wants it
   * remeshed before the next step; the call is skipped when nothing changed.
   * Only MFTLRemeshEnsemble() is run -- all bodies take part, as they would in
   * the caller's own remeshing pass.
   */
  const struct MFTL_RemeshParameters *remesh;
} MFTL_TopologyOptions;

/**
 * @struct MFTL_TopologyStats
 * @brief What one call did. Every field is GLOBAL, identical on every rank.
 */
typedef struct MFTL_TopologyStats
{
  int n_events;  ///< Events after agreement and de-duplication.
  int n_breakup;  ///< Of which breakup events.
  int n_coalescence;  ///< Of which coalescence events.
  int n_clusters;  ///< Clusters reconstructed.
  int n_clusters_refused;  ///< Clusters left untouched because a body had a rim off the domain walls.
  int n_bodies_reconstructed;  ///< Bodies that went into a reconstruction.
  int n_bodies_out;  ///< Bodies that came out of it.
  int n_bodies_created;  ///< Of those, bodies given a NEW id.
  int n_bodies_retired;  ///< Ids that went in and no longer exist.
  int n_bodies_discarded;  ///< Closed components below `min_body_triangles`, deleted.
  int n_label_rounds;  ///< Rounds the distributed component labelling needed.
  double volume_before;  ///< Volume of the reconstructed clusters before (summed; see `conserve_volume`).
  double volume_raw;  ///< The same, straight out of marching cubes -- the error the correction removes.
  double volume_after;  ///< The same, at the end.
  int lattice_cells[3];  ///< Lattice cells per axis.
  double t_detect;  ///< Seconds in detection and agreement (max over ranks).
  double t_reconstruct;  ///< Seconds in rasterisation, exchange and marching cubes.
  double t_separate;  ///< Seconds in labelling, id assignment and object construction.
  double t_redistribute;  ///< Seconds in the closing redistribution.
  double t_remesh;  ///< Seconds in the optional remesh.
  double t_total;  ///< Seconds in the whole call.
} MFTL_TopologyStats;

/**
 * @brief The defaults: 60 degrees, bias 1.5 over a radius of 1.5, snap 0.02,
 *        16-triangle satellites dropped, no remesh -- and ZERO lengths, which
 *        the caller must set.
 * @return The option set to start from.
 */
MFTL_TopologyOptions MFTLTopologyDefaultOptions(void);

/**
 * @brief The reconstruction lattice for a domain and a target spacing.
 *
 * Axis k has `cells[k] = cuts[k] * max(1, round(L_k / (cuts[k] * dx)))` cells
 * of `spacing[k] = L_k / cells[k]`, anchored at `global_min[k]`. Two properties
 * follow and the whole reconstruction rests on them: a node plane lies on every
 * domain wall, so a surface meeting a wall is cut exactly there, and a node
 * plane lies on every rank's cut plane, so every lattice cell belongs to
 * exactly one rank.
 *
 * @param      domain  The domain.
 * @param      dx      Target spacing, > 0.
 * @param[out] spacing Actual spacing per axis.
 * @param[out] cells   Cells per axis.
 * @rank_local Pure function of the domain.
 */
void MFTLTopologyLattice(const MFTL_Domain *domain, double dx, double spacing[3], int cells[3]);

/**
 * @brief Detects the thin features of the ensemble and agrees on them.
 *
 * The returned list is IDENTICAL on every rank: gathered, sorted and with
 * duplicates removed (a gap seen from both sides, or by two ranks, is one
 * event; event points in one cell of a quarter of max(smaller threshold, dx)
 * are merged). Pairs sharing a vertex are never events.
 *
 * @param      ens      The ensemble. Needs current derived connectivity; reads
 *                      positions only.
 * @param      opts     The options; the thresholds, the normal tolerance and
 *                      `dx` (which sets the merging distance) are read.
 * @param[out] events   `malloc`'d list, free() it. #MFTL_NULL when empty.
 * @param[out] n_events Its length.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid arguments (on every rank).
 * @collective
 */
int MFTLTopologyDetectEvents(MFTL_Ensemble *ens, const MFTL_TopologyOptions *opts, MFTL_TopologyEvent **events, int *n_events);

/**
 * @brief Reconstructs the listed bodies from their signed distance field,
 *        biased at the listed events, and splits the result into bodies.
 *
 * The core of MFTLEnsembleApplyTopologyChanges(), exposed because it is useful
 * on its own -- re-gridding a body, or reconstructing one with no events at all
 * to test the pipeline -- and because it is where every distributed property of
 * the module lives.
 *
 * CLUSTERS are the connected components of the listed bodies under the events
 * between two of them -- of either kind: a coalescence joins two bodies, and a
 * liquid film between a drop and a bubble inside it is a breakup event between
 * two bodies that bound one liquid from opposite sides. Each cluster's bodies
 * are rasterised together, so the field is the signed distance to the UNION of
 * their surfaces, signed by the nearest one -- which is correct for nested
 * bodies as well, where a min over per-body fields would not be.
 * Ids: in each cluster the largest resulting component keeps the id of the
 * largest body that went in, the second the second, and so on; components
 * beyond that get new ids, and bodies beyond that are retired. So the jet a
 * drop pinches off stays the jet, and the pool a drop falls into stays the pool.
 *
 * Closes with MFTLRedistribute() to the canonical partition, current triangle
 * geometry and global counts -- and then, when `opts->remesh` is set, one
 * MFTLRemeshEnsemble() over the whole ensemble.
 *
 * @param ens      The ensemble. After the call every #MFTL_Object pointer of a
 *                 listed body is invalid -- use MFTLEnsembleFindLocalObject() --
 *                 and so are all local indices, as after any redistribution.
 * @param bodies   `[n_bodies]` body ids to reconstruct, identical on every rank.
 * @param n_bodies Their number.
 * @param events   `[n_events]` events, identical on every rank; each biases the
 *                 field of the cluster its bodies are in. May be #MFTL_NULL.
 * @param n_events Their number.
 * @param opts     The options.
 * @param[out] stats Optional.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR (on every rank).
 * @collective
 */
int MFTLEnsembleReconstructBodies(MFTL_Ensemble *ens, const int *bodies, int n_bodies, const MFTL_TopologyEvent *events, int n_events,
                                  const MFTL_TopologyOptions *opts, MFTL_TopologyStats *stats);

/**
 * @brief Detects, reconstructs and remeshes: one topology-change step.
 *
 * MFTLTopologyDetectEvents(), then MFTLEnsembleReconstructBodies() on every
 * body an event touches (which remeshes when `opts->remesh` is set). A call
 * that finds no event changes nothing at all: no redistribution, no remesh,
 * every pool untouched.
 *
 * Call it where a remeshing pass would go: after the vertices moved and the
 * ghost positions were synchronised, before the geometry a solver reads.
 *
 * @param ens   The ensemble. Same pointer rules as MFTLEnsembleReconstructBodies().
 * @param opts  The options.
 * @param[out] stats Optional.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR (on every rank).
 * @collective
 */
int MFTLEnsembleApplyTopologyChanges(MFTL_Ensemble *ens, const MFTL_TopologyOptions *opts, MFTL_TopologyStats *stats);

#endif  // MFTL_TOPOLOGY_CHANGE_H
