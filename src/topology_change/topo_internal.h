/**
 * @file topo_internal.h
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief Internal interfaces of the topology-change module.
 *
 * ## THE FILES AND WHAT EACH MAY DO
 *
 *   topo_comm.c         EVERY MPI call of the module, and nothing else. The
 *                       same discipline as src/remeshing/remesh_comm.c: the
 *                       algorithm files are pure local computation over data a
 *                       collective handed them, which is what keeps the
 *                       deadlock argument checkable by reading one file.
 *   topo_detect.c       the pair predicate and the event list's canonical form
 *   topo_cluster.c      lattice, clusters (union-find), extents, windows
 *   topo_reconstruct.c  rasterisation, the bias, marching cubes, local meshes
 *   topo_split_bodies.c component labelling, ids, object construction
 *   topo_driver.c       the two public entry points, in collective order
 *
 * ## THE COLLECTIVE ORDER
 *
 * Every rank executes the same sequence of collectives in the same order,
 * whatever it holds. Nothing here returns early on a LOCAL count: a rank with
 * no triangles of a cluster still enters every exchange with empty buffers.
 */

#ifndef MFTL_TOPO_INTERNAL_H
#define MFTL_TOPO_INTERNAL_H

#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_meshgen.h"
#include "mftl_topology_change.h"
#include "mftl_types.h"

/**
 * @brief Whether @p r is this rank or one of its graph neighbours, i.e. whether
 *        a record for it can travel on `comm_graph`.
 * @param dom The MPI/physical world.
 * @param r   A rank.
 * @return `true` if the neighbourhood exchange reaches @p r.
 */
static inline bool MFTLTopoIsNeighbour(const MFTL_Domain *dom, int r) { return r == dom->rank || MFTLDomainNeighborIndex(dom, r) != MFTL_ERROR; }

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The lattice and the clusters
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_TopoLattice
 * @brief The global reconstruction lattice and this rank's place on it.
 */
typedef struct MFTL_TopoLattice
{
  double origin[3];  ///< Coordinate of node 0 (the domain's lower corner).
  double spacing[3];  ///< Cell size per axis.
  int cells[3];  ///< Cells per axis.
  int n_global[3];  ///< Nodes per axis: cells (periodic) or cells + 1.
  bool periodic[3];  ///< Domain periodicity.
  int per_rank[3];  ///< Cells per rank box per axis (cells / cuts).
  int coords[3];  ///< This rank's Cartesian coordinates.
  double r_cut;  ///< Rasterisation cutoff.
  int pad;  ///< Window padding in nodes beyond the box.
} MFTL_TopoLattice;

/** @brief One cluster: bodies reconstructed together, and their events. */
typedef struct MFTL_TopoCluster
{
  int *bodies;  ///< Body ids, ascending.
  int n_bodies;  ///< Their number.
  int *events;  ///< Indices into the event list.
  int n_events;  ///< Their number.
  bool refused;  ///< A body has a rim off the domain walls: left untouched.

  /* The global extent, as a centre and half-width per axis; `full[k]` when the
   * cluster covers the periodic axis (MFTL_PeriodicExtent::covers). */
  double centre[3];  ///< Extent centre.
  double half[3];  ///< Extent half-width.
  bool full[3];  ///< Covers the periodic axis: wraps it, or is at least 3/4 of it wide.
  bool empty;  ///< No rank holds a triangle of it.

  /* The volume measure: the flux of F = (x_k - ref) e_k, which the walls the
   * cluster's rims lie on do not contribute to. See MFTLTopoChooseFlux(). */
  int flux_axis;  ///< k, or -1 when no axis gives an exact volume.
  double flux_ref;  ///< ref.
  int rim_walls;  ///< Walls any rim of the cluster lies on, as bits 2 * axis + side.
  double volume_old;  ///< Volume before the reconstruction.

  bool has_window;  ///< This rank's window is non-empty.
  MFTL_SdfGrid grid;  ///< This rank's window of the field.
  MFTL_LevelSetSurface surf;  ///< Marching cubes output on this rank's cells.
} MFTL_TopoCluster;

/** @brief Unions bodies joined by a two-body coalescence event; clusters ascend by smallest body id. */
int MFTLTopoBuildClusters(const int *bodies, int n_bodies, const MFTL_TopologyEvent *events, int n_events, MFTL_TopoCluster **out);

/** @brief Frees the clusters and everything they own. */
void MFTLTopoFreeClusters(MFTL_TopoCluster *cl, int n);

/** @brief Index of the cluster body @p gid is in, or -1. */
int MFTLTopoClusterOf(const MFTL_TopoCluster *cl, int n, int gid);

/** @brief Fills the lattice for @p dom at spacing @p dx; `coords` comes from topo_comm. */
void MFTLTopoLatticeInit(MFTL_TopoLattice *lat, const MFTL_Domain *dom, double dx);

/**
 * @brief This rank's contribution to every cluster's extent:
 *        #MFTL_PERIODIC_EXTENT_SLOTS MIN values per cluster.
 */
void MFTLTopoLocalExtents(const MFTL_Ensemble *ens, const MFTL_TopoCluster *cl, int n_cl, double *out);

/** @brief Turns reduced extents into centre/half/full per cluster. */
void MFTLTopoResolveExtents(const MFTL_Domain *dom, MFTL_TopoCluster *cl, int n_cl, const double *reduced);

/** @brief Largest distance any OWNED triangle of a cluster body reaches beyond this rank's box. */
double MFTLTopoLocalOverhang(const MFTL_Ensemble *ens, const MFTL_TopoCluster *cl, int n_cl, const MFTL_TopoLattice *lat);

/** @brief Allocates this rank's window for every cluster it can contribute to or needs. */
void MFTLTopoMakeWindows(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl);

/** @brief Picks each cluster's volume measure from its rim walls and extent. */
void MFTLTopoChooseFlux(const MFTL_Domain *dom, MFTL_TopoCluster *cl, int n_cl);


/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Detection
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief A growable list of events. */
typedef struct MFTL_TopoEventList
{
  MFTL_TopologyEvent *e;  ///< The events.
  int n;  ///< Count.
  int cap;  ///< Capacity.
} MFTL_TopoEventList;

/** @brief Runs the pair search on this rank; events are appended, un-deduplicated. */
int MFTLTopoDetectLocal(MFTL_Ensemble *ens, const MFTL_TopologyOptions *opts, MFTL_TopoEventList *out);

/**
 * @brief Snaps every event point to a cell of size @p quantum, orders the list
 *        canonically and merges events on the same (bodies, kind, cell).
 */
void MFTLTopoCanonicaliseEvents(const MFTL_Domain *dom, double quantum, MFTL_TopoEventList *list);

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Reconstruction
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Rasterises this rank's OWNED triangles of every cluster into its window. */
void MFTLTopoRasterise(const MFTL_Ensemble *ens, MFTL_TopoCluster *cl, int n_cl, const MFTL_TopoLattice *lat);

/**
 * @brief Packs every known window node that another rank needs:
 *        records `(cluster, node key, value)`, bucketed by destination rank.
 *
 * @param dom              The domain: rank map and neighbourhood.
 * @param lat              The lattice.
 * @param cl               The clusters, with their windows rasterised.
 * @param n_cl             Their number.
 * @param[out] dest_counts `[nproc]` records per destination.
 * @param[out] records     `malloc`'d, grouped by destination in rank order.
 * @return Whether any destination lies outside this rank's graph neighbourhood.
 */
bool MFTLTopoPackNodeExchange(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, const MFTL_TopoCluster *cl, int n_cl, int *dest_counts, double **records);

/** @brief Merges received `(cluster, node key, value)` records into the windows. */
void MFTLTopoMergeNodes(const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, const double *records, int n_records);

/** @brief Adds each cluster's event bias to its known window nodes. */
void MFTLTopoApplyBias(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, const MFTL_TopologyEvent *events,
                       const MFTL_TopologyOptions *opts);

/** @brief Marching cubes over this rank's cells plus one ghost layer, per cluster. */
void MFTLTopoExtract(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, double snap);

/** @brief The rank owning lattice cell key @p cell. */
int MFTLTopoCellOwner(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, long long cell);

/** @brief Largest bias magnitude any event can apply. */
double MFTLTopoMaxBias(const MFTL_TopologyOptions *opts);

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Separation
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_TopoPiece
 * @brief One cluster's local marching cubes mesh, ready for labelling.
 */
typedef struct MFTL_TopoPiece
{
  int rank;  ///< The rank this piece lives on; a ghost is a triangle it does not own.
  MFTL_Object *mesh;  ///< Scratch object: all local triangles, twins linked. Not in any ensemble.
  int *tri_owner;  ///< [n_tris] owning rank of each triangle.
  int *vert_owner;  ///< [n_verts] owning rank of each vertex.
  long long *tkey;  ///< [n_tris] lattice key: 16 * cell + slot.
  long long *vkey;  ///< [n_verts] lattice key of the edge (or centroid) each vertex is on.
  long long *label;  ///< [n_tris] component label (smallest triangle key in the component).
  int *comp;  ///< [n_tris] local component index.
  int n_comp;  ///< Local components.
  long long *owned_key;  ///< Sorted keys of the OWNED triangles.
  int *owned_idx;  ///< Their triangle index, parallel to owned_key.
  int n_owned;  ///< Owned triangles.
} MFTL_TopoPiece;

/** @brief Builds each cluster's scratch mesh from its marching cubes soup. */
void MFTLTopoBuildPieces(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, MFTL_TopoPiece *pieces);

/** @brief Frees the pieces. */
void MFTLTopoFreePieces(MFTL_TopoPiece *pieces, int n);

/**
 * @brief One local round of labelling: every local component takes the
 *        smallest label of any of its triangles. Returns whether anything changed.
 */
bool MFTLTopoLabelLocal(MFTL_TopoPiece *pieces, int n_cl);

/** @brief Packs `(cluster, triangle key, label)` for every GHOST triangle, by owner. */
void MFTLTopoPackLabelRequests(const MFTL_TopoPiece *pieces, int n_cl, int nproc, int *dest_counts, double **records);

/**
 * @brief Owner side of one labelling round: lowers the labels of the owned
 *        triangles named in @p req and writes each one's (possibly lowered)
 *        component label into @p reply. Returns whether anything changed.
 */
bool MFTLTopoServeLabelRequests(MFTL_TopoPiece *pieces, int n_cl, const double *req, int n_req, double *reply);

/** @brief Holder side: takes the owners' replies for its ghost triangles, in request order. */
bool MFTLTopoApplyLabelReplies(MFTL_TopoPiece *pieces, int n_cl, int nproc, const double *reply);

/**
 * @brief Records `(cluster, label, owned triangles, open)` for every distinct
 *        global label with an owned triangle on this rank.
 */
int MFTLTopoComponentRecords(const MFTL_TopoLattice *lat, const MFTL_TopoPiece *pieces, int n_cl, double **records);

/** @brief A decided component: which body it becomes, or none. */
typedef struct MFTL_TopoComponent
{
  int cluster;  ///< Cluster index.
  long long label;  ///< Global label.
  long long n_tris;  ///< Global triangle count.
  bool open;  ///< Has a boundary on a wall.
  int body;  ///< Body id it becomes, or -1 if discarded.
} MFTL_TopoComponent;

/**
 * @brief Decides every component's fate and id, identically on every rank.
 *
 * @param all        Gathered component records.
 * @param n_all      Their number.
 * @param cl         The clusters; refused ones are skipped.
 * @param n_cl       Their number.
 * @param bodies     `[n_bodies]` the listed body ids, sorted and unique.
 * @param body_size  `[n_bodies]` global triangle count of each listed body, in the order of @p bodies.
 * @param n_bodies   Their number.
 * @param min_tris   Closed components with fewer triangles are discarded.
 * @param n_bodies_global In: the id watermark. Out: raised by the new ids.
 * @param[out] comps Sorted by (cluster, label); `malloc`'d.
 * @param[out] stats Created/retired/discarded counts are written.
 * @return The number of components.
 */
int MFTLTopoDecideComponents(const double *all, int n_all, const MFTL_TopoCluster *cl, int n_cl, const int *bodies, const long long *body_size, int n_bodies,
                             int min_tris, int *n_bodies_global, MFTL_TopoComponent **comps, MFTL_TopologyStats *stats);

/** @brief The decided component (cluster, label), or #MFTL_NULL. */
const MFTL_TopoComponent *MFTLTopoFindComponent(const MFTL_TopoComponent *comps, int n, int cluster, long long label);

/**
 * @brief Builds this rank's objects for every kept component it owns anything
 *        of, with PROVISIONAL ids, and swaps them for the old bodies in the
 *        ensemble.
 *
 * @param ens        The ensemble; every listed body of a reconstructed cluster is erased from it.
 * @param lat        The lattice, for the wall flags of rim edges.
 * @param cl         The clusters.
 * @param n_cl       Their number.
 * @param pieces     `[n_cl]` this rank's marching cubes output, labelled.
 * @param comps      The decided components, sorted by (cluster, label).
 * @param n_comps    Their number.
 * @param[out] created `malloc`'d body ids of the new objects, sorted.
 * @param[out] vkeys `[n_new]` per new object, the lattice key of each vertex (malloc'd).
 * @param[out] tkeys `[n_new]` per new object, the lattice key of each triangle.
 * @return Number of new objects created on this rank; @p created lists their body ids.
 */
int MFTLTopoBuildObjects(MFTL_Ensemble *ens, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, MFTL_TopoPiece *pieces,
                         const MFTL_TopoComponent *comps, int n_comps, int **created, long long ***vkeys, long long ***tkeys);

/** @brief Packs `(body, kind, key)` id requests for every ghost of the new objects, by owner. */
void MFTLTopoPackIdRequests(const MFTL_Ensemble *ens, const int *created, int n_created, long long **vkeys, long long **tkeys, int nproc, int *dest_counts,
                            double **records);

/** @brief Owner side: answers `(body, kind, key)` with the densified id. */
void MFTLTopoServeIdRequests(const MFTL_Ensemble *ens, const int *created, int n_created, long long **vkeys, long long **tkeys, const double *req, int n_req,
                             double *reply);

/** @brief Holder side: writes the replies into the ghosts, in request order. */
void MFTLTopoApplyIdReplies(MFTL_Ensemble *ens, const int *created, int n_created, int nproc, const double *reply);

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Communication (topo_comm.c -- the only file with MPI)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief This rank's Cartesian coordinates. */
void MFTLTopoCommCoords(const MFTL_Domain *dom, int coords[3]);

/** @brief Wall-clock seconds. */
double MFTLTopoCommTime(void);

/** @brief In-place Allreduce of @p n doubles. op: 0 = MIN, 1 = MAX, 2 = SUM. */
void MFTLTopoCommAllreduceDouble(const MFTL_Domain *dom, double *v, int n, int op);

/** @brief In-place Allreduce of @p n long longs. op: 0 = MIN, 1 = MAX, 2 = SUM. */
void MFTLTopoCommAllreduceLongLong(const MFTL_Domain *dom, long long *v, int n, int op);

/** @brief In-place logical OR over ranks. */
bool MFTLTopoCommAny(const MFTL_Domain *dom, bool local);

/**
 * @brief Allgatherv of records of @p stride doubles.
 * @return The gathered records, `malloc`'d, in rank order; @p n_out records.
 */
double *MFTLTopoCommAllgather(const MFTL_Domain *dom, const double *mine, int n_mine, int stride, int *n_out);

/**
 * @brief Personalised exchange of records of @p stride doubles.
 *
 * @param dom         The domain: communicator and neighbourhood graph.
 * @param records     Grouped by destination rank, `dest_counts[r]` records for rank r.
 * @param dest_counts `[nproc]` records per destination.
 * @param stride      Doubles per record.
 * @param global_path Whether any rank has a destination outside its neighbourhood;
 *                    agreed inside, so a local value is enough.
 * @param[out] n_recv Records received.
 * @param[out] src_counts Optional `[nproc]`: records received from each rank.
 * @return Received records, `malloc`'d, grouped by source rank in rank order.
 */
double *MFTLTopoCommExchange(const MFTL_Domain *dom, const double *records, const int *dest_counts, int stride, bool global_path, int *n_recv, int *src_counts);

/**
 * @brief Sends @p reply (one record of @p stride doubles per received request,
 *        in receive order) back to the requesters. The inverse of
 *        MFTLTopoCommExchange() with the counts swapped.
 *
 * @param dom         The domain.
 * @param reply       One record per received request, in receive order.
 * @param src_counts  `[nproc]` requests received from each rank, as
 *                    MFTLTopoCommExchange() reported them.
 * @param stride      Doubles per record.
 * @param global_path Whether the request exchange went global; agreed inside.
 * @return The replies, in the order the requests were packed.
 */
double *MFTLTopoCommReply(const MFTL_Domain *dom, const double *reply, const int *src_counts, int stride, bool global_path);

#endif  // MFTL_TOPO_INTERNAL_H
