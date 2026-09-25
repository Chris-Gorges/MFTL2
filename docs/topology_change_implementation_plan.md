# MFTL Topology Changes Implementation Plan: SDF-Based Marching Cubes

This document specifies the technical design, mathematical formulations, algorithmic workflow, and MPI parallel architecture for implementing topology changes (breakup and coalescence) in the MultiFlow Triangulation Library (MFTL) using Signed Distance Field (SDF) rasterization and Marching Cubes (MC) iso-surface extraction.

---

## 1. Architectural Strategy: Component Grids vs. Global Grid

To maintain optimal memory scaling and avoid communication bottlenecks across large MPI process counts, **independent Cartesian Eulerian sub-grids are allocated per interacting component cluster**, rather than a single domain-wide Eulerian mesh:

1. **Avoid Wasteful Allocations:** A single bounding box covering distant interacting bodies (e.g., body A coalescing on Rank 0 and body B pinching off on Rank 100) spans nearly the full simulation domain, allocating billions of inactive cells.
2. **Cluster Localization:** Disjoint-set grouping identifies connected interaction clusters $C_k = \{B_1, B_2, \dots\}$. Each cluster gets an independent, tight Eulerian grid bounded by $[\mathbf{x}_{\min}^{C} - 3\Delta x, \, \mathbf{x}_{\max}^{C} + 3\Delta x]$.
3. **Dispersed Migration Pattern:** If a cluster fits within a rank's neighborhood, bodies are temporarily migrated to the rank owning their shared centroid (reusing MFTL's dispersed strategy pattern in `src/remeshing/remesh_dispersed.c`). All SDF queries, Marching Cubes, and Breadth-First Search (BFS) body splits execute locally without MPI overhead.
4. **Spanning Path:** If an interacting structure exceeds the rank box size, a localized Cartesian grid is instantiated collectively across the involved sub-communicator ranks, partitioned along existing Cartesian cuts.

---

## 2. Mathematical Formulations

### 2.1 Detection Criteria

#### Intra-Body Pinch-Off (Breakup)
For two triangles $T_i, T_j$ on the same body $B$:
$$\text{dist}(T_i, T_j) \le \delta_{\text{breakup}} \quad \text{and} \quad \mathbf{n}_{T_i} \cdot \mathbf{n}_{T_j} \le -\cos(\theta_{\text{tol}}) \quad \text{with } T_j \notin \text{star}(T_i)$$
Triangles sharing a vertex or edge (1-ring neighborhood) are excluded to prevent self-detection on curved surfaces.

#### Inter-Body Thin-Film Rupture (Coalescence)
For two triangles $T_i \in B_A$ and $T_j \in B_B$:
$$\text{dist}(T_i, T_j) \le \delta_{\text{coalescence}} \quad \text{and} \quad \mathbf{n}_{T_i} \cdot \mathbf{n}_{T_j} \le -\cos(\theta_{\text{tol}})$$
where $\theta_{\text{tol}} \approx 60^\circ$ enforces opposing normal vectors ($\mathbf{n}_{T_i} \cdot \mathbf{n}_{T_j} < 0$).

### 2.2 Point-to-Triangle Distance Projection
Given query grid node $\mathbf{x}$ and triangle $T = (\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)$ with edges $\mathbf{e}_0 = \mathbf{v}_1 - \mathbf{v}_0$ and $\mathbf{e}_1 = \mathbf{v}_2 - \mathbf{v}_0$:
$$\mathbf{p}(s, t) = \mathbf{v}_0 + s\mathbf{e}_0 + t\mathbf{e}_1, \quad s \ge 0, \; t \ge 0, \; s + t \le 1$$
We minimize $\Vert{}\mathbf{p}(s, t) - \mathbf{x}\Vert{}^2$ over the 7 Voronoi regions (face interior, 3 edges, 3 vertices) using clamped quadratic minimization. The projection yields the closest point $\mathbf{p}_{\text{closest}} \in T$ and distance:
$$d(\mathbf{x}, T) = \Vert{}\mathbf{x} - \mathbf{p}_{\text{closest}}\Vert{}$$

### 2.3 Sign Determination via Angle-Weighted Pseudo-Normals
The sign is determined using the pseudo-normal $\mathbf{n}_{\text{pseudo}}$ evaluated at $\mathbf{p}_{\text{closest}}$:
$$\text{sign}(\mathbf{x}) = \text{sign}\left( (\mathbf{x} - \mathbf{p}_{\text{closest}}) \cdot \mathbf{n}_{\text{pseudo}}(\mathbf{p}_{\text{closest}}) \right)$$
* **Face interior:** $\mathbf{n}_{\text{pseudo}} = \mathbf{n}_T$ (unit face normal).
* **Edge interior:** Angle-weighted average of the two incident face normals.
* **Vertex:** Area/angle-weighted vertex normal accumulated during `MFTLEnsembleUpdateVertexNormals`.

For multi-body coalescence, the signed fields merge via:
$$\phi_{\text{merged}}(\mathbf{x}) = \min_{B \in C_k} \phi_B(\mathbf{x}) \quad (\text{with } \phi \le 0 \text{ inside})$$

### 2.4 Marching Cubes Edge Cut
For an active cell edge with endpoints $\mathbf{x}_A, \mathbf{x}_B$ having opposite signs ($\phi_A \cdot \phi_B < 0$):
$$\mathbf{x}_{\text{cut}} = \mathbf{x}_A + \frac{-\phi_A}{\phi_B - \phi_A}(\mathbf{x}_B - \mathbf{x}_A)$$
If $\vert{}\phi\vert{} < 10^{-7}$, clamp to $\pm 10^{-7}$ to avoid division by zero and degenerate triangles.

---

## 3. Detailed Algorithmic Workflow

[ Step 1: Pre-Filter & Proximity Detection ]│  - Traverse LBVH / spatial halo for intra-body and inter-body pairs│  - Check distance <= delta and normal alignment▼[ Step 2: Interaction Clustering & Scheduling ]│  - Group interacting bodies via Union-Find│  - Classify clusters: Dispersed (Local) vs Spanning (Distributed)▼[ Step 3: Temporary Eulerian Grid Setup ]│  - Calculate bounding box [min - 3dx, max + 3dx]│  - Allocate 3D contiguous float buffer initialized to +FLT_MAX▼[ Step 4: Bounded Triangle Rasterization ]│  - Loop over triangles; compute AABB dilated by 2.5 * dx│  - Query grid points in AABB; evaluate Eberly projection & pseudo-normal│  - Update phi = min(phi, signed_dist)▼[ Step 5: Marching Cubes Iso-Surface Extraction (phi = 0) ]│  - Classify 8-corner bitmask (256-case lookup table)│  - Linear interpolation along cut edges; deduplicate cut vertices│  - Emit into local MFTL AoS/SoA pool layouts (enforce 3t+k invariants)▼[ Step 6: Topological Body Separation & ID Assignment ]│  - Half-edge twin BFS traversal to isolate disconnected manifold shells│  - Prefix scan (MFTLEnsembleExscanBodies) to assign dense, unique global IDs▼[ Step 7: Post-Processing & Global Redistribution ]│  - Isotropic remeshing (splits, collapses, flips, tangential smoothing)│  - Free temporary Eulerian memory│  - MFTLRedistribute restores canonical domain ownership
### Step 1: Contact and Breakup Detection
1. **Intra-body Breakup:** For each owned body, query its `MFTL_LBVH` with an inflated threshold $\delta_{\text{breakup}}$. Traverse dual-nodes; prune pairs belonging to the same triangle or sharing an edge. At the leaf stage, compute centroid distance and verify $\mathbf{n}_A \cdot \mathbf{n}_B \le -\cos(\theta_{\text{tol}})$. Early-exit per body on first verified pair.
2. **Inter-body Coalescence:** Invoke the broad-phase spatial halo candidate pair query (`MFTLExtendSpatialHaloNearBodies`). Traverse dual-LBVH trees between candidate bodies with envelope $\delta_{\text{coalescence}}$. Flag pairs meeting distance and opposing-normal thresholds.

### Step 2: Clustering & Migration Routing
1. Run a local Union-Find (Disjoint-Set) algorithm over flagged body IDs to form connected interaction groups $C_k$[cite: 3].
2. For each cluster $C_k$, compute its global bounding box and centroid via `MFTLEnsembleReduceBodies`.
3. If $C_k$ fits within a single rank's box (dispersed condition):
   - Migrate participating `MFTL_Object` records to the rank owning the cluster centroid.
   - Rank executes Steps 3 to 6 in pure shared-memory local execution.
4. If $C_k$ spans across ranks, instantiate a sub-communicator grid over participating ranks.

### Step 3: Eulerian Patch Sizing and Allocation
1. Define local patch grid boundaries:
   $$x_{\min} = \lfloor (b_{x,\min} - 3\Delta x)/\Delta x \rfloor \Delta x, \quad x_{\max} = \lceil (b_{x,\max} + 3\Delta x)/\Delta x \rceil \Delta x$$
2. Allocate contiguous memory for field $\phi$:
   ```c
   size_t n_nodes = (size_t)dims[0] * dims[1] * dims[2];
   float *phi = (float *)malloc(n_nodes * sizeof(float));
   for (size_t i = 0; i < n_nodes; ++i) phi[i] = FLT_MAX;
Step 4: Bounded Triangle Rasterization (SDF)Ensure vertex pseudo-normals are populated (guaranteed by MFTLComputeVertexNormals).   Loop over cluster triangles:Compute triangle bounding box $[\mathbf{t}_{\min}, \mathbf{t}_{\max}]$.Dilate bounds by cutoff radius $R_{\text{cut}} = 2.5\Delta x_{\text{MC}}$.Clamp to integer cell ranges:$$i_{\min} = \max(0, \lfloor (t_{x,\min} - R_{\text{cut}} - x_{\min})/\Delta x \rfloor)$$$$i_{\max} = \min(N_x - 1, \lceil (t_{x,\max} + R_{\text{cut}} - x_{\min})/\Delta x \rceil)$$Within this localized cell box, evaluate point-to-triangle projection:Cdouble p_closest[3];
double dist = MFTL_PointTriangleDistance(p_grid, v0, v1, v2, p_closest);
if (dist < fabs(phi[idx])) {
  double disp[3] = { p_grid[0] - p_closest[0], 
                     p_grid[1] - p_closest[1], 
                     p_grid[2] - p_closest[2] };
  double n_pseudo[3];
  MFTL_InterpolatePseudoNormal(T, p_closest, n_pseudo);
  double sign = (MFTL_DOT(disp, n_pseudo) >= 0.0) ? 1.0 : -1.0;
  phi[idx] = (float)(sign * dist);
}
Step 5: Marching Cubes Iso-Surface ExtractionLoop over Cartesian cells $i \in [0, N_x-2], j \in [0, N_y-2], k \in [0, N_z-2]$.Build 8-bit cube index: $\text{mask} = \sum_{m=0}^7 (\phi_m < 0.0) \ll m$[cite: 3].Skip empty cells (mask == 0 || mask == 255)[cite: 3].For active edges dictated by the 256-entry lookup table, compute $\mathbf{x}_{\text{cut}}$[cite: 3].Deduplicate vertices on shared cell edges using an integer edge hash map (edge_hash = (edge_axis << 60) | (k << 40) | (j << 20) | i).Append vertices to MFTL_VertexPool and triangles to MFTL_TrianglePool.   Strictly construct implicit half-edge records at indices $3t, 3t+1, 3t+2$ with outbound references satisfying MFTL Invariant 1.   Step 6: Topological Body Separation (BFS)Marching Cubes outputs a single triangulated pool that may contain multiple disconnected shells[cite: 3].Allocate a bitset visited of size tris.n_local[cite: 3].For each unvisited triangle, spawn a new MFTL_Object:   Run queue-based Breadth-First Search (BFS) over triangle half-edge twins: neighbor = obj->tris.he[3*t + k].twin_idx / 3[cite: 1, 3].Enqueue adjacent unvisited triangles until the connected component is exhausted[cite: 3].Assign global IDs using MFTLEnsembleExscanBodies to ensure dense, gap-free IDs across ranks without MPI_COMM_WORLD scans.   Step 7: Mesh Smoothing and Global RedistributionCall MFTLRemeshObject on every newly generated body. Set feature_angle_threshold = 180.0 to disable rigid feature locking on smooth fluid interfaces (preventing artificial freezing). Run edge splits, collapses, flips, and tangential spring smoothing[cite: 1, 3].   Retire old progenitor bodies via MFTLObjectReleasePools.   Register new bodies into ens->local_objects.   Invoke MFTLRedistribute(ens):   Evaluates centroid/vertex spatial routing[cite: 1].Restores the canonical geometric decomposition across MPI ranks[cite: 1].Reconstructs ghost communication schedules and twin connectivity[cite: 1].4. Source Tree & Header ArchitectureFollowing MFTL's modular layering (Layer 3: Module APIs)[cite: 1]:include/
  mftl_topology_change.h        // Public Layer 3 API
src/
  topology_change/
    CMakeLists.txt
    topo_detect.c               // Intra-body and inter-body proximity sweeps
    topo_cluster.c              // Disjoint-set clustering & sub-communicator setups
    topo_sdf_rasterize.c        // Eberly projections & pseudo-normal sign checks
    topo_marching_cubes.c       // 256-case MC tables, edge interpolation, pool packing
    topo_split_bodies.c         // Half-edge twin BFS body component separator
tests/
  test_topology_breakup.c       // Liquid ligament breakup validation
  test_topology_coalescence.c   // Two-sphere collision and bridge verification
Public API (include/mftl_topology_change.h)C#ifndef MFTL_TOPOLOGY_CHANGE_H
#define MFTL_TOPOLOGY_CHANGE_H

#include "mftl_ensemble.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double delta_breakup;         /* Pinch-off distance threshold */
  double delta_coalescence;     /* Film rupture distance threshold */
  double normal_tol_deg;        /* Opposing normal alignment threshold (deg) */
  double dx_mc;                 /* Cartesian cell spacing for Marching Cubes */
  bool   enable_remeshing;      /* Run isotropic remesher after MC reconstruction */
} MFTL_TopologyOptions;

/**
 * Returns default topology change options.
 */
MFTL_TopologyOptions MFTLTopologyDefaultOptions(void);

/**
 * Executes full parallel topology check, localized SDF reconstruction,
 * Marching Cubes iso-surface extraction, BFS body splitting, and redistribution.
 *
 * Collective call over ens->comm.
 */
int MFTLEnsembleApplyTopologyChanges(MFTL_Ensemble *ens, 
                                     const MFTL_TopologyOptions *opts);

#ifdef __cplusplus
}
#endif

#endif /* MFTL_TOPOLOGY_CHANGE_H */
5. Invariants and GuardrailsTo prevent regressions and maintain library integrity, the implementation must respect MFTL invariants:Implicit Half-Edge Triple Alignment: Marching Cubes must emit half-edges in contiguous blocks of three per triangle (3t, 3t+1, 3t+2). Arithmetic indexing (MFTLHeTriIdx, MFTLHeNextIdx) must never be broken.Ownership Prefix: After MC extraction, all created entities must be owned locally (n_local == n). Ghost layers must only be populated via MFTLRedistribute.No Dynamic Hash Maps in Hot Paths: During Marching Cubes rasterization, edge deduplication should use direct Cartesian coordinate keys or bounded lookup structures.Normals are Unit Vectors: Marching Cubes triangles must be normalized using unclamped sqrt(|v|^2) and checked for positive outward winding against the signed distance field gradient.Clean Partition Settlement: Reconstructed bodies must call MFTLObjectMarkPartitionDirty to force owner-map re-evaluation during the concluding redistribution.