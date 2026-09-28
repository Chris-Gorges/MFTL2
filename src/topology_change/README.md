`src/topology_change/` — Breakup and coalescence {#mftl_mod_topology_change}
==============================================

**Public header:** `include/mftl_topology_change.h` · **Layer:** 3 ·
**Module group:** @ref mftl_topology_change

A thread of liquid thinner than $\delta_b$ breaks; a film between two surfaces
thinner than $\delta_c$ ruptures. Both are decided by the same detection, and both
are carried out by the same reconstruction: rasterise the affected bodies' signed
distance field on a lattice, bias it at every event, and triangulate its zero level
set with marching cubes. **Nothing migrates.** Every step runs on the partition the
caller has, and the result does not depend on the process count.

| File | Job | MPI? |
|---|---|---|
| `topo_detect.c` | the event predicate, and the canonical event list | no |
| `topo_cluster.c` | the lattice, clusters (union-find), extents, rims, the volume measure | no |
| `topo_reconstruct.c` | rasterisation into windows, the node exchange, the bias, marching cubes on own cells | no |
| `topo_split_bodies.c` | distributed component labelling, the id decision, building the objects | no |
| `topo_driver.c` | the three public entry points and the order of everything | no |
| `topo_comm.c` | **every MPI call the module makes** | **yes** |

`topo_internal.h` declares the helpers the files share. The module is a client of
five others, and every piece that is not specific to topology change lives in the
module it belongs to rather than here:

| Piece | Where | Why there |
|---|---|---|
| closest point on a triangle, and WHICH feature it is | `MFTLClosestPointOnTriangle` (`src/geometry/`) | pure geometry |
| signed distance rasterisation, `MFTL_SdfGrid`, the merge rule | `MFTLRasterizeSignedDistance`, `MFTLSdfMerge` (`src/geometry/`) | one object, local, no MPI |
| angle-weighted vertex normals | `MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE)` | a weighting of an existing function, not a second function |
| the marching cubes table and the extractor | `MFTLMarchingCubesCase`, `MFTLExtractLevelSet`, `MFTLBuildMeshFromLevelSet` (`src/mesh_creation/`) | a mesh generator from a field |
| connected components | `MFTLLabelConnectedComponents` (`src/topology/`) | a half-edge traversal |
| self-proximity | `MFTL_BVHBuildParams.self_pairs` / `pair_filter` (`src/bvh/`) | the contact pipeline, one flag wider |

## Notation {#topo_notation}

| Symbol | Meaning | Where it lives |
|---|---|---|
| $\delta_b$, $\delta_c$ | breakup and coalescence thresholds | `opts.delta_breakup`, `opts.delta_coalescence` |
| $\theta$ | opposing-normal tolerance | `opts.normal_tol_deg` |
| $\Delta x_k$ | lattice spacing on axis $k$ | `MFTLTopologyLattice` |
| $\varepsilon_b$, $\varepsilon_c$ | bias magnitudes, $\tfrac{\beta}{2}\delta_b$ and $\tfrac{\beta}{2}\delta_c$ | `opts.bias_factor` $=\beta$ |
| $\rho$ | bias bump radius, $\gamma\max(\delta,\Delta x)$ | `opts.bias_radius_factor` $=\gamma$ |
| $r_\text{cut}$ | rasterisation cutoff | `MFTL_TopoLattice.r_cut` |
| $\phi$ | the signed distance, negative inside | `MFTL_SdfGrid.phi` |

---

## The pipeline {#topo_pipeline}

```
DETECT      opposing triangle pairs within delta, within and between bodies
AGREE       gather, canonicalise: an identical list on every rank
CLUSTER     union-find over bodies joined by a two-body event
RASTERISE   owned triangles into a window per cluster; exchange and merge overlap nodes
BIAS        a bump at every event
EXTRACT     marching cubes on own cells plus one ghost layer
LABEL       components by their smallest triangle key, until nothing changes
BUILD       decide ids, build objects, dense ids by holder scans
REDISTRIBUTE to the canonical partition
REMESH      optional; then the size floor a second time
VOLUME      three Newton steps per cluster
```

<!-- figure: topo-reconstruction
The reconstruction, run in 2D on a pinching dumbbell. (a) The neck is thinner
than $\delta_b$, and its opposing triangles are each behind the other.
(b) The signed distance is rasterised only within $r_\mathrm{cut}$; beyond it,
the bulb centres included, nodes stay unknown. (c) A bump of height
$\varepsilon_b$ around the event lifts the neck above zero (dashed: the input
surface). (d) The face rule extracts two closed loops. Every vertex lies on a
lattice edge and is named by it, which is why neighbouring ranks need no
stitching.
-->

`MFTLEnsembleApplyTopologyChanges` is `MFTLTopologyDetectEvents` followed by
`MFTLEnsembleReconstructBodies` on every body an event touches. A call that
detects nothing changes nothing: no redistribution, no remesh, no pool touched.
`MFTLEnsembleReconstructBodies` is public on its own because it is where every
distributed property lives, and because re-gridding a body with no event at all is
the cleanest end-to-end test of the pipeline there is.

---

## Detection {#topo_detect}

Two triangles $A$, $B$ — of one body or of two — are an EVENT when

1. they share no vertex;
2. their unit normals oppose, $\mathbf n_A\cdot\mathbf n_B \le -\cos\theta$;
3. each lies on the same side of the other, along its own normal:

   $$
   s_A = \mathbf n_A\cdot(\mathbf c_B-\mathbf c_A), \qquad
   s_B = \mathbf n_B\cdot(\mathbf c_A-\mathbf c_B)
   $$

   both negative — each is BEHIND the other, the gap is inside the liquid, a thread
   or a sheet: **breakup**. Both positive — the gap is outside, a film:
   **coalescence**. Mixed signs are a fold seen edge-on and are not events;
4. their distance is within the threshold of THAT kind.

Condition 3 is what lets one predicate serve both directions. A Rayleigh–Taylor cap rolling onto its own
stem and a jet's neck thinning are both "two opposing triangles within $\delta$ on
one body"; only the side the gap is on tells them apart, so "intra-body means
breakup, inter-body means coalescence" is wrong in both directions — a film inside
one body coalesces, and a liquid film between a drop and a bubble inside it is a
breakup between two bodies.

<!-- figure: topo-events
An event is classified by the side its gap is on, not by how many bodies it
touches. Opposing triangles each behind the other enclose liquid: a breakup.
Each in front of the other, they enclose a film of the other phase: a
coalescence. So (c) a cap rolled onto its own stem coalesces within one body,
and (d) the liquid film between a drop and the bubble inside it breaks between
two bodies.
-->

The pairs come from `MFTLEnsembleIntersectObjects` with `self_pairs` set — the
existing distributed contact pipeline, with its halo, its broad phase and its
ownership rule (a pair is reported by the owner of its A-side triangle) — and the
opposing-normal test runs as a `pair_filter` BEFORE the narrow phase, since
`MFTL_TriTriProximity` costs 176 ns per pair (@ref mftl_mod_bvh) against one dot
product, and most candidate pairs on a curved surface are
neighbours facing the same way. The A-side tree is built with a zero
envelope (the B side carries the whole $\max\delta$), which took the candidates from
129 to 45 per triangle at the same hit count.

**The list is canonical.** One neck is found once per triangle pair around it, and a
pair across a rank boundary in general by both owners. Every event point is snapped
to a cell of $\tfrac14\max(\min\delta,\Delta x)$ and events are merged on
(bodies, kind, cell), keeping the smallest gap — locally, and again over the gathered
list. The result is a pure function of the geometry: identical on every rank and at
every process count. The $\Delta x$ in that quantum is load-bearing: on the threshold
alone one jet neck was shredded into 779 events.

**Clusters** are the connected components of the listed bodies under the events
between two of them — of EITHER kind. The two-body breakup case is the drop with a
bubble inside it: the liquid film between them is bounded by both, and only a field
over both surfaces sees the film.

---

## The lattice {#topo_lattice}

$$
  \texttt{cells}_k = \texttt{cuts}_k\cdot\max\!\Big(1,\ \operatorname{round}\frac{L_k}{\texttt{cuts}_k\,\Delta x}\Big),
  \qquad \Delta x_k = \frac{L_k}{\texttt{cells}_k},
$$

anchored at `global_min`. Two properties follow and everything rests on them:

- **a node plane lies on every domain wall**, so a surface that meets a wall is cut
  exactly there — an open surface's rim is reconstructed ON its wall, at lattice
  resolution, and carries `MFTL_FLAG_BOUNDARY` from the lattice key of its edge;
- **a node plane lies on every rank's cut plane**, so every lattice cell belongs to
  exactly one rank, and "who triangulates this cell" has one answer everywhere.

The lattice is GLOBAL rather than one grid per cluster because of the second
property: a per-cluster grid anchored at the cluster's box
puts cell faces at arbitrary offsets from the cut planes, so the question "which rank
extracts this cell" needs either a negotiation or an overlap. Memory is not the price
it looks like — a rank allocates only a WINDOW (`MFTLSdfGridInitWindow`): its own box
of the cluster's extent plus the cutoff, indexed modulo the period on a periodic axis.
A rank whose box does not meet a cluster allocates nothing for it.

---

## The signed distance {#topo_sdf}

For a node $\mathbf x$ and a triangle $T$, `MFTLClosestPointOnTriangle` returns the
closest point $\mathbf p$ and the FEATURE it lies on — face interior, one of three
edges, one of three corners — using Ericson's region test; a degenerate triangle is
reduced to its nearest edge, and a corner result is bit-exact. The sign is that of
$(\mathbf x-\mathbf p)\cdot\mathbf n_\text{pseudo}$, with the pseudo-normal of the
feature:

| feature | $\mathbf n_\text{pseudo}$ |
|---|---|
| face | $\mathbf n_T$ |
| edge | $\mathbf n_T + \mathbf n_\text{twin}$ |
| corner | the ANGLE-weighted vertex normal |

This is Bærentzen & Aanæs' result: with angle weighting the sign is exact for any
closed, consistently oriented mesh, where area weighting is wrong at a spiky vertex —
on randomly spiked spheres, of about 15 000 query points whose nearest feature is a
vertex, angle weighting signs none wrongly and area weighting 122
(`test_vertex_signs_on_a_spiky_mesh_need_angle_weighting`, which keeps the area run
as a control that must fail).
Angle weighting is a second mode of `MFTLComputeVertexNormals`, not a second
function, and `MFTLUpdateGhostVertexNormals` makes it exact on ghosts. The edge
pseudo-normal needs the twin, which is why rasterisation reads the half-edge records
rather than `tris.neighbors` alone.

**A cluster is rasterised as a UNION.** The field is the signed distance to the
union of all its bodies' surfaces, signed by the nearest one — not the minimum of
per-body fields, which is wrong for nested bodies: inside a drop that contains a
bubble, $\min(\phi_\text{drop},\phi_\text{bubble})$ is negative inside the bubble.

<!-- figure: union-field
Why a cluster is rasterised as a union. A 1D cut through a drop containing a
bubble, whose surface faces into the bubble. The distance to the union of both
surfaces, signed by the nearest one, is negative exactly in the liquid. The
minimum of the per-body fields, each known within $r_\mathrm{cut}$ of its own
surface, is negative inside the bubble, and next to the drop as well.
-->

**The merge is commutative and associative, so the field is partition-independent
to the bit.** `MFTLSdfMerge` keeps the smaller $|\phi|$ and breaks a tie towards
NEGATIVE. Each rank rasterises only the triangles it OWNS — each triangle exactly
once in the job — into its window, and sends every node another rank needs; the
merged value is the same whatever order the contributions arrive in, and equal to a
serial rasterisation of the whole mesh. The alternative, pulling every neighbour's
nearby triangles and rasterising them locally, ships a triangle near a corner to
seven ranks and rasterises it on eight; pushing moves one double per boundary node.

**There is no flood fill.** A node farther than $r_\text{cut}$ from every triangle
stays `MFTL_SDF_UNKNOWN`, and with

$$
  r_\text{cut} = \big\|(\Delta x_0,\Delta x_1,\Delta x_2)\big\| + \max(\varepsilon_b,\varepsilon_c) + \tfrac12\Delta x
$$

a cell with an unknown corner provably cannot be crossed by the biased zero level
set, so marching cubes skips it without needing to know which side it is on. A
flood fill would be a global, sequential sign propagation across ranks — exactly what
the cutoff removes.

---

## The bias {#topo_bias}

Rasterising and re-extracting a thread of radius $0.4\,\Delta x$ reproduces it or
loses it depending on where the lattice nodes happen to fall. The event, not the
lattice, should decide, so every event adds a smooth bump to the field around its
point before extraction:

$$
  \phi \leftarrow \phi + \varepsilon\, w\!\left(\frac{|\mathbf x-\mathbf x_e|}{\rho}\right),
  \qquad
  w(r) = \begin{cases} 1 & r \le \tfrac12 \\ \operatorname{smoothstep}(1 \to 0) & \tfrac12 < r < 1 \\ 0 & r \ge 1 \end{cases}
$$

with $\varepsilon = +\varepsilon_b$ for a breakup (the thread's inside becomes
outside) and $-\varepsilon_c$ for a coalescence (the film becomes inside).
Overlapping bumps combine by the maximum per sign, not by summing, so ten events on
one neck bias it once. $\beta > 1$ decides a feature just at the threshold with
margin. $\rho$ is the trade the header documents with measurements: full weight out
to $\tfrac12\rho = 0.75\,\delta$ by default reaches past the surface of any thread
thinner than $\delta$, and a wider bump spends the bias on the bodies either side
(2.9 % against 4.0 % volume per bulb at $\gamma = 1.5$ against 2 on the pinched
dumbbell, before the correction below puts it back).

---

## Marching cubes {#topo_mc}

The 256-case table is **generated** from one rule applied to each face of the cell,
depending only on that face's four corner signs, so two cells sharing a face draw the
same segments on it and the surface cannot crack between cells; loops are fanned from
an apex none of whose diagonals lies in a face, so no triangle lies in a cell face.
`src/mesh_creation/marching_cubes.c` has the full argument; `test_marching_cubes.c`
enumerates all 256 cases for closure, orientation and the absence of in-face
triangles.

**A vertex is named by the lattice edge it lies on, and a triangle by
`16 * cell + slot`.** Those keys are what makes the distributed extraction need no
stitching: each rank extracts its own cells plus one layer of its neighbours', so the
triangles around every vertex it owns are present — the vertex-complete halo, built
by the extraction — and two ranks that both produce a vertex produce it from the same
two node values and name it the same.

**Positions are computed from WRAPPED node indices.** On a periodic axis the same
lattice edge is reached as node $n$ by one rank and as node $n - N$ by the rank across
the seam; computing the position from the unwrapped index gives two positions one
period apart that the owner test then sends to two different owners. Measured on the
travelling-wave example at np = 2: holes and $\chi = 15$ for an annulus. Guarded by
`test_seam_vertices_get_one_owner_on_every_rank`, which fails when the wrap is
removed.

---

## Bodies from triangle soup {#topo_split}

**A component's label is the smallest triangle key it contains** — a pure function of
the component, so every holder converges on the same number without electing
anything. Each round: every LOCAL component takes the smallest label of its
triangles (`MFTLLabelConnectedComponents` did the local connectivity), every holder of
a ghost triangle tells its owner the label it has, and the owner lowers its component
to the smallest it hears and answers. A component crossing $k$ rank boundaries needs
at most $k+1$ rounds, and the loop ends on a global "nothing changed". Objects are
built per GLOBAL label, not per local component: a U whose bend is on the next rank is
two local components and one body.

**Who keeps which id** is decided identically on every rank from one `Allgatherv` of
`(cluster, label, owned triangles, open)` records — one per component and holder, a
number sized by the change, not by the ensemble. Per cluster, components largest
first take the ids of the bodies that went in, largest first; components beyond that
get fresh ids; ids beyond that are retired. So the jet a drop pinches off stays the
jet, and the pool a drop falls into stays the pool. A CLOSED component below
`min_body_triangles` is deleted (an open one never is).

**Vertex and triangle ids are made dense** with `MFTLEnsembleExscanBodies` — each
owner numbers what it owns from its holder-scan base — and a ghost asks its owner for
the id of the entity with its key. The closing `MFTLRedistribute` then rebuilds the
canonical halo from correct ids. `MFTLObjectSetCounts` marks every new object
partition-dirty, so the redistribution's body skip cannot mistake one for settled.

---

## Rims {#topo_rims}

A level set has no boundary except where it meets the lattice's edge, so an open
surface is representable exactly when its rims lie on domain walls — which, on the
lattice above, is where node planes are. The reconstruction therefore reproduces a
rim on its wall and flags it from the lattice key; a cluster containing a rim
anywhere else is REFUSED (`n_clusters_refused`) and left untouched rather than
corrupted. Detected by one flag per cluster on a reduction already being taken.

---

## After the extraction {#topo_after}

The closing `MFTLRedistribute` restores the canonical partition, and when
`opts.remesh` is set one `MFTLRemeshEnsemble` runs over the whole ensemble.

**The size floor is applied twice.** A satellite a couple of lattice cells across
comes out of marching cubes as a few dozen slivers and clears `min_body_triangles`;
remeshed at a target edge length near its diameter it becomes a tetrahedron, which
the collapse approval refuses to dissolve and which then survives as four triangles
each facing against all three neighbours. That is exactly what the Rayleigh–Taylor
example reported at np = 4 as "4 inverted triangles". So the count is taken again
after the remesh, from one sum per component of THIS reconstruction.
`test_a_satellite_the_remesher_collapses_is_discarded` fails without the second
count.

**The volume goes back LAST.** Marching cubes cuts chords across a curved surface, the
bias moves liquid, and the remesh that cleans the slivers up smooths some away on its
own; on the liquid jet, without the correction, 60 of a final 81 volume units were
lost over 45 reconstructions. Per cluster, the enclosed volume is measured before and
after (a closed body directly; an open one together with the walls its rims are on,
as the flux of $(x_k - x_{k,\min})\,\mathbf e_k$ along an axis whose far wall carries
no rim), and restored by a uniform offset $d$ along a displacement field $\mathbf u$:
the area-weighted vertex normals, with a rim vertex's wall-normal component removed.
The step is Newton's, on the exact first variation of the volume,

$$
  \frac{dV}{dd} = \sum_t \mathbf S_t\cdot\frac{\mathbf u_0+\mathbf u_1+\mathbf u_2}{3},
$$

($\mathbf S_t$ the area vector), which each rank sums over the triangle corners it
owns — every (triangle, corner) pair is present once in the job because the halo is
vertex-complete. Dividing by the area instead converges only linearly, at about 0.03
per step. Measured from a 15 % deficit: $2.8\cdot10^{-5}$ and $10^{-9}$ after two and
three Newton steps, against $1.3\cdot10^{-4}$, $3\cdot10^{-6}$, $8\cdot10^{-8}$ after
two, three, four area steps. The count is FIXED at three: a stopping test on a reduced
double would be two ranks disagreeing in the last ulp about whether to enter another
collective.

---

## Design alternatives {#topo_departures}

Each choice below was either made before implementation or forced by a
measurement.

| Alternative | Chosen | Why |
|---|---|---|
| migrate a dispersed cluster to one rank; a sub-communicator grid for a spanning one | nothing migrates; every rank works on its own box | one code path at every body size; no second partition to keep consistent; the field is partition-independent by the merge rule |
| one grid per cluster, `[min − 3Δx, max + 3Δx]` | one global wall-aligned lattice, windows per rank | a node plane on every cut plane and every wall: unique cell owners, rims on walls |
| `min` over per-body fields | the distance to the UNION, signed by the nearest surface | nested bodies (a bubble in a drop) |
| sign "vertex: area/angle-weighted" | angle-weighted only, ghosts synchronised | area weighting is measurably wrong at spiky vertices |
| cutoff $2.5\Delta x$ | $r_\text{cut}$ above, from the cell diagonal and the bias | a proof that no cell with an unknown corner is crossed |
| topology decided by where lattice nodes fall | an explicit $\pm\varepsilon$ bias at each event | the event decides, with margin |
| intra-body = breakup, inter-body = coalescence | the side of the gap decides, either kind within or between bodies | RT caps coalesce within one body; a drop's film around a bubble breaks between two |
| a typed 256-row table | generated from one face rule | no crack between cells by construction; all 256 cases tested |
| ghosts only through `MFTLRedistribute` | a halo built by the extraction, then a redistribution | component labelling and dense ids need the halo before the redistribution can run |
| `float` field | `double` | the partition-independence argument is bit-level |

---

## What is tested, and how {#topo_tests}

`tests/test_topology_change.c` runs every scene twice — distributed, and on rank 0
alone with `MPI_COMM_SELF` — and compares a CENSUS of the result: per body its
vertex, triangle and boundary counts, Euler characteristic and volume, and the
sorted positions of every vertex, quantised so that a last-bit difference in a
volume-correction sum cannot reorder a tie. At np = 1, 2, 4, 8 and 27 the distributed
surface equals the single-rank surface to $10^{-12}$. The scenes are placed where the
cut planes of every tested grid meet, on the periodic seam, and across walls; each
also checks the partition invariants and dense ids. The unit pieces are tested where
they live: `test_geometry.c` (closest point, every feature region, degenerate
triangles), `test_signed_distance.c` (the merge laws, sign against the winding
number on hard shapes, rasterising in pieces bit-identical to all at once),
`test_marching_cubes.c` (all 256 cases), `test_bvh_self.c` (self-proximity as a SET
against an exhaustive sweep, at np up to 27, with a cross-rank non-vacuity count),
`test_topology.c` (component labelling) and `test_vertex_normals.c` (angle weighting).

The three open-surface examples carry it into flows, each with a non-vacuity flag
that fails a run in which the body count or genus never changed:
`test_liquid_jet_topology` (a drop pinches off and travels to the outlet),
`test_rayleigh_taylor_topology` (a blob pinches off the rolled-up cap at
t ≈ 3.5) and `test_travelling_waves_topology` (a plunging lip reconnects with the face).

## Costs, and what is not done {#topo_deferred}

- **A reconstructed body is re-triangulated EVERYWHERE**, not only at the event:
  vertices, ids and pool order are new, and the surface moves by the marching cubes
  error, $O(\Delta x^2\kappa)$. Bodies outside every cluster are not touched at all.
  A local reconstruction that re-extracts only a neighbourhood of each event and
  stitches it to the old mesh would avoid that; it needs a seam between a lattice
  surface and a remeshed one, and was not attempted.
- **Detection runs every call** and is the part that scales with the whole
  ensemble's surface rather than with the change: one self- and pair-proximity query
  over every body. It dominates a call that finds nothing, which is most calls.
- **There is no continuous-time detection.** Two surfaces that approach by more than
  $\delta$ in one step are not found, exactly as for contact detection
  (@ref bvh_deferred).
- **Rims off the walls are refused, not handled.** A level set cannot represent them;
  a surface with a free rim needs a different reconstruction.
