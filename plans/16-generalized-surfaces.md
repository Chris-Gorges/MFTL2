# 16 — Generalized surfaces: periodic and open triangulated objects

Status: **ACCEPTED / COMPLETE (2026-09-25, §28)** — 16A ACCEPTED (independent review 2026-09-24, §18.1); §3.6 revision 3.2
INDEPENDENTLY REVIEWED AND FROZEN (2026-09-24, §19, errata E1–E8); 16B IMPLEMENTED (2026-09-24,
§20) and ACCEPTED by the independent review of 2026-09-24 after fixes (§21); 16C IMPLEMENTED
(2026-09-24, §22) and ACCEPTED by the independent review of 2026-09-24 after one fix (§23);
16D IMPLEMENTED (2026-09-24, §24) and **ACCEPTED by the independent review of 2026-09-25 after
fixes R16D-1 to R16D-5 (§25)**; 16E IMPLEMENTED (2026-09-25, §26) and **ACCEPTED by the
independent review of 2026-09-25 after corrections C16E-1 to C16E-6 (§27)**, gap C16E-7 closed in 16F (§28.1); **16F executed and ACCEPTED
(2026-09-25, §28); Stage 16 ACCEPTED / COMPLETE**.
Depends on: 13 (periodicity, accepted), 15 (MFTL1→MFTL2 swap, accepted). Absorbs R6.2 of
`plans/15` (guaranteed ghost completion).
Precedes: 17 (code-quality cleanup + manual — the manual must describe this model), 18
(performance, to be planned).
Implemented by: 16A (§17), 16B (§20, review fixes §21), 16C (§22, review fix §23), 16D (§24, review fixes §25), 16E (§26, review corrections §27), 16F (§28)
Reviewed by: independent design reviews of 2026-09-23 — revision 1 (not ready: M1–M9),
revision 2 (not ready: MF1–MF4); revision 3 resolves MF1–MF4 and records the should-fix items
(§16); independent review of 2026-09-24 — 16A accepted, §3.6 (revision 3.1) not ready (§18);
§3.6 revision 3.2 reviewed and frozen (§19); independent review of 16B of 2026-09-24 — accepted
after fixes (§21); independent review of 16C of 2026-09-24 — accepted after one fix (§23);
independent review of 16D of 2026-09-25 — accepted after fixes (§25); independent review of 16E
of 2026-09-25 — accepted after corrections (§27); 16F final validation of 2026-09-25 (§28)
Acceptance: **PASSED — Stage 16 ACCEPTED / COMPLETE (2026-09-25).** 16A–16F accepted; every
requirement is IMPLEMENTED + TESTED, EXPLICITLY UNSUPPORTED or DEFERRED to Stage 17/18 (§28.14).

Revision history:
- **Revision 1** (initial plan): reviewed, not ready (M1–M9).
- **Revision 2**: resolved M1–M9; the second review found MF1–MF4 plus should-fix items.
- **Revision 3** resolves MF1–MF4 and incorporates the should-fix items.
- **Revision 3.1** (2026-09-23, §3.6 only): reviewed 2026-09-24, not sound (MF-36-1, MF-36-2,
  S-36-3 to S-36-5; §18.2).
- **Revision 3.2** (2026-09-24, §3.6 only): independently reviewed 2026-09-24, sound, frozen
  (§19; errata E1–E8, no rule changes).
- **Revision 3.2** (2026-09-24, §3.6 only; this text): resolves them (§3.6.11). Every rule has
  an executable model (`tools/phase_set_prototype/`, 47/47 cases). Awaiting a fresh independent
  review.

§15 maps every review item to the section that resolves it; §16 is the design-review record.

## 0. Objective and compatibility contract

MFTL2 and multiflow must support, in ONE surface model:
1. **periodic triangulated objects** — one continuous physical surface crossing periodic
   boundaries, including surfaces that **wrap** around one or more periodic directions;
2. **open triangulated surfaces** — genuine boundary edges, no enclosed volume;
3. their combination (an open surface crossing a seam, split over several MPI ranks).

Hard compatibility contract:
- Closed, non-periodic behaviour stays **bit-identical** to the accepted Stage-15 state.
  - Library: full `ctest` incl. `test_stage15`, LAPACK on and off.
  - multiflow: Release A/B of FT, FT_Sharp_CSF, FT_Sharp_Integral and IBM-MFTL on 1 and 4
    ranks, against the Stage-15 build with contract-off flags.
  - The new per-vertex image counters are 0 on non-periodic axes and change no arithmetic.
- Periodic fronts are **explicitly enabled** per run; open surfaces are **explicitly declared**
  per body. No class is ever inferred from a mesh that merely fails to close, and a boundary,
  seam or rim is never inferred from coordinate location.
- Boundaries are never faked with artificial triangles, virtual twins or duplicated geometry.
- **Physics choices live in multiflow**; MFTL2 provides topology and geometry only.

## 1. Audit findings (evidence)

### 1.1 MFTL2
| Area | Finding | Evidence |
|---|---|---|
| Missing twin | `twin_idx = -1` means "no twin stored locally" — ghost rim **or** physical boundary. | `mftl_core.h:127`, `topology.c:57–85` |
| `boundary` field | means "no local twin"; set for every rim half-edge in packing/redistribution. | `communication.c:283,899,1275,1337,1970,2027,2962`, `topology.c:59,83` |
| `twin_gid` | −1 = "unknown" (open mesh **or** twin elsewhere). | `mftl_core.h:131–137` |
| `glue_wings` (collapse) | rewires `twin_idx` but **never updates `twin_gid`**; `mftlRefreshTwinGids` refreshes only half-edges with a local twin → a boundary-adjacent collapse leaves a twin-less half-edge naming a dead half-edge. | `remeshing_core.c:282–302,412–422,950–957` |
| Soup builder | builds open meshes: half-edges unmatched on every rank keep `twin_gid = -1`; no declaration/check. | `soup.c:231–275` |
| Integrity check | a missing twin is accepted whenever `boundary` is set (any rim). | `topology.c:299–302` |
| Ring walks / `IsBoundaryVertex` | stop at any missing twin; rim and boundary indistinguishable. | `mftl_topology.h:200–300`, `remeshing_core.c:474–485` |
| Curvature | a non-closed ring → vertex **silently skipped**. | `curvature.c:72–92` |
| Smoothers | need closed rings; serial variants refuse ghosts. | `volume_conserving.c:60–86,167–186` |
| Remeshing | partial boundary logic (`common == 1`, "no boundary into interior"); no true-boundary-edge collapse test; parallel sweeps gate on a local twin. | `remeshing_core.c:596–740`, `tests/test_remeshing.c:330–498` |
| Remesh drivers | compute `MFTLObjectVolume` **unconditionally** as the recovery target; recovery loops call `MFTLVolumeRecovery`. | `remeshing_core.c:1433,1612`, `remeshing_parallel.c:475,786,962–969` |
| Integrals | single collective anchor unwrap. `MFTLCheckMinimumImageConstraint` rejects bodies with unwrapped extent ≥ half a period. | `geometry_object.c:40,84–117,234–300` |
| Point query | ray parity after a single-anchor unwrap of all vertices. | `point_query.c:2–20,229+,387` |
| Neighbour graph | periodic wrap of Cartesian neighbours, deduplicated (1-slab: self; 2-slab: same rank both sides). | `ensemble.c:87–150` |
| MPI pack | `PackedVertex` has `pos`, `flags`, `id_he_out`, … (no image data); `PackedHalfEdge` has `twin_gid` (int) and `boundary`. | `mftl_communication.h:25–100` |
| Native I/O | MFTL1 format; no class, no periodic data. | `io.c:914–940` |

### 1.2 multiflow / adapter
| Area | Finding | Evidence |
|---|---|---|
| Front periodicity | forced off (`periodic = 0`). | `mftl_adapter.c:97–100` |
| FT overlap mesh | `FTDomain` forced non-periodic; periodic branches commented out. | `initializepboxmesh_ft.c:163–177` |
| Box adaptation | FT box fitted to raw front bounding boxes. | `initializepboxmesh_ft.c:180+,508` |
| Clamp | every vertex clamped 1e-7 inside the box after remeshing. | `mftl_adapter_mftl2.c:1024` |
| Adapter integrals | own "MFTL1 formulas over owned triangles" → **bypass any library class check**. | `mftl_adapter_mftl2.c:1079ff` |
| Adapter searches | proximity / thin-region grids hash raw coordinates. | `mftl_adapter_mftl2.c:806–890` |
| Parity remesher | `mftl1_criteria = 1` set for every body. | `mftl_adapter_mftl2.c` (`mftl2_params_in`) |
| Raw positions | 131 raw position uses in 22 physics files. | grep (§1.3 of revision 1) |
| Scalar transport | `getScalarGradients` **aborts** (`exit(1)`) when an edge has no neighbour; the matrix assumes two triangles per edge. | `surfacescalartransport.c:413–422` |
| Surface tension | Frenet element force Σ σ L ν per triangle edge; interior edges cancel; boundary edges leave a line force. | `spreadfromfronttofluid.c:236–290` |
| IBM mass properties | GTS injection and MoveIBM properties use `ObjectVolume` / `MomentOfInertia`. | `injectparticlesfromGTSfile.c`, `calculatemoveibmparticlepropertiesfrommftl.c` |

## 2. 16A — Topology and twin-state invariant

### 2.1 The three half-edge states (exact invariant)

For every live half-edge `h` in a **quiescent** mesh (§2.3):

| State | `twin_idx` | `twin_gid` | Meaning |
|---|---|---|---|
| **LOCAL** | `≥ 0` | `== half_edges[twin_idx].global_id` (≥ 0) | twin stored on this rank (owned or ghost) |
| **REMOTE** | `−1` | `≥ 0`, and a live half-edge with this gid exists on some rank, in a triangle whose owner is known (§2.4) | MPI rim |
| **BOUNDARY** | `−1` | `≤ −2` (`MFTL_TWIN_NONE = −2` base; `−2 − tag` encodes the boundary tag, §2.6) | genuine physical boundary |

- No fourth state: `twin_gid == −1` ("unknown") and a `twin_gid` naming a dead or non-existent
  half-edge are **illegal** in a quiescent mesh.
- Symmetry: LOCAL(h) ⇒ `twin(twin(h)) == h`. REMOTE(h) ⇔ the remote twin is REMOTE or LOCAL on
  its own rank and names `h` back. BOUNDARY is a property of the undirected edge, identical on
  every rank that stores it.
- The existing `boundary` field is redefined as the cache `boundary == (twin_gid ≤ −2)`. All
  ≈ 45 missing-twin/boundary sites are classified in 16A.1 (rim uses switch to
  `twin_idx == -1 && twin_gid >= 0`), with a debug cross-check at every site.
- Edge identity: `edge_id = min(gid, twin_gid)` for LOCAL/REMOTE, and `edge_id = gid` for
  BOUNDARY. Unchanged for closed meshes.
- **Explicit state predicates** replace every implicit sign test: `MFTLHeIsLocal(h)`,
  `MFTLHeIsRemote(h)`, `MFTLHeIsBoundary(h)`, `MFTLHeBoundaryTag(h)`, and
  `MFTLHeTwinGid(h)` (defined only for LOCAL/REMOTE; asserts otherwise).
  - **All 34 `twin_gid` readers** are audited in 16A together with the ≈ 45 missing-twin sites.
    `twin_gid` is a plain `int`, so the encoding fits, but existing readers assume "−1 or a gid".
  - Confirmed sites that break under the encoding and must be converted:
    - `MFAdapterEdgeId` (`mftl_adapter_mftl2_inline.h:170`; `min(gid, twin_gid)` would give every
      boundary edge of a loop the same id −2 − tag);
    - `MFAdapterMFTL2FindEdgeById` (`mftl_adapter_mftl2.c:395`, matches `twin_gid == edge_id`);
    - `MFTLObjectAppendTriangleCopy` (`soup.c:356`, tests `!= MFTL_ERROR`);
    - `MFTLHeTwinGid` (`mftl_topology.h:121`);
    - the rim pack/unpack paths.
  - A grep-based CI check forbids new raw comparisons of `twin_gid` outside the predicates.

### 2.2 Maintenance by every topology-changing operation

| Operation | Required twin-state handling |
|---|---|
| **Split** of interior edge (both halves local; parallel sweeps already gate on this) | new inner pairs are LOCAL with gids set pairwise; the wing half-edges keep their state. |
| **Split** of boundary edge | the two new boundary halves are BOUNDARY with the **parent's tag**; new inner pair LOCAL. |
| **Collapse** / `glue_wings(x = twin(he_ca), y = twin(he_bc))` | the merged edge's state follows from the outer half-edges: both LOCAL → x,y become a LOCAL pair (gids set to each other); one BOUNDARY, other LOCAL → the LOCAL one becomes BOUNDARY with the tag of the boundary one; both BOUNDARY → **forbidden** by the link condition (§4.4) — never reached; **any REMOTE outer half-edge → the collapse is rejected** before any change (the gate requires the full augmented link locally, §4.4.3). `glue_wings` sets `twin_gid` explicitly in every case; the finaliser no longer "repairs" twin gids. |
| **Flip** | only interior LOCAL edges whose two triangles and four wing half-edges are stored; the flipped pair stays LOCAL (gids unchanged); the wings keep their states. Boundary edges are never flipped. |
| **Builder** (`MFTLBuildFromSoup`) | local pairing → LOCAL; global rim resolution (global discovery mode, §2.4) → REMOTE with the partner gid. An edge unmatched everywhere becomes BOUNDARY **only if the body is declared open AND the edge is declared boundary** by the input: an explicit boundary-edge list `(vgid_a, vgid_b, tag)`, or — as an explicit per-input user declaration — `OpenSurface <body> BoundaryFromMesh` ("every unmatched edge of this input file is a boundary", auto-tagged §2.6). Any other unmatched edge is a collective error. Nothing is inferred silently. |
| **Triangle copy / append** (`MFTLObjectAppendTriangleCopy`) | copies are staged; `MFTLObjectCommitCopy(dst, mode)` (collective, **global discovery mode**, §2.4) closes the batch. `mode = ASSEMBLE`: every non-LOCAL half-edge must resolve to a live partner in `dst`, else collective error. `mode = CUT`: the **cut set is explicit by construction** — exactly those half-edges whose source twin exists in the source object but was not copied (recorded at copy time from the source `twin_gid`) become BOUNDARY with a new tag. `dst` must be declared open (error otherwise), and any other unresolved half-edge is an error. The copy is not quiescent until committed. |
| **Component copy / merge** (`MFAdapterBodyCopyTriangleFrom`, `MergeInto`, breakup, coalescence) | = copy + commit, followed by **image-counter canonicalisation** (§3.3). Copying whole connected components is ASSEMBLE; copying a sub-patch is CUT. multiflow's breakup copies components (ASSEMBLE). |
| **Redistribution / ghost creation** | `twin_gid` travels unchanged in `PackedHalfEdge`; `twin_idx` is re-derived locally; states are recomputed (LOCAL if paired locally, otherwise from `twin_gid`); then ghost completion (§2.5). |
| **Remesh finaliser** | compaction must keep the states; it ends with the collective validation of §2.3. |

### 2.3 Quiescent points and collective validation
A mesh is **quiescent** after: build/commit, `MFTLRedistribute*` (after ghost completion), remesh
finalisation, serial-bracket end, and native-mesh read. Within a remesh pass (dead entities,
VISITED flags) the invariant is not required.

`MFTLValidateTwinStates(obj)` (collective) runs at every quiescent point in Debug builds and
wherever `MFTL_VALIDATE` is enabled. It is on by default in the multiflow Debug configuration and
in all Stage-16 tests.
- Local checks: every half-edge's state; LOCAL symmetry; BOUNDARY `twin_idx == -1`.
- Remote liveness: every REMOTE gid is confirmed live by the discovery protocol (§2.4, a
  validation-only query with no data transfer), and the partner names `h` back.
- Result: success, or a collective error listing the offending gids on rank 0.

Release builds run the cheap local checks at every quiescent point; the remote-liveness check
comes for free wherever ghost completion runs (§2.5).

### 2.4 Remote-twin discovery (MPI algorithm)
Input on each rank: a set Q of `(twin_gid, requesting half-edge, edge midpoint X_m in local
continuous coordinates)`. There are two modes.

**Global mode** (always used before triangle-centre ownership is established: builder, staged
copies, ASSEMBLE/CUT commits, component assembly, marching-cubes assembly, serial-bracket
scatter; and as the fallback of the owned mode):
1. `MPI_Allreduce` of the local request count. If the global count is zero, return (all ranks
   take the same branch).
2. `MPI_Allgatherv` of the requested gids.
3. **Any rank holding a live, non-ghost (authoritative) copy** of the requested half-edge
   answers, whether or not it owns the triangle by centre. Staged, not-yet-redistributed copies
   count as authoritative on the rank that staged them. Answers are
   `(gid, holding rank [, packed triangle + vertices in completion mode])`, returned with
   `MPI_Alltoallv`.
4. Duplicate authoritative copies of one gid (possible for staged copies) must be bitwise
   identical in topology and position; this is checked, with a collective error otherwise.

**Owned mode** (after redistribution, when every triangle has exactly one owner by the centre
rule):
1. **Round 1 — geometric candidates.**
   - Let ℓ_max be the **global maximum live edge length** (`MPI_Allreduce`, refreshed at every
     redistribution) and ε the ghost-position staleness margin (the maximum vertex displacement
     since the last `MFTLUpdateGhostPositions`, bounded by the application's CFL step; 0 right
     after an update).
   - The twin triangle's centroid lies within ℓ_max/3 of the edge midpoint (centroid − midpoint
     = (d − m)/3 with |d − m| ≤ ℓ_max).
   - Candidate ranks are those whose (rectilinear, periodic) box intersects the periodic ball
     B(X_m, ℓ_max/3 + 2ε). This is well defined because ℓ_max < half a period.
   - A cached gid→owner hint is tried first; correctness never depends on it.
   - Requests go to the candidates in a sparse exchange (`MPI_Alltoall` counts, then
     `Isend`/`Irecv`). A rank answers for gids it holds as a live half-edge of a triangle it
     **owns**.
2. **Round 2 — global fallback: the global mode above, always executed** (its Allreduce makes
   it collective). It carries zero data whenever round 1 was complete, which is guaranteed for
   fresh ghost positions.

**Completion guarantee and errors**
- After either mode, every requested gid is resolved (holder or owner known), or it has no live
  authoritative copy anywhere.
- An unresolved gid in the builder or a CUT commit becomes BOUNDARY **only** under the explicit
  declarations of §2.2.
- In every other context an unresolved non-boundary twin is a **collective error** (with
  offending gids reported). It is never converted silently.
### 2.5 Guaranteed ghost completion (R6.2)
The guarantee after every `MFTLRedistribute*` (and after serial-bracket end):
- (G1) every edge of every **owned** triangle is LOCAL or BOUNDARY;
- (G2) the complete one-ring (fan) of every owned vertex is stored.

Implementation:
- The existing ghost rule gives G2 (vertex owners receive incident triangles) and most of G1.
- Remaining REMOTE edges of owned triangles are resolved by the §2.4 protocol in
  ghost-completion mode; the imported triangles become ghosts.
- One round suffices: G1 concerns only owned triangles' direct neighbours, and imported ghosts'
  own rims are allowed to stay REMOTE.
- This replaces the adapter's second redistribution pass, and `MFAdapterOwnedTrianglesMissNeighbours`
  becomes an assertion.

### 2.6 Boundary tags, boundary vertices and loops
- **Tags:** a BOUNDARY half-edge stores `twin_gid = −2 − tag`, tag ≥ 0. There is no layout
  change: `twin_gid` is already an int in the pack. Tags come from the input (e.g. per-loop
  labels), or are auto-assigned at build as the loop index ordered by minimum vertex gid.
  Splits inherit the tag, collapses along a boundary keep it, and CUT commits create new tags.
  Tags give multiflow a persistent handle for per-loop boundary conditions (§6, §7).
- **Manifoldness** (validated at build and after every remesh): each edge has 1 or 2 triangles;
  each vertex has exactly one fan; the surface is orientable with consistent winding. Bow-tie
  vertices, non-manifold edges and Möbius strips are rejected.
- **Boundary vertex:** a vertex with exactly one outgoing and one incoming BOUNDARY half-edge
  (manifold). Its `out_he_idx` points to its outgoing BOUNDARY half-edge. This is maintained by
  every operation, and `PackedVertex.id_he_out` carries it through MPI. It is guaranteed for
  **owned** vertices (their fan is complete by G2); ghost vertices may have partial fans and are
  never used as loop-walk pivots.
- **Ring completeness:** `MFTLVertexRing(obj, v, …)` returns `RING_CLOSED` (interior, closed
  fan), `RING_OPEN_COMPLETE` (fan runs BOUNDARY-to-BOUNDARY) or `RING_INCOMPLETE` (the walk hits
  a REMOTE half-edge). It replaces `mftlRingIsClosed` / `mftlVcRingTris` / `IsBoundaryVertex`.
- **Distributed loop construction** (`MFTLBoundaryLoops`, collective):
  1. Every owned boundary vertex `v` emits `(edge_id(in_v), edge_id(out_v), gid(v), tag, winding
     increment of out_v)`.
  2. `MPI_Allgatherv` of these records. Boundary size is O(N^{1/2}); the cost is documented,
     and pointer-jumping list ranking replaces it only if Stage 18 measures a need.
  3. Every rank chains the records into loops.
  - Each loop has: ordered edge ids and vertex gids, tag set (normally one tag), length
    (continuous coordinates), and a **winding vector** Σ s(e) of the edge lattice shifts
    (§3.3).
  - A loop is well defined because every boundary vertex has exactly one successor.
  - Loops are independent of rank count and seams.

## 3. 16B — Periodic image representation

### 3.1 Representation
- `MFTL_Vertex` gains `int img[3]`; `PackedVertex` gains `int img[3]`. This is a **layout
  change**: the MPI datatype is rebuilt and resized to `sizeof`.
- The **continuous coordinate** of a vertex is `X = pos + img ⊙ L` (period vector `L`, `img = 0`
  on non-periodic axes). `pos` stays **wrapped** in `[min, max)` on periodic axes (Stage 13
  storage convention); `img` records which periodic image the continuous surface point belongs
  to.
- **Edge lattice shift** (derived, defined in §3.3): for an edge u→v, `d(u,v)` is the
  minimum-image displacement (well defined: every edge is shorter than half a period, §3.5), and
  `s(u,v) = round((X_v − X_u − d(u,v)) ⊘ L) ∈ Z³`. An edge with `s_k ≠ 0` is a **cut edge**
  along k. After canonicalisation, cut edges exist only where the surface genuinely winds (§3.3).

### 3.2 Lifecycle of the counters and the displacement rule

**Displacement rule (applies to every vertex update).** A vertex is only ever moved by a
**minimum-image displacement δ from an anchor vertex's continuous coordinate**:
`X_new = X_anchor + δ`, then wrap (`k = floor((pos − min) ⊘ L)`, `pos −= k ⊙ L`, `img += k`).

- Kernels that currently return a *wrapped* position (paraboloid/barycentric fits via
  `mftlParaboloidFitPoint`, TSUR/Kuprat moves, tangential smoothing, `MFTLVolumeRecovery`)
  are converted in 16B:
  - their result `p_out` is turned into `δ = minimage(p_out − pos_anchor)` before
    `(pos, img)` is updated;
  - `MFTLApplyPeriodicity` on a vertex position is no longer allowed outside the wrap function.
- `MFTLTranslateObject` / `MFTLMoveBody` / rotation / scaling operate on continuous
  coordinates (relative to the body reference point of §3.5) and then wrap.
- A displacement must be shorter than half a period (checked; the CFL guarantees it for
  advection).

| Event | Rule |
|---|---|
| **Continuous input** (generators, GTS, user input with continuous coordinates) | `img = floor((X − min) ⊘ L)`, `pos = X − img ⊙ L` on periodic axes; then canonicalisation (§3.3). |
| **Wrapped input without counters** (v1 files, user wrapped input; each independently supplied piece) | initial `img = 0`, then canonicalisation (§3.3) assigns consistent counters. It *is* the lift, so there is no separate lift routine. |
| **Soup build** | vertices duplicated across ranks must agree on `(pos, img)` after the per-piece wrap (checked, collective); then canonicalisation. |
| **Marching-cubes assembly** | vertices emitted with continuous coordinates from periodic cell indices; pieces from different ranks may be gauged independently → canonicalisation after assembly. |
| **Vertex motion** (advection, smoothing, fits, VCS/TSUR, application writes) | displacement rule. The adapter's `MFAdapterVertSetPos` / `VertPosMut` + `MFAdapterBodyWrapVertices` apply it with the vertex's own previous continuous coordinate as anchor. |
| **Split** (new M on a→b) | anchor a: `X_M = X_a + ½·d(a,b) + δ_fit` (δ_fit by the displacement rule). The edge a→M then has zero lattice shift; M→b inherits the shift of a→b. |
| **Collapse** (a→b, survivor b) | anchor b: `X_b' = X_b + δ` (δ = minimum-image displacement to the fitted midpoint). |
| **Flip** | no vertex change. |
| **Redistribution / ghosts / `UpdateGhostPositions`** | `img` travels with `pos` in `PackedVertex`; ghost copies carry the owner's `img`. |
| **Copy / merge / component copy / coalescence** | `img` copied; then canonicalisation (the pieces may be gauged independently). |
| **Native I/O v2 / restart** | stores `pos`, `img`, the tags, the declared class and the period box. On any rank count the reader rebuilds, redistributes and canonicalises. Canonicalisation is idempotent on a canonical file, so a restart reproduces the counters exactly; the D8/A5 tests compare them. |

Every new or moved vertex receives a `(pos, img)` pair whose continuous coordinate is its
intended physical position in the image continuous with its anchor. That holds because each
update is anchored and short.

### 3.3 Winding (homological definition) and canonicalisation

**Edge lattice shift.** For an edge u→v on periodic axis k:
`s_k(u,v) = round((X_v,k − X_u,k − d_k(u,v)) / L_k) ∈ Z`, where d is the minimum-image
displacement (unambiguous because every edge is shorter than half a period). `s` is an integer
1-cochain on the mesh graph with `s(v,u) = −s(u,v)`.

**Winding (definition).** For a cycle γ of mesh edges, `W_k(γ) = Σ_{e∈γ} s_k(e)`. A connected
component **winds along k** iff `W_k(γ) ≠ 0` for some cycle γ. Equivalently, `s_k` is not a
coboundary: there is no single-valued vertex potential `φ_k` with `s_k(u,v) = φ_k(v) − φ_k(u)`
on every edge.

- This is a topological invariant of the embedded surface: independent of the counter values
  and invariant under counter changes that preserve continuous positions modulo the lattice.
  - Around any cycle, the minimum-image displacements sum to `L_k·W_k(γ)` (the true
    displacement around the cycle), while continuous displacements telescope to 0.
- The **winding lattice** of a component is the subgroup of Z³ spanned by the `W(γ)` of its
  independent cycles.

**Canonicalisation `MFTLCanonicaliseImages(obj)`** (collective) runs after every event that can
join independently gauged pieces:
- soup construction and every read;
- marching-cubes assembly;
- `MergeInto`, ASSEMBLE/CUT commits, component copy, breakup, coalescence;
- initial lifting of independently supplied pieces.

Algorithm (correctness first):
1. **Gather** the edge graph with shifts to rank 0: per canonical edge
   `(gid_u, gid_v, s_x, s_y, s_z)`, 20 bytes per edge, O(E) at assembly only. A distributed
   frontier BFS is a Stage-18 option.
2. Per connected component, a **deterministic BFS spanning tree** rooted at the minimum vertex
   gid gives the potential `φ(root) = 0`, `φ(v) = φ(u) + s(u,v)` along tree edges.
3. New counters: `img' = img − φ`. Positions are unchanged, and every continuous coordinate
   moves by a whole lattice vector, i.e. the same physical point.
   - Tree edges then have zero shift.
   - A non-tree edge e = (u,v) keeps the **cycle residual** `r(e) = s(e) − (φ(v) − φ(u))`,
     which is the winding of its fundamental cycle.
4. **Whole-body re-gauge:** add one integer vector `g` to every `img` of the component so that
   the root's counter returns to its pre-canonicalisation value (or to 0 on first construction).
   This keeps counters small and the body where it was.
5. Rank 0 scatters `img'` and the per-component winding lattice. The **cached per-axis wrap flags**
   are set: `WRAP_k ⇔` some residual has `r_k ≠ 0`.
6. After canonicalisation, **cut edges (edges with `s_k ≠ 0`) exist exactly on the non-tree
   edges with non-zero residual** — artificial cuts from different piece offsets are gone.

**Why the cached flags stay valid in the time loop** (no canonicalisation needed until the next
assembly event):
- The displacement rule never creates a lattice shift on an edge along an axis where no cut
  edge exists (split, collapse and moves are anchored and short).
- Link-condition-legal collapses, splits and flips are homotopy-preserving edits of an embedded
  surface with short edges, so they preserve every cycle's winding.
- Hence "WRAP_k ⇔ a cut edge along k exists" remains true between canonicalisations. Along a
  wrapping axis cuts may spread, which is harmless: the flag is unchanged, and the next
  canonicalisation compacts them.
- A debug check verifies the flag against a cut-edge scan after every remesh.
- Topology changes (coalescence, breakup, MC) are assembly events and re-canonicalise.

### 3.4 Per-axis classification and diagonal winding
Per body and periodic axis k: `WRAP_k` iff the projection of the component winding lattice onto
k is non-zero (from the residuals); otherwise `COMPACT_k`. A compact axis has **globally
consistent continuous coordinates, of any extent** (no half-period object restriction).

**Diagonal winding** — e.g. a ribbon whose lattice is generated by (1,1,0) — sets both
`WRAP_x` and `WRAP_y`. This is conservative. The only consequence is `UNSUPPORTED` (never a
wrong value) for the phase-set formula of §3.6, which needs an axis that is compact for every
member.

Supported surfaces: compact seam-crossing objects; objects spanning one direction (ribbon,
`WRAP_x`); multi-direction wrapping (films `WRAP_x ∧ WRAP_y`, `WRAP_xyz`). Loop windings (§2.6)
use the same shifts.

### 3.5 Local minimum-image restriction, reference points and precision
- Every **edge** is shorter than half the period on each periodic axis (checked after remeshing
  and advection). Operation stencils (fans, two-ring fit stencils) are checked; an operation
  whose stencil violates this is skipped (counted diagnostic).
- The **object-extent restriction is removed**; all single-anchor unwraps (integrals, centroid,
  inertia, point queries, the extent part of `MFTLCheckMinimumImageConstraint`) are replaced.
- **Precision:** continuous coordinates are never formed as `pos + img·L` with large `img`
  inside sums.
  - Integrals, centroids and ray queries use `Y = X − X_ref`, where `X_ref` is a
    **body/set reference point** (the continuous coordinate of the lowest-gid owned vertex,
    agreed collectively).
  - `Y` is formed from integer differences: `(img − img_ref)·L + (pos − pos_ref)`, which is
    exact in the integer part.
  - Whole-body re-gauging (§3.3 step 4) additionally keeps `img` small.
  - Test B10 checks a body with `img ≈ 10⁶`.
- Spatial search trees over wrapping bodies use per-triangle boxes in local frames plus the
  Stage-13 periodic interval tests (correct; efficiency is a Stage-18 item).

### 3.6 Phase-boundary sets: status model, criterion, evaluation (revision 3.2)

> **Revision 3.2 (2026-09-24) — INDEPENDENTLY REVIEWED 2026-09-24 (§19): SOUND, FROZEN.** No
> must-fix item. The review's errata E1–E6 (§19.4) correct two justification sentences, state the
> status precedence and record scope/implementation notes; they change no rule. The prototype was
> extended (72 cases incl. interlocking overhangs, wall pieces, axis permutation, winding-derived
> flags, random re-gauging; §19.3).
> - Revision 3.2 rewrites the *evaluation* of §3.6. It resolves the items of the review of
>   revision 3.1 (§18.2; mapping in §3.6.11). The definition — status model, Φ(S) = 0 criterion,
>   consistent orientation — is unchanged.
> - Every rule below has an executable counterpart in `tools/phase_set_prototype/p36.py` (steps
>   R1–R10). The validation cases of §3.6.10 run with `python3 cases.py`. Revision 3.1 is kept
>   as `p31.py` / `cases31.py` for the review record.

#### 3.6.1 Definitions

**Status model** (all closed-phase queries: volume, centre, inside/outside, indicator):
- `MFTL_DEFINED` — meaningful and evaluated;
- `MFTL_UNDEFINED` — the topology/physics does not define it;
- `MFTL_UNSUPPORTED` — mathematically defined, but this implementation cannot evaluate the
  case yet (explicit, never a value).

multiflow treats UNDEFINED and UNSUPPORTED both as configuration errors, with different
messages.

**Phase-boundary set** `S`: an explicit list of (body, orientation sign σ = ±1) entries plus
optional **wall closures** `W = (w, side)`, side ∈ {min, max}, on a *non-periodic* axis w.
- Declaring a closure on a periodic axis is a configuration error.
- Each body is by default the singleton set with σ = +1: the phase is on the side into which
  its stored normals do *not* point.

**Assumptions** (documented, not checked beyond cheap tests):
- members are embedded — no self-intersections and no mutual intersections;
- members are orientable manifolds (validated, §2.6);
- members are connected components. A body with several components contributes each
  component separately, with the body's σ.

**Notation.**
- `k` is the evaluation axis (§3.6.3); `i, j` are the two lateral axes.
- `A_k = L_i L_j` is the cell face area normal to k.
- `V_cell = A_k L_k`, or `V_box = A_k (max_k − min_k)` when k is walled.
- `Y = X − X_ref` are continuous coordinates relative to the set reference point (§3.5).
- With wall closures, `ref_k` is the coordinate of the lowest closure wall.

#### 3.6.2 Closing S at declared walls (S-36-5)

Every open member must be closed by wall patches before anything else is evaluated.

1. **Loops.** Every boundary loop of every member must lie on a declared closure: all its
   vertices within `τ_loop` of `c_w`.
   - A loop on no declared closure → UNDEFINED ("open surface not closed").
   - A loop that touches two walls (along a box edge) → UNSUPPORTED.
   - A loop with a non-zero winding vector (§2.6/§3.3; it spans a periodic direction of the
     wall plane) → UNSUPPORTED.
2. **Orientation of the wall loops.** `Γ_W = Σ_m σ_m ∂m|_W`, where each loop is oriented as
   the boundary of its oriented member.
3. **The wall patch.**
   - `P_W` is the 2-chain in the wall plane with the domain-outward unit normal `n_W`
     (`−e_w` at min, `+e_w` at max) and multiplicity

     `μ_W(q) = ε_W · wind(Γ_W, q) + n_W`, with `ε_min = +1`, `ε_max = −1`.
   - `wind` is the lateral winding number, measured with the orientation (e_i, e_j) for which
     (e_i, e_j, e_w) is right-handed. On a wall plane that is periodic in both lateral axes, it
     is summed over the lateral images of the (non-winding, hence closed) lifted loops.
   - This is the unique chain with `∂P_W = −Γ_W` up to the full-wall multiplicity
     `n_W ∈ {0, 1}`, so `Ŝ = members ∪ P_W` is a closed 2-cycle.
4. **Admissible multiplicities.**
   - `n_W` must make `μ_W(q) ∈ {0, 1}` for every q.
   - If the wall plane has a non-periodic lateral axis, then `n_W = 0` (the patch cannot reach
     other walls).
   - No admissible `n_W` → UNDEFINED (loops nested with inconsistent orientation).
5. **Flux selection.**
   - Φ_w of a patch is `s_W ∫ μ_W dA = s_W (ε_W A(Γ_W) + n_W A_w)`, where
     `s_W = n_W · e_w = ∓1` and `A(Γ)` is the signed shoelace area.
   - Patches contribute to no other axis.
   - Among the admissible assignments {n_W}, exactly those with `Φ(Ŝ) = 0` (§3.6.3) are
     valid. One → use it. None → UNDEFINED (Φ ≠ 0). Several → UNSUPPORTED ("ambiguous
     closure"; e.g. a closed drop with closures on both walls: either the drop, or the box
     minus the drop).
6. **What a patch adds.**
   - Crossings: a line along w through lateral point q crosses P_W at `c_w` iff `μ_W(q) = 1`,
     with sign `s = −1` at a min wall (entering the phase along +w) and `s = +1` at a max wall.
   - Volume and centre (k = w): `∫_{P_W} Y_k n_k dA = s_W (c_w − ref_k) ∫ μ_W dA` and
     `∫_{P_W} Y_k² n_k dA = s_W (c_w − ref_k)² ∫ μ_W dA`. These vanish for the min closure
     wall, because `ref_k = c_w` there.
7. **Components of Ŝ.** The connected components of the glued surface: members, glued along
   their loops to the connected pieces of `{μ_W = 1}` that those loops bound. A patch piece
   without loops (a full wall, `n_W = 1`) is a component of its own.
8. **When a wall-closed set is DEFINED.** Steps 1–5 succeed with a unique assignment, and
   §3.6.3–§3.6.6 succeed. Wall closures on two different axes → UNSUPPORTED.

Examples:
- **B6** — a closed film with normal +z over a z-min closure, no loops:
  - `μ = n_min`, and flux +A − n_min·A = 0 selects `n_min = 1`, i.e. the full wall;
  - V = height of the film.
- **Film with normal −z** over a z-min closure only: no admissible solution → UNDEFINED.
- **Cap on a wall orifice** (B14): `μ = disk` → a compact component (cap ∪ disk).
- **Inverted cap** (σ = −1) with closures on both walls:
  - `μ_min = 1 − disk` and `n_max = 1`;
  - components: (cap ∪ wall minus disk), which is a layer, and the full top wall;
  - V = V_box − V_cap.

#### 3.6.3 Criterion, evaluation axis and component classes

**Criterion (mathematical, unchanged).** With the assumptions of §3.6.1, S bounds a finite phase
region iff:
1. Ŝ is closed (§3.6.2);
2. **Φ(Ŝ) = 0** — `Φ_j = Σ_m σ_m ∮_m n_j dA + Σ_W Φ_j(P_W) = 0` for every axis j. Φ_j is the
   pairing of the homology class with `dx_{j+1}∧dx_{j+2}`; H₂ of the (partially) periodic box
   is torsion-free, so this is exactly vanishing homology;
3. **consistent orientation** — the level function (§3.6.5) takes exactly two values.

Φ ≠ 0 → UNDEFINED. Φ = 0 with inconsistent orientation → UNDEFINED.

*[Erratum E2, §19.4 — status precedence: closure errors (§3.6.2 steps 1–4), then Φ(Ŝ) ≠ 0
(UNDEFINED), then the admissible axis (UNSUPPORTED), then the component classes and degenerate
compacts (UNSUPPORTED), then the degenerate total volume (UNDEFINED); every line-based step comes
after these.]*

**Evaluation axis k (admissible axis).**
- With wall closures: k = w.
- Otherwise: a non-periodic axis if there is one, else a periodic axis with `COMPACT_k` for
  every member (§3.4). Ties go to the smallest axis index.
- None → UNSUPPORTED. Examples: two opposite triply-wrapping tilted planes; diagonal-winding
  members.

**Component classes** (Φ_k of a component includes its patch pieces):
- **compact:** `Φ_k(c) = 0`;
- **layer:** `|Φ_k(c)| = A_k` within `τ_Φ = 1e-10·A_k + C·ε·∮|n_k| dA`. This is the only
  non-zero value for a connected embedded component that is compact along k (degree ±1;
  confirmed in §18.2);
- anything else → UNSUPPORTED.

**Compact components.**
- `v_c = Σ σ ∮_c Y_k n_k dA` in the component's own gauge, patch terms included. It is
  gauge-free because `Φ_k(c) = 0`.
- `|v_c| ≤ τ_V` → UNSUPPORTED (degenerate component).
- **Phase side** `e_c = sign(v_c)`: +1 means the phase is inside c.

**Geometric facts used below.** They hold in the cover that unrolls k; for walled k this is the
domain itself.
- **(F1)** A compact component's lift is null-homologous and bounds exactly one region `I(c)`
  that is bounded along k, with `|v_c| = vol(I(c))`.
- **(F2)** Layer lifts are disjoint separating sheets, totally ordered along +k. Between two
  consecutive lifts of one layer lies exactly one lift of every other layer. The crossings of
  one lift on one line span less than L_k. *[Erratum E1: this last sentence is not proven and
  no step uses it.]*
- **(F3)** No `I(c)` contains a layer lift.
- **(F4)** Removing regions with connected boundaries does not disconnect. Hence:
  - without layers, the complement of the outermost insides — the **exterior E** — is
    connected;
  - with layers, each **slab region** between consecutive layer lifts, minus the outermost
    insides in it, is connected.

  The complement of Ŝ is therefore a forest: slab regions (or E) at the roots, and nested
  compact insides below them.

#### 3.6.4 Lines and genericity (MF-36-2)

All evaluation uses lines parallel to e_k:
- on a periodic k: closed loops of length L_k;
- on a walled k: segments from min_k to max_k.

**Crossings.** A crossing is a transversal intersection with a triangle of a member (its height
in that member's own continuous gauge) or with a patch.
- Its sign is `s = σ_m · sign(n_k)` for a member triangle, and as in §3.6.2 step 6 for a patch.
- `s = −1` means entering the phase along +k.

**Generic line.** All of the following hold:
- every triangle hit has barycentric coordinates `> η_b`;
- `|n_k|/|n| > η_n` at every hit;
- q lies farther than `η_b` times the local edge length from every wall loop;
- all crossing heights — reduced mod L_k on a periodic k — are pairwise more than `η_z L_k`
  apart, with `η_z L_k A_k ≤ 0.1 τ_V` (so any region with V > τ_V is resolved).

Three kinds of lines; `N_try = 64` candidates each, otherwise UNSUPPORTED ("no generic line"):
1. **Canonical line** (only when layers exist). Candidates
   `q_n = q_0 + (frac(nφ₁)L_i, frac(nφ₂)L_j)`, n = 0, 1, …, from the set reference point's
   lateral coordinates q_0. Any line meets every layer, so a far jump is harmless here.
2. **Member line** of a compact component c.
   - Candidates are interior points P of c's *own* triangles.
   - Triangles are taken in increasing gid order, skipping those with `|n_k|/|n| ≤ η_n`.
   - Barycentric weights: `b_0 = (⅓, ⅓, ⅓)`, then `b_n ∝ (1 + frac(nφ₁), 1 + frac(nφ₂), 1)`.
   - The first line that is generic with respect to *all* components is used.
   - Every candidate passes through c (at height `t_P` in c's own gauge), so it can never miss
     the member or its local window. A reference vertex or edge is never used.
3. **Query line** of a point Q.
   - `d = dist(Q, Ŝ)`. If `d ≤ τ_d`: Q is **ON_BOUNDARY**.
   - Otherwise candidates `Q_n = Q + r_n(cos θ_n e_i + sin θ_n e_j)`, with `r_0 = 0`,
     `r_n = ½ d · frac(nφ₁)` and `θ_n = 2π frac(nφ₂)`.
   - The segment Q Q_n does not meet Ŝ, so the answer is exact for every candidate. The
     perturbation can never leave Q's region.

Distributed evaluation: each rank intersects its owned triangles. The crossing records
(height, sign, component, triangle gid) are gathered and ordered deterministically. Every
decision is taken identically on all ranks.

#### 3.6.5 Orientation consistency of the whole set (MF-36-1)

**Level function.** Ŝ oriented defines `λ: D∖Ŝ → Z` up to a constant: λ increases by 1 across a
crossing with `s = −1` along +k. S is consistently oriented iff λ takes exactly two consecutive
values, normalised to {0, 1}. The phase region is `{λ = 1}`. By (F4), λ is constant on every
slab region, on E, and on the inner region of every compact (inside c, outside its children).
Consistency is therefore checked in two parts, and **no signed-volume test is involved**.

**(C1) Layers — all crossings of the canonical line**, layers + compacts + patches (as in
revision 3.1):
- periodic k: sorted by height mod L_k, the signs strictly alternate cyclically (even length);
- walled k: start at level 0 below min_k, apply `−s` at each crossing, stay in {0, 1}, and end
  at 0.
- Failure → UNDEFINED.

By (F2), on every line the last crossing of a layer lift and the first crossing of the next
lift above it are adjacent. So one line compares every pair of neighbouring layers, overhangs
included. *[Erratum E1, §19.4: the adjacency statement is false for interlocking layers (case
I1: crossing order Y1 Y2 Y2 Y1 Y1 Y2). The conclusion holds for a different reason: the line
crosses every layer lift, both slab regions adjacent to each crossing appear on it, so
alternation fixes every slab level.]*

**(C2) Compacts — the nesting tree.** For each compact c, on its member line through P:
1. **Enclosers.** Every (c′, p), c′ ≠ c compact, `p ∈ Z` (p = 0 on walled k), with
   `#{crossings t′ of c′ on the line : t′ + p L_k < t_P}` odd. The parity rule says P lies in
   `I(c′ + pL_k)`; at most one p per c′ can satisfy it.
2. `depth(c)` = the number of enclosers. **Parent** = the encloser (c′, p) with
   `depth(c′) = depth(c) − 1`. It must be unique, otherwise UNSUPPORTED.
3. **Surrounding level** `λ_out(c)`:
   - with a parent (c′, p): `λ_out = 1` if `e_{c′} = +1`, else 0 (the parent's inner side);
   - no parent, with layers: the nearest *layer* crossing strictly below `t_P` on the line
     (periodic: the smallest `(t_P − t′) mod L_k`; walled: the largest `t′ < t_P`, and none
     means level 0). `λ_out = 1` iff that crossing has `s = −1`;
   - no parent, no layers: `λ_out = λ_E`.
4. **Rule:** `e_c = +1 ⟺ λ_out(c) = 0`. A drop of phase sits in non-phase surroundings; a
   cavity of non-phase sits in phase. Otherwise → UNDEFINED.
5. **Exterior level** (no layers):
   - every outermost compact implies `λ_E = 0` (if e = +1) or `λ_E = 1` (if e = −1);
   - all must agree, otherwise → UNDEFINED. This is review case 8c: two unrelated drops with
     opposite σ;
   - `λ_E = 1` (the matrix is the phase) requires every axis to be periodic. Otherwise the
     exterior touches an undeclared wall → UNDEFINED;
   - an empty set → UNDEFINED.

**Proposition.** (C1) ∧ (C2) ⟺ consistent orientation.
- By (F1)–(F4) the regions of D∖Ŝ are exactly the slab regions (or E) and the inner regions of
  the compacts, arranged in the tree.
- (C1) fixes the slab levels. (C2) fixes each inner region relative to its surroundings and the
  common exterior level.
- No other adjacency exists.

The cases the rule must separate:
- one drop → `λ_E = 0`;
- several disjoint drops of the same phase → all imply `λ_E = 0`, DEFINED;
- a drop with a cavity, or a drop in a cavity in a drop → alternating e along the tree, DEFINED;
- nested components with the same e → UNDEFINED;
- two unrelated components with opposite σ → the exterior levels disagree → UNDEFINED;
- a signed-volume cancellation such as `V₁ − V₂` can never pass.

#### 3.6.6 Volume

1. `V_raw = Σ_m σ_m ∮_m Y_k n_k dA` (each member in its own gauge) + the patch terms
   (§3.6.2).
2. **Degeneracy first.**
   - Representative `V_rep = V_raw mod V_cell` on a periodic k, `V_raw` on a walled k.
   - `V_rep ≤ τ_V` or `V_rep ≥ V_cell − τ_V` (resp. `V_box − τ_V`) → UNDEFINED ("degenerate:
     empty or full region"). This is tested before any line is cast.
   - `τ_V = 1e-12·V_cell + C·ε·Σ_m ∮ |Y_k n_k| dA`, as before. *[Erratum E7, §19.4: the
     executable model and the implementation use τ_V = 1e-10·V_cell (and τ_Φ = 1e-9·A_k); the
     choice only moves the degenerate thresholds of cases L9 and D1–D4.]*
3. **Value**, after (C1) and (C2) have passed:
   - periodic k with layers: `V = V_raw mod V_cell` (gauge-free; §18.2 confirmed);
   - periodic k without layers: `V = V_raw + λ_E V_cell` (exact, because
     `λ = λ_E + Σ_c e_c 1_{I(c)}`). This covers a single bubble with σ = −1 in a fully periodic
     cell (the matrix phase), and resolves S-36-4;
   - walled k: `V = V_raw` (λ = 0 at both ends).

#### 3.6.7 Canonical relative gauge (placement) and centre

The placement assigns a shift `π_c ∈ L_k Z` to each component. There is no placement on a
walled k: π = 0.

1. **Layers (S-36-3).**
   - On the canonical line, take the *layer* crossings only.
   - **z₀** is the midpoint of the first segment, in order of reduced height, that follows a
     leaving (`s = +1`) layer crossing. It is a point at level 0; by (C1) such a segment
     exists.
   - For each layer Y with crossings y_1…y_r on the line (own gauge):

     `a_Y(z₀) = max{ p ∈ Z : #{ y_i + p L_k < z₀ } odd }`, and `π_Y = a_Y(z₀) L_k`.
   - The placed layer lifts are then the n lifts immediately below z₀ (F2): consecutive,
     with level 0 below the lowest and above the highest.
   - This uses the complete ordered crossing sequence. It never assumes that a window starts
     outside a layer's crossing span, so overhangs that straddle the reduction point (case O3)
     are handled. Interlocking layers — a layer reaching into another layer's overhang pocket —
     are handled too: `a_Y(z₀)` only uses the parity of one lift's crossings below z₀ and the
     total order of the lifts (cases I1–I8, §19.3; erratum E1).
2. **Compacts**, in order of increasing depth:
   - parent (c′, p): `π_c = π_{c′} − p L_k`;
   - no parent, with layers: let Y be the layer of the nearest crossing below `t_P`, at own
     height y, whose lift `y + q L_k` lies just below `t_P`. Then `π_c = π_Y − q L_k`;
   - no parent, no layers: `π_c = 0`.

   Both rules are exact adjacency relations of the member line through c. There is no window
   and no reference vertex.
3. **Invariant.** In the placed gauge, `Σ σ ∮ (Y_k + π) n_k dA = V`. This is checked, and a
   mismatch is an internal error.
4. **Centre along k:** `C_k = ref_k + (1/2V) [Σ σ ∮ (Y_k + π)² n_k dA + patch terms]`,
   reported modulo L_k on a periodic k.
5. **When a centre component is DEFINED.**
   - **Number of phase components:** `N_ph = n_layers/2 + #{c : e_c = +1} + λ_E` (every phase
     slab; every drop starts a new component; the matrix).
   - **Periodic axis j** (including j = k):
     - if some member has `WRAP_j`, or `λ_E = 1`: the region wraps j, so C_j is UNDEFINED;
     - else if `N_ph ≠ 1`: C_j is UNSUPPORTED (per-component centres available);
     - else DEFINED.
   - **C_j for j ≠ k** uses the same construction with j as the line axis. j is admissible,
     because no member wraps it.
   - **Non-periodic axis:** DEFINED whenever V is, even for a disconnected region.

#### 3.6.8 Inside/outside, indicator, set-level operations

**Inside/outside** is gauge-free, so no placement is needed.
- On the query line through Q_n (§3.6.4), take the nearest crossing strictly below Q_k
  (periodic: cyclically mod L_k). Q is inside iff that crossing has `s = −1`.
- If there is no crossing below: walled k → 0 (the bottom of the domain); periodic k → λ_E.
  The latter is only possible without layers, on a line that meets no component.
- `MFTLPhaseSetInside` returns DEFINED with INSIDE / OUTSIDE / ON_BOUNDARY.

**Indicator seeding** uses the same rule. **Set-level volume recovery** (a uniform normal shift
by ΔV / A(S)) and the other operations that need a relative placement use §3.6.7.

#### 3.6.9 Status summary

| Situation | Status |
|---|---|
| open member not closed by a declared wall; loop multiplicities outside {0,1}; no patch choice with Φ = 0; Φ(Ŝ) ≠ 0 | UNDEFINED |
| (C1) or (C2) fails; exterior levels disagree; λ_E = 1 with an undeclared wall; empty set | UNDEFINED |
| degenerate volume (V within τ_V of 0 or V_cell / V_box) | UNDEFINED |
| no admissible axis; `\|Φ_k(c)\| ∉ {0, A_k}`; degenerate compact (`\|v_c\| ≤ τ_V`); unresolvable nesting | UNSUPPORTED |
| winding wall loops; loops on two walls; closures on two axes; ambiguous patch choice | UNSUPPORTED |
| no generic line within N_try candidates | UNSUPPORTED |
| centre along a periodic axis of a disconnected region | UNSUPPORTED (component centres available) |
| centre along a periodic axis that the region wraps (wrapping member, or λ_E = 1) | UNDEFINED |
| everything else | DEFINED |

Surface-only quantities (area, area moments, flux of lattice-invariant fields, loops) are
always DEFINED.

#### 3.6.10 Validation cases (tests B11, B14, B15, B16; executable model)

`tools/phase_set_prototype/cases.py` builds each case on triangulated meshes and runs the model
`p36.py`. DEFINED values are compared (to 1e-10; the centre modulo L_k on a periodic k) with the
same mesh in an explicitly given physical placement, independent of the model's own rules.
Point queries are checked against known answers. The 16B tests B11/B14/B15/B16 (§10) reproduce
every case on the MFTL2 implementation, on np 1/2/4/8.

| ID | Case | Expected |
|---|---|---|
| C1 | one compact drop | DEFINED, centre DEFINED |
| C2 | one compact bubble σ = −1, fully periodic (matrix) | DEFINED V = V_cell − V_b; centre UNDEFINED (wraps) |
| C3 | seam-crossing sphere (x and z seams) | DEFINED, centre DEFINED (mod period) |
| C4 | two disjoint drops, same phase | DEFINED V₁ + V₂; centre UNSUPPORTED |
| C5 | **review case 8c**: disjoint drops σ = +1 / σ = −1 | **UNDEFINED** (exterior levels disagree) |
| C6 | drop with a cavity stored one period up | DEFINED, cavity placed by parent rule |
| C7 | drop ⊃ cavity ⊃ drop | DEFINED V₁ − V₂ + V₃; centre UNSUPPORTED (2 components) |
| C8 | nested drops with the same σ (review 8b) | UNDEFINED |
| C9 | bubble σ = −1 containing a drop σ = +1 | DEFINED V = V_cell − V_o + V_i; centre UNDEFINED |
| C10 | closed tube wrapping x | DEFINED |
| L1 | one wrapping sheet | UNDEFINED (Φ ≠ 0) |
| L2 | two-sheet slab, ceiling stored one period up | DEFINED V = A h, C_z analytic |
| L3 | complement of the slab | DEFINED |
| L4 | three sheets | UNDEFINED (Φ ≠ 0) |
| L5 | four sheets (−,+,−,+) | DEFINED; centre UNSUPPORTED |
| L6 | four sheets (−,−,+,+) | UNDEFINED (C1) |
| L7 | six alternating sheets in mixed images | DEFINED |
| L8 / L9 | slab of thickness 1e-6 / 1e-11 | DEFINED / UNDEFINED (degenerate) |
| L10 | two bodies of two disconnected sheets each (−,− / +,+) | UNDEFINED (components are members → as L6; erratum E8) |
| M1 | **B15**: wavy floor + ceiling + bubble in the dip (bubble stored +1) | DEFINED, bubble at z = 0.1, centre analytic |
| M2 | **review case 4b**: wide x-tube near a floor that is wavy along y (tube stored +1) | DEFINED, C_z = 0.5485 (3.1: −1.095) |
| M3 | slab + tube of the other phase inside | DEFINED |
| M4 | slab + drop σ = −1 in the non-phase region (review 8d) | UNDEFINED |
| M5 | slab + phase drop in the non-phase region | DEFINED; centre UNSUPPORTED |
| M6 / M7 | bubble in a slab; first candidate member line through layer vertices | DEFINED; M7 uses the second candidate |
| O1 / O2 | overhanging (folded) floor, 3 crossings per line | DEFINED |
| O3 | **S-36-3**: overhang straddling the z seam on the canonical line (3.1: no valid shift) | DEFINED |
| U1 | one tilted triply-wrapping plane | UNDEFINED |
| U2 | two opposite tilted planes (valid, no admissible axis) | UNSUPPORTED |
| W1 | **B6**: film over a z-min closure (full-wall patch) | DEFINED V = 0.6 |
| W2 / W3 | film with the wrong orientation / without a closure | UNDEFINED |
| W4 | film, closures on both walls | DEFINED (flux selects the patch) |
| W5 / W6 | **B14**: cap on a wall orifice / with closures on both walls | DEFINED |
| W7 / W8 | inverted cap, z-min closure only / both closures (complement) | UNDEFINED / DEFINED |
| W9 | open member without a declared closure | UNDEFINED |
| W10 | half tube along x standing on the wall (loops wind; valid) | UNSUPPORTED |
| W11 / W12 | drop / bubble σ = −1 in a walled domain | DEFINED / UNDEFINED |
| W13 | B6 film + cavity below it | DEFINED |
| W14 | two drops in a walled domain (centre along the walled axis) | DEFINED |
| W15 | drop with closures on both walls (drop or box minus drop) | UNSUPPORTED (ambiguous) |

Point queries are included in C1–C3, C6, C7, L2, M1, O1, O2, W1, W5 and W8, among them lines
through mesh vertices (M1, L2). **Result of the model: see §3.6.12.**

#### 3.6.11 Resolution of the review of revision 3.1 (§18.2)

| Item | Resolution in revision 3.2 |
|---|---|
| MF-36-1 incomplete orientation check without layers | (C2) nesting tree with a common exterior level, §3.6.5. No signed-volume test (C5 = 8c now UNDEFINED) |
| MF-36-2 local rays leave the member | member lines through interior points of the member's own triangles; exact parent/adjacency placement, no windows or reference vertices (§3.6.4, §3.6.7); query lines inside the clearance ball (M2 = 4b correct; M7) |
| S-36-3 window start inside an overhang | placement from the sheet index `a_Y(z₀)` at a level-0 point of the complete crossing sequence (§3.6.7; O3) |
| S-36-4 single bubble in a fully periodic cell | `V = V_raw + λ_E V_cell` with λ_E from (C2) (C2, C9) |
| S-36-5 wall-closure consistency | wall patch as a chain `μ_W = ε_W wind + n_W ∈ {0,1}`, flux selection, orientation, flux / volume / centre terms, DEFINED vs UNSUPPORTED (§3.6.2; W1–W15) |

Unchanged and confirmed by the review:
- Φ(S) = 0 as the separation criterion;
- the volume modulo V_cell with layers;
- degree ±1 layers;
- all-crossings alternation for layers, with overhangs;
- slab stacks;
- tubes wrapping a periodic direction;
- the three statuses.

API (unchanged names):
- `MFTLPhaseSetClassify(S, &info)` returns the status, Φ, the evaluation axis, the wall patches
  (n_W), the components (class, e, depth, parent), λ_E and the placed shifts;
- `MFTLPhaseSetVolume`, `…Centre(S, C[3], status[3])` and `…Inside(S, Q, &where)` each return
  one of the three statuses.
- Per-body wrappers use the singleton set.
- The existing unchecked functions keep their closed-P0 behaviour bit-identically and abort
  with a message on UNDEFINED/UNSUPPORTED.

#### 3.6.12 Model run (2026-09-24)

`python3 tools/phase_set_prototype/cases.py` (numpy only; revision-3.2 model `p36.py`):
**47 passed, 0 failed.** Output: `tools/phase_set_prototype/cases.out`.
- Every DEFINED value agrees with the explicit physical placement to ≤ 7e-16 (V) and ≤ 1.0e-15
  (C_k).
- Every expected UNDEFINED / UNSUPPORTED status is reproduced with the reason given above.
- M7 rejects the first member-line candidate (it passes through layer vertices) and uses the
  second.
- **Independent review run (2026-09-24, §19.3):** `cases_review.py` adds 25 cases (interlocking
  overhangs I1–I8 and I1q, seam-crossing wavy layers and stacks S1–S5, wall pieces W16–W19,
  degeneracy D1–D4): 25 passed. All 72 cases also pass with the wrap flags derived from the §3.3
  winding (`run_derived.py`), rotated onto x and onto y (`run_axes.py 0|1`) and with every
  member re-gauged by random lattice vectors (`run_regauge.py`, 8 seeds).
- The superseded revision 3.1 (`p31.py`) on the same review cases:
  - 8c: DEFINED V₁ − V₂ once its local-ray defect is fixed, UNSUPPORTED as written;
  - 4b: C_z = −1.095;
  - O3: no valid layer shift (W₀ = 0.1197 is inside the fold's crossing span);
  - seam sphere and B15 bubble as written: UNSUPPORTED.

## 4. 16C — Generalized geometric kernels and remeshing

### 4.1 Geometry kernels
All kernels evaluate in local frames:
- `MFTLTriangleUnwrapped`, `MFTLVertexDelta` (minimum image), `MFTLEdgeVector`,
  `MFTLVertexContinuous(v)` (returns X);
- `MFTLBoundaryEdgeGeometry(h, &len, t[3], nu[3], mid_X[3])`: edge tangent `t` (oriented along
  the half-edge), triangle normal `n`, **in-surface outward conormal `ν = t × n`**, midpoint in
  continuous coordinates.

Global integrals use continuous coordinates on compact axes and phase sets on wrapping axes
(§3.6). Proximity / thin-region searches use periodic grid hashing with minimum-image distances
(in the library, replacing the adapter's raw hashes).

### 4.2 Normals and curvature at boundaries
- Vertex normals: angle-weighted over the available fan (`RING_OPEN_COMPLETE`).
- Mean and Gaussian curvature: computed only for `RING_CLOSED` vertices. Boundary vertices get
  `MFTL_VFLAG_CURV_UNDEFINED` (vertex `flags`, packed). An optional **extrapolated** curvature
  (paraboloid fit on the open stencil, ≥ 6 points) is stored separately with
  `MFTL_VFLAG_CURV_EXTRAPOLATED`.
- The geodesic turning angle `π − Σ corner angles` is provided for boundary vertices (used by
  Gauss–Bonnet validation and corner detection).
- **No silent skip:** a skipped vertex is always flagged; `RING_INCOMPLETE` on an owned vertex is
  an invariant violation after G2.

### 4.3 Boundary curve
Each boundary loop (by tag) has a **boundary-curve mode**:
- **REPRESENTED** (default): the curve is the interpolating Catmull–Rom spline through the
  current boundary vertices of the loop, evaluated in local frames. It passes through existing
  vertices, so repeated remeshing does not shrink the loop beyond the interpolation error.
- **CONSTRAINED**: the application registers a projector
  `project(tag, X_in, X_out, user)` (e.g. a wall plane, a circle, a pinned support line). New and
  moved boundary vertices are projected onto it. multiflow uses this for PINNED supports (§7.2).

**Corners:** a boundary vertex is a corner if its turning angle exceeds `boundary_corner_angle`
(default 30°) or if it is tagged `MFTL_VFLAG_CORNER` by input/application (packed flag). Corners
are never moved, merged or collapsed.

### 4.4 Remeshing rules for surfaces with boundary

#### 4.4.1 Link condition
The ad-hoc rules of revision 1 are replaced. Following Dey et al., augment the complex with one
**virtual vertex ω**: add ω adjacent to every boundary vertex, and a virtual triangle (u,v,ω) for
every boundary edge (u,v). The augmented complex is a closed pseudo-manifold, used only inside
the test. The collapse of edge ab is **topologically legal iff**

  `Lk(a) ∩ Lk(b) = Lk(ab)` in the augmented complex (vertices and edges),

where Lk(ab) = {c, d} for an interior edge and {c, ω} for a boundary edge (c, d the opposite
vertices).

This single test rejects:
- **narrow-strip pinching:** an interior edge joining two boundary vertices has ω in both links
  but not in Lk(ab);
- **merging two boundary loops:** same reason;
- **bow-tie creation**;
- **triangular holes and 3-edge loops closing**: the third hole vertex is in both links;
- **isolated triangles / whole components vanishing**: the edge cω in both links;
- **ear-triangle degeneration** where it would produce a dangling edge.

An ear collapse (ab and bc boundary, ca interior) is legal and correct: the result keeps a
boundary edge cb with one triangle.

On top of the link condition: corners are never collapsed, merged or moved. A boundary-edge
collapse places the survivor on the boundary curve (§4.3). An interior edge with exactly one
boundary endpoint keeps the boundary vertex **at its position** as the survivor.

#### 4.4.2 Other operators
- **Split:** boundary edges may split (the new vertex is on the boundary curve; twin states per
  §2.2); interior splits are unchanged.
- **Flip:** boundary edges are never flipped. Flips of edges next to the boundary are allowed,
  with valence targets 6 (interior), 4 (straight boundary) and `round(θ/60°) + 1` (boundary
  vertex with interior angle θ, minimum 2), plus the usual quality/dihedral gates.
- **Tangential smoothing:** boundary vertices move only along the boundary curve (arc-length
  redistribution) or not at all (CONSTRAINED/PINNED loops may freeze them); corners are fixed.
- **Volume-conserving smoothers (TSUR/Kuprat):** only on interior edges whose endpoints are both
  `RING_CLOSED`; they never move boundary vertices.

#### 4.4.3 Serial and parallel
The same test functions are used everywhere. The parallel gate requires the complete augmented
links of both endpoints locally (owned fans, G2) and all outer wing half-edges LOCAL or BOUNDARY
(§2.2). The split candidate freeze of Stage 15 is kept.

#### 4.4.4 Remesh-mode selection and class-dependent drivers
- **Mode selection per body** (MF3), independent of run periodicity and of seam contact:
  - **closed bodies → Stage-15 parity remeshing (`mftl1_criteria`) by default, periodic or
    not**. The parity path is topological (vertex ids, closed rings) and minimum-image based;
    in 16B its fits and moves adopt the displacement rule (§3.2), and nothing else changes. On
    non-periodic runs it stays bit-identical to Stage 15.
  - **open bodies → generalized boundary-aware remeshing** (§4.4.1–4.4.3), always.
  - **closed bodies → generalized remeshing only by explicit opt-in** per body
    (`RemeshMode <body> GENERALIZED`).
  - Consequence: periodic translation-invariance and periodic-vs-large-box comparisons (B1, C5,
    D4, D5) compare like with like, with the same criteria on both sides.
- **Legacy limitation of the parity path (documented, no fix).** The parity collapse test checks
  the vertex part of the link condition (`common == 2`), not the edge part.
  - In a closed 2-manifold, an edge-part violation for edge ab means triangles abc, abd, acd and
    bcd all exist. Then every edge of {a,b,c,d} already has its two triangles, so the four
    triangles form a closed tetrahedral component in which every vertex has valence 3.
  - MFTL1's collapse check requires valence ≥ 4 at both endpoints
    (`(N1 − 2) ≥ 2`, `(N2 − N1) ≥ 2`, `mftl1_criteria.c`), so such a collapse can never be
    performed.
  - The gap is therefore harmless for closed manifold meshes and is recorded as a legacy
    limitation. Open bodies never use the parity path.
- The remesh drivers call `MFTLPhaseSetClassify` on the singleton set **before** computing a
  recovery target. Global volume recovery (`MFTLVolumeRecovery` in `MFTLRemeshEnsemble` and the
  parallel drivers) runs only when the volume is defined. Otherwise `skip_volume_recovery` is
  forced, and `MFTLObjectVolume` is not called (fixes the unconditional calls at
  `remeshing_core.c:1433`, `remeshing_parallel.c:786`).
- Set-level recovery (e.g. a slab of two bodies) is offered through the phase-set API: a
  uniform normal shift of all members by ΔV / A(S).
- **Class transitions:** a body that acquires or loses a winding (only possible through a
  topology change, §3.3), or whose phase set becomes undefined, is reported to the application
  (return status plus a diagnostic), never silently accepted.

## 5. 16D — multiflow integration

### 5.1 Options and declarations
- `FRONTTRACKING PeriodicFronts <axes>` (explicit). Validated: each axis must be periodic in the
  fluid.
- Per body: `OpenSurface <body> [BoundaryFromMesh]`. Boundary edges come either from an
  explicit boundary-edge/loop list in the input (tags from the input), or — as an explicit
  per-input declaration — `BoundaryFromMesh` ("every unmatched edge of this input is a
  boundary", auto-tagged). Without either, an unmatched edge is an error (§2.2). Optionally
  `PhaseSet <id> <body[:sign]>… [WallClosure <axis> <min|max>]`.
- Physical boundary models per body/tag (§6, §7).
- A body's class is checked at initialisation against every enabled module (§5.2). A conflict
  stops with a message naming the module and the body.

### 5.2 Capability matrix (enforced at initialisation)
| Module | Closed P0 | Closed periodic (compact/wrap) | Open |
|---|---|---|---|
| Advection | yes | yes | yes (free boundary) / constrained (PINNED/CONSTRAINED loops) |
| Spreading / interpolation (Peskin) | yes | yes (periodic stencil) | yes |
| Surface tension (Frenet element) | yes | yes | only with a boundary model, §7 (default REJECT); needs no indicator (a surface pinned on all its loops bounds no phase; §27.1) |
| CSF / sharp CSF (IRL) curvature | yes | yes if the phase set is defined | only PINNED on a wall closure (§7.3), else REJECT |
| Integral surface tension | yes | if the phase set is defined | REJECT |
| Indicator (Poisson), IRL fractions | yes | iff the phase-set status is DEFINED (§3.6); UNDEFINED/UNSUPPORTED → configuration error | only with a wall closure, else REJECT |
| Marching cubes, breakup/coalescence | yes | if the phase set is defined (periodic MC ids) | REJECT |
| Volume recovery | yes | iff V(S) is defined (set-level) | only wall-closed, else REJECT |
| Surface scalar transport | yes | yes | with explicit loop BCs, §6 (default REJECT) |
| Surface viscosity | yes | yes | REJECT (traction BC not designed) |
| Remeshing | parity (unchanged) | parity by default; generalized by explicit opt-in (§4.4.4) | generalized (§4.4) |
| IBM markers (static, GTS) | yes | yes | only marker/force coupling OR explicit mass properties, §5.5 |
| MoveIBM | yes | compact only | REJECT unless explicit mass properties and no inside/outside use |
| Point-inside queries | yes | iff the phase set is defined | REJECT |

### 5.3 Adapter
- **Class checks in the adapter itself**: `MFAdapterObjectVolume`, `SurfaceCentre` (volume-based
  users), `MomentOfInertia` and the volume-recovery helpers call the MFTL2 phase-set API (or
  check the body class) and fail explicitly. They no longer compute an unchecked formula. The
  closed-P0 arm stays bit-identical (the current MFTL1 formulas are kept for that arm).
  Checked variants return `MFADAPTER_UNDEFINED`.
- **Per-body remesh mode** (§4.4.4): closed → parity by default (periodic or not); open →
  generalized; closed + `RemeshMode GENERALIZED` → generalized. The selection depends only on
  the declared open/closed state and the explicit option — never on run periodicity or seam
  contact.
- Checked adapter queries return one of `MFADAPTER_DEFINED` / `UNDEFINED` / `UNSUPPORTED`
  (mirroring §3.6).
- The 1e-7 clamp only on non-periodic axes.
- Periodic kernels in all adapter geometry (AABB, move/rotate/scale about continuous centres,
  proximity, thin-region, sphere generation with continuous input).
- New accessors: `MFAdapterTriVertexPositions` (local frame), `MFAdapterVertContinuous`,
  `MFAdapterVertDelta`, `MFAdapterEdgeVector`, `MFAdapterBoundaryEdgeGeometry`,
  `MFAdapterBodyWrapVertices`, `MFAdapterBoundaryLoops`, `MFAdapterBodyClass`, `MFAdapterPhaseSet*`.
- The two-pass ghost workaround is removed (§2.5).

### 5.4 Periodic infrastructure
- `FTDomain` periodic on enabled axes (restore and validate the branches), with wrapped cell
  indices in point→cell mapping, periodic ghost boxes and `sendnonlocal*` across the seam.
- No `PBoxAdaptation` along periodic axes; the MFTL2 period equals the fluid period.
- After advection: `MFAdapterBodyWrapVertices` (updates counters).
- Output (VTU): continuous local frames per triangle, seam triangles duplicated per image
  (output only).
- Logs: continuous centroids on compact axes; undefined values are printed as "undefined".
- Restart: native I/O v2.

### 5.5 IBM
Open IBM bodies are accepted only if **either**:
- the module uses pure marker/force coupling and never asks for enclosed mass properties, **or**
- the input supplies mass, centre of mass and inertia explicitly (`IBMMassProperties …`).

GTS injection and MoveIBM property routines are rejected otherwise; they are never derived from
an open surface. MoveIBM fictitious-cell (inside/outside) logic requires a defined phase set.

### 5.6 Physics conversion (no scattered corrections)
- All 131 raw-position sites are converted to the adapter accessors. Physics code never
  subtracts positions itself.
- Specifics:
  - Peskin stencils over wrapped cell ranges;
  - `EdgePlanar` via `MFAdapterEdgeVector`;
  - IRL cell clipping with the triangle unwrapped near the cell;
  - integral-surface-tension distances;
  - scalar-transport edge/centroid vectors;
  - marching cubes emits **continuous** vertex coordinates derived from periodic cell indices
    (a surface through the seam is one mesh, correctly wound), followed by canonicalisation
    because pieces assembled from several ranks may be gauged independently (§3.3);
  - `detectbodies` through seam twins;
  - breakup/coalescence through the library's periodic proximity.

## 6. 16E(a) — Open-surface scalar transport (multiflow)

MFTL2 provides only geometry: for each boundary half-edge its tag, `L_e`, `t`, `ν`, the midpoint
X_m, the in-surface distance from the triangle centroid to X_m and the triangle height
`h_T = 2A_T / L_e` (local frame). *[Correction C16E-1, 16E review §27.2: the scheme uses the normal
distance `δ_e = (X_m − X_c)·ν = h_T / 3`, see "Degenerate Dirichlet distance" below.]*

Boundary condition per body and tag (`SurfaceScalarBC <body> <tag|ALL> NEUMANN q |
DIRICHLET c_D`); an open body without a BC for every tag is rejected. Robin is reserved.

| Where | Interior edge (unchanged) | NEUMANN (flux q, zero-flux q = 0) | DIRICHLET (c_D) |
|---|---|---|---|
| Gradient reconstruction (Green–Gauss, `getScalarGradients`) | face value ½(c_T + c_N) | face value `c_f = c_T − q δ_e / D` (so `c_f = c_T` for q = 0) | face value `c_f = c_D` |
| Muradoglu correction term | Σ L ν | the boundary edge's L ν **included** (closure of Σ L ν over the element) | included |
| Diffusion matrix | off-diagonal coupling to N | no coupling; RHS `− q L_e` | diagonal `+ D L_e / δ_e`; RHS `+ D L_e c_D / δ_e` |
| Advective (normal-corrected) flux | as now | zero across boundary edges (material boundary) | zero |
| Conserved update / diagnostics | mass = Σ c A | mass change = −Σ q L Δt (zero for q = 0) | mass change = Σ D L (c_D − c_T)/δ_e Δt (inflow positive); `checkScalar` compares against the reference mass plus the accumulated boundary flux |

**Conventions.**
- **Sign:** the Neumann flux `q` is the diffusive flux per unit boundary length **positive
  outward**, along the in-surface conormal ν: `q = −D ∂c/∂ν`. A positive q removes mass;
  `q = 0` is zero flux.
- **Degenerate Dirichlet distance** *(corrected in the 16E review, C16E-1 and C16E-2, §27.2; the
  original text is kept struck through for the record)*:
  - ~~`δ_e` is the in-surface distance from the triangle centroid to the boundary-edge midpoint;
    bounded below by `δ_min = 0.1 · h_T`.~~ Both details were erroneous: the centroid lies at the
    normal distance `h_T/3` from the edge line, so `δ_e ≥ h_T/3` and a floor `0.1 h_T` can never
    act; and the midpoint distance `√(h_T² + s²)/3` (apex offset `s` along the edge) makes the
    two-point flux inconsistent on skewed boundary triangles (first-order convergence).
  - **`δ_e` is the normal distance from the triangle centroid (the scalar location) to the
    boundary edge, `δ_e = (X_m − X_c)·ν = h_T/3`**, `h_T = 2A_T / L_e`. The two-point flux
    `D L_e (c_T − c_D)/δ_e` and the Neumann face value `c_T − q δ_e / D` are then exact for a linear
    field that is constant along the edge, for any apex position.
  - **Sliver floor (Dirichlet coefficient only): `δ_eff = max(δ_e, 0.1 · L_e)`.** A sliver has a
    small normal height relative to its tangential extent, so the floor is tied to the tangential
    scale of the boundary edge, not to the normal height itself; it bounds the coefficient
    `D L_e / δ_eff ≤ 10 D`. It acts when `h_T < 0.3 L_e`. Every use increments a diagnostic
    counter (logged per step with the largest `L_e/δ_eff`).
  - `A_T` below the degenerate-area tolerance is an invariant violation (remeshing removes such
    triangles).

The `exit(1)` on a missing neighbour is replaced by the BC dispatch. On closed bodies the
missing-neighbour case remains an invariant violation.

Tests:
- zero-flux mass conservation to 1e-12 per step on a moving, remeshing open patch;
- 1-D Dirichlet diffusion on a triangulated strip with c = c₀ and c₁ at the two loops: the
  steady state is linear in the strip coordinate to discretisation order (≥ 1.9 convergence),
  and the transient matches the analytic Fourier series to ≤ 1e-3 on the finest mesh;
- a mixed NEUMANN/DIRICHLET strip.

## 7. 16E(b) — Open-surface surface tension (multiflow)

### 7.1 Principle
For constant σ, the element force of a triangle is `Σ_edges σ L ν_e ≈ ∫_T σ κ n dA`. Summed over
an open surface, interior edges cancel and the total is `Σ_boundary σ L_e ν_e`: the line force
at the boundary. MFTL2 only provides `t`, `ν`, `L_e` and loops. **multiflow never applies an
open-boundary line force implicitly.** A free liquid-interface edge is not physically
meaningful and is not offered.

### 7.2 Modes (per body and tag: `SurfaceTensionBoundary <body> <tag|ALL> <mode> [support]`)
- **REJECT** (default): an open body with Frenet/CSF surface tension and no mode stops at
  initialisation.
- **PINNED <support>**:
  - boundary vertices are kinematically constrained to the declared support: a static wall
    line, or an IBM/DEM body whose velocity they follow. Their loop is CONSTRAINED to the
    support's geometry (§4.3). *[16E review, §27.8: Stage 16 supports STATIC only; an IBM/DEM
    support is general/future scope (no gate or test of §9/§10 requires it) and is explicitly
    UNSUPPORTED in Stage 16 (rejected at initialisation, §12).]*
  - The element forces are spread to the fluid unchanged (they are the discrete ∫σκn over the
    elements).
  - The line force `R = −Σ_loop σ L_e ν_e` is the **reaction** on the support. It is added to
    the support body's force balance (IBM/DEM) or reported per step for a static wall.
  - Momentum budget: fluid force + R = 0 to round-off (tested).
- **CONTACT_LINE** (future stage): a prescribed/dynamic contact angle; reserved keyword,
  rejected in Stage 16.

### 7.3 CSF / sharp CSF (curvature-based)
These need the curvature at boundary vertices and an indicator.
- Supported only for **PINNED on a wall closure** (the phase set is defined via §3.6): boundary
  curvature from `MFTL_VFLAG_CURV_EXTRAPOLATED` (open-stencil paraboloid fit), the indicator from
  the wall-closed phase set. *[16E review, §27.9: multiflow's CSF keeps its own IRL paraboloid
  fit (the same curvature source as closed-surface CSF, open stencil at the rim, `NeighbourhoodOrder
  ≥ 2`) and IRL fractions checked against the wall-closed volume; both are legitimate
  approximations, and the existing multiflow formulation is retained.]*
- Otherwise REJECT. Integral surface tension: REJECT for open surfaces.

### 7.4 Tests
- **Spherical cap pinned on a circular ring** (library + multiflow). Library: the summed element
  force equals the analytic ring force `2π r σ sin θ` (along the axis) to discretisation order,
  and each loop's `Σ L ν` matches the analytic conormal integral.
- multiflow: a static cap on a wall orifice; the pressure jump converges to `2σ/R`; the reaction
  equals `−2π r σ sin θ` within 1 %, and fluid force + reaction = 0 to 1e-12.
- Rejection tests for REJECT, CONTACT_LINE and the open integral-surface-tension case.

## 8. Boundary / seam / MPI-rim distinction (invariant table)

| Concept | Representation | Determined by |
|---|---|---|
| **Physical boundary** | `twin_idx == −1`, `twin_gid ≤ −2` (tag) | topology only (builder/commit/operators) |
| **MPI rim** | `twin_idx == −1`, `twin_gid ≥ 0`, live remote twin | ownership/ghosting only |
| **Periodic seam** | ordinary LOCAL/REMOTE twins; a seam is a geometric crossing of the domain boundary, not a mesh attribute; lattice shifts `s(e)` derive from the counters, and winding from cycle sums (§3.3) | topology plus counters |

- Coordinates are never consulted to decide any of these.
- A seam edge can simultaneously be an MPI rim (REMOTE) — the concepts compose.
- A physical boundary can run across a seam and across rank cuts:
  - its half-edges are BOUNDARY on every rank;
  - its loop is built by the collective §2.6 algorithm;
  - its winding comes from the counters.
- One- and two-slab periodic decompositions (a rank is its own neighbour, or the same rank on
  both sides; `ensemble.c:112–150`) are explicitly tested.

## 9. Substages, dependencies and gates

| Sub | Content | Depends | Gate (must pass before the next substage) |
|---|---|---|---|
| **16A** Topology/invariants | three twin states; classification of ≈ 45 sites; `boundary` cache; tags; operator maintenance (§2.2); `MFTLValidateTwinStates`; discovery protocol; ghost completion G1/G2; builder declared class and manifold/orientability validation; copy/commit modes; `MFTLVertexRing`; boundary loops; I/O v2 skeleton | 13, 15 | closed-P0 **bit-identical** (full ctest, test_stage15, adapter tests, multiflow A/B); T-A tests (§10) incl. invariant fuzz (closed and open, non-periodic) on 1/2/4/8 ranks |
| **16B** Periodic image representation | `img` counters (vertex + pack + I/O v2), counter-aware wrap API and displacement rule, homological winding + canonicalisation after every assembly event, cached per-axis flags, reference-point evaluation, removal of the extent restriction and single-anchor paths, phase sets with the DEFINED/UNDEFINED/UNSUPPORTED API and canonical relative gauge, class-guarded remesh drivers | 16A | closed-P0 bit-identical; T-B tests B1–B16 (B11/B14–B16 reproduce every case of §3.6.10) on 1/2/4/8 ranks incl. 1- and 2-slab decompositions |
| **16C** Kernels + remeshing | local-frame kernels, boundary geometry, curvature validity/extrapolation, boundary curve modes, corners, augmented link condition (serial + parallel), boundary split/flip/smoothing rules, periodic proximity | 16A, 16B | closed-P0 bit-identical (the augmented link condition is used by the generalized mode only — open bodies and closed bodies opted in; the closed parity path is unchanged, periodic or not); T-C tests |
| **16D** multiflow integration | options/declarations, capability matrix, adapter class checks, per-body criteria, clamp, periodic `FTDomain`/PBox/`sendnonlocal*`, wrap after advection, raw-site conversion, MC continuous coordinates, IBM rules, output/restart | 16C | multiflow closed non-periodic A/B bit-identical; T-D tests |
| **16E** Boundary models | scalar-transport BCs (§6), surface-tension modes and CSF-at-wall (§7) | 16D | T-E tests |
| **16F** Validation | full matrix §10, MPI matrix, performance check, documentation (plans, adapter header, manual input for Stage 17) | all | every row of §10; performance target §11 |

No substage starts before the previous gate passes.

## 10. Validation matrix (quantitative)

Conventions:
- Geometry tolerances are relative to the natural scale (R, R², 2/R); fields are normalised by
  the field maximum.
- **Exact-parity tests use dyadic domains and exactly representable translations** (e.g.
  domain [0,1)³, translations by multiples of 2⁻⁵, vertex coordinates on a 2⁻⁴⁰ grid), so that
  operation decisions are bitwise reproducible. Non-dyadic cases use statistical criteria.
- All MPI tests run on np 1/2/4/8, plus 1-slab and 2-slab periodic decompositions where marked.

**T-A (topology/invariants)**
- A1: twin-state validation after every operation type (split, collapse, flip, build, copy
  ASSEMBLE/CUT, redistribution, serial bracket) on closed and open meshes.
- A2: **randomized invariant fuzz** — 10³ steps of random vertex jitter plus remesh plus
  redistribution on closed and open bodies, np 1/2/4/8. After every step check: twin states
  (§2.3), **exactly-once ownership** of vertices and triangles, G1/G2, boundary-edge count
  changing only by boundary splits/collapses, loop count, tags and winding.
- A3: discovery protocol — REMOTE resolution entirely by round 1 for fresh positions; forced
  fallback via deliberately stale ghost positions; a stale/dead twin gid detected as an error.
- A4: builder rejections — Möbius, bow-tie, non-manifold edge, declared-closed with a hole,
  declared-open without a boundary.
- A5: restart on a **different rank count** (write np 4, read np 1/2/8): identical topology,
  tags, counters, surface scalar.
- A6: **global discovery before ownership** — builder and ASSEMBLE/CUT commits with triangles
  deliberately held on non-owner ranks (staged copies, random placement): every twin is
  resolved by the global mode; CUT creates BOUNDARY exactly on the recorded cut set; an
  unmatched edge without a boundary declaration is a collective error; `BoundaryFromMesh`
  accepts it.
- A7: **twin_gid reader audit** — boundary edges of one loop get distinct edge ids
  (`MFAdapterEdgeId`), `FindEdgeById` finds boundary edges, copy of a boundary edge keeps its
  tag; the CI grep check passes.

**T-B (periodic representation)** (np 1/2/4/8, 1- and 2-slab)
- B1: compact seam crossing — object A away from seams vs B translated across one seam, two
  seams and a corner (x, xy, xyz), dyadic. Area, volume, continuous centroid (mod period),
  normals, curvature, a remesh pass (operation counts identical, volume ≤ 1e-12).
- B2: a large compact object (extent > half period, < period) → accepted, with correct volume
  (the extent restriction is removed).
- B3: x-spanning ribbon: `WRAP_x`, two boundary loops each with winding (1,0,0), area exact,
  no boundary edge on the seam.
- B4: xy-wrapping film in an xyz-periodic domain: `WRAP_x ∧ WRAP_y`, χ = 0, **volume
  `MFTL_UNDEFINED`** (Φ_z ≠ 0).
- B5: two opposite films bounding a slab: the phase set is finite, V = analytic `A·h` to
  1e-12; the x/y centres are undefined and the z centre is analytic.
- B6: a film in a z-walled domain with a wall closure: V = analytic volume below the film to
  1e-12.
- B7: a single triply-wrapping plane x+y+z = c (Φ ≠ 0) → UNDEFINED; two opposite such planes
  (Φ = 0, no admissible axis) → **UNSUPPORTED**, not undefined.
- B8: counter lifecycle — a vertex advected across seams 100 times: `img` updated exactly,
  X continuous (no jump) to round-off. Canonicalising wrapped input (all `img = 0`) reproduces
  the counters of the same surface given with continuous coordinates, up to one whole-body
  lattice shift. Canonicalisation is idempotent.
- B10: **precision** — a sphere drifting with `img ≈ 10⁶`: volume, area and centroid (mod
  period) equal the `img = 0` case to 1e-13 relative (reference-point evaluation, §3.5);
  whole-body re-gauging keeps `img` bounded.
- B11: **phase-set status cases** (§3.6):
  - seam-crossing sphere DEFINED;
  - single xy sheet UNDEFINED;
  - two-sheet slab DEFINED, V = A·h to 1e-12, z-centre analytic, x/y centre UNDEFINED;
  - four alternating sheets DEFINED with two slabs;
  - three sheets UNDEFINED (Φ ≠ 0);
  - four sheets (−,−,+,+) UNDEFINED (orientation);
  - a slab whose sheets are stored in different periodic images (one sheet shifted by ±L_z in
    its counters) gives the same V and centre (canonical relative gauge);
  - a compact sphere inside a slab;
  - UNSUPPORTED tilted planes (B7).
- B15: **§3.6 revision cases** — model cases M1–M7 and O1–O3 of §3.6.10:
  - the wavy-floor slab with a compact bubble in a dip (bubble stored one period away; centre
    analytic to 1e-12, bubble placed by the adjacency rule of §3.6.7);
  - **review case 4b** (a wide x-tube near a floor that is wavy along y);
  - an overhanging layer with three crossings per line, including one straddling the seam on
    the canonical line (O3);
  - a member line whose first candidate passes through layer vertices (M7);
  - near-degenerate volume → UNDEFINED; |Φ_k| ≠ A_k → UNSUPPORTED;
  - two separate slabs → V DEFINED, set centre along z UNSUPPORTED.
- B16: **orientation of the whole set and wall closures** — model cases C1–C10, U1–U2 and
  W1–W15 of §3.6.10:
  - one drop;
  - a single σ = −1 bubble in a fully periodic cell (matrix);
  - several disconnected drops;
  - nested cavities (two and three levels);
  - **review case 8c** (disjoint drops of opposite σ → UNDEFINED);
  - nested drops of equal σ → UNDEFINED;
  - B6 film and B14 cap with every closure combination;
  - winding wall loops and an ambiguous closure → UNSUPPORTED.

  Every case of §3.6.10 is a required 16B test, with the model's status and values as the
  expected result (np 1/2/4/8).
- B12: **two compact drops merging across a seam**, initially expressed in different periodic
  images (one drop's counters offset by one period): after coalescence (ASSEMBLE + MergeInto)
  and canonicalisation the merged body is `COMPACT` on all axes with no cut edges, and its
  volume equals the sum of the two drop volumes (to the merge-operation tolerance). Without
  canonicalisation the regression check detects the artificial cut edges.
- B13: **diagonal winding** — a ribbon with winding lattice (1,1,0): `WRAP_x ∧ WRAP_y`
  (conservative); its loops' windings are (1,1,0); the phase-set queries on it return
  UNSUPPORTED/UNDEFINED as specified, never a value.
- B14: **wall closure** (§3.6.2) — a spherical cap with its circular loop on a z-wall:
  - wall-patch flux from the shoelace area equals πr²;
  - V equals the analytic cap volume to discretisation order;
  - the inverted cap with closures on both walls gives V_box − V_cap;
  - a loop that winds along x on the wall → UNSUPPORTED;
  - B6 (film over a closure) selects the full-wall patch by the flux condition.
- B9: seam coincident with an MPI rank cut; periodic corner (xyz); 1-slab and 2-slab.

**T-C (kernels/remeshing)**
- C1: planar n×n patch: 4n boundary edges, 1 loop, χ = 1, area exact to 1e-14, normals exact,
  interior mean curvature 0 to 1e-12, boundary vertices flagged.
- C2: hemisphere (refine 2–5): loop length → 2πR and area → 2πR² (rate ≥ 1.9); interior mean
  curvature → 2/R; **discrete Gauss–Bonnet** Σ interior angle defects + Σ boundary turning
  angles = 2πχ to 1e-12 (χ = 1). Closed sphere χ = 2, tube χ = 0, film χ = 0.
- C3: link-condition cases, each asserting reject/accept and a valid result:
  - **one-triangle-wide strip** (all pinching collapses rejected);
  - **ear triangles** (legal ear collapse accepted; isolated triangle rejected);
  - **nearby independent boundary loops** one edge apart (merge rejected);
  - a 3-edge triangular hole (closing rejected);
  - a corner (never collapsed).
- C4: boundary-preserving remeshing, 20 cycles on a perturbed hemisphere and planar patch:
  - serial and parallel; loop count and tags preserved;
  - boundary vertices on the analytic curve within 1e-12 (planar/CONSTRAINED) or the
    interpolation error ≤ C h³ (REPRESENTED);
  - corners bit-fixed; area within the fit error; mesh integrity each cycle.
- C5: an open patch crossing a seam (O + periodic) vs translated away (dyadic): loops, tags,
  areas, normals, curvature flags and values identical, and a remesh pass identical.
- C6: **geometry scales 1e-4 and 1e4** for C1–C4 and B1 (all tolerances relative) — exposes
  remaining absolute thresholds.
- C7: rank independence of C1–C5 (topology exact for the serial kernel; parallel sweeps
  statistical within 1 %).

**T-D (multiflow integration)**
- D1: capability matrix — every REJECT row of §5.2 stops with its message; every allowed row
  initialises.
- D2: **mixed open + closed ensemble** in one body list (remesh, redistribute, output, restart).
- D3: seam-crossing spreading — a uniform force integrates to the analytic total to 1e-12 and
  equals the unshifted case per (shifted) cell; indicator equal to 1e-10; `sendnonlocal*`
  round trips exact.
- D4: droplet across an xyz corner vs centred (static and oscillating), **same remesh mode on
  both sides** (parity, §4.4.4): spurious currents, frequency and amplitude to 1e-6; volume
  drift to 1e-10.
- D5: bubble advected through a seam for ≥ 200 steps vs a larger non-periodic box (**same
  remesh mode on both sides**): centroid
  trajectory 1e-6 L, volume/area 1e-6, triangle count within 1 %, no constraint violations,
  exactly-once ownership each step.
- D6: marching-cubes pinch-off with the neck on a seam vs unshifted: same body count, volumes to
  1e-6.
- D7: a passive open patch in a LeVeque vortex, out and back: returns to the initial shape to
  the Stage-13 level; boundary preserved.
- D8: **restart on a different rank count** for a periodic run and an open run.
- D9: remesh-mode selection — a closed body in a periodic run uses parity by default and
  generalized only with `RemeshMode GENERALIZED`; an open body always uses generalized; a
  non-periodic closed case stays bit-identical to Stage 15.

**T-E (boundary models)**
- E1: zero-flux scalar mass conservation to 1e-12 per step (moving, remeshing open patch).
- E2: 1-D Dirichlet diffusion on a strip (§6); mixed Neumann/Dirichlet; a non-zero Neumann
  flux q > 0 removes mass at exactly `q·L_loop` per unit time (sign convention); a sliver
  boundary triangle uses the floor `δ_eff = 0.1 L_e` and is counted (C16E-2, §27.2).
- E3: pinned spherical cap on a circular ring — force balance (§7.4).
- E4: rejection of REJECT/CONTACT_LINE/integral-ST/open-surface-viscosity configurations.

**Compatibility gate (every substage):** full MFTL2 ctest (Release + ASan/UBSan, LAPACK on and
off), `test_stage15`, adapter tests 1–4 ranks, multiflow Release A/B **bit-identical** to the
Stage-15 build. Regenerated MFTL2 goldens (owner decision, `plans/15` R3) must pass unchanged
once they exist.

## 11. Risks and performance

**Risks**
- Classification of the ≈ 45 missing-twin sites and the 34 `twin_gid` readers (highest; 16A
  gate, explicit predicates, CI grep check, A7).
- Counter consistency between canonicalisations (§3.3 argument) — guarded by B8, B12 and the A2
  fuzz, which checks after every step that no cut edge appears on an axis flagged COMPACT.
- The discovery protocol before ownership (A6) and under stale ghost positions (A3).
- Front↔fluid periodic transposes (D3 exactness before any physics).
- Phase-set evaluation (B11, B14–B16):
  - the definition is exact (Φ = 0 + orientation);
  - the evaluation is restricted to admissible axes, with everything else UNSUPPORTED — never
    a value;
  - revision 3.2 is backed by an executable model whose cases the 16B tests must reproduce
    (§3.6.10).
- Marching-cubes continuous coordinates and canonicalisation after assembly (D6, B12).
- Physics boundary models are deliberately minimal (PINNED only); contact lines are out of
  scope.

**Performance**
- Twin states: an int compare per predicate.
- `img`: +12 bytes per vertex and per packed vertex; no effect on non-periodic arithmetic.
- The wrap after advection is O(N_local).
- Discovery: one sparse exchange per redistribution, replacing the second full redistribution
  (a net gain); the global fallback normally carries zero data. The global mode (builder,
  commits) costs O(unresolved rim) Allgatherv.
- Canonicalisation: O(E) gather at assembly events only (construction, MC, merges); a
  distributed BFS is a Stage-18 option.
- Loops: an Allgatherv of O(boundary) records, only when loops are requested.
- Periodic accessors are inline, with the domain flag hoisted out of loops.
- Target: closed non-periodic wall time unchanged within noise; periodic ≤ 10 % over the
  equivalent non-periodic run; open-surface overhead ≤ 5 % over closed of equal size.

## 12. Intentionally unsupported in Stage 16
- Contact-line dynamics (CONTACT_LINE reserved). Consequently the loops of a wall-closed member
  must be PINNED STATIC (a free loop would leave its wall closure; §27.7).
- PINNED supports on IBM/DEM bodies (moving-support kinematics and support-force coupling; §27.8).
- Free liquid-interface edges.
- Surface viscosity on open surfaces.
- Integral surface tension on open surfaces.
- Phase-set quantities with status UNDEFINED (not separating, inconsistent orientation, open
  without wall closure).
- Phase-set quantities with status UNSUPPORTED (§3.6.9):
  - no admissible axis, e.g. tilted triply-wrapping pairs, diagonal winding;
  - wall loops that wind periodically, loops on two walls, closures on two axes;
  - ambiguous wall closures;
  - components with |Φ_k| ∉ {0, A_k} or zero enclosed volume;
  - no generic line within N_try candidates;
  - the centre of a disconnected region along a periodic axis.
- Mass properties derived from open surfaces.
- Non-manifold, non-orientable and self-intersecting meshes (embeddedness is assumed, not
  checked).
- `mftl1_criteria` on open bodies (closed bodies may use it periodic or not; §4.4.4).

## 13. Required data-structure changes (complete list)
1. `MFTL_Vertex.img[3]` and `PackedVertex.img[3]` — the **only structural change**.
2. Semantic changes without layout change:
   - the three `twin_gid` states (−2 − tag encoding) with explicit predicates;
   - the `boundary` field as a derived cache;
   - new vertex flag bits (`CURV_UNDEFINED`, `CURV_EXTRAPOLATED`, `CORNER`) in the existing
     packed `flags`;
   - a local, unpacked vertex field for the extrapolated curvature (recomputed on demand).
3. New object-level descriptors:
   - `declared_open` and declared boundary edges;
   - the class/winding cache (per-axis flags, winding lattice);
   - remesh mode;
   - phase sets (separate objects);
   - boundary-curve modes and projectors per tag.
4. Native I/O v2 (counters, tags, class, period box). v1 is read as closed with `img = 0`,
   followed by canonicalisation.

## 14. Final plan review (answers)
1. **Half-edge/twin invariant.**
   - Every live half-edge in a quiescent mesh is exactly one of:
     - LOCAL (`twin_idx ≥ 0`, consistent `twin_gid`);
     - REMOTE (`twin_idx = −1`, `twin_gid ≥ 0` naming a live authoritative half-edge that
       names it back);
     - BOUNDARY (`twin_idx = −1`, `twin_gid = −2 − tag`).
   - "Unknown"/stale is illegal, and all readers use explicit predicates.
   - Every operator maintains the states (§2.2). BOUNDARY arises only from explicit
     declarations (open body plus declared boundary edges, or a recorded CUT set).
   - `MFTLValidateTwinStates` checks the invariant collectively at quiescent points. It
     distinguishes a dead/stale remote twin (an unresolved non-negative gid → error) from a
     boundary (negative encoding, never queried).
2. **Periodic image representation.**
   - Persistent per-vertex per-axis counters `img[3]`, with wrapped `pos` and continuous
     coordinate `X = pos + img ⊙ L` (evaluated relative to a reference point).
   - Every vertex update is a minimum-image displacement from an anchor.
   - Counters are packed and stored by I/O v2, and canonicalised after every assembly event.
   - Winding is homological: the cycle sums of integer edge lattice shifts, i.e. whether the
     shift cochain is a coboundary.
   - After canonicalisation, cached per-axis flags (cut edges ⇔ winding) are valid until the
     next assembly event.
   - There is no object-extent restriction, only the per-edge half-period rule.
3. **Wrapping/non-separating vs finite-volume boundaries.**
   - A phase-boundary set is separating iff its members are closed (or wall-closed, §3.6.2),
     its net normal-flux (homology) vector Φ(S) is zero, and its orientation is consistent.
   - Consistency is checked for the whole set (§3.6.5): layer alternation on one line, plus a
     nesting tree of the compact components with a common exterior level. No signed-volume
     test is used.
   - DEFINED quantities are evaluated along an admissible axis, in a canonical relative gauge
     built from exact adjacency relations (§3.6.7).
   - Separating sets without an admissible axis are UNSUPPORTED; non-separating or
     inconsistently oriented sets are UNDEFINED.
4. **Open-surface physics in multiflow.**
   - Directly: advection and spreading/interpolation.
   - Frenet surface tension with PINNED boundaries (reaction on the support); CSF/sharp CSF
     only PINNED on a wall closure.
   - Surface scalar transport with NEUMANN (outward-positive q) / DIRICHLET loop BCs.
   - IBM markers for force coupling, or with explicit mass properties.
   - Indicator/IRL/volume recovery only for a wall-closed DEFINED set.
   - Remeshing always generalized.
5. **Explicit rejections.**
   - Volume, centre, inside/outside, indicator, IRL, marching cubes, breakup/coalescence,
     volume recovery and point queries whenever the phase-set status is UNDEFINED or
     UNSUPPORTED.
   - Integral surface tension and surface viscosity on open surfaces; surface tension without a
     boundary model (REJECT default); mass/inertia from open surfaces; `mftl1_criteria` on open
     bodies.
   - Enforced at multiflow initialisation (§5.2) and by the checked MFTL2/adapter APIs (§3.6,
     §5.3).
6. **MPI ghost completion.**
   - After every redistribution: G1 (every edge of every owned triangle LOCAL or BOUNDARY) and
     G2 (complete fans of owned vertices).
   - G2 from the ghost rule. G1 from the owned-mode discovery: round 1 geometric candidates with
     the global ℓ_max/3 + 2ε ball; round 2 the global mode, always executed; an unresolved gid is
     an error.
   - Before ownership exists (builder, commits, assembly), only the global mode is used, and any
     live authoritative copy answers.
7. **Fundamental data-structure change:** only the vertex image counters. Everything else —
   boundary edges and tags, rims, seams, loops, winding classes, phase sets, remesh modes — is
   semantics on the existing half-edge representation, with no new entity types, no virtual
   twins, and no half-edge or triangle pack change. Manifold, orientable, embedded meshes
   remain required.

## 15. Review cross-reference
| Review item | Resolution |
|---|---|
| M1 twin-gid maintenance | §2.1–§2.3 (three states, per-operator rules incl. `glue_wings`, copy/commit modes, collective validation) |
| M2 twin discovery | §2.4–§2.5 (geometric candidates, global fallback, completion guarantee G1/G2) |
| M3 lattice representation | §3 (persistent per-axis counters, pack/I-O change, per-axis classes, extent restriction removed) |
| M4 wrapping volume/indicator | §3.6 (phase-boundary sets, admissible axis, Φ = 0, wall closures) |
| M5 adapter bypass | §5.3 (adapter class checks; closed-P0 arm unchanged) |
| M6 scalar transport | §6 (NEUMANN/DIRICHLET design, `exit(1)` replaced, tests) |
| M7 surface tension | §7 (geometry-only MFTL2, REJECT/PINNED, reaction to support, CSF only wall-closed) |
| M8 collapse topology | §4.4.1 (augmented link condition, serial + parallel) |
| M9 IBM | §5.5 |
| Should-fix 1–7 | §2.6 (fan/loop scope), §8 + T-B9 (degenerate graphs), §4.3/§4.4 (valence targets, boundary curve, class-guarded recovery), §5.3 (per-body criteria), §3.5 (search-tree efficiency), §5.2 (advection with constraints), §3.2/§13 (I/O v2) |
| **MF1** winding + canonicalisation (review of revision 2) | §3.3 (homological definition, `MFTLCanonicaliseImages` after every assembly event, whole-body re-gauge, cached flags), §3.4; tests B8, B12, A2 |
| **MF2** phase sets / status / gauge | §3.6 (DEFINED/UNDEFINED/UNSUPPORTED, Φ = 0 criterion without an axis requirement, canonical relative gauge, alternation, (0, V_cell) check, centre in the same gauge); tests B7, B11, B13, B14 |
| **MF3** remesh-mode selection | §4.4.4, §5.2, §5.3 (closed → parity periodic or not; open → generalized; closed generalized opt-in; legacy tetrahedron argument); tests D4, D5, D9 |
| **MF4** discovery before ownership | §2.2 (builder/CUT explicit boundary declarations), §2.4 (global mode with authoritative holders, owned mode with global ℓ_max + margin, fallback, errors); test A6 |
| **MF-36-1 … S-36-5** (review of §3.6 revision 3.1, §18.2) | §3.6 revision 3.2, mapping in §3.6.11; cases B15/B16 = §3.6.10 |
| Should-fix (review of revision 2) | `twin_gid` reader audit and predicates §2.1 + A7; precision via reference points §3.5 + B10; displacement rule §3.2; diagonal winding §3.4 + B13; wall-closure conventions §3.6 + B14; embeddedness assumption §3.6/§12; Neumann sign + Dirichlet δ floor §6 + E2; additional validation cases §10 |

## 16. Design-review record

| Round | Date | Outcome |
|---|---|---|
| Review of revision 1 | 2026-09-23 | not ready — must-fix M1–M9 |
| Review of revision 2 | 2026-09-23 | not ready — must-fix MF1–MF4 plus should-fix items |
| Revision 3 | 2026-09-23 | MF1–MF4 resolved (§15); should-fix items incorporated |
| Review of §3.6 revision 3.1 (+ 16A acceptance) | 2026-09-24 | §3.6 not ready: must-fix MF-36-1, MF-36-2; should-fix S-36-3 to S-36-5 (§18.2). 16A accepted (§18.1) |
| §3.6 revision 3.2 | 2026-09-24 | resolves MF-36-1 … S-36-5 (§3.6.11); executable model 47/47 (§3.6.12); **awaiting a fresh independent review** |

**Status (updated 2026-09-24):**
- The review of 2026-09-24 (§18.2) found §3.6 revision 3.1 **not sound**.
- **Revision 3.2** of §3.6 resolves the review items. It was **independently reviewed on
  2026-09-24 and is FROZEN** (§19, errata E1–E8, no rule changes). The Stage-16 design is frozen.
- 16B is implemented (§20) and awaits its independent review.
- 16A is accepted (§18.1).
- The 2026-09-23 status below is kept as history.

**Status (2026-09-23):**
- The final narrow check (before 16A) confirmed §3.3 (winding/canonicalisation) as sound.
- It found a must-fix in the §3.6 evaluation: global-window placement of compact members, and
  first-crossing layer ordering.
- §3.6 is **revised (revision 3.1) and REQUIRES A FRESH INDEPENDENT REVIEW BEFORE 16B BEGINS**.
- The complete Stage-16 design is therefore **not frozen yet**.
- 16A (topology/twin-state invariants) does not depend on §3.6 and is implemented (§17).

**Unresolved design questions: none that block implementation.** The remaining items are
deliberate, documented scope limits, each returning an explicit status or error, never a value:
- evaluation of separating phase sets without an admissible axis (UNSUPPORTED);
- periodically winding wall loops (UNSUPPORTED);
- contact-line physics (reserved keyword, rejected);
- surface viscosity and integral surface tension on open surfaces (rejected);
- a distributed canonicalisation BFS (a Stage-18 performance option; the gather-based version
  is correct).

Two choices are **owner decisions with safe defaults**:
- the corner-angle threshold (default 30°);
- the `δ_min = 0.1·h_T` Dirichlet floor.

Both are parameters, validated by C3/C4 and E2.

## 17. 16A implementation status and test results

**Status: 16A IMPLEMENTED (2026-09-24) and ACCEPTED by the independent review of 2026-09-24
(§18.1). 16B–16F NOT STARTED. Stage 16 as a whole is INCOMPLETE; its Acceptance is NOT PASSED.**

> **§3.6 revision 3.1 was reviewed on 2026-09-24 and is NOT frozen (§18.2).** Revision 3.2 awaits
> a fresh independent review. 16A does not depend on §3.6. Nothing of 16B (image counters,
> winding, canonicalisation, phase sets) was implemented.

### 17.1 What was implemented (MFTL2)
| Item (§2) | Implementation |
|---|---|
| Three states + encoding | `MFTL_TWIN_UNKNOWN (−1)`, `MFTL_TWIN_NONE (−2)`, BOUNDARY code `−2 − tag` (`mftl_constants.h`); struct documentation (`mftl_core.h`). |
| Explicit predicates | `MFTLHeIsLocal / IsRemote / IsBoundary / IsUnresolved / BoundaryTag / SetBoundary / SetRemote`, `MFTLHeTwinGid` (LOCAL/REMOTE gid, `MFTL_ERROR` for BOUNDARY — a boundary code is never returned as a gid), `MFTLHeTwinCode` (raw code for transport/copies), `MFTLHeEdgeId` (`min(gid, twin)` for LOCAL/REMOTE, `gid` for BOUNDARY) — `mftl_topology.h`. |
| `boundary` field | now the cache `MFTLHeIsBoundary`; set by `MFTLRefreshBoundaryCache` (called by `MFTLRebuildTwins`, the remesh finaliser, commits and builders), never a rim marker. |
| Boundary vertex convention | `MFTLRefreshBoundaryCache` points every vertex with an outgoing BOUNDARY half-edge at it; split and flip keep it. |
| Declared class | `MFTL_Object.declared_open` + `MFTLObjectDeclareOpen`; only declared-open bodies may carry BOUNDARY half-edges (integrity check and validation). `MFTLGenerateFace` declares its body open. |
| Serial builder | `MFTLBuildTopologyFromTriples` now returns `int`: duplicate directed half-edge (non-manifold edge / inconsistent orientation / Möbius) → error; unmatched edge on a closed body → error; declared open → BOUNDARY with tags = loop index ordered by the smallest vertex gid; bow-tie → error; fills the gid lookups. GTS reader uses it (open files accepted only for declared-open bodies). |
| Soup builder | `MFTLBuildFromSoupEx(…, MFTL_BoundaryDecl*)` (global discovery mode before ownership): unpaired half-edges gathered on all ranks **plus** locally paired half-edges on the same edges (duplicated triangles), so every twin is found wherever it is held; different half-edges with the same direction → error; unmatched edges become BOUNDARY only for a declared-open body **and** an explicit declaration (`from_mesh` = BoundaryFromMesh, or an explicit edge list — union over ranks; missing or extra declared edges → error); everything collective. `MFTLBuildFromSoup` = closed. The native mesh reader uses `from_mesh` for declared-open bodies. |
| Copies / commit | `MFTLObjectAppendTriangleCopy` stages the source state code; `MFTLObjectCommitCopy(dst, ASSEMBLE|CUT)` resolves every non-LOCAL, non-BOUNDARY half-edge in global mode against live authoritative (non-ghost) half-edges of all ranks (name-back and duplicate-signature checks); CUT turns the unresolved copies of existing source twins into BOUNDARY with one new tag (max+1) and requires a declared-open destination; anything else is a collective error. |
| Operators | split: interior halves/inner pairs LOCAL with explicit gids; BOUNDARY edge halves inherit the tag; REMOTE never split. `glue_wings`: LOCAL+LOCAL → LOCAL pair, LOCAL+BOUNDARY → LOCAL becomes BOUNDARY with that tag (asserts no REMOTE / double BOUNDARY). Collapse gate `mftlCollapseStatesOk` (no REMOTE edge/wing, no triangle with two BOUNDARY wings) before both the MFTL2 and the MFTL1-parity criteria; the MFTL2 link test now uses complete neighbour sets for boundary vertices and rejects interior edges joining two boundary vertices (pinch). Flip: only LOCAL edges; complete neighbour sets for the duplicate-edge test; boundary vertices keep their outgoing BOUNDARY half-edge. The finaliser only refreshes LOCAL pairs. |
| Transport | pack uses `MFTLHeTwinCode`; unpack/ghost append keep the code, resolve `twin_idx` only for codes ≥ 0; states re-derived by `MFTLRebuildTwins`. |
| Discovery (§2.4) | `MFTLInternalGlobalLookup` (global mode: Allreduce count → Allgatherv requests → Allgatherv answers from every live authoritative holder, with name-back and duplicate-signature results). Owned mode inside ghost completion: candidates = ranks whose (periodic) box meets B(X_m, ℓ_max/3 + 2ε), ℓ_max = global max live edge length; replies only from owners; then the mandatory global fallback (any authoritative holder). Unresolved → collective error. Test hooks: `MFTLDiscoverySetForceFallback`, `MFTL_DiscoveryStats`. Round 1 is skipped (fallback only) under x-slab load balancing, whose ownership is not the Cartesian box. |
| Ghost completion G1/G2 (§2.5) | `MFTLCompleteGhosts`: imports as ghosts the twin triangles of REMOTE edges of owned triangles (G1) and of REMOTE edges incident to owned vertices (G2), repeating until no rank requests. Called at the end of `MFTLRedistributeListOfObjects` (one batched Allreduce decides which objects need rounds) and `MFTLRedistributeObject`. |
| Validation (§2.3) | `MFTLCheckTwinStatesLocal` (states, LOCAL symmetry/vertices, cache, declared class, directed duplicates, bow-ties on complete owned fans, boundary out-half-edge convention, optional G1/G2) and collective `MFTLValidateTwinStatesEx` (+ REMOTE liveness and name-back, BOUNDARY agreement between copies). Quiescent hook after every redistribution: full collective validation in Debug builds or with `MFTL_VALIDATE=1`; Release runs the cheap local subset (states, symmetry, G1). |
| Rings / loops (§2.6) | `MFTLVertexRing` (CLOSED / OPEN_COMPLETE / INCOMPLETE; `IsBoundaryVertex` uses it); `MFTLBoundaryLoops` / `MFTLFreeBoundaryLoops` (records from owned boundary vertices, Allgatherv, chaining; ordered by smallest vertex gid; tag, mixed-tag flag, minimum-image length). The winding vector is a 16B item (not provided). |
| Reader audit (§2.1) | `tools/check_twin_gid_readers.sh` (ctest `TwinGidReaderAudit`): no raw `twin_gid` comparison outside the predicate header, the twin-state module and the transport layer; multiflow passes it too. |

multiflow adapter: `MFAdapterEdgeId` → `MFTLHeEdgeId` (distinct ids for boundary edges);
`FindEdgeById` rim match through `MFTLHeIsRemote`/`MFTLHeTwinGid`; canonical-edge and adjacent-
triangle helpers through `MFTLHeIsLocal`; edge statistics through `MFTLHeEdgeId`; triangle
pack/unpack carry `MFTLHeTwinCode`; `MFAdapterOwnedTrianglesMissNeighbours` is the G1 assertion
(LOCAL or BOUNDARY); staged copies (soup build, copy, merge, unpack) are closed by
`MFTLObjectCommitCopy(ASSEMBLE)` before the next redistribution (abort on failure) and the former
second redistribution pass is removed (the library completes the ghost layer).

### 17.2 Audit of the missing-twin / twin_gid sites
| Class | Sites | Treatment |
|---|---|---|
| Writers | `topology.c` (RebuildTwins, BuildTopologyFromTriples), `soup.c` (builder, copy), `sphere.c`, `generators.c`, `remeshing_core.c` (new_halfedge, split, glue_wings, finaliser), `communication.c` (pack ×2, unpack ×4, ghost append, remap) | converted: explicit state codes (§17.1). |
| Decisions that must tell BOUNDARY from REMOTE | CollapseApproved (`is_boundary_edge`, IsBoundaryVertex, link test), MFTLSplitApproved, FlipApproved / FlipApprovedWithParams, glue_wings, MFTLCheckMeshIntegrity, MFTLFlattenConnectivity neighbour, GTS reader, adapter `MissNeighbours` | converted to predicates / `MFTLVertexRing`. |
| `twin_gid` readers | MFTLHeTwinGid (redefined), pack ×2 (TwinCode), soup copy (TwinCode + IsRemote), native edge id (EdgeId), adapter EdgeId / FindEdgeById / EdgeLengthStats / pack / unpack; test_stage15 helpers (closed meshes, unchanged semantics) | converted; grep check in ctest. |
| Adjacency walkers that stop at "no stored twin" | ring helpers in `mftl_topology.h`; `curvature.c` (closed rings only, 4 sites); `volume_conserving.c` (20); `vertex_fit.c` (11: a missing twin = never fit / feature); `remeshing_core.c` feature census and serial sweeps; `remeshing_parallel.c` interior gates (6: require LOCAL twins, ghost-free); collapse fan loop | unchanged: they test `twin_idx` for *storage* (both REMOTE and BOUNDARY mean "not stored") and refuse to use a missing neighbour; behaviour at a true boundary is 16C work (curvature validity, boundary smoothing). |
| MFTL1 parity criteria (`mftl1_criteria.c`, 17) | closed rings required (`mftl1RingClosed`), wings via `mftl1Across` | unchanged: closed-body parity path (§4.4.4); boundary-adjacent operations are rejected there. |

### 17.3 Tests and results
New test `tests/test_stage16a.c` (ctest `Stage16A-np1/2/4/8`), every CHECK collective, the full
collective `MFTLValidateTwinStatesEx` after every tested consistent point:
- **closed_sphere** — closed regression: valid with G1/G2, exactly-once ownership, no BOUNDARY,
  REMOTE only on ghost triangles, edge ids unchanged (`min(gid, twin)`); a stale/non-existent
  REMOTE gid and an UNKNOWN state are detected (A3).
- **open_patch** — curved patches with one and two boundary loops: states, exactly-once, boundary
  count, `MFTLBoundaryLoops` (count, sizes, tags 0/1, length, identical on all ranks), distinct
  boundary edge ids (A7); re-decomposition + second redistribution.
- **builder_rejections** (A4) — Möbius strip, bow-tie, non-manifold edge, closed body with a hole
  rejected; the same patch declared open accepted.
- **soup_before_ownership** (A6) — soup with random placement, arbitrary owner labels and
  duplicated triangles: open without declaration / on a closed body → error; BoundaryFromMesh and
  an explicit list (given on one rank only) accepted with the right tags; a list missing an edge or
  naming an interior edge → error; closed sphere soup: every twin REMOTE-resolved before
  ownership, then redistribution valid and exactly-once.
- **copy_commit** — ASSEMBLE of a whole body; CUT of half a sphere (BOUNDARY exactly on the
  recorded cut set, one loop, tag 0); ASSEMBLE of a half → error; CUT into a closed body → error;
  CUT of half an open patch (old boundary keeps tag 0, the cut gets tag 1, one mixed loop).
- **discovery** (A3, G1/G2) — default grid, 1-slab and 2-slab periodic-x layouts, closed sphere
  (crossing the seam in the periodic layouts) and open patch: ghost triangles stripped → G1 holes →
  completion resolves 100 % in owned-mode round 1 for fresh positions (fallback 0); forced fallback
  resolves 100 % in round 2; stale positions (all local copies shifted by 1.4) → round 1 misses,
  the mandatory fallback completes; validation with G1/G2 after each.
- **boundary_operators** — boundary-edge splits (tag inherited, new vertex OPEN_COMPLETE with its
  outgoing BOUNDARY half-edge), boundary flips rejected, interior flip keeps states, two collapses
  of an interior edge with one BOUNDARY wing (boundary count and tags preserved), collapses whose
  triangle has two BOUNDARY wings rejected by the gate (both sides).
- **fuzz** (A2) — random jitter + `MFTLRemeshEnsembleParallel` (MFTL2 criteria, and the MFTL1
  parity criteria every third step for the closed body) + redistribution, closed sphere and open
  two-loop patch; after every step: full validation with G1/G2, exactly-once ownership, loop count
  and tags, distinct boundary edge ids.

| Gate item | Result |
|---|---|
| MFTL2 Release `ctest` (LAPACK on) | 62/62 pass (incl. Stage15-np1..4, Stage16A-np1/2/4/8, TwinGidReaderAudit) |
| MFTL2 Release `ctest`, LAPACK off | 62/62 pass |
| MFTL2 Debug + ASan/UBSan/LSan `ctest` (full collective validation at every redistribution of every test) | 62/62 pass; no UBSan `runtime error`, no ASan error; LSan clean with the project's MPI suppressions |
| `test_stage16a` Debug+sanitizers, np 1/2/4/8 | 339 / 370 / 370 / 370 checks, 0 failed |
| Randomized fuzz, 1000 steps, closed + open, `MFTL_VALIDATE=1`, np 1/2/4/8 (Release) | 0 invariant violations (10 002 checks per rank count) |
| multiflow adapter tests (np 1–4), also with `MFTL_VALIDATE=1` | 78 / 168 / 261 / 360 checks, 0 failed; 0 twin-state diagnostics |
| multiflow Release A/B, 16A vs Stage-15 binaries run side by side: FT np1/np4, FT_Sharp_CSF np1/np4, FT_Sharp_Integral np1/np4, IBM-MFTL A np1/np4, IBM-MFTL B np4 | **bit-identical** (every HDF5 dataset, worst difference 0); FT np4 also bit-identical to the archived Stage-15 run and with `MFTL_VALIDATE=1` (0 violations) |
| MOVEIBM2 np1/np4, TOPO np1 (900 s cap) | both builds hit the cap exactly as the archived Stage-15 runs (MOVEIBM2 np1: 3 partial files, identical; np4 and TOPO: no output before the cap) — **not informative** |
| Redistribution overhead (MPIStressTest, 50 bodies, 8 ranks, idle machine) | 5.2 s vs 5.0 s for the Stage-15 library (+4 %: one batched Allreduce per redistribution + the Release local check) |

### 17.4 Deviations from §2 (to be judged by the acceptance review)
1. `MFTLHeTwinGid` returns `MFTL_ERROR` for BOUNDARY instead of asserting (§2.1): existing readers
   then compute the correct BOUNDARY edge id without a code change; `MFTLHeTwinCode` exposes the raw
   code where it is needed (transport, copies).
2. Owned-mode round 1 uses `MPI_Alltoall` + `MPI_Alltoallv` rather than `Isend/Irecv`, has no
   cached gid→owner hint, and uses ε = 0 (the library cannot know the application's staleness);
   correctness rests on the always-executed global fallback, as specified. Under x-slab load
   balancing round 1 is skipped.
3. Ghost completion also covers G2 actively (REMOTE edges incident to owned vertices are resolved,
   iterating until no rank requests) instead of relying on the ghost rule alone.
4. Release builds run the cheap local subset (states, LOCAL symmetry, G1) at quiescent points; the
   directed-duplicate, bow-tie and boundary-vertex checks run in Debug / `MFTL_VALIDATE=1`
   (measured cost reason, §17.3).
5. The MFTL2 (non-parity) collapse link test now uses complete neighbour sets at boundary vertices
   and rejects interior edges joining two boundary vertices; flips use complete neighbour sets.
   These are the minimal rules needed to keep open bodies manifold under the existing remesher
   (found by the np-4 fuzz); the augmented link condition proper remains 16C (§4.4.1). Closed
   bodies are unaffected (identical neighbour sets).
6. The loop winding vector (§2.6) needs 16B's lattice shifts and is not provided; tags are not
   persisted by the native mesh file (v1 format — I/O v2 is 16B; a declared-open body re-derives
   tags as loop indices on reading). A5 (restart on a different rank count) is therefore not part
   of this 16A delivery.

### 17.5 Issues relevant to independent acceptance of 16A
- The multiflow topology-change path (marching-cubes reconstruction, breakup copies → ASSEMBLE
  commit, removal of the adapter's second redistribution pass) is exercised by the adapter tests
  and the library tests, but **no multiflow end-to-end case with topology changes finished within
  the A/B time cap** (TOPO, MOVEIBM2 — identical cap behaviour in Stage 15). A longer TOPO run (or a
  shorter topology-change case) should be compared before 16A is accepted.
- The periodic tests cover compact closed bodies across a seam and 1-/2-slab layouts; surfaces that
  *wrap* a periodic direction need the 16B image counters and are not tested.
- ~~§3.6 (revision 3.1) still requires a fresh independent review before 16B begins.~~ Reviewed
  on 2026-09-24: not ready (§18.2).

## 18. Independent review of 16A and of §3.6 revision 3.1 (2026-09-24)

Reviewer: Claude Code (independent of the implementation session). Nothing was committed. The
reviewer built everything from the working tree and reran the tests (§18.1.3).

### 18.1 16A — verdict: ACCEPTED (with should-fix items)

#### 18.1.1 What was checked (code reading)
- **States and encoding.** LOCAL/REMOTE/BOUNDARY are disjoint by construction: REMOTE is
  `twin_gid ≥ 0`, BOUNDARY is `twin_gid ≤ −2`, and UNKNOWN is `−1`. `MFTLRebuildTwins` keeps
  the code of an unpaired half-edge, so a formerly LOCAL half-edge becomes REMOTE, never BOUNDARY.
- **Every writer of a BOUNDARY code was traced:**
  - builders — explicit declaration only, and only on a declared-open body;
  - the CUT commit;
  - boundary-edge split — inherits the parent code;
  - `glue_wings` — LOCAL + BOUNDARY; asserts no REMOTE wing and not two BOUNDARY wings;
  - `MFTLGenerateFace` — declares its body open.
- **A stale or dead REMOTE gid is never turned into BOUNDARY on any path except one.** A CUT
  commit fed from a source that is not quiescent can do it (S-A1). Everywhere else a stale gid
  stays REMOTE, and the validation / ghost completion report it as an error.
- **Readers.**
  - The predicates, `MFTLHeTwinGid` (returns `MFTL_ERROR` for BOUNDARY) and `MFTLHeEdgeId` are
    correct.
  - The adapter's `EdgeId`, `FindEdgeById`, `Canonical`, `EdgeAdjTri`, pack/unpack and
    `MissNeighbours` are converted.
  - The reader audit script passes.
- **Operators.**
  - Split sets both new inner pairs and the split halves explicitly (LOCAL), or both halves
    BOUNDARY with the parent's tag.
  - The collapse gate `mftlCollapseStatesOk` runs before both criteria paths.
  - The parallel collapse, split and flip gates already require a LOCAL twin and ghost-free
    fans, so REMOTE never reaches an operator.
  - Flip accepts only LOCAL edges and keeps the boundary vertex's outgoing BOUNDARY half-edge.
- **`MFTLObjectCommitCopy`.**
  - ASSEMBLE rejects every unresolved code, including UNKNOWN, a partner that does not name the
    half-edge back, and duplicate copies that differ.
  - CUT requires `declared_open`; otherwise it is a collective error. It therefore cannot open a
    closed body.
- **Discovery and ghost completion.**
  - The global mode answers from any non-ghost holder. The owned mode is followed by a global
    fallback that always runs.
  - An unresolved gid, or a round that imports nothing, is a collective error.
  - Every branch is decided from a globally reduced value, so the ranks cannot diverge.
- **Boundary loops.** Records come from owned boundary vertices; exactly-once ownership and G2
  make each record unique. Checking that incoming edge ids are unique makes the successor map a
  permutation, so the chaining produces cycles only.
- **Vertex rings.** `MFTLVertexRing` rewinds and replays the fan correctly. Its fan count
  compared with the incident-triangle count detects bow-ties.

#### 18.1.2 Reviewer tests (scratch, not in the repository)
- **Serial remesher on an open two-loop patch, 40 steps.** Alternating strong coarsening and
  refinement.
  - Invariants held at every step:
    - `MFTLCheckTwinStatesLocal` + `MFTLCheckMeshIntegrity`;
    - the collective validation;
    - exactly 2 loops, tags 0/1, no mixed tags.
  - Boundary edges were split in 14 steps. **No boundary edge was ever collapsed**, see S-A4.
- **An open two-loop patch crossing both the x and y periodic seams.** Default, 1-slab and
  2-slab layouts, np 1/2/4/8, four jitter + parallel remesh + redistribution steps each.
  - Every check passed:
    - the collective validation with G1/G2;
    - exactly-once ownership;
    - 2 loops, identical on all ranks, with total loop length equal to the unshifted case to
      1e-12;
    - distinct boundary edge ids.
  - Result: boundary loops distributed over ranks and across seams are correct.
- **Stale-code commits.**
  - ASSEMBLE with a staged twin gid that exists nowhere → collective error (np 1/2/4/8).
  - CUT with the same corruption on a would-be cut edge → **silently BOUNDARY** (np 1/4/8; S-A1).

#### 18.1.3 Regression runs (reviewer builds)
- MFTL2 `ctest`: 62/62 pass in each of four builds:
  - Release, LAPACK off;
  - Release, LAPACK on;
  - Debug + ASan/UBSan/LSan, LAPACK off;
  - Debug + ASan/UBSan/LSan, LAPACK on.
- Sanitizers: 0 `runtime error` / ASan / LSan reports.
- `Stage16A` (Debug + sanitizers): np1 339, np2/np4/np8 370 checks, 0 failed.

#### 18.1.4 multiflow topology-change coverage
- **Root cause of the TOPO "stall".** It is not front work. `DetectBreakupFTBodies`
  (`breakup.c:112–127`) sends to **every** rank including itself with a blocking `MPI_Send`,
  and no receive is posted for the self-message. `DetectCoalescenceFTBodies`
  (`coalescence.c:127–140`) does the same.
  - Under MPICH 4.1.2 the process spins in `PMPI_Send` (gdb backtrace) at step 1, before marching
    cubes starts.
  - The code dates from 2024-11 (5d484ad9) and does not depend on MFTL.
  - The stall is identical for MFTL1 HEAD, Stage 15 and 16A. **Pre-existing multiflow bug; it
    does not block 16A.**
- **Second pre-existing crash.** After the reconstruction, `CleanupFluidCellsinOverlapMeshBoxes`
  dereferences never-allocated arrays of the marching-cubes overlap mesh (`SendFluidCells`,
  `nPDomainToFluidProcToProcTable`), called from `executemarchingcubes.c:371`. The result is
  SEGV, identical on Stage 15.
- **End-to-end A/B with both bugs guarded in scratch copies.** Patches: skip the self-send; guard
  the NULL cleanup. Stage-15 stack (MFTL2 HEAD + multiflow HEAD) against the 16A stack (working
  trees).
  - Case: the TOPO example shortened to 20 steps. AMR max level 2, `FTDomainMCToFTDomainResolution 1`.
  - Two drops coalesce through marching cubes at step 1. Path: distributed soup build →
    `DetectBodies` copies and merges (2 iterations at np4) → ASSEMBLE commit → redistribution
    with ghost completion → `MissNeighbours` assertion. Then 19 steps of remeshing, smoothing
    and redistribution on the reconstructed body.
  - np1 and np4: **bit-identical** (23 HDF5 files and every front VTU/PVTU file).
  - np4 with `MFTL_VALIDATE=1`: identical output and 0 twin-state diagnostics.
- **Remaining gap:** a **breakup** into several bodies (component copies into distinct bodies)
  has not been run end to end. The adapter tests (copy into two halves + merge) and the library
  tests (`copy_commit`, `soup_before_ownership`) cover these operations directly, and nothing
  indicates a regression. Close the gap when the multiflow bugs above are fixed (Stage 17 or a
  multiflow maintenance task). MOVEIBM2 was not re-investigated.

#### 18.1.5 Should-fix items (not blocking; fix in 16C or Stage 17)
- **S-A1.** `MFTLObjectCommitCopy(CUT)` turns *every* unresolved non-negative staged code into
  BOUNDARY. §2.2 allows this only for copies of an *existing* source twin. With a source that is
  not quiescent, a dead twin gid becomes a physical boundary without any message.
  - Fix: pass the source to the commit and require the twin to be live and authoritative in the
    source (global lookup), or document and assert that the source must be validated.
  - multiflow uses only ASSEMBLE.
- **S-A2.** The return codes of `MFTLRedistribute*` are ignored by these callers:
  - the serial bracket;
  - `MFAdapterMFTL2RedistributeObject`;
  - `MFAdapterBodyListUpdateGhostPositions`.

  In Release, a failed completion or local check is printed but has no effect. Fix: abort in
  the adapter, and propagate the code in the bracket.
- **S-A3.** `MFTLCompleteGhosts` stops after 64 rounds and returns success even if requests
  remain, and the Release check does not test G2. Fix: return an error when the cap is reached.
- **S-A4.** Boundary edges are never collapsed: `mftlEdgeIsGeometricallySharp` treats a
  missing twin as a feature. The `is_boundary_edge` / `common == 1` branch is therefore dead,
  and under coarsening loops only grow. Topologically safe, but the boundary-edge collapse is
  still to come in 16C (augmented link condition, §4.4.1).
- **S-A5.** An interior collapse with one boundary endpoint moves the boundary survivor to the
  midpoint, contrary to §4.4.1 ("keeps the boundary vertex at its position"). This is geometry
  only and belongs to 16C.

### 18.2 §3.6 revision 3.1 — verdict: NOT SOUND (not frozen; 16B blocked)

#### 18.2.1 Method
A prototype applies the text **literally** to triangulated members in a triply periodic unit
cell with k = z:
- exact per-triangle integrals of `z n_z` and `z² n_z`;
- vertical ray casting with lateral periodic images;
- the golden-ratio candidate sequence and the η_b / η_n / η_z genericity tests;
- W₀ and the layer, local-window and nesting placement as written.

Every DEFINED result is compared with the value for the physically correct placement of the
same mesh (`tools/phase_set_prototype/`: `p36.py`, `cases.py`, output `cases.out`; run with
`python3 cases.py`, numpy only; modes `literal` = the text as written, `F12` = MF-36-2 + S-36-3
fixes, `fixed` = additionally MF-36-1 / S-36-4).

#### 18.2.2 Confirmed as sound
- The criterion (closed / wall-closed, Φ = 0, consistent orientation) and the H₂ argument.
- Degree ±1 of connected layers.
- A proof that an embedded layer's crossings on one line span less than L_k.
- `V_raw` gauge-free modulo `V_cell`, with a unique representative in (0, V_cell).
- All-crossings alternation. It detects:
  - (−,−,+,+) layer stacks;
  - nested drops with the same σ;
  - a wrongly signed drop in a slab.
- Verified cases (the three results after S-36-3 use its corrected window start, the others the
  literal text):

  | Case | Result |
  |---|---|
  | Slab, and its complement, with a sheet stored in another image | exact V and centre |
  | Overhanging floor with three crossings | exact |
  | Six alternating sheets in mixed images | V exact |
  | Three sheets | UNDEFINED |
  | Single sheet | UNDEFINED |
  | Closed tube wrapping x (Φ = 0, laterally wrapping compact member), alone and inside a slab | exact |
  | Seam-crossing sphere | exact |
  | B15 wavy floor + bubble | exact |

#### 18.2.3 Must-fix
- **MF-36-1 — the orientation check is incomplete without layers; a wrong DEFINED value.**
  Alternation is only checked on rays. Two compact members whose rays share no crossing are
  never compared. The `(0, V_cell)` test catches only the case where every member is inverted.
  - Example: two laterally disjoint drops, σ = +1 (r = 0.2) and σ = −1 (r = 0.1).
    - The level function takes the values −1, 0 and +1 (inconsistent), so the correct status is
      UNDEFINED.
    - §3.6 (with MF-36-2 fixed) returns **DEFINED, V = V₁ − V₂** (prototype case 8c).
  - Fix: pin the exterior level globally.
    1. Compute the nesting depth of each compact component from its local ray.
    2. Require `sign(σ_m · own volume) = s₀ · (−1)^depth` with one global s₀.
    3. With layers, every ray crosses every layer, so the check is already complete.
- **MF-36-2 — local rays leave their member; UNSUPPORTED for the worked cases and a wrong
  centre.**
  - The text asks for "a local generic ray through its own reference point (the lowest-gid
    vertex; the same deterministic perturbation rule)".
  - The ray through a vertex is never generic. Every local ray is therefore shifted by
    (frac(iφ₁)·L_x, frac(iφ₂)·L_y), i ≥ 1, which is about (0.62, 0.41) of a period at i = 1.
  - Results of the literal text:
    - **UNSUPPORTED** for the plan's own DEFINED cases: seam-crossing sphere, the B15 bubble, two
      separate drops, the x-tube.
    - For a laterally extended member (a tube wrapping x, near a floor that is wavy along y),
      the member is placed with a window taken at another lateral position than its reference
      vertex. It lands one period off: **centre −1.095 instead of 0.548** (case 4b).
  - Fix:
    - The local ray goes through an interior point of one of the member's own triangles, with
      a deterministic perturbation inside that triangle, or the next triangle in gid order.
    - The member is placed by its own crossing on that ray, not by its reference vertex.
    - Point queries use a ray through the query point. A degenerate hit is resolved by symbolic
      tie-breaking, or by the nearest generic crossing above or below — never by a lateral jump.

#### 18.2.4 Should-fix (fix in the same revision)
- **S-36-3.** The window start "smallest-position entering crossing" can lie strictly inside the
  crossing span of an overhanging layer.
  - Example: a fold straddling the z seam, with lifted crossings 0.90 / 0.95 / 1.05 and a
    ceiling at 1.5. Then W₀ = 0.05, and no integer shift puts {0.90, 0.95, 1.05} into
    [0.05, 1.05), so the prescribed placement has no solution.
  - Fix: use the smallest entering crossing that is not strictly inside any layer's span,
    otherwise UNSUPPORTED. The prototype with this rule reproduces every layer case above.
- **S-36-4.** For a periodic k without layers, `V_raw < 0` is taken as "inverted orientation" →
  UNDEFINED.
  - A consistently oriented matrix, such as one bubble with σ = −1 in a triply periodic cell, is
    a valid finite region with V = V_cell + V_raw.
  - Fix: after MF-36-1, use `V_raw mod V_cell`. Allow the exterior as phase (s₀ = −1) only when
    it touches no undeclared wall.
- **S-36-5.** A closed film over a z-wall (B6) has no boundary loops, so "the region enclosed by
  the loops" is empty and Φ_z ≠ 0 → UNDEFINED, which contradicts B6.
  - Fix: choose the wall patch among the complementary candidates so that the net flux
    vanishes; for B6 that is the full wall.

§3.6 needs revision 3.2 addressing MF-36-1, MF-36-2 and S-36-3 to S-36-5, plus prototype-backed
test cases for B11/B15, including cases 4b and 8c above. That revision needs another independent
review before 16B starts.

> **Follow-up (2026-09-24): revision 3.2 of §3.6 has been written.**
> - Resolution mapping: §3.6.11. Validation cases: §3.6.10. Model run: §3.6.12 (47 passed,
>   0 failed).
> - 8c = C5, 4b = M2, and the S-36-3 straddle = O3.
> - The review verdict above applies to revision 3.1. **Revision 3.2 awaits a fresh
>   independent review.**

## 19. Independent review of §3.6 revision 3.2 (2026-09-24)

Reviewer: Claude Code, fresh review of the written §3.6 (revision 3.2) and of
`tools/phase_set_prototype/p36.py`. 16A (accepted, §18.1) was not reopened. Nothing committed.

### 19.1 Verdict: SOUND — revision 3.2 is FROZEN
No formulation step can return a wrong classification, volume, centre or inside/outside result
within the stated assumptions (embedded, orientable, connected members). Errata E1–E6 (§19.4)
correct justification text, fix the status precedence and record scope and implementation
notes; no rule changes. 16B may start.

### 19.2 What was checked
**Plan ↔ prototype.** Every step of §3.6.2–§3.6.8 was traced to `p36.py` (R1–R10): canonical and
member lines (candidate sequences, genericity η_b, η_n, η_z), encloser parity, depth, parent,
surrounding level, the exterior level, the volume formulas, z₀ and `a_Y(z₀)`, the compact
placement rules, the centre statuses and the point query. One mismatch: the model glued every
loop of a wall to one patch piece, while §3.6.2 step 7 glues members to the *connected pieces*
of {μ_W = 1}. It only affected `N_ph` (the centre status along a periodic lateral axis); the
model now builds the pieces from the nesting tree of the wall loops (E6), cases W16–W18.

**A. Whole-set orientation and nesting.** The regions of D∖Ŝ are the slab regions (or E) and the
compact insides (F1–F4). F4 holds in general: the outer collar of a connected two-sided closed
surface is connected, so removing I(c) never disconnects its surroundings (also for compacts
that wrap laterally, e.g. two sheets joined by a tunnel, and in walled domains). Enclosers of a
point form a chain, so the parent is unique. The rule `e_c = +1 ⇔ λ_out = 0` plus the common
exterior level is equivalent to a two-valued level function. e_c comes from each component's own
volume; no step adds volumes of different components before the orientation check, so no
signed-volume cancellation can validate an inconsistent set. Cases: one drop (C1), one bubble with
the matrix as phase (C2), several drops (C4, W16), nested cavities and inclusions (C6, C7, C9,
W17, W18), one wrongly oriented disjoint component (C5, M4, W19), nested equal σ (C8).

**B. Layer crossings and overhangs.** (C1) needs only that the line crosses every layer lift;
both slab regions adjacent to each crossing appear on the line (E1: the adjacency sentence of
§3.6.5 is false for interlocking layers, the conclusion is not). The placement uses the parity of
one lift's crossings below z₀ and the total order of the lifts; neither needs the crossings of
different lifts to be separated. **New interlocking-overhang cases:** normal offsets of a folded
curve (min. curvature radius 0.048; offsets 0.012–0.04), so the upper layer lies inside the lower
layer's pocket; the canonical line the model actually uses was steered through the fold and shows
the interleaved order Y1 Y2 Y2 Y1 Y1 Y2 (I1) and Y1 Y2 Y3 Y3 Y2 Y1 Y1 Y2 Y3 Y4 (I3):
- I1 thin folded slab (ceiling one period up), I2 its complement, I3/I6 interlocking 4-layer
  stacks in mixed images (both phase choices), I4 three layers (Φ ≠ 0), I5 (−,−,+,+), I7 bubble in
  the overhang pocket (stored +1), I8 fold straddling the z seam, I1q point queries between every
  pair of consecutive crossings of the fold line;
- S1 wavy floor crossing the z seam (ceiling stored +5 periods), S2 4-layer wavy stack with a
  layer across the seam, S3 overhanging floor in a 4-layer stack.

All DEFINED values agree with the explicit physical placement to ≤ 2.4e-15. The 1-D crossing
representation classifies interlocking configurations correctly; no UNSUPPORTED fallback needed.

**C. Compacts inside layered regions.** Member lines always pass through the member; the
parent/adjacent-layer placement is an exact adjacency relation, so a re-gauge by m periods
changes t_P and q by m and leaves the placed position unchanged (checked: 8 random re-gauge seeds,
every member shifted by up to ±3 periods in every periodic direction). Cases: bubble in the dip of
a wavy layer (M1), near a seam and crossing it (S4, S5: stored three periods away), in an overhang
pocket (I7), first candidate line through layer vertices (M7), wrapping tube near a wavy layer
(M2 = 4b).

**D. Volume modulo V_cell.** A member re-gauged by m L_k changes `∮Y_k n_k` by `m L_k Φ_k`, i.e. by
a multiple of V_cell for a layer and by 0 for a compact; a reference-point change multiplies
Φ(Ŝ) = 0. Hence `V_raw mod V_cell` is gauge-free, `V_raw + λ_E V_cell` is exact without layers,
and for every consistent set the representative equals the true volume, so the degeneracy test
before any line is exact. Full-domain phase with a bubble: C2 gives V_cell − V_b, centre UNDEFINED
(wraps). Degeneracy: L9, D1–D4.

**E. Axis independence.** The text is written for a generic k (lateral pair (i, j) with
(e_i, e_j, e_k) right-handed). The model now selects the axis by the §3.6.3 rule (`p36axis.py`)
and evaluates it through the orientation-preserving cyclic permutation; all 72 cases rotated onto
x and onto y reproduce statuses, V, C_k and every point query (compact-only sets select x, the
smallest admissible axis — 13 cases in the y run).

**F. Wrap classification input.** §3.6 needs, per member, COMPACT_k (no cycle with W_k ≠ 0), which
is exactly the §3.3 flag: s_k is a coboundary ⇔ the member's k-coordinate is single-valued in its
own gauge. `winding.py` derives the flags from the mesh (wrapped storage, edge lattice shifts,
deterministic BFS potentials, cycle residuals); they equal the declared flags for every member of
every case (lattices e.g. (1,0,0) tube, (1,1,0) diagonal tube, (−1,0,1)/(0,1,0) tilted plane), and
all 72 cases give the same results with the derived flags. A diagonal tube (lattice (1,1,0)) is
classified along z (DEFINED); two opposite tilted planes derive (1,1,1) → UNSUPPORTED.

**G. Wall closures.** Signs: with ε_min = +1, ε_max = −1 and wind measured in (e_i, e_j) the patch
has the domain-outward normal and closes the members consistently (W5: cap ∪ disk with the disk
normal −z; W8: wall minus disk). Flux `s_W(ε_W A(Γ_W) + n_W A_w)`, volume and centre terms
`s_W (c_w − ref_k)^p ∫μ` (p = 1, 2) were re-derived. Full versus partial wall: n_W ∈ {0, 1} with
μ ∈ {0, 1} pointwise, selected by the flux (W1, W4, W6, W8, W15). Several loops on one wall are
supported through the nesting-tree pieces (W16 two drops, W17 annulus, W18 with a cavity, W19
inconsistent → UNDEFINED). Scope: loops that wind along the wall, loops touching two walls and
closures on two axes → UNSUPPORTED (W10). A member whose loops lie on both walls (a liquid bridge)
is admissible in the text but has only vertical member triangles → UNSUPPORTED "no generic line"
(E4).

**H. Degenerate lines.** Lines with no crossing only occur for query lines without layers
(λ_E) or on a walled k (level 0). A generic line has all hits at barycentric > η_b and
|n_k|/|n| > η_n, so no hit can be missed or duplicated by rounding (a hit within η_b of an edge
flags the line non-generic); vertical triangles containing the line and loops within η_b of the
line are non-generic too. Crossings closer than η_z L_k (η_z L_k A_k ≤ 0.1 τ_V) are rejected, so any
region with V > τ_V is resolved. All candidate sequences are deterministic (golden-ratio lattice
from the reference point, gid-ordered member triangles, clearance-ball query points); after
N_try candidates the status is UNSUPPORTED. Vertex/edge lines are exercised by M6, M7, L2, I-cases
(the reference vertex itself is always candidate 0 and always rejected).

### 19.3 Prototype additions (`tools/phase_set_prototype/`)
| File | Content |
|---|---|
| `p36.py` | per-triangle minimum-image line tests and point distances (any member gauge); wall loops in minimum image; **connected wall pieces** from the loop nesting tree (§3.6.2 step 7) |
| `interlock.py`, `cases_review.py` | 25 new cases: I1–I8, I1q, S1–S5, W16–W19, D1–D4 (`cases_review.out`: 25 passed) |
| `winding.py`, `check_winding.py`, `run_derived.py` | §3.3 winding from the mesh; flags derived = declared for every member; 72/72 cases with derived flags |
| `p36axis.py`, `run_axes.py` | §3.6.3 axis selection; 72/72 cases rotated onto x and onto y |
| `run_regauge.py` | every member re-gauged by random lattice vectors; 72/72 for 8 seeds |

`cases.py` (47 cases) is unchanged and still passes: 47/47.

### 19.4 Errata (no rule changes)
- **E1** §3.6.5 (C1) and §3.6.7: the sentence "on every line the last crossing of a layer lift
  and the first crossing of the next lift above it are adjacent" is false for interlocking layers
  (I1). The conclusions hold by the arguments of §19.2 B. The F2 remark on the span of one lift's
  crossings is unproven and unused.
- **E2** Status precedence (made explicit): closure errors → Φ(Ŝ) ≠ 0 (UNDEFINED) → admissible axis
  (UNSUPPORTED) → classes and degenerate compacts (UNSUPPORTED) → degenerate total volume
  (UNDEFINED) → line-based steps. A single drop below τ_V is therefore UNSUPPORTED (D1).
- **E3** Walled k with V_raw ≤ 0 is reported as a degenerate/empty volume; the status (UNDEFINED)
  is right, the message should say "inconsistent orientation or phase at an undeclared wall" (W12).
- **E4** Scope: a compact component whose member triangles are all within η_n of parallel to e_k
  (e.g. a liquid bridge between two closure walls) has no member line → UNSUPPORTED. A later
  extension may take candidates on the component's patch pieces.
- **E5** Implementation: the placed coordinate `Y_k + π` is formed in the integer part of the image
  arithmetic, `((img − img_ref)_k + π/L_k)·L_k + (pos − pos_ref)_k` (§3.5). Floating shifts of
  the input lose ≈ 1e-10 relative in the centre of the 1e-6 slab L8 (seen only in the re-gauge
  harness, which shifts float coordinates).
- **E6** Prototype: the per-wall single-piece gluing is replaced by the connected pieces of
  {μ_W = 1} (the rule of §3.6.2 step 7).

Found during the 16B implementation (2026-09-24), after the review verdict above; no rule changes:
- **E7** Tolerance values: the model (and the implementation) use τ_V = 1e-10·V_cell and
  τ_Φ = 1e-9·A_k; §3.6.6 states 1e-12·V_cell. Only the degenerate thresholds move (L9, D1–D4).
- **E8** Case L10 of §3.6.10 was built from two bodies of two *disconnected* sheets each, and the
  model kept each body as one member (|Φ_k| = 2A → UNSUPPORTED). §3.6.1 says every connected
  component is a member, so the case is four layers (−,−,+,+) → UNDEFINED, exactly as L6. The
  implementation follows the text; the model now splits members into components
  (`split_components` in `p36.py`) and L10 expects UNDEFINED. The `|Φ_k| ∉ {0, A_k}` branch is
  unreachable for connected embedded members (degree ±1, §18.2) and stays as a safety net. This
  item was missed by the review (§19.2 "plan ↔ prototype") and caught by the implementation
  cross-check (§20).

## 20. 16B implementation status and test results (2026-09-24)

**Status: 16B IMPLEMENTED; ACCEPTED by the independent review of 2026-09-24 after the fixes of
§21 (which supersede items 3 and 4 of §20.5).** 16A stays accepted (§18.1). §3.6 revision 3.2 is frozen (§19) with errata E1–E8.

### 20.1 Representation and lifecycle (§3.1–§3.2, §3.5)
| Item | Implementation |
|---|---|
| Counters | `MFTL_Vertex.img[3]`; `PackedVertex.img[3]` and `GhostUpdatePayload.img[3]` with their MPI datatypes; `img` travels with `pos` at every pack, unpack, redistribution, ghost append, ghost-position update and triangle copy. |
| Wrap / displacement API (`mftl_geometry.h`) | `MFTLWrapPosImg` (wrap counted into img; position arithmetic identical to `MFTLApplyPeriodicity`), `MFTLImgFollow` (a position replaced by an independently wrapped short move), `MFTLWrapContinuous` (continuous input → (pos, img)), `MFTLVertexRelative` (Y from the exact integer image difference), `MFTLEdgeShift` (s(u,v) = img_v − img_u + j). |
| Every position writer | split (anchor a), collapse (anchor b), tangential smoothing (two drivers), Kuprat / volume recovery, TSUR (pointer writes), paraboloid/barycentric fits (including the temporary collapse-fit position, whose counters are restored), generators, sphere, cube tessellation (seam-safe midpoints on periodic domains), GTS and soup input (lifting), translation + `MFTLWrapObjectIntoDomain` (multi-period wrap counted), scaling and rotation of continuous coordinates on periodic domains. Non-periodic arithmetic is unchanged. |
| Integrals / queries | `MFTLObjectVolume`, `MFTLSurfaceCenter`, `MFTLMomentOfInertia` and `MFTLPointInsideObject` use continuous coordinates from the counters relative to the agreed reference vertex; the point query tests every lattice image of the probe inside the body's continuous bounding box. **The object-extent restriction is removed** (`MFTLCheckMinimumImageConstraint` keeps the edge check). Wrapping bodies: the unchecked integrals abort with a message pointing to the phase-set API. The overlap module's temporary re-imaging holds the counters at 0 (its periodic proximity rewrite is 16C). |
| Class guard | `MFTLObjectHasVolume` (closed and compact): the remesh drivers take a volume target and run volume recovery only for such bodies. |
| Native I/O v2 | `MFTLWriteNativeMeshV2` (class, period box, exact `%.17g` positions + counters, boundary tags, explicit vertex triples); `MFTLReadNativeMesh` reads v1 and v2 on any rank count through `MFTLBuildFromSoupImg` (explicit counters, explicit boundary declaration from the tags) and canonicalises. |

### 20.2 Winding and canonicalisation (§3.3–§3.4; `src/general/images.c`)
- `MFTLCanonicaliseImages`: every edge once (the canonical half-edge of an authoritative
  triangle) as (u, v, s) and every authoritative vertex as (gid, img) gathered to rank 0;
  duplicates must agree (collective error otherwise); deterministic BFS spanning trees rooted at
  the smallest gid (neighbours in gid order); `img' = img − φ`; the residuals of the non-tree
  edges give the winding lattice (stored as sign-normalised generators) and the cached flags
  `obj->wrap[3]`; the new counters are broadcast and applied to every copy (owned and ghost).
  Idempotent. No-op on non-periodic domains (no communication; bit-identical, no cost).
- `MFTLObjectRegaugeImages`: the same with every component's root counter brought to 0.
- Called after every assembly event: soup build (hence every read), `MFTLObjectCommitCopy`
  (copies, merges, breakup/coalescence through the adapter), generators, sphere, GTS.
- `MFTLCheckWrapFlags` (no cut edge along a compact axis) runs in the full quiescent validation
  (Debug / `MFTL_VALIDATE=1`) after every redistribution; `MFTLCountCutEdges` for tests.
- Boundary loops carry their winding vector (Σ s over the loop, §2.6).

### 20.3 Phase-boundary sets (§3.6; `include/mftl_phase_set.h`, `src/general/phase_set.c`)
API: `MFTLPhaseSetCreate / AddMember(body, σ) / AddWallClosure(axis, side) / Classify(&info) /
Volume / Centre(C[3], status[3]) / Inside(Q, &where) / Destroy`, statuses
`MFTL_DEFINED / MFTL_UNDEFINED / MFTL_UNSUPPORTED`. Every call is collective: the set's
authoritative triangles are gathered to every rank in member-gauge continuous coordinates (from
the counters, relative to the set reference point, each body shifted by its reference image
offset, E5) and the evaluation runs identically everywhere. Steps R1–R10 of the model, axis-
generic: members = connected components; loops continuous along the loop with winding; wall
patches with the nesting-tree pieces, admissible n_W and the flux selection; components of Ŝ;
admissible axis (closures → w; else a non-periodic axis; else a periodic axis along which no
member has a cut edge — exactly COMPACT_k after canonicalisation); classes; degenerate checks;
canonical, member and query lines with the model's genericity rules and deterministic candidate
sequences; alternation; nesting tree, exterior level and orientation rule; volume; z₀ / `a_Y`
placement and compact placement; centre along k and, for the other axes, the §3.6.7 rules
(UNDEFINED when wrapped or λ_E = 1, UNSUPPORTED when N_ph ≠ 1, otherwise the same construction
with that axis as line axis); point query in the clearance ball.
Deviation (performance only, recorded): the triangles are gathered to every rank instead of
gathering per-line crossing records (§3.6.4); results are identical, cost O(N) per query
(Stage-18 item).

### 20.4 Tests and results
`tests/test_stage16b.c` (ctest `Stage16B-np1/2/4/8`, `Stage16B-slab1-np4`, `Stage16B-slab2-np4`,
`Stage16B-restart-write-np4` → `Stage16B-restart-read-np1/2/8`):
- **seam sphere** (B1/B9): a sphere at mid-domain and translated across the x, xy and xyz seams
  (dyadic): V and A identical to 1e-14/1e-13, centre (mod period) identical to 1e-13, no cut
  edge, flags 0, phase-set V and centre DEFINED and equal to the object integrals;
  **B2** large compact sphere (extent 0.8 > half period): V exact to 1e-13.
- **large counters** (B10): every counter shifted by (10⁶, −10⁶, 10⁶): V, A equal to 1e-13
  relative, surface and phase-set centres equal modulo the period to 1e-12;
  `MFTLObjectRegaugeImages` brings every counter back to |img| ≤ 1 with unchanged V, A.
- **lifecycle** (B8): 100 translations across seams with redistributions: no cut edge, the
  reference vertex's continuous coordinate advanced by exactly 100 d, V and A unchanged; zeroed
  counters (wrapped input) show artificial cut edges, canonicalisation removes them and
  reproduces the counters up to one lattice vector (identical on all ranks), idempotent.
- **merge** (B12): drop A + drop B across a seam given as wrapped input without counters
  (artificial cuts detected by `MFTLCheckWrapFlags`); copies + ASSEMBLE commit (canonicalises):
  no cut edge, flags 0, V = V_A + V_B to 1e-14.
- **ribbons** (B3/B13): open ribbons winding x, y, z and diagonal (1,1,0): flags exactly the
  winding directions, two boundary loops with winding ±dir, area exact, phase set UNDEFINED
  (open, not closed by a declared wall).
- **film** (B4): xy film in an xyz-periodic cell: WRAP_x ∧ WRAP_y, `HasVolume` 0, Φ_z = ±1,
  phase set UNDEFINED.
- **phase-set cases** (B5–B7, B11, B14–B16): all 69 model cases (47 of §3.6.10 + 22 of the
  review §19.3: interlocking overhangs, wavy seam-crossing stacks, compacts at seams and in
  overhang pockets, wall pieces, degeneracy), each built as distributed MFTL2 bodies (soup with
  random placement and duplicates, then redistribution) in its own gauges and **rotated so that
  its evaluation axis is z, x and y**: status, V (≤ 1e-10), centre status and value along the
  model's axis (≤ 1e-9 mod period), N_ph and every point query equal the model's answers.
- **restart** (A5, §3.2): written on 4 ranks (native v2), read on 1, 2 and 8 ranks: counters and
  positions identical (exact), surface scalar identical, declared class and tags (loops 0, 1 with
  winding) restored, full twin-state validation passes.

| Gate item | Result |
|---|---|
| MFTL2 Release `ctest` (LAPACK on) | 72/72 (incl. Stage15, Stage16A, Stage16B np1/2/4/8, 1- and 2-slab, restart write np4 → read np1/2/8, TwinGidReaderAudit) |
| MFTL2 Release `ctest`, LAPACK off | 72/72 |
| MFTL2 Debug + ASan/UBSan/LSan `ctest` (full collective validation + wrap-flag check at every redistribution) | 72/72; no UBSan report, no ASan error, LSan clean with the project's MPI suppressions (checked explicitly on test_stage16b np1/4 and test_stage16a np1/8) |
| `test_stage16b` Release, np 1/2/4/8 | 781 checks each (np1 655 + 126 representation), 0 failed; 69 cases × 3 axes identical to the model |
| Prototype (model with component splitting, E8) | cases.py 47/47, cases_review.py 25/25; 72/72 with winding-derived flags, rotated onto x and y, and for 8 random re-gauge seeds |
| multiflow adapter tests np 1–4 (also with `MFTL_VALIDATE=1`) | 78 / 168 / 261 / 360 checks, 0 failed, no diagnostics |
| multiflow Release A/B vs the Stage-15 binaries: FT, FT_Sharp_CSF, FT_Sharp_Integral (np1/np4), IBM-MFTL A (np1/np4), B (np4) | **bit-identical** (every HDF5 dataset, worst difference 0) |
| Existing periodic tests | PeriodicEndToEnd and Overlap pass; the old "over-large sphere is refused" check now asserts B2 (accepted, V equal to the non-periodic generation to 1e-12) |

### 20.5 Deviations and open points (for the independent 16B review)
1. **Evaluation by replication** (§20.3): every phase-set query gathers the set's triangles to
   every rank (O(N) communication and memory per query) instead of per-line crossing records.
   Correct and deterministic; a Stage-18 performance item.
2. **Canonicalisation gathers to rank 0** (O(E) at assembly events), as §3.3 allows; distributed
   BFS is a Stage-18 option. Skipped entirely on non-periodic domains.
3. **Transforms on periodic domains** operate on each vertex's own continuous coordinate
   pos + img·L (rank-local, no collective reference); rotation/scaling about a centre are
   therefore defined in the continuous frame of the stored counters. Non-periodic transforms are
   unchanged.
4. **Unchecked functions on wrapping bodies** (`MFTLObjectVolume`, `MFTLSurfaceCenter`,
   `MFTLMomentOfInertia`, `MFTLPointInsideObject`) abort with a message (plan §3.6.11); bodies that
   were never canonicalised are treated as compact (legacy construction paths).
5. **Overlap / proximity** keep the Stage-13 single-anchor re-imaging (counters held at 0 during
   it); the periodic proximity rewrite is 16C.
6. **multiflow adapter**: front periodicity stays off (16D); the adapter's own triangle packing
   carries positions only, which is exact while every counter is 0. No adapter change was needed
   for 16B; the rebuild picks up the new struct layouts.
7. **Tolerances** follow the model (E7).
8. The phase-set member line takes candidates from all members of a component (member order,
   then gid); the model used the component's first member only. Identical on every exported case
   (components with several members only arise with wall pieces).
9. **Performance**: the stress test (50 bodies, 8 ranks) is unchanged by 16B on non-periodic
   domains (canonicalisation is a no-op there); periodic cost was not measured beyond the tests.

**Stage 16 remains INCOMPLETE**; its Acceptance is not passed (status: §21, §22).

## 21. Independent review of 16B (2026-09-24)

**Verdict: 16B ACCEPTED after the fixes below** (all inside the 16B implementation; no frozen
Stage-16 rule and no part of §3.6 revision 3.2 was changed). 16A stays accepted; §3.6 stays frozen.

### 21.1 Findings and fixes
| # | Finding (evidence) | Fix |
|---|---|---|
| R16B-1 | **Unknown winding treated as compact.** `MFTLObjectIsCompact` returned 1 for a body that was never canonicalised (`wrap_valid = 0`), and several builders/mutators left stale flags. A seam-crossing sphere built through the serial builder reported V = 0.183 instead of 0.033; an xy film built the same way reported `HasVolume = 1`. Contradicts §3.3 ("no silent unknown → non-wrapping"). | `IsCompact` is 0 whenever a periodic body's winding is unknown. New collective `MFTLObjectEnsureWinding` (no-op and no communication on non-periodic domains; else canonicalises if any rank holds an invalid state). `wrap_valid` is invalidated by `MFTLBuildTopologyFromTriples`, `MFTLObjectClear` and triangle copies; every consumer (unchecked integrals, point query, phase-set gather, remesh drivers, volume recovery, overlap) ensures the winding first. A canonicalisation that changes a body's winding lattice sets `class_transition` and prints a rank-0 diagnostic; `MFTLObjectTakeClassTransition` reports it (§4.4.4). |
| R16B-2 | **Periodic rotation/scaling were gauge-dependent** (§20.5 item 3): transforming each vertex's own `pos + img·L` about a centre gave different geometry for the same body in gauges differing by a lattice vector (rotation: relative volume/position differences 0.58 and 0.36 in the review probe). | Transforms act on continuous coordinates relative to the collectively agreed reference point (`MFTLObjectReference`), with the pivot taken at the minimum image of the centre relative to it: `X = X_ref + P + M(Y − P)`, `img = img_ref + wrap`. A transform must map every winding generator onto a lattice vector (else `MFTL_ERROR`, body unchanged); a changed lattice invalidates the winding. `MFTLScaleObject` / `MFTLRotateObject` now return `int`. Non-periodic transforms unchanged (bit-identical). |
| R16B-3 | **Overlap on a wrapping body aborted** inside `MFTLSurfaceCenter`. | `MFTLEnsembleCalculateOverlap` ensures the winding and returns `MFTL_ERROR` with a message for non-compact bodies. |
| R16B-4 | **Generated test data broke the build**: the phase-set model cases were generated in an `ALL` target requiring NumPy, and were not regenerated when the model changed. | Option `MFTL_PHASE_SET_MODEL_TESTS` (AUTO/ON/OFF) with a configure-time NumPy probe; generation is a ctest fixture (`Stage16B-export-model-cases`) required by the model-case tests, so CI regenerates every run; without NumPy (AUTO) the library and all other tests build and the model cases are skipped with a message; ON without NumPy is a configure error. Exporter output is deterministic (identical on repeated runs). |

### 21.2 Review items
1. **img invariant** — counters travel with positions at all nine communication sites that
   carry vertex positions (§20.1 "Counters"); verified by code inspection and the B8/B10 tests.
   Sound.
2. **Canonicalisation / winding validity** — sound after R16B-1: no path reads flags of unknown
   validity; unknown is never compact.
3. **Winding** — homological (cycle sums of integer edge shifts, BFS residual lattice);
   independent of gauge and decomposition (tested np 1/2/4/8 and three gauges). Sound.
4. **Transforms** — gauge invariant after R16B-2: rotation, scaling and translation of the same
   body in gauges 0, (3,−2,1) and (10⁶,−10⁶,7) give identical positions modulo the period and
   identical V, A; lattice-incompatible transforms of a wrapping film are rejected (0.5 rad
   rotation, x-scaling) and compatible ones accepted (90° rotation, z-scaling).
5. **Phase sets** — the implementation follows §3.6.1–§3.6.10 as frozen; analytic
   cross-checks (slab minus bubble volume, slab centre, inside/outside, matrix phase,
   inconsistent orientation → UNDEFINED) agree with closed-form values; all model cases pass.
6. **Restart** — v1 and v2 read on any rank count; a v1 file of a cleared/rebuilt body restores
   and canonicalises.
7. **Build infrastructure** — fixed (R16B-4). The library never needs NumPy.
8. **Old per-body functions** — abort (unchecked integrals) or return an error (transforms,
   overlap) with a message on wrapping bodies, as §3.6.11 states.
9. **Regression** — closed non-periodic paths bit-identical (gate below).

New test: `tests/test_stage16b_gauge.c` (ctest `Stage16B-gauge-np1/2/4/8`, 109 checks each):
gauge invariance of transforms, winding validity for serial-builder bodies, film transform
compatibility, clear + v1 restart, analytic phase sets.

### 21.3 Pre-existing points noted (not 16B defects; not changed)
- The legacy parity flip dihedral gate (`remeshing_core.c`) compares normals with a sign
  convention that accepts some folding flips; kept unchanged on the closed default path for
  bit-identity with Stage 15 (the generalized path uses a corrected gate, §22). Stage-17 item.
- Absolute area threshold `1e-12` in `mftlTriQualityFromPos` (scale dependence). Stage-17 item.
- The parallel parity remesh path leaves a volume-recovery residual of 1e-10 to a few 1e-7
  relative (identical before and after 16B/16C).

## 22. 16C implementation status and test results (2026-09-24)

**Status: 16C IMPLEMENTED; ACCEPTED by the independent review of 2026-09-24 after one fix (§23).** 16A and 16B stay
accepted (§18.1, §21). No frozen rule was changed.

### 22.1 What was implemented (MFTL2)
| Item (§4) | Implementation |
|---|---|
| Kernels (§4.1) | `src/general/kernels.c`: `MFTLVertexDelta`, `MFTLEdgeVector`, `MFTLVertexContinuous`, `MFTLTriangleUnwrapped`, `MFTLBoundaryEdgeGeometry(h, &len, t, ν = t × n, mid_X)`, `MFTLVertexAngleSum` (returns the ring class), `MFTLVertexTurningAngle` (π − Σ angles), `MFTLBoundaryNeighbours`. |
| Periodic proximity (§4.1) | `MFTLFindProximityPairs`, `MFTLFindThinRegionPairs`: periodic cell hashing with minimum-image distances (rank-local candidates; callers gather where a global answer is needed). |
| Normals / curvature (§4.2) | Angle-weighted normals over open complete fans; curvature only on closed rings; boundary vertices flagged `MFTL_VFLAG_CURV_UNDEFINED` (set and cleared every evaluation); optional `MFTLComputeExtrapolatedCurvatures` (quadratic fit on the open stencil, ≥ 6 points) into `curv_extrap` with `MFTL_VFLAG_CURV_EXTRAPOLATED`. |
| Boundary curve (§4.3) | `MFTLObjectSetBoundaryCurve(obj, tag, mode, projector, user, freeze)`. REPRESENTED: uniform Catmull–Rom through the loop's vertices in the local frame, corners as breakpoints (linear end extrapolation); a curve point whose control stencil is not stored locally is not evaluated and the operation is skipped (never a chord). CONSTRAINED: the application projector is applied to every new or moved boundary vertex; `freeze` fixes the loop's vertices. |
| Corners (§4.3) | Turning angle > `boundary_corner_angle` (new remesh parameter, default 30°) or `MFTL_VFLAG_CORNER` (packed, kept through copies); never moved, merged or collapsed. |
| Link condition (§4.4.1) | `MFTLAugmentedLinkConditionOk`: Lk(a) ∩ Lk(b) = Lk(ab) with the virtual vertex ω (vertices and edges), serial and parallel. Collapse targets: boundary edge → curve point; interior edge with one boundary endpoint → the boundary vertex stays (**S-A5 fixed**); interior → midpoint / fit. Boundary edges now collapse (**S-A4 fixed**: sharpness is evaluated only for interior edges). |
| Other operators (§4.4.2) | Boundary split onto the curve; boundary edges never flipped; flips next to the boundary use the valence targets 6 / 4 / round(θ/60°)+1 (min 2); boundary smoothing by arc-length redistribution along the curve (moves only where neighbouring spacings differ by > 10 %, see 22.4) and never for corners or frozen loops; Kuprat/TSUR only on interior edges with both endpoints closed. |
| Orientation guards | Generalized mode: collapse, split fit, flip (dihedral gate on the consistently oriented new pair, position-based normals) and smoothing (candidate and again at application — Jacobi-safe) reject any move that inverts a triangle **or makes it degenerate** (2A / l_max² ≤ 1e-8; a sign test alone accepted an ear on a straight boundary pulled flat onto the line). |
| Parallel safety | Generalized smoothing (interior and boundary) moves a vertex only if every vertex of its fan is owned by the same rank (`MFTLVertexFanOwned`): concurrent moves on two ranks are seen only after the ghost update and folded a shared triangle (observed on 4 ranks, LAPACK off). Rim vertices are smoothed in later passes, after the shift scheme moves the rank boundaries. No effect on 1 rank. Collapse and split gates already require fully owned fans. |
| Valence | The generalized collapse is rejected if the survivor would have more than 16 neighbours (val(a) + val(b) − 2); repeated coarsening had produced valence > 30 and overflowed the smoothers' fixed one-ring buffers (assertion on 8 ranks). The generalized smoother skips fans of ≥ 32 triangles (refinement around a high-valence vertex) instead of overflowing its buffers. |
| Mode selection (§4.4.4) | `MFTLObjectSetRemeshMode(obj, MFTL_REMESH_MODE_DEFAULT / GENERALIZED)`, `MFTLObjectRemeshGeneralized`: open bodies always generalized, closed bodies parity by default, generalized by opt-in. The parity path is unchanged (bit-identical, periodic or not). |
| Class guards / recovery (§4.4.4) | `MFTLRemeshRecoveryTarget` / `MFTLRemeshRecover` in the serial and both parallel drivers: legacy volume recovery for `HasVolume` bodies, set-level recovery (`MFTLPhaseSetVolumeRecovery`, uniform normal shift ΔV/A, Newton ≤ 8 steps) for a DEFINED singleton set of a wrapping closed body, otherwise skipped and counted (`stat_recovery_skipped`). Class transitions are reported (§21, R16B-1). |

### 22.2 Tests (`tests/test_stage16c.c`, ctest `Stage16C-np1/2/4/8`, 243 checks each)
- **C1 planar** open patch: loop count, χ = 1, area/normals exact, interior curvature 0,
  boundary vertices CURV_UNDEFINED, corners detected.
- **C2** octahedral-cap hemisphere, levels 2–5: loop length → 2πR, area → 2πR² (rate ≥ 1.9),
  interior curvature → 2/R; discrete Gauss–Bonnet on sphere, hemisphere, open tube and film to
  1e-12 (+ scale term); extrapolated boundary curvature; boundary conormals.
- **C3 link**: one-triangle strip pinches rejected, ear legal, isolated triangle rejected, two
  holes one edge apart not merged, triangular hole not closed, corners never collapsed; S-A5.
- **C4 cycles**: 20 refine/coarsen cycles, serial and parallel, planar patch (boundary on the
  square to 1e-12, corners bit-fixed, area exact, boundary edges both coarsen and refine) and
  perturbed hemisphere (REPRESENTED ≤ C h³ with C = 1; CONSTRAINED circle projector ≤ 1e-12);
  full validation every cycle (no bow-tie, no pinch, no inverted triangle).
- **C5 seam**: open patch across the x and y seams vs translated by (½, ½) (dyadic): loops, tags,
  length, area, per-vertex normals, curvature flags/values and turning angles identical
  (≤ 3e-15); the remesh pass is identical on 1 rank (operation counts and triangles) and
  statistically equal on > 1 rank (12 parallel sweeps, triangle counts within 5 %; observed
  ≤ 3.5 %).
- **C6 scales**: C1, C2 (Gauss–Bonnet), C4 (planar) and B1 at scales 1e-4 and 1e4.
- **periodic**: wrapping ribbon (loop windings preserved through remeshing), closed film
  (parity and generalized; volume recovery skipped where undefined), closed seam-crossing
  sphere under parity remeshing vs its dyadic translation (exact on a 2⁻⁴⁰ grid without fits on
  1 rank; statistical otherwise), closed tube wrapping x (set-level recovery restores the
  phase-set volume), proximity and thin-region search across seams.

### 22.3 Results
| Gate item | Result |
|---|---|
| MFTL2 Release `ctest`, LAPACK on | 81/81 (incl. Stage15, Stage16A np1/2/4/8, Stage16B np1/2/4/8 + slabs + restart, Stage16B-gauge np1/2/4/8, Stage16C np1/2/4/8) |
| MFTL2 Release `ctest`, LAPACK off | 81/81 (after the fixes of 22.1 "Parallel safety"/"Valence" and the statistical bound of 22.4 item 4, which this build exposed) |
| MFTL2 Debug + ASan/UBSan/LSan `ctest` (LAPACK on) | 81/81, 0 sanitizer reports; the LAPACK-off Stage16C cycles also run clean under the sanitizers on 8 ranks |
| `test_stage16c`, np 1/2/4/8, all three builds | 243 checks each, 0 failed |
| `test_stage16b_gauge`, `test_stage16b`, `test_stage16a` | 109, 781 (+ slab/restart), 339/370 checks, 0 failed, np 1/2/4/8 |
| C5 seam patch, 1 rank | loops, length, area, per-vertex quantities and the remesh pass (operation counts, 2048 triangles) identical |
| Seam sphere (parity), dyadic, 1 rank | V, A, triangles and operation counts identical to its translation |
| Closed parity path | unchanged code; the parallel icosphere volume test bit-identical between the installed 16B library and 16C (np 2/4/8) |
| multiflow adapter tests np 1–4 (also `MFTL_VALIDATE=1`) | 78 / 168 / 261 / 360 checks, 0 failed, no diagnostics |
| multiflow Release A/B (16B install vs 16C): FT, FT_Sharp_CSF, FT_Sharp_Integral (np1/np4), IBM-MFTL A (np1/np4), B (np4) | **bit-identical** (every HDF5 dataset, worst difference 0; wall times equal within 1 %) |

### 22.4 Deviations and open points (for the independent 16C review)
1. **REPRESENTED curve drift under repeated cycles.** §4.3 says that repeated remeshing does not
   move the loop "beyond the interpolation error". Measured on the perturbed hemisphere
   (R = 0.3, h_max ≈ 0.093, 20 cycles; final code, LAPACK on / off): worst rim deviation
   3.6e-3 / 3.1e-3 (np1), 8.4e-3 / 7.5e-3 (np2), 6.9e-3 / 6.3e-3 (np4), 7.9e-3 / 7.5e-3 (np8),
   i.e. up to 2.8 % of R, against the test bound h_max³/R² = 9.0e-3 (C = 1). **The margin is
   thin (7 % at np2): a longer run would exceed this bound.**
   Before the skip of unevaluable curve points (22.1) and the smoothing dead band, the 4-rank deviation grew by ≈ 5e-4 per cycle (1.0e-2
   after 20 cycles). Each re-sampling of the
   reconstructed curve (boundary split, boundary collapse, smoothing) adds a small systematic
   interpolation error, and the parallel shift scheme performs more passes per cycle, so the
   deviation grows with the number of cycles in parallel rather than saturating. The smoothing
   dead band (move only when |L₂ − L₁| > 0.1 (L₁ + L₂)) reduces unnecessary re-sampling; keeping
   the survivor of a boundary collapse instead of the curve midpoint was tried and made it worse
   (1.5e-2), so it was not adopted. Long runs whose boundary must stay on a known curve should
   use CONSTRAINED (exact to 1e-16 in the same test). No rule changed; recorded for the review.
2. **Legacy parity flip gate** keeps its pre-existing sign issue on the closed default path
   (bit-identity, §21.3); the generalized path uses the corrected gate.
3. **Absolute thresholds** (`area < 1e-12` in the quality function) remain; the generalized
   path is protected by the application-time orientation re-check, and the scale tests pass at
   1e-4 and 1e4. Stage-17 item.
4. **Parallel statistics**: parallel remeshing depends on the decomposition, so C5 and the
   parity seam sphere compare statistically (triangle counts within 5 %, observed ≤ 3.5 %; volume
   within 1e-6 relative, reflecting the pre-existing parallel parity recovery residual).
5. **Overlap** (`MFTLEnsembleCalculateOverlap`) supports compact bodies only and returns an error
   for wrapping ones (R16B-3); the multiflow proximity use moves to the library kernels in 16D.
6. **Proximity search is rank-local**; a global answer needs the caller to gather (the test
   uses the serial bracket).
7. The generalized split places boundary vertices on the curve without a fit; interior splits
   and collapses fall back to the midpoint when the paraboloid fit would invert a triangle.
8. **Pre-existing in the parity path (not changed, bit-identity):** its parallel tangential
   smoothing has the same cross-rank race (owned vertices on both sides of a rank rim move in the
   same sweep, checked only against stale ghosts) and no valence cap beyond MFTL1's collapse
   rules. It has not been observed to fold a closed body in the existing suites; recorded for
   Stage 17/18.
9. The fold, degeneracy and valence defects above surfaced only in the LAPACK-off build on 4 and
   8 ranks (different fit positions change the operation sequence); they are genuine and fixed
   for every build.

**Stage 16 remains INCOMPLETE** (status: §23, §24); its Acceptance is not passed.

## 23. Independent review of 16C (2026-09-24)

**Verdict: 16C ACCEPTED after one fix (R16C-1).** No frozen Stage-16 rule was changed; 16A and 16B
stay accepted; §3.6 stays frozen. Reviewer: Claude Code (independent of the implementation
session). Nothing was committed.

### 23.1 Finding and fix
| # | Finding (evidence) | Fix |
|---|---|---|
| R16C-1 | **One-ring buffer overflow in the generalized remesher.** Refinement around a vertex left coarse by repeated coarsening adds one neighbour per split of a link edge (the opposite vertex of a split gains an edge), so valence grows without bound; 16C capped only the collapse survivor (16) and the smoother (fans ≥ 32 skipped). `FlipApproved` gathers the one-rings of all four vertices into a 32-entry stack buffer (`MFTLGetVertexOneRingVertices` only `assert`s its bound): reviewer probe R1 (seam-crossing open patch with two holes, 16 coarsen/refine cycles, 8 ranks, Debug) aborted with `Buffer overflow in MFTLGetVertexOneRingVertices` (backtrace `FlipApproved` ← `mftlParallelRemeshPass`); in Release the assertion is compiled out, i.e. a stack-buffer overflow. | (a) `mftlRingVerticesFull` counts the fan first (`MFTLVertexRing` without a buffer) and returns −1 instead of overflowing; every caller rejects the operation on −1. (b) `FlipApproved` uses a 128-entry buffer (identical for every valence < 32). (c) Generalized mode only: a split never raises an opposite vertex above valence 24 (`MFTLGeneralizedSplitValenceOk`, also re-checked in the parallel split apply loop, since several splits of one pass may touch the same opposite vertex), and a flip never raises c or d above 24; flips still reduce higher valences. Topological, translation- and decomposition-independent; the parity path is untouched. Regression test `valence` in `test_stage16c.c` (open wheel of 30 spokes whose rim splits each add a neighbour to the centre): the pre-fix code aborts on 4 ranks; with the fix the maximum valence stays at the initial 30 on 1/2/4/8 ranks. |

### 23.2 Review items
1. **Minimum-image / continuous coordinates.** Every 16C position writer (boundary split/collapse
   targets, `MFTLSetVertexPosFromFrame`, fits through `mftlSetPosFollow`, generalized smoothing)
   converts its result into a minimum-image displacement from the vertex's current position before
   `(pos, img)` is updated (`MFTLImgFollow` after a wrapped write; fit outputs are wrapped first).
   Reviewer probe R5: an open seam patch remeshed 3 passes with every image counter offset by
   (3, −2, 1) gives identical operation counts, bitwise identical positions and counters offset by
   exactly (3, −2, 1) for every surviving and new vertex.
2. **Augmented link condition** (`MFTLAugmentedLinkConditionOk`): vertex and edge parts with the
   virtual vertex ω traced for interior edges (c, d), boundary edges (c, ω), pinches (ω in both
   links, not in Lk(ab)), triangular holes (edge {x, ω} in both links), isolated triangles and
   ears. Correct. Interior edges joining two boundary vertices are rejected (pinch); an interior
   collapse whose `a` is on the boundary and `b` interior is rejected ("boundary never moves into
   the interior"); corners are never collapsed.
3. **S-A4** fixed: boundary edges coarsen (planar cycles: 32 boundary edges → [13, 75]).
   **S-A5** fixed: an interior collapse with a boundary survivor keeps it bitwise (serial test, and
   reviewer probe R3 in parallel: every rim vertex of a seam-crossing cap tagged `CORNER`, 12
   aggressive cycles on 1/2/4/8 ranks: every tagged vertex survives with bitwise identical position).
4. **Boundary topology / loops.** Reviewer probe R1: an open planar patch crossing the x and y seams
   with two square holes one cell apart and one hole one cell from the outer boundary (4 loops,
   χ = −2), 16 cycles alternating targets 0.2 / 0.02 with and without fits, serial kernel on 1
   rank and parallel on 2/4/8, shifted and unshifted: after every cycle full twin-state validation,
   4 loops with unchanged tags and zero windings, χ unchanged, no mixed tags, every boundary vertex
   on its original outline to 1e-12 (corners bitwise), area unchanged to 1e-12, no cut edge,
   `MFTLCheckWrapFlags` clean. No bow-tie, pinch or loop merge in any run.
5. **Wrapping surfaces.** Probe R2: ribbons wrapping x and crossing the y seam, 7 and 2 vertex rows
   (the latter one triangle wide), 12 cycles with coarsening to 0.3 (ribbon length 1): 2 loops with
   windings ±(1,0,0), χ = 0, wrap flags (1,0,0), no class transition, on 1/2/4/8 ranks.
6. **Closed-body parity path** unchanged by 16C (and by R16C-1 for valences < 32, where the old
   code was undefined behaviour anyway); closed seam-crossing sphere vs centred body: implementation
   test (parity, dyadic) and reviewer probe R4 (the same sphere opted into GENERALIZED): identical
   V, A, triangle and operation counts on 1 rank; statistical on > 1 rank.
7. **Class guards.** `MFTLVolumeRecovery` and the remesh drivers take a volume target only for
   `MFTLObjectHasVolume` bodies, set-level recovery for DEFINED singleton sets, otherwise skip and
   count (probe R6: open patch, serial and parallel drivers, `stat_recovery_skipped > 0`, no
   `MFTLObjectVolume` call). *Observation (16B, not changed):* the unchecked `MFTLObjectVolume` still
   returns a number for a declared-open body (it aborts only for wrapping bodies, §20.5 item 4);
   no library path reaches it, and the 16D adapter class-checks before calling it.
8. **Translation invariance.** Exact (bitwise) only while every coordinate stays representable: in
   probe R1 (dyadic, midpoints, no smoothing) the shifted and unshifted runs are bitwise identical
   for 9 full cycles; afterwards repeated bisection exceeds 53 bits and the two binades round
   differently (1 operation in 3 500 after 16 cycles). Statistical thereafter, as §10 prescribes.
   Converged parallel runs (with fits) agree within 1.5 % in triangle count across 1/4/8 ranks and
   both translations.
9. **§22.4 item 1 (REPRESENTED drift)** confirmed: a single uniform Catmull–Rom insertion on a circle
   is off by only ≈ 0.023 θ⁴ R (≈ 6e-5 here), so the 3.6e-3 – 8.4e-3 rim deviation is the
   accumulation of repeated re-sampling (inward on a convex loop), not interpolation error. The
   frozen REPRESENTED model (§4.3: the curve is defined by the current vertices only) cannot avoid it
   without a persistent reference curve, i.e. a design change; **erratum E9 (no rule change):** the
   sentence "repeated remeshing does not shrink the loop beyond the interpolation error" holds per
   operation, not cumulatively. Recorded as a limitation; CONSTRAINED (exact to 1e-16) is the mode
   for boundaries on a known curve (16E PINNED uses it).

### 23.3 Reviewer runs
| Item | Result |
|---|---|
| MFTL2 ctest after R16C-1: Release (LAPACK on), Release LAPACK off, Debug + ASan/UBSan/LSan | 81 / 81 in each, 0 sanitizer reports |
| `test_stage16c` (now 245 checks incl. `valence`), Release and Debug + sanitizers, np 1/2/4/8 | 0 failed; REPRESENTED rim deviation identical to §22.3 (3.6e-3 / 8.4e-3 / 6.9e-3 / 7.9e-3) |
| Reviewer probes R1–R6 (scratch, not in the repository), Release np 1/2/4/8 and Debug + ASan/UBSan np 1/2/4/8 | 0 invariant failures; the only failing checks were the reviewer's own unconverged cross-decomposition triangle-count comparisons (single parallel passes), replaced by converged comparisons |
| Stage16A / 16B / 16B-gauge / slab / restart ctest entries | included in the 81 above, pass |

## 24. 16D implementation status and test results (2026-09-24)

**Status: 16D IMPLEMENTED (2026-09-24), awaiting independent review.** 16A, 16B and 16C stay
accepted (§18.1, §21, §23); §3.6 stays frozen; no frozen rule was changed. 16E and 16F are not
started. Nothing was committed (multiflow `berend-mftl` at 468d2deb plus working-tree changes:
32 files, +1920 / −153, new `src/include/ftperiodic.h` and
`src/interfacial/fronttracking/generalizedsurfaces.c`).

### 24.1 What was implemented (multiflow)
| Item (§5) | Implementation |
|---|---|
| Options (§5.1) | `FRONTTRACKING PeriodicFronts <XYZ>` (validated against the fluid periodicity; rejected with `FixedPBox`); per body `OpenSurface <body> BoundaryFromMesh`, `BodyMesh <body> <file.gts>` (continuous coordinates; an open body requires it), `RemeshMode <body> GENERALIZED\|DEFAULT`. Per-body option arrays are broadcast (`broadcastoptions.c`; without it every np > 1 run crashed). |
| Capability matrix (§5.2) | `CheckFrontTrackingCapabilities` (`generalizedsurfaces.c`): each body is classified (closed P0 / closed periodic compact / closed periodic wrapping / open) and checked against the 13 enabled-module rows; a conflict stops the run naming module, body and reason. The phase set of the closed bodies (orientation +1, the multiflow convention) is classified on periodic runs; phase-dependent modules require DEFINED. Checked again after every remesh and marching-cubes step (`CheckFrontClassTransitions`; transitions are never accepted silently). ≥ 5 front-tracking boxes per periodic axis are required (a Peskin stencil must not meet a box and its own image). |
| Adapter (§5.3) | Class-checked `ObjectVolume` / `SurfaceCentre` / `MomentOfInertia` (closed-P0 arithmetic unchanged; otherwise the checked variants, abort unless DEFINED); checked variants return `MFADAPTER_DEFINED/UNDEFINED/UNSUPPORTED`; `MFAdapterSurfaceArea` for every class (minimum-image edge vectors for a wrapping body); `PointQueryCreate` requires `HasVolume`. Per-body remesh mode: closed → parity, open or `RemeshMode GENERALIZED` → generalized (independent of periodicity). The 1e-7 clamp only on non-periodic axes. Periodic AABB (continuous, full axis when wrapping), move (then `MFAdapterBodyWrapVertices`), rotate/scale about continuous centres, proximity and thin-region pairs via the library kernels, sphere generation in continuous coordinates, `EdgeLength`/`VertSetPos` minimum image. New accessors: `MFAdapterTriVertexPositions`, `VertContinuous`, `VertDelta`, `PointDelta`, `EdgeVector`, `WrapPoint`, `BodyWrapVertices`, `EdgeIsBoundary`/`BoundaryTag`, `BoundaryEdgeGeometry`, `BoundaryLoops`, `BodyClass`, `BodyTakeClassTransition`, `BodyDeclareOpen`, `BodySetGeneralizedRemesh`, `BuildFinalizeOpen`, `PhaseSet*`, `BodyListEnsureBackend`. Packed triangles carry the vertex image counters; open or periodic bodies are written in native v2. The single-pass redistribution with the G1 assertion (§2.5) was already in place since 16A. |
| Periodic infrastructure (§5.4) | Front-tracking and marching-cubes overlap meshes periodic along the declared axes and spanning the fluid period (no PBox adaptation there); periodic ghost boxes, including a rank's own images (`FillPeriodicSelfImages`: with one slab along an axis the DMDA ghost boxes are the rank's own boxes and were empty); `ftperiodic.h` (`FTStencilLimits`, `FTMinImage`, `FTWrapPoint`; each reduces to the Stage-15 expression on a non-periodic axis). Wrap after advection, volume recovery and marching-cubes recovery. VTU output: per-triangle continuous local frames, seam triangles duplicated per image (output only). Restart: native v2 (class, counters, tags). |
| Physics conversion (§5.6) | Peskin interpolation (6 kernels) and spreading over periodic box stencils with minimum-image kernel offsets; `EdgePlanar` and the USER edge centre through the adapter; Poisson indicator with a translation-covariant pinned point (the image antipodal to the first closed body, kept for the run; `FTIndicatorSeedPoint*`); marching cubes: ghost corner layers and global edge ids on wrapped indices, corner interpolation on wrapped points, soup assembly with minimum-image edges and canonicalisation; body detection through seam twins; coalescence distance by minimum image, candidates from the library's periodic proximity. |
| Passive front | `solvefronttracking.c`: with neither indicator nor surface tension the front is advected and remeshed only (previously a configuration error). This is the configuration an open surface can run in without the 16E models. |
| IBM (§5.5) | IBM bodies are never declared open and IBM decompositions stay non-periodic, so the MoveIBM / GTS-injection mass properties only ever see closed P0 bodies; an open GTS on an IBM body fails in `MFAdapterReadGTS` (message now says so). `IBMMassProperties` is therefore not needed and not implemented. |

### 24.2 MFTL2 changes made in 16D
- `MFTLReadGTS`: the signed-volume winding flip is applied only to closed bodies (an open surface
  keeps its file orientation); on a declared-open body an edge used by one triangle is a boundary
  edge (the pre-check rejected every open file, although the builder supports them); the
  orientation BFS skips boundary edges. New test `test_read_open_gts` in `test_io.c` (seam-crossing
  open patch, both file orientations: rejected when not declared, kept orientation, 1 loop of 32
  edges, np 1/2/4/8).
- (R16C-1, §23.1, belongs to the 16C review.)

### 24.3 16D defects found and fixed during validation
| # | Defect (evidence) | Fix |
|---|---|---|
| D-1 | **Periodic stencil on the wrong side of the seam.** Probes are wrapped to the canonical image; a rank whose ghost range lies across the seam (np 8 = 2×2×2 split: the x ∈ [1, 2] rank holds boxes [8, 22), a probe at x = 0.001 has box 0) got an empty or partial stencil. Seen as an open-patch vertex crossing the x seam that moved half a step too little (7.1e-4 off its straight boundary on np 8 only; np 1/2/4 never split x). | `FTStencilLimits` uses the image of the centre box nearest to the rank's ghost range (unchanged when it lies inside). Now every vertex of the translated patch moves identically to 2.9e-13 on np 1/2/4/8. |
| D-2 | **Generalized flips never happened in multiflow.** Without tangential relaxation the adapter passed `max_quality_threshold = 0`, and the generalized flip gate rejects `q > max_quality_threshold`. (The parity flip does not read the field.) | The adapter keeps the MFTL2 defaults unless multiflow sets the thresholds. Minimum angle of the remeshed open patch 3° → 14° (np 1). |
| D-3 | Checked surface centre aborted for a wrapping body (`MFTLSurfaceCenter` requires compactness) instead of returning UNDEFINED. | Compactness is checked first; the area comes from minimum-image edge vectors. |

### 24.4 Pre-existing multiflow bugs (§18.1.4), fixed separately
Both are pre-existing (identical on MFTL1 HEAD and Stage 15, not caused by Stage 15/16) and were
fixed minimally, apart from the 16D changes:
1. **Blocking self-`MPI_Send` in breakup/coalescence detection** (`breakup.c`, `coalescence.c`,
   since 5d484ad9): the sender skips itself (`if (t == Run->rank) continue;` — its own list is
   already complete). Confirmed again during 16D validation: the Stage-15 build still hangs at
   step 1 of the non-periodic marching-cubes case on one rank; the fixed build completes it.
2. **Marching-cubes overlap-mesh cleanup crash** (`cleanupfluidcellsinoverlapmesh.c`): the
   send-list cleanup is skipped when `SendFluidCells` was never allocated.
The rank-local candidate search of coalescence detection (pairs on different ranks are not found;
np 4/8 detect no coalescence in the two-drop cases, periodic or not) is a pre-existing limitation
and was not changed.

### 24.5 Deviations and limitations (explicitly rejected, never approximated)
- Open boundaries only via `BoundaryFromMesh`; explicit boundary-edge/loop lists, the
  `PhaseSet … WallClosure` option and wall closures are not implemented (they carry 16E physics).
- Modules not converted to periodic fronts are **rejected** on any periodic front ("UNSUPPORTED on
  periodic fronts by this implementation"): CSF / sharp CSF, IRL reconstruction, integral surface
  tension, surface scalar transport, surface viscosity, NOA advection, generic acceleration, force
  computation, divergence-free interpolation. Their raw-position sites are therefore not converted
  (the plan's "all 131 sites" is met for every module that can run on a periodic front).
- A closed body **wrapping** a periodic axis is rejected for every phase-dependent module even
  when its phase set is DEFINED (the Eulerian indicator and volume paths assume compact fronts);
  surface-only use stays allowed.
- Open bodies run only as passive fronts (every enabled physics row of §5.2 rejects them until 16E).
- Periodic marching-cubes resolution: MC cell = SmallestDX / `FTDomainMCToFTDomainResolution` over
  the whole period (the non-periodic sizing — fluid cell count across the bodies' union bounding
  box — is not translation invariant for seam-crossing bodies). With factor 1 the periodic MC is
  coarser than the non-periodic default (two-drop case: volume error before recovery −47.5 % vs
  −24 %); factor 3 gives −7.5 % (after recovery 0.07 % vs 0.34 %).
- Remeshing results depend on the decomposition: the serial kernel (np 1) and the parallel shift
  scheme (np > 1, one sweep per call in multiflow) differ, and a surface lying on a rank plane is
  remeshed much less on np > 1 (open patch at z = 1.0 on the z split: 742 / 624 / 596 / 601
  triangles on np 1/2/4/8; off the plane 745 / 710 / 701 / 712). Geometry (translation, boundary,
  area) is unaffected. Parity remeshing is round-off chaotic (observed in 16D: seam vs centred
  runs diverge whenever it runs, including the marching-cubes pre-refinement for factors > 1).

### 24.6 Test results
| Test | Result |
|---|---|
| multiflow Release A/B vs Stage-15 binaries (final working tree): FT, FT_Sharp_CSF, FT_Sharp_Integral np 1/4, IBM-MFTL A np 1/4, B np 4 | **bit-identical** (every HDF5 file, `h5diff` 0 differences); 0 build warnings |
| Adapter unit tests (`tools/mftladapter_tests`), np 1/2/3/4 | 131 / 273 / 418 / 569 checks, 0 failed. New 16D tests: seam-crossing sphere vs centred (volume, centre mod period, area, inertia, continuous AABB, local frames, move across three seams, point queries, pack/unpack with counters, native v2 restart), open seam patch (rejected undeclared, class, UNDEFINED volume/inertia, centre, loop, conormals, generalized remeshing keeps loop and area), wrapping tube (class, set-level volume, UNDEFINED centre, full-axis box), phase sets (inside across the seam, UNSUPPORTED centre), rank-local creation after `EnsureBackend`, **D7** LeVeque out and back, **D2** mixed open + closed list |
| D7: passive open patch across the x seam, LeVeque out and back (T = 1, 200 RK4 steps, z excursion 0.70) | without remeshing: back to 9.9e-11 (loop 4.8 + 7.5e-11); with generalized remeshing: off-plane 1.5e-2, loop +5.4e-3, area +2.6e-3 at L = 0.08, second-order in L (3.4e-2 / 1.5e-2 / 4.1e-3 for L = 0.16 / 0.08 / 0.04); np 1–4 consistent |
| D3/D4: closed droplet across the xyz corner vs centred (Laplace, 20 steps, Frenet + Poisson indicator) np 1/2/4/8 | velocity ≤ 1.1e-14, surface tension ≤ 1.4e-13, pressure ≤ 1.1e-12 (after removing the constant), indicator ≤ 2e-7 |
| Periodic centred vs non-periodic centred droplet; LAPP0 s16 vs s15 | identical logs |
| Open patch across the x/y seams (FT passive, uniform flow, generalized remeshing every 5 steps), np 1/2/4/8 | no remeshing: all 169 vertices translate identically (2.9e-13); with remeshing: boundary on the translated rectangle to 1.2e-13, loop exactly 4w, corners exact, areas of the four rank counts within 2e-4 of each other |
| D8: restart on a different rank count — periodic droplet (np 4 → 1/2/8), open patch (np 4 → 1/2/8) | droplet identical; open patch: class restored, translation, loop and rectangle exact |
| D6 (partial): two drops coalescing through marching cubes across the x seam vs centred, np 1 (and y seam np 1/2, earlier) | factor 1: same body count, volume errors equal to 3e-8; factor 3: same body count, volumes within 2.8e-3 (chaotic pre-refinement; a 1e-9 perturbation of the centred case alone changes them by 5e-4). A pinch-off with the neck on the seam was not run. |
| D1: rejections | open + Frenet surface tension, open + indicator, open + volume recovery, open without `BodyMesh`, wrapping closed tube + indicator, periodic front + CSF/IRL: each stops with its message; undeclared open GTS fails to read |
| D9: remesh-mode selection | adapter test (closed parity, open generalized) and the FT log line per body; non-periodic closed cases bit-identical (A/B) |
| MFTL2 ctest (Release LAPACK on, Release LAPACK off, Debug + ASan/UBSan/LSan) | 81 / 81 each (incl. Stage16A/B/C np 1/2/4/8, MeshIO np 1/2/4/8 with the new open-GTS test) |

### 24.7 Open for the 16D review
- D6 as specified (pinch-off with the neck on a seam) and D5 (≥ 200-step bubble advected through a
  seam vs a larger non-periodic box) were not run; D3's `sendnonlocal*` round trip is covered only
  indirectly (np 8 translation exactness).
- Multi-rank coalescence detection (pre-existing, §24.4).

## 25. Independent review of 16D (2026-09-25)

**Verdict: 16D ACCEPTED after fixes (R16D-1 to R16D-5).** No frozen Stage-16 rule was changed;
16A, 16B and 16C stay accepted (R16D-1 is a latent 16C initialisation defect, fixed without a
rule change); §3.6 stays frozen. 16E and Stage 17 are not started. Reviewer: Claude Code
(independent of the implementation session). Nothing was committed or pushed.

### 25.1 Findings and fixes
| # | Finding (evidence) | Fix |
|---|---|---|
| R16D-1 | **Uninitialised corner threshold of the generalized remesher** (MFTL2, 16C origin). `MFTLInitRemeshParams` never set `boundary_corner_angle` (nor the output `stat_recovery_skipped`); the multiflow adapter and most tests build `MFTL_RemeshParameters` on the stack, so `MFTLBoundaryVertexKind` compared the turning angle with stack garbage. Reviewer probe (open square patch, adapter `MFAdapterRemesh`, load-balanced shift scheme, 4 ranks): at the first coarsening cycle two of the four corners were collapsed (turning angle 1.571 rad classified *regular*), area 4 → 3.99710, loop 8 → 7.95728, a boundary vertex 4.9e-3 inside the outline; the serial kernel happened to read a harmless value. | Both fields are initialised (`boundary_corner_angle = 0` → default 30°, `stat_recovery_skipped = 0`). The parity path never reads them (Stage-15 bit-identity kept). Regression test `params` in `test_stage16c.c` (poisoned struct; fails with the old header: 3 of 4 checks). After the fix: area and loop exact, all corners kept, in every mode on 1/2/4/8 ranks. |
| R16D-2 | **Generalized remeshing under multiflow's default `Optimizer 0`** (adapter). `MFADAPTER_REMESH_PARALLEL` runs the MFTL1 "LoadBalance = NO" interior-only single pass on the native decomposition: a triangle touching a rank cut is never remeshed. On a static decomposition the cut-straddling edges stay frozen for good and repeated splits next to them flatten the neighbours: open patch across the x/y cuts, 4/8 ranks, refinement to L = 0.05 — max edge stays 0.177 (initial diagonal), min angle 1.33° → 0.00° (collinear triangles), 408/661 triangles < 20°. This is the cause of the reported "surface on a rank plane is remeshed much less" (§24.5). | A generalized body (open, or closed opted in) always gets the two-pass shift scheme (`p.load_balance = 1` in `MFAdapterRemesh`); closed parity bodies keep the Stage-15 path. Regression test `test16d_rank_cut_remesh` in `tools/mftladapter_tests` (fails without the fix on 4 ranks: min angle 0.02°, max edge 0.177; passes with it: ≥ 9.7°, every edge refined, outline and area exact). |
| R16D-3 | **Class re-check bypassed after marching cubes** (multiflow). Marching cubes replaces the changed bodies by *new* objects; their class is set by their first canonicalisation and never flagged by `MFAdapterBodyTakeClassTransition`, so `CheckFrontClassTransitions` never fired. Reviewer case D6W: a capsule along x whose two tips face each other 0.04 apart across the seam; breakup detection triggers marching cubes, which joins the tips into a tube **wrapping x** (wrap flags 100, 144 cut edges, centre UNDEFINED); the run continued with the Poisson indicator and Frenet surface tension on a closed wrapping body, which §5.2/§24.5 reject. | `CheckFrontCapabilitiesAfterTopologyChange` (generalizedsurfaces.c) re-checks every body against the enabled modules after each reconstruction on a periodic front (on a non-periodic front every reconstructed body is closed P0; only the transition check runs there, Stage-15 path unchanged). D6W now stops right after the reconstruction, before any module uses the body, naming module, body and class (1, 2 and 4 ranks). |
| R16D-4 | **Periodic marching cubes with `FTDomainMCToFTDomainResolution > 1` is not reliable.** On a periodic front 16D applies the factor to the marching-cubes grid over the whole period (the non-periodic relation-1 grid is never refined by it: 32 cells across the bodies' box whatever the factor). With factor 2/3 the soup is not closed on smooth, well-resolved bodies (icosphere dumbbell, 20 480 triangles, edges ≤ 0.03: 11 / 73 unmatched edges; the same numbers seam-crossing and centred, i.e. not a seam effect) and the run aborts in `Polygonise`. A reviewer experiment widening the marching-cubes cell band from 2 to 4 boxes removes the cracks, so the cause is the band of the Stage-15 cell list, not periodic indexing. The factor-3 remedy proposed in §24.5 is therefore not usable. | Rejected at initialisation: `PeriodicFronts` + topology change + factor > 1 is a configuration error (deterministic, before any step). Factor 1 is the supported periodic resolution (§25.5). Enabling finer periodic marching cubes needs a wider cell band and its own validation (16F/17). |
| R16D-5 | **Volume after a periodic marching-cubes reconstruction** (multiflow). At factor 1 the periodic grid is the front-tracking box, coarser than the non-periodic bounding-box grid; the single first-order MFTL1 recovery shift overshoots for under-resolved bodies: two drops of 4.8 boxes per diameter coalescing (MCPXc) −47.5 % → **+5.37 %** after recovery (non-periodic default: −24.3 % → +0.34 %). | On a periodic front `FTVolumeRecovery` repeats the shift until the target is met (≤ 1e-10, at most 20 shifts); a non-periodic front keeps the single MFTL1 shift (bit-identical). MCPXc and MCP0c: +2.6e-11 after recovery, seam and centred equal. |

Test hygiene (no behaviour change): `test_stage16a.c` computed its loop checksum in signed
`long` arithmetic; with R16D-1 the fuzz loops are longer and UBSan reported a signed overflow in
the test (library clean). The checksum is now unsigned.

### 25.2 D5 — bubble advected through the seams (≥ 200 steps, crossing and recrossing)
Case: fluid [0, 2)³, 32³, fully periodic, both phases ρ = 1 (μ 0.05 / 0.1), σ = 0.2, drop
D = 0.6 (1 280 triangles), Frenet surface tension + Poisson indicator, parity remeshing every 10
steps; uniform initial velocity (5, 3, 4) reversed by a uniform momentum source (−25, −15, −20):
the centroid moves 1.5625 → 2.0675 → 1.5625 in x (and across y = 2 and z = 2 at the turning
point), i.e. the drop crosses the x seam and the xyz corner and comes back; 200 steps,
dt = 2e-3. Reference: the identical run shifted by 12 cells (0.75, dyadic) on every axis, which
never touches a seam (same fluid discretisation — a larger non-periodic box would change the
physics by more than the 1e-6 criterion); also the same drop with a non-periodic front on the
periodic fluid (Stage-15 path). Diagnostics every step (reviewer `USERWriteDataLog`): checked
volume and centroid (mod period), area, owned triangle/vertex gids (exactly once), min/max angle,
edge statistics, `MFTLValidateTwinStatesEx(GHOSTS)`, `MFTLCheckWrapFlags`, cut edges, counter
range, mean and spurious velocity.

| Comparison | centroid /L | volume | area | triangles | spurious velocity | invariants |
|---|---|---|---|---|---|---|
| seam vs centred, np 1 / 2 / 4 / 8 | ≤ 2.0e-12 | ≤ 4.2e-11 | ≤ 2.9e-11 | identical | equal to 1e-8 | 0 violations, every step |
| same, active remeshing (1 280 → 3 200 triangles), np 4 | 1.9e-12 | 5.9e-12 | 4.0e-12 | identical | equal to 1e-9 | 0 |
| np 8 vs np 1 (seam run; centred run) | 5.5e-13; 7.7e-13 | 1.4e-11; 3.0e-11 | ≤ 2.1e-11 | identical | ≤ 1e-8 | 0 |

Velocity interpolation, spreading, pressure/indicator and geometry are therefore translation
exact to round-off through two seam crossings of the corner; the volume drift (4.1e-4, no volume
recovery) is identical in both runs, i.e. no cumulative seam error; the image counters reach 1 and
return, wrap flags stay 000, no cut edge. **D5 passes** (criterion 1e-6). The non-periodic-front
reference is reported in §25.8.

### 25.3 D6 — marching-cubes pinch-off with the neck on a seam
Case: closed dumbbell (`BodyMesh`, lobes R = 0.35 at ±0.4, neck radius 0.03) in the periodic
32³ box, breakup criterion 1.5 dx, factor 1. Placements: neck on the x seam (S1), dumbbell centred
on the xyz corner so that both lobes also straddle the y and z seams (S3), centred (C1); a
pole-clustered mesh (5 696 triangles) and a well-shaped icosphere mesh (20 480). Every run: breakup
detected at step 1, the periodic marching cubes splits the body into **2 bodies** (same count in
every placement and rank count), winding valid (compact, no cut edge after canonicalisation),
redistribution clean (0 invariant violations over 10 steps), the capability re-check (R16D-3)
passes, both lobes DEFINED.

| Comparison (final tree, with R16D-5) | volume | centroid /L |
|---|---|---|
| S1 vs C1, np 1 / 4 / 8 | 1.2e-9 / 1.4e-8 / 1.2e-13 | ≤ 5.3e-9 |
| S3 vs C1, np 1 / 4 / 8 | 9.5e-8 / 1.0e-7 / 8.9e-8 | ≤ 2.8e-6 |
| icosphere mesh, np 4: S1 / S3 vs C1 | 2.9e-12 / 7.1e-8 | ≤ 1.3e-5 |
| before R16D-5 (single MFTL1 shift): S3 vs C1, np 1 / 2 / 4 / 8 | 1.8e-5 – 6.8e-5 | ≤ 3.4e-6 |
| raw marching-cubes output, S3 vs C1 (debug dump, float32) | same 574 vertices / 1 144 triangles per body, positions equal to 1e-6 | |
| S3 vs C1 with an order-independent (Jacobi) recovery (reviewer experiment) | 2.6e-7 (np 1), 1.5e-7 (np 4) | ≤ 6e-7 |
| C1 np 4 vs np 1 (no seam): MFTL1 recovery / Jacobi recovery | 9.2e-4 / 6.2e-7 | |

The marching-cubes geometry is translation exact; the S3 residual is created by the
**order-dependent MFTL1 post-processing** (in-place sequential normal shift and Gauss–Seidel
Kuprat smoothing over a vertex order that differs when the body's cells wrap), the same mechanism
that makes the centred case differ by 9e-4 between 1 and 4 ranks. It is pre-existing Stage-15
behaviour and cannot change on the non-periodic path (parity); with R16D-5 the periodic recovery
is iterated to the target, which removes the volume part of it. **D6 passes** on the final tree:
same body count in every placement and rank count, marching-cubes geometry equal, volumes equal to
≤ 1.4e-8 (neck on the seam) and ≤ 1.1e-7 (corner) — criterion 1e-6; the centroid residual of the
corner case (≤ 2.8e-6 L, shape of the Kuprat smoothing order) is below the rank-count spread of the
centred case itself (1.8e-5 L).
Class re-check after the topology change: D6W (R16D-3).

### 25.4 Rank-count-dependent remeshing — diagnosis
Two MPI-boundary defects (**B**) and benign order dependence (**A**):
- **B, fixed:** R16D-1 (corners collapsed nondeterministically in the parallel generalized path)
  and R16D-2 (interior-only single pass under the default `Optimizer 0`: cut-straddling edges
  frozen, neighbours flattened to 0°). The implementer's multiflow cases (open patch advected in
  the periodic box, generalized remeshing every 5 steps) before → after the fixes:
  - patch straddling the z = 1 rank plane (OPENX), triangles on np 1/2/4/8:
    742 / 624 / 596 / 601 → **717 / 717 / 760 / 741**; splits 678 / 494 / 414 / 440 →
    699 / 733 / 763 / 725;
  - the same patch off the plane (OPENX06): 745 / 710 / 701 / 712 → 723 / 719 / 765 / 761;
  - on and off the plane now agree within 2.6 % at every rank count; area equal to 1e-4; min
    angle 7.5–13.5°, at most 2 triangles below 10°.
- **B, remaining, bounded (documented):** on narrow load-balanced slabs (8 slabs across a 2 × 2
  patch, slab ≈ 2 target edges) the two-pass scheme cannot approve collapses whose fans straddle
  both the original and the shifted cut in one call; converged triangle count 969 on 8 ranks vs
  694 on 1. Handing the converged mesh to the serial kernel removes 19 % more (→ 781), i.e. part of
  the excess is suppressed coarsening. It does not grow with time, never produces invalid or
  degenerate geometry (outline and area exact), and multiflow's `Cleanfrequency` (gather + serial
  boundary cleanup, MFTL1 design) removes it.
- **A, benign:** the remaining spread (781 vs 694, min angle 4.7° vs 12.8°) is also a fixed point
  of the serial kernel on the other mesh: different legal operation order, same geometry.
- The generalized remesher without tangential relaxation (multiflow's default) produces slivers on
  curved closed bodies opted into GENERALIZED even serially (sphere: min angle 0.27°); open planar
  patches are fine (≥ 12.8°). Rank-independent; recorded for 16F/17 (tangential relaxation).

### 25.5 Periodic marching-cubes resolution — conclusion
- Factor 1 (marching cubes at the front-tracking box size on periodic axes) is translation
  invariant and gave the correct topology in every D6 placement (neck on a seam, corner,
  centred; both meshes; 1/2/4/8 ranks) and in the two-drop coalescence across the seam. The
  pre-recovery deficit is a resolution effect: −10.6 % for the dumbbell (11 boxes per lobe
  diameter; non-periodic default −8.3 %), −47.5 % for the 4.8-box drops (non-periodic −24.3 %,
  whose grid spans the bodies' box and is finer there).
- The recovered solution was **not** reliable for under-resolved bodies (+5.4 %); fixed by R16D-5
  (iterated recovery on periodic fronts: 2.6e-11). For resolved bodies the single shift gave
  +0.22 % (non-periodic +0.13 %); now ≤ 1e-10.
- A neck thinner than about one box is removed by design (breakup criterion ≥ 0.5 dx); a
  periodic run needs the bodies that change topology resolved by ≳ 10 boxes per diameter for the
  marching-cubes shape (not only the volume) to be accurate — recorded as a validation requirement
  for the manual (Stage 17).
- Factor > 1 is rejected on periodic fronts (R16D-4).

### 25.6 Explicit rejections and open surfaces
Reviewer matrix, 2 ranks (every case stops at initialisation before the first step, naming
module, body, class and reason; allowed combinations run):
- periodic closed drop + CSF, integral ST, IRL, surface scalar transport, surface viscosity, NOA
  advection, generic acceleration, force computation, divergence-free interpolation: each
  rejected;
- open patch (non-periodic front) + Frenet ST, CSF, integral ST, Poisson indicator, IRL, volume
  recovery, marching cubes, scalar transport, surface viscosity, NOA, generic acceleration, force
  computation: each rejected; + divergence-free interpolation: allowed (surface-only row);
- closed tube wrapping x + indicator: rejected; the same tube passive: runs; after a topology
  change that creates a wrapping body: rejected (R16D-3);
- periodic front + factor > 1: rejected (R16D-4).
The passive open front (no indicator, no surface tension) runs only advection and generalized
remeshing: no indicator solve, no spreading, no recovery; volume UNDEFINED, centre DEFINED; no
invariant violation (288 → 1 152 triangles, min angle 17.6°). Frenet ST without the indicator is
stopped at step 1 by the pre-existing multiflow configuration check (identical on non-periodic
fronts; nothing is spread). IBM bodies are never open and IBM decompositions stay non-periodic
(§24.1), so no enclosed-mass path can see an open or periodic body.

### 25.7 Pre-existing breakup/coalescence fixes (§24.4)
- Unmodified Stage 15 on the non-periodic two-drop case MC0c (1 rank) crashes at the first
  marching-cubes call (SEGV → `MPI_Abort` 59, then blocks until the 300 s timeout): the cleanup of
  the never-allocated marching-cubes overlap mesh. The self-send skip is required by the protocol
  (the receive loop probes only the other ranks, so a blocking self-send is never matched).
- Stage 15 + only these two fixes vs the 16D tree: **bit-identical** on MC0c at 1 rank (marching
  cubes three times, every HDF5 file, identical recovery logs) and 4 ranks. Both fixes are
  minimal and do not change any other Stage-15 behaviour.
- Rank-local coalescence detection (pairs on different ranks not found) is pre-existing and
  outside Stage 16 (unchanged).

### 25.8 Regression (final tree)
| Gate | Result |
|---|---|
| MFTL2 ctest, Release (LAPACK on), Release (LAPACK off), Debug + ASan/UBSan/LSan | 81 / 81 in each; 0 sanitizer reports (after the unsigned checksum in `test_stage16a.c`) |
| `test_stage16c` new `params` check | passes; fails (3 of 4) against the unfixed header |
| Adapter unit tests, np 1/2/3/4 | 134 / 279 / 427 / 581 checks, 0 failed (incl. `test16d_rank_cut_remesh`; 4 failures on 4 ranks without R16D-2) |
| multiflow A/B vs the Stage-15 binaries, final tree: FT, FT_Sharp_CSF, FT_Sharp_Integral np 1/4, IBM-MFTL A np 1/4, B np 4, Laplace LAPP0 np 4 | **bit-identical** (every HDF5 file, `h5diff`); 0 build warnings |
| Non-periodic marching cubes MC0c np 1/4 vs Stage 15 + the two pre-existing fixes | bit-identical (Stage 15 alone crashes, §25.7) |
| D5, np 1/2/4/8 (+ remeshed variant np 4; + non-periodic-front reference np 4) | §25.2; seam run vs the Stage-15 non-periodic-front path: centroid 1.4e-11 L, volume 7.5e-10 over 200 steps |
| D6, np 1/4/8, two meshes; D6W class transition (1/2/4 ranks) | §25.3; D6W stops at the capability check; factor 3 rejected at initialisation |
| Marching cubes across the x seam vs centred (two drops, MCPXc / MCP0c, np 1) | same body count, pre-recovery −47.53 % both, after recovery 2.6e-11 both |
| Open seam patch (OPENX, OPENX06) np 1/2/4/8 | §25.4 |
| D8 restart on a different rank count: periodic droplet np 4 → 1/2/8; open patch np 4 → 1/2/8 | droplet: restarts agree to 5e-15 among 1/2/8 ranks and with the non-periodic-front restart (the offset to the continuous run is pre-existing, §25.9); open patch: class restored, 0 invariant violations, geometry within the remeshing spread (area 7e-5) |
| Rejection matrix (§25.6), 2 ranks | every row as specified, before the first step |
| multiflow Debug + ASan/UBSan (np 2): D5 seam, D6 corner and neck-on-seam, passive open patch, open seam patch with generalized remeshing, D6W | 0 ASan, 0 UBSan reports. *[Corrected by the 16E review, §27.4: this statement was inaccurate. multiflow's CMakeLists replaced `CMAKE_C_FLAGS`, so the `-fsanitize=address,undefined` (and `-ffp-contract=off`) given there never reached multiflow's compile lines (the reviewer build cache shows `C_FLAGS` without them); the binary was only linked with the sanitizer runtime (`CMAKE_EXE_LINKER_FLAGS`, and the instrumented MFTL2 archive's interface). Only the MFTL2 library code was instrumented in these runs. The multiflow code of these cases was first run genuinely instrumented in the 16E review (§27.10).]* |

### 25.9 Pre-existing observations (not 16D defects; not changed)
- The MFTL1 post-marching-cubes recovery and Kuprat smoothing are order dependent: after a
  reconstruction the volume differs by up to 9e-4 between 1 and 4 ranks on a non-periodic-like
  (centred) case. Stage-15 parity; candidate for Stage 18.
- Marching cubes without a `FRONTREMESHING` block dereferences the NULL remesh settings (SEGV,
  non-periodic too).
- Restart: the fluid state after a restart differs from the continuous run by 6e-7 in volume and
  3e-5 in spurious velocity at the first restarted step, identically on periodic and non-periodic
  fronts and independent of the rank count (restarts on 1/2/8 ranks agree to 5e-15).
- The debug marching-cubes case histogram counts periodic ghost cells as well (statistic only).
- The logged continuous centroid is expressed in the frame of the body's reference vertex and
  jumps by one period when that vertex wraps (equal mod L; logs only).

### 25.10 Items for 16E / 16F (not blocking 16E)
- Finer periodic marching cubes (factor > 1) needs a wider cell band and its own validation.
- Parallel coarsening on narrow load-balanced slabs (§25.4) and generalized-remesh quality
  without tangential relaxation on curved closed bodies.
- The phase-dependent modules still reject closed bodies that wrap a periodic axis even when the
  phase set is DEFINED (§24.5); open-surface physics is 16E.

## 26. 16E implementation status and test results (2026-09-25)

**Status: 16E IMPLEMENTED (2026-09-25); ACCEPTED by the independent review of 2026-09-25 after
corrections C16E-1 to C16E-6 (§27).** 16A–16D stay ACCEPTED
(§18.1, §21, §23, §25); §3.6 stays frozen and no Stage-16 rule was changed. 16F and Stage 17 are not
started. Nothing was committed or pushed (MFTL2 `main` and multiflow `berend-mftl` working trees).

### 26.1 What was implemented

**MFTL2 (geometry only).**
- `MFTLBoundaryEdgeCentroidDistance(obj, he, &delta, &h_T)` (`kernels.c`, `mftl_geometry.h`): the
  in-surface distance δ_e from the centroid of a boundary half-edge's triangle to the edge midpoint
  and the height h_T = 2A_T/L_e, in the local frame (§6: "MFTL2 provides δ_e").
- `tests/test_stage16e.c` (ctest `Stage16E-np1/2/4/8`, 101 checks each): planar patch (zero element
  force on every triangle, loop Σ L ν = 0), pinned spherical caps (summed element force = boundary
  line force to round-off; convergence to the analytic ring force), the cap across the x/y seams and
  with reversed orientation (identical resultant), δ_e and h_T against a direct evaluation incl. a
  sliver strip (h/L = 1e-4).

**multiflow adapter** (`mftl_adapter.h`, `mftl_adapter_mftl2.c`): `MFAdapterBoundaryEdgeCentroidDistance`,
`MFAdapterVertBoundaryTags` (tags of a boundary vertex's two boundary edges),
`MFAdapterBodySetBoundaryConstraint` (CONSTRAINED boundary curve + freeze, §4.3),
`MFAdapterPhaseSetVolumeRecovery` (set-level recovery, §4.4.4). The MFTL1 post-remesh clamp (1e-7
inside the box) is applied to open bodies as a clamp into the *closed* box: a loop pinned on a wall
closure must stay within τ_loop = 1e-9 L of the wall (§3.6.2 step 1); closed bodies unchanged.

**multiflow options** (`FRONTTRACKING` block; fixed-size tables in `FTOptions`, carried by the byte
broadcast):
- `SurfaceScalarBC <body> <tag|ALL> NEUMANN <q> | DIRICHLET <c> | ROBIN` (ROBIN reserved, rejected);
- `SurfaceTensionBoundary <body> <tag|ALL> PINNED STATIC | PINNED IBM <id> | PINNED DEM <id> |
  CONTACT_LINE | REJECT` (IBM/DEM supports and CONTACT_LINE rejected, see §26.5);
- `PhaseSetMember <body> <+1|-1>` (an open body joins the dispersed-phase set with its σ; closed
  bodies are members with σ = +1, the multiflow convention) and `WallClosure <X|Y|Z> <MIN|MAX>`
  (set-level closures). This is the multiflow form of §5.1's `PhaseSet … WallClosure …`: multiflow
  has one dispersed phase. An explicit tag takes precedence over `ALL`; duplicates are errors.

**Surface scalar transport (§6)** (`surfacescalartransport.c`). A boundary edge takes the condition
of its loop's tag; the discretisation is exactly the §6 table: face value c_T − qδ_e/D (NEUMANN) /
c_D (DIRICHLET) in the Green–Gauss gradient (the boundary edge's L ν is in the Muradoglu closure);
no off-diagonal coupling; NEUMANN RHS −qL_e; DIRICHLET diagonal +DL_e/δ_e and RHS +DL_e c_D/δ_e;
no advective (NOA) flux across a boundary edge; δ_e from MFTL2 with the floor δ_min = 0.1 h_T
(counted and logged). Sign: q > 0 is flux outward along the MFTL2 conormal ν, i.e. out of the
surface domain. Mass budget of an open body: the step conserves Σ c_old A_old + Δt·inflow exactly
(inflow = −Σ qL − Σ DL(c_T − c_D)/δ_e with the new values); the sub-iteration criterion uses it,
and the log reports the step error and the drift against (initial mass + accumulated inflow).
The former `exit(1)` on a missing neighbour is replaced by the dispatch; a missing neighbour across
a non-boundary edge is an invariant violation (error). Closed bodies use the unchanged code. The
centre-to-centre vector uses the minimum image on periodic fronts, so scalar transport is now
enabled on periodic fronts (closed and open); its overlap-mesh remapping (volume-conserving
Laplacian smoothing, marching cubes) is still rejected there.

**Surface tension, PINNED STATIC (§7.2)** (`openboundarymodels.c`). The support of a pinned loop is
the loop as declared (read from the mesh or the restart), gathered as a polyline in continuous
coordinates. Kinematics: the owned vertices of pinned loops are restored after the time
integration (before the wrap, so the counters are unchanged); for remeshing the loop is
CONSTRAINED to the support (closest-point projection onto the polyline, minimum image) and frozen.
The element forces are spread unchanged; the non-cancelling line force of each boundary edge,
σ L ν with the same σ and conormal as its element (EdgePlanar), gives the support reaction
R = −Σ σ L ν, reported per step per (body, tag) together with the momentum budget (fluid force =
Σ over cells of the spread density × volume; element force; fluid force + ΣR). Fix found during
validation: `SpreadFrontPropertiesCartesian` uses unnormalised kernel weights, exact only when the
whole stencil lies inside the domain; the rim triangles of a surface pinned on a wall lost part of
their force (fluid + R = 6e-4 of R). For open bodies the weights are now normalised over the
existing cells (as `SpreadFrontPropertiesGeneral` always does); closed bodies keep the Stage-15
weights.

**CSF / sharp CSF and IRL fractions (§7.3).** Enabled only for an open body that is PINNED STATIC on
all loops, a wall-closed member of a DEFINED dispersed-phase set, on a non-periodic front (the IRL
paths are not converted to periodic fronts, 16D), with `NeighbourhoodOrder ≥ 2` (the multiflow
paraboloid fits need ≥ 6 stencil points and otherwise terminate the run; a boundary vertex's one-ring
does not guarantee that). The reaction is the geometric line force −σ Σ L ν of the pinned loop
(MFTL2 conormal), reported per step; CSF has no element forces, so no momentum-budget identity.
*Deviation (for review):* the boundary curvature comes from multiflow's own open-stencil IRL
paraboloid fit (VOLUMEFIT/VERTEXFIT on the N-ring), not from MFTL2's
`MFTL_VFLAG_CURV_EXTRAPOLATED`, and the indicator from the IRL fractions + flood fill (checked
against the §3.6 wall-closed volume, §26.3), not evaluated from the phase set.

**Wall closures and the dispersed-phase set (§3.6.2).** The set = closed bodies (σ = +1) + declared
open members (σ) + declared closures; it is classified at initialisation whenever an open member
exists (previously only on periodic fronts; closed-only runs build the identical set). Uses:
- Poisson indicator: on the faces of declared closure walls the boundary treatment is the Neumann
  condition consistent with the spread gradient (∂I/∂n = G·n: the flux term and the matching RHS
  term cancel, so the face contributes nothing); every other wall keeps the Stage-15 treatment. The
  spread indicator gradient of an open member uses its σ.
- indicator seed: in wall-closed mode every candidate is tested exactly with
  `PhaseSetInside` (the Stage-15 corner first, or on periodic fronts the point antipodal to the first
  member, so translated configurations pin at the translated point); kept while outside.
- volume: the open member's own wall-closed volume (singleton + closures) for the start volume,
  the stored volume and volume recovery, which uses the library's set-level recovery (interior
  vertices shifted uniformly, boundary vertices fixed on the wall).
- after every remeshing the wall-closed set is re-classified; if it is not DEFINED the run stops.

**Capability matrix (§5.2) — 16E rows for open bodies** (`generalizedsurfaces.c`, before the first
step and again on class transitions; every message names module, body, class and the missing BC,
support or closure):

| Module | Open body: allowed iff |
|---|---|
| Frenet surface tension | every loop PINNED STATIC, and the Poisson indicator on (multiflow spreads Frenet forces only with it), which in turn needs the wall closure below |
| CSF / sharp CSF, IRL fractions | every loop PINNED STATIC + wall-closed member of a DEFINED set + non-periodic front + `NeighbourhoodOrder ≥ 2` |
| Poisson indicator | member (`PhaseSetMember`) + ≥ 1 `WallClosure` + the set DEFINED + every loop PINNED STATIC (only a pinned loop is held on its wall; a free contact line is not modelled) |
| Volume recovery | as the indicator, and the body's own wall-closed volume DEFINED |
| Surface scalar transport | every loop has NEUMANN/DIRICHLET (ROBIN reserved); periodic fronts allowed |
| Integral ST, surface viscosity, marching cubes, NOA, generic acceleration, force computation | REJECT (unchanged) |
| Divergence-free interpolation, advection, spreading | allowed (unchanged) |

Declaration checks: boundary models and `PhaseSetMember` only on declared-open bodies and existing
tags; `WallClosure` only on a non-periodic fluid wall, only with an open member, not with
`PBoxAdaptation`/`FixedPBox`, and the front box must end at that wall; loops on no/the wrong
closure, winding loops, loops on two walls, closures on two axes and ambiguous closures return the
§3.6 statuses (UNDEFINED/UNSUPPORTED) and are rejected with the library's reason.

**IBM (§5.5):** unchanged. IBM bodies are never declared open; no mass property is derived from an
open surface; PINNED supports on IBM/DEM bodies are rejected.

### 26.2 Surface-scalar validation (multiflow, 16E validation binary with `FT_SCALAR_TRANSPORT`)
| Case | Result |
|---|---|
| 1. zero-Neumann open patch, moving (Taylor–Green + uniform flow), generalized remeshing every 4 steps, crossing the x seam (periodic fronts), 80 steps, np 1/2/4/8 | per-step mass error ≤ 3.7e-14, drift vs initial mass ≤ 2.9e-13 (plan E1: 1e-12 per step) |
| 2. Neumann q = 0.1 on the planar patch (loop length 2) | dM/dt = −qL exactly: M(t) − (0.25 − 0.2 t) ≤ 4e-16 |
| 3. Dirichlet, open cylinder (two loops, c = 1 / 0), steady | area-weighted L2 error vs linear 7.6e-5 / 9.4e-6 / 1.2e-6 for 8/16/32 rings: rate 3.0 (plan ≥ 1.9) |
| 3. transient vs Fourier series, finest mesh, t = 0.5 | max error 2.5e-4 (Δt 1e-3), 1.6e-4 (Δt 5e-4) (plan ≤ 1e-3) |
| 4. mixed Dirichlet / Neumann (c = 1; q = 0.05) | L2 5.2e-4 / 1.3e-4 / 3.4e-5: rate 2.0; step mass error ≤ 6e-12 (KSP tolerance) |
| 5. Dirichlet cylinder crossing the x seam vs centred | fields equal to 3.0e-15; error vs linear identical to the non-periodic case |
| 6. the mixed and the seam cases on 1/2/4/8 ranks | fields equal to ≤ 3.7e-11 (KSP tolerance) |
| 7. missing BC; BC on a closed body; BC for an absent tag; ROBIN | rejected at initialisation with body/tag messages |
| δ_min floor | never invoked (0 edges in every run) — see §26.6 B-1 |

### 26.3 Surface-tension and wall-closure validation
Cap on the z-min wall of [0,1]²×[0,0.5] (no-slip walls), equal phases, σ = 0.1, Frenet + Poisson
indicator, `PhaseSetMember 0 1`, `WallClosure Z MIN`, `PINNED STATIC`, 20 steps, np 4 unless noted.

| Case | Result |
|---|---|
| 9. hemisphere R = 0.25, 16/32/64 cells per unit (24/48/96 rim edges) | reaction R_z error vs 2πrσ sinα: −1.80 / −0.45 / −0.11 % (2nd order); Δp vs 2σ/R: −3.61 / −0.97 / −0.25 %; Σ I·V vs the §3.6 wall-closed volume 8.8e-3 / 1.1e-3 / 1.4e-4 |
| 9. momentum budget, every step, every cap case | \|fluid force + R\| ≤ 2.8e-16 (R ≈ 0.156) |
| 9. cap α = 0.35π, remeshed every 2 steps (720 → 1463 triangles) | contact line on its support ≤ 1e-12, set DEFINED throughout; R_z 1.6 % off at 48 rim edges (first order for α ≠ π/2: the boundary conormal uses the one-sided triangle normal — library test: 5.2 % → 0.54 % from 16 to 128 edges, rate 1.0; hemisphere rate 2.0) |
| 10. hemisphere across the x/y seams vs centred (periodic x, y; walls z) | R identical to 16 digits, Δp equal to 1e-12, indicator volume equal to 2.5e-9 |
| 11. reversed triangle orientation with `PhaseSetMember 0 -1` | R, Δp and indicator volume identical to the original; with σ = +1 the set is UNDEFINED (W7) and the run is rejected |
| 12. open + Frenet without `SurfaceTensionBoundary`; CONTACT_LINE; PINNED IBM; no indicator | each rejected at initialisation |
| cap in a Taylor–Green flow with remeshing: without / with wall-closed volume recovery | volume drift 3.3e-3 / 8.7e-12 over 50 steps; contact line on the support ≤ 1e-12 |
| CSF + IRL (VOLUMEFIT, order 3), hemisphere, 16/32/64 | Δp error 4.46 / 2.27 / 0.69 %; IRL indicator volume vs wall-closed volume 2.0e-3 / 3.3e-4 / 3.2e-5; np 1 and np 4 identical; contact line on the support ≤ 1.6e-16 |
| 13./14. wall-closure statuses | cap with z-min: DEFINED (V = mesh volume); closures on both z walls: DEFINED (W6, V = cap); closure on the wrong wall: UNDEFINED; closure on a periodic axis: configuration error; half tube along x on the wall (winding loops): UNSUPPORTED; closure without an open member, with PBoxAdaptation: rejected |
| 15. periodic seam-crossing wall-closed cap | = case 10 |
| Library (Stage16E, np 1/2/4/8) | planar: element forces and Σ L ν = 0 exactly; cap: F_elem = Σ L ν to ≤ 3.7e-14; seam/reverse: identical resultant |

8. *[Superseded by the 16E review, §27.1: Frenet on a pinned open surface no longer needs the
indicator; the pinned film and free cap are regressions T1/T2.]* *Pinned planar patch in multiflow* is not runnable: multiflow spreads Frenet forces only with the
Poisson indicator, which an open surface can use only wall-closed, and a planar patch cannot be
wall-closed non-degenerately (a loop on one wall is a degenerate film; loops on two walls are
UNSUPPORTED). The planar force balance is covered by the library test and by the budget identity,
which holds for every pinned surface.

Rejection matrix (np 2, 27 configurations): every REJECT row and every declaration error stops at
initialisation with its message; the allowed configurations (scalar on an open patch with BCs,
both-wall closure, CSF with order 3) run.

### 26.4 Regression
| Gate | Result |
|---|---|
| MFTL2 ctest, Release (LAPACK on), Release (LAPACK off), Debug + ASan/UBSan/LSan | 85 / 85 each (81 + Stage16E np 1/2/4/8); 0 sanitizer reports |
| Adapter unit tests np 1/2/3/4 | 146 / 303 / 463 / 629 checks, 0 failed (new: centroid distance, vertex tags, CONSTRAINED + frozen loop under refinement, wall-closed cap across the seams: volume, set-level recovery, rim fixed, inside/outside) |
| multiflow A/B vs the accepted 16D final outputs (= Stage 15): FT, FT_Sharp_CSF, FT_Sharp_Integral np 1/4, IBM-MFTL A np 1/4, B np 4, Laplace LAPP0 np 4, marching cubes MC0c np 1/4 | **bit-identical** (every HDF5 file, `h5diff`); 0 build warnings |
| D5 (bubble through the seams, 200 steps) seam and centred, np 8; D6 neck on the seam, corner, centred, np 4; D6W | reviewer diagnostics (`front_log.txt`) and every HDF5 file **identical** to the accepted 16D final runs; D6W stops at the capability re-check as before |
| multiflow Debug with ASan/UBSan *instrumented* (flags in `CMAKE_C_FLAGS_DEBUG`, see §26.6 O-2), np 2: moving/remeshing scalar patch across the seam, mixed-BC cylinder, periodic wall-closed pinned cap with remeshing and volume recovery, CSF cap | 0 ASan, 0 UBSan reports |

### 26.5 Intentionally unsupported (explicit rejection, never approximated)
- `CONTACT_LINE` (reserved), dynamic/moving contact lines, wetting models; `ROBIN` scalar BC.
- **PINNED on an IBM/DEM support** (plan §7.2 names it): not implemented in 16E — it needs support
  velocities in the advection and the reaction in the IBM/DEM force balance (IBM scope unchanged).
  Rejected with its own message. *Scope gap for the review to confirm.*
- A free (unpinned) boundary loop on a wall closure: the wall-closed set needs the loop on the wall
  to 1e-9 L; only PINNED STATIC loops are held there. Rejected for every phase-dependent module.
- ~~Frenet surface tension on an open surface without the Poisson indicator~~ (supported since the
  16E review, §27.1; the fluid is then single-phase).
- CSF / sharp CSF / IRL on periodic fronts (16D) and with `NeighbourhoodOrder < 2` on open surfaces.
- Integral surface tension, surface viscosity, marching cubes, NOA, generic acceleration and force
  computation on open surfaces; point-inside queries of open surfaces; mass properties of open
  surfaces (IBM bodies are never open).
- Surface-scalar remapping through the overlap mesh (volume-conserving smoothing, marching cubes) on
  periodic fronts.

### 26.6 Findings
- **B-1 (contradiction in the frozen §6 text, not changed):** δ_e, the distance from the centroid
  to the edge midpoint, is always ≥ h_T/3 (the centroid lies at distance h_T/3 from the edge line),
  so the floor δ_min = 0.1 h_T can never be invoked; the E2 sub-test "a sliver boundary triangle uses
  the δ_min floor and is counted" is unattainable (library test `delta`: min δ_e/h_T = 1/3 exactly,
  also on a strip with h/L = 1e-4). The floor and its counter are implemented as specified (0 uses
  in every run). For a sliver whose long edge is the boundary edge, δ_e ≈ h_T/3 → 0 and the
  Dirichlet coefficient D L/δ_e grows like 3DL/h_T; it only strengthens the diagonal (the implicit
  system stays diagonally dominant), and a degenerate A_T is an invariant violation. If a floor is
  intended to bound the coefficient, it must be defined relative to L_e (e.g. 0.1 L_e), which is a
  model change for the reviewer/owner.
- A-1 (fixed): unnormalised Cartesian spreading lost the force of stencils cut by a wall (§26.1).
- A-2 (fixed): the post-remesh 1e-7 clamp pulled pinned loops off their wall closure (§26.1).
- A-3 (fixed): the indicator gradient of an open member ignored its σ (reversed cap: I ≈ 0).
- A-4 (fixed): a `CHECK` inside rank-local loops deadlocked the first version of the library test.
- O-1: surface scalar transport needs `FT_SCALAR_TRANSPORT` (without it `scalar_old` is never set
  and the scalar is zeroed after the first step, closed bodies included); multiflow's CMakeLists
  overwrites `CMAKE_C_FLAGS`, so the define must go through `CMAKE_C_FLAGS_<CONFIG>` or
  `compileoptions.h`. Pre-existing; the 16E validation binary sets it.
- O-2: for the same reason the 16D review's multiflow "Debug + ASan/UBSan" binary (§25.8) was
  linked against the sanitizer runtime but not instrumented (`-fsanitize` in `CMAKE_C_FLAGS`).
  The 16E sanitizer runs use `CMAKE_C_FLAGS_DEBUG`.
- O-3: `ErrorH` calls `exit(1)` on every rank; `mpirun` occasionally returns 0 for such a run
  (one rejection case). The message is deterministic; exit codes of rejected runs are not reliable.
- O-4: the Frenet line force of a boundary edge uses its one-sided triangle normal, so the resultant
  of a non-hemispherical pinned cap converges at first order (library test: rate 1.0; hemisphere
  2.0). This is the discretisation order of the Frenet element model at an open boundary.

### 26.7 Open for the 16E review (all resolved in §27)
- Confirm the scope decisions: PINNED on IBM/DEM supports rejected; CSF via multiflow's own IRL
  open-stencil fit instead of MFTL2's extrapolated curvature (§26.1); pinned loops required for every
  wall-closed member.
- Decide B-1 (the Dirichlet floor).
- The multiflow validation cases, meshes, generators and diagnostics are in the implementation
  session's scratch (not in the repositories): `gen16e.py`, `mk16e.py`, `uf16e/`, `cases/`.

## 27. Independent review of 16E (2026-09-25)

**Verdict: 16E ACCEPTED after corrections C16E-1 to C16E-6.** 16A–16D stay accepted (C16E-6 is a
latent 16C/16D library clamp, fixed for declared-open bodies only; closed behaviour unchanged);
§3.6 stays frozen. The two corrections to frozen Stage-16 text (§6: the Dirichlet distance and its
sliver floor) are narrowly scoped corrections of erroneous details, recorded in place. 16F and
Stage 17 are not started. Reviewer: Claude Code (independent of the implementation session).
Nothing was committed or pushed.

### 27.1 Capability-matrix fidelity: Frenet surface tension and the indicator (C16E-3)
- **Finding.** §26.1/§26.3 made Frenet surface tension on an open body require the Poisson
  indicator, and hence a wall closure ("a pinned planar patch cannot run in multiflow"). The frozen
  §5.2 row allows Frenet on an open body "only with a boundary model, §7" and §7.2 needs only PINNED
  loops; the wall closure belongs to the indicator/CSF/volume rows. The implementation was
  stricter than the matrix.
- **Analysis.** The dependency is an inherited code structure, not physics or numerics:
  `SolveFronttracking` had a single branch "Poisson indicator AND Frenet" (the two are spread in
  one pass by `SpreadFrontProperties*`), and every other combination with surface tension ended in
  "Something is wrong for the settings". The spread Frenet density enters the momentum equation
  only as the body-force source `SURFACETENSION{X,Y,Z}` (`SetSourceTerms`); the indicator only
  mixes the phase properties (`INDICATOR = 0` gives phase-1 properties everywhere) and drives the
  indicator-based models (CSF, integral, marching cubes). A surface pinned on all its loops exerts
  its element forces `Σ σ L ν` on the fluid and `R = −Σ_loop σ L ν` on its support whether or not it
  bounds a phase.
- **Fix** (multiflow; the indicator/CSF/volume phase requirements are unchanged):
  - `solvefronttracking.c`: Frenet without the indicator spreads the element forces (the budget of
    §7.2 is recorded as before); no indicator solve. `EnsureTriangleConnectivity` also for it.
  - `createeulerianvectors.c`: the spreading target vectors exist in that configuration too.
  - capability rule (`generalizedsurfaces.c`): open + Frenet needs every loop PINNED STATIC only.
    Without the indicator the fluid is one phase, so the phases must have equal density and
    viscosity (a property jump needs a phase region, i.e. the indicator; rejected otherwise with a
    message), and marching cubes needs the indicator. The indicator row still requires
    `PhaseSetMember` + `WallClosure` + a DEFINED set + pinned loops.
- **Regressions (testcases/FT_Stage16E, §27.3):** T1_PINNED_FILM (a bumped film pinned on its
  square loop in the middle of the domain, Frenet, no indicator; np 1 and 4): fluid force + R
  ≤ 1.6e-16 every step (R_z(0) = 0.0388), R_z(0) vs the analytic conormal integral
  `4σ∫ s/√(1+s²) dη` +6.0e-4, spread = element force to 1.9e-16, contact line exactly on the support,
  the film relaxes (height 0.050 → 0.043, area decreasing), identical R on 1 and 4 ranks;
  T2_PINNED_CAP_FREE_24/48/96 (cap pinned on a ring, no wall, no indicator): budget ≤ 2.1e-16,
  R_z error −7.07 / −3.30 / −1.59 % (rate 1.10, 1.05: first order, O-4); R16 (phases of different
  density without the indicator) and R17 (no boundary model) rejected; A1 (the former §26.3
  rejection R04: wall-closed cap, Frenet, no indicator) runs.

### 27.2 Dirichlet distance and sliver floor (C16E-1, C16E-2; §6 corrected)
- **C16E-2 (the floor).** Confirmed B-1: the centroid lies at the normal distance h_T/3 from the
  edge line, so `δ_e ≥ h_T/3` and the frozen floor `0.1 h_T` could never act. Derivation of the
  correction: `δ_e` is a normal (height-type) distance; a sliver is a triangle whose normal height
  is small relative to its tangential extent; a floor expressed in the same normal height can
  therefore never detect it, and must be tied to a tangential scale. The boundary-edge length
  `L_e` is the tangential scale of the flux through the edge (no other already-defined tangential
  triangle scale is more appropriate): `δ_eff = max(δ_e, 0.1 L_e)` (Dirichlet coefficient only),
  bounding `D L_e/δ_eff ≤ 10 D`. Every use is counted and logged with the largest `L_e/δ_eff`.
- **C16E-1 (the distance itself; found while checking the discrete equations).** The frozen text
  (and the implementation) used the distance from the centroid to the edge **midpoint**,
  `√(h_T² + s²)/3` for an apex offset `s` along the edge. For a linear field constant along the
  edge (every DIRICHLET condition, which is constant per loop), `c_T − c_D = g_ν h_T/3` exactly, so
  the two-point flux `D L (c_D − c_T)/δ` is exact only with the **normal** distance `h_T/3`; the
  midpoint distance under-estimates the boundary flux by `h_T/√(h_T² + s²)` on skewed triangles
  (the same holds for the Neumann face value `c_T − qδ/D`). Evidence (steady Dirichlet on the open
  cylinder, every boundary triangle skewed by s = L/2, S3K): midpoint distance L2 3.32e-3 /
  1.68e-3 / 8.49e-4 for 8/16/32 rings (**rate 0.98**); normal distance 8.87e-5 / 1.14e-5 /
  1.45e-6 (rate 2.96, 2.98). On the symmetric mesh (apex above the midpoint, the only mesh the
  §26 validation used) both agree (rate 3.0). Now `δ_e = h_T/3` (from MFTL2's h_T; the MFTL2 API is
  unchanged).
- **Deliberate sliver (S7_SLIVER_8/16/32):** one boundary triangle with height/edge ≈ 0.02–0.05
  (a ring-1 vertex moved to 2 % of the ring spacing). The floor is used on exactly that edge every
  step (counter 1), the coefficient ratio is exactly 10 (unfloored: 57 / 96 / 132), the solution is
  finite and converges (L2 1.96e-2 / 5.71e-3 / 8.82e-4). On every regular mesh (S3, S3K, S5, S1D:
  a remeshed moving patch with max L/δ_eff 6.9) the floor is never used, so well-shaped triangles
  are unchanged.
- **Honest assessment of the floor.** The implicit Dirichlet term only strengthens the diagonal, so
  the unfloored coefficient caused no instability in any run (the unfloored sliver runs converge,
  L2 7.2e-3 / 1.9e-3 / 5.1e-4). The floor bounds the coefficient as required but weakens the
  coupling on a true sliver (it acts when h_T < 0.3 L_e, e.g. isosceles base angles below ≈ 31°),
  which here raises the sliver-mesh error by a factor 1.7–2.7. It is a conditioning safeguard, not
  an accuracy improvement; remeshing keeps boundary triangles above the threshold in practice.

### 27.3 Reproducible validation (repository files)
All 16E multiflow validation now lives in `multiflow/testcases/FT_Stage16E/` (README): the user
files (diagnostics `f16e_log.txt`, initial fields), `stage16e.py` (mesh generators, the case table
with rank counts, reference solutions and tolerances, a parallel runner and the checker) and
`run_suite.sh` (configure + build + gen + run + check). Groups: `quick` (the regression set,
~10 min on 16 cores), `full` (+ 64-cell hemisphere, 32-ring sliver, the 1/2/4/8 matrix of S1/S6),
`scalar`, `tension`, `closure`, `reject`. The implementation session's scratch is no longer needed.
The library part is `tests/test_stage16e.c` (ctest, np 1/2/4/8; +8 checks: the closure transition).
Result on the final trees: see §27.10.

### 27.4 multiflow build flags (C16E-4, C16E-5)
- **C16E-4.** `CMakeLists.txt` did `set(CMAKE_C_FLAGS ${mypetscccflags})`, discarding every
  externally supplied flag (command line, preset, environment). Now the PETSc flags come first and
  `CMAKE_C_FLAGS` is appended (`set(CMAKE_C_FLAGS "${mypetscccflags} ${CMAKE_C_FLAGS}")`), so
  user flags reach every compile and link line and win over conflicting PETSc defaults; CMake still
  appends `CMAKE_C_FLAGS_<CONFIG>` (Release `-O3 -DNDEBUG` unchanged: with no user flags the compile
  line is the Stage-15 one, A/B bit-identical §27.10). Same for the Fortran flags.
- **C16E-5.** `FT_SCALAR_TRANSPORT` is the CMake option `MF3_FT_SCALAR_TRANSPORT`
  (`target_compile_definitions(MF3 PRIVATE FT_SCALAR_TRANSPORT)`; FATAL_ERROR without MFTL2), set by
  the two scalar-transport example presets and `generate_cmake_presets.sh`. O-1 was confirmed
  (without it `scalar_old` stays 0 and the transport is silently wrong, closed bodies included): a
  run with `SurfaceDiffusionCoefficient` on a build without it now stops at the capability check.
- **16D record corrected.** The 16D reviewer's build cache shows `CMAKE_C_FLAGS=-ffp-contract=off
  -fsanitize=address,undefined …` but compile lines without them: the §25.8 "multiflow Debug +
  ASan/UBSan" runs instrumented only the MFTL2 archive. §25.8 carries the correction.
- **Proof of instrumentation:** configured with `-DCMAKE_C_FLAGS="-fsanitize=address,undefined
  -fno-omit-frame-pointer"` against the instrumented MFTL2 (`ENABLE_SANITIZERS`); the verbose
  compile line of `surfacescalartransport.c` carries `-fsanitize=address,undefined` and
  `-DFT_SCALAR_TRANSPORT`, the link line `-fsanitize=address,undefined`; the multiflow objects
  contain `__asan_report_*`/`__ubsan_handle_*` calls (surfacescalartransport.o 11,
  openboundarymodels.o 12, spreadfromfronttofluid.o 16; the Release control object 0); `ldd MF3`
  lists libasan.so.8 and libubsan.so.1, and a run with `ASAN_OPTIONS=verbosity=1` prints the ASan
  runtime messages. Results §27.10.

### 27.5 Scalar boundary conditions (independent check of the discrete equations)
- Equation: `(A c)^{n+1}/Δt − (A c)^n/Δt + Σ_edges F_out = 0` with the interior two-point flux
  (diagonal `+αDL/d`, off-diagonal `−αDL/d`, over-relaxed non-orthogonal correction in the RHS).
  Boundary edge: NEUMANN `F_out = q L` → RHS `−qL`; DIRICHLET `F_out = D L (c_T − c_D)/δ_eff` →
  diagonal `+DL/δ_eff`, RHS `+DL c_D/δ_eff`. Signs and units consistent (`c·m²/s` per unit length ×
  length) with the interior terms.
- `q > 0` is outward (along ν, out of the surface domain): S2 `M(t) − (0.25 − 0.2 t)` ≤ 4.2e-16
  (dM/dt = −qL, L = 2), np 1 and 2.
- Orientation reversal (normals and loop orientation reversed, same tags and data): ν = t × n is
  unchanged, and EdgePlanar orients ν away from the triangle; S3R equals S3 to 2.8e-15.
- Each boundary edge belongs to exactly one triangle and every sum runs over owned triangles; the
  budget identity holds on 1–8 ranks (S5_16 rank counts equal to ≤ 3.7e-11, the KSP tolerance).
- Zero Neumann: S1 (moving, remeshing every 4 steps 288 → 524 triangles, across the x seam) step
  error ≤ 2.8e-14, drift ≤ 1.6e-13 on 1 and 4 ranks. The drift is measured after the remeshing in
  every step, so the MFTL2 scalar remapping of splits/collapses/flips is conservative.
- Dirichlet budget on a moving, remeshing patch (S1D): step error ≤ 1.2e-13, drift ≤ 6.2e-13.
- Seam: S6 seam-crossing vs centred fields equal to 3.3e-15 (np 1, 4); no boundary contribution is
  lost or duplicated.
- Exactly one condition per loop: an explicit tag overrides `ALL`, a duplicate (body, tag) is an
  input error, a missing condition (R4), a condition on a closed body (R5), on an absent tag (R6)
  and ROBIN (R7) are rejected.
- Convergence: Dirichlet steady rate 3.0 (symmetric and skewed meshes), mixed 1.96/1.99,
  transient vs the Fourier series max error 2.5e-4 (≤ 1e-3).

### 27.6 Pinned surface tension
- Element forces unchanged: `force = Σ_edges σ L ν_e`, ν_e from `EdgePlanar` (edge normal = the sum
  of the adjacent triangle normals; the single triangle's normal on a boundary edge). Interior
  edges cancel exactly between their two triangles (same edge normal, opposite orientation), so
  `Σ_T force_T = Σ_boundary σ L ν` is the non-cancelling line force, and each boundary edge is
  booked once (with the element that owns it) into its (body, tag) support: `R = −Σ σ L ν`.
- Sign: ν is the outward conormal (in the surface, pointing away from its triangle). The fluid
  receives the element forces, whose sum is `+σ Σ L ν`; the membrane pulls its support towards
  itself, i.e. the support receives `R = −σ Σ L ν`, and fluid force + R = 0 to round-off
  (≤ 5e-16 in every case). Hemisphere on the wall: ν points down at the rim, the fluid force is
  `−2πRσ e_z`, `R_z = +2πRσ` (the membrane pulls the ring up).
- Orientation reversal (T5, σ = −1): R, Δp and Σ I V identical (5.5e-18, 0, 8.6e-16).
- MPI ownership: the same R on 1 and 4 ranks (T1 8.9e-16, T2 2.5e-17 relative).
- Seam: T4 hemisphere across the x/y seams vs centred: R 1.1e-15, Δp 1.0e-12, Σ I V 2.6e-9.
- Pinned kinematics: the owned pinned vertices are restored after the time integration (before the
  wrap), and remeshing places new boundary vertices by projection onto the support polyline and
  freezes them. **C16E-6 (found here):** the MFTL2 parallel remesh driver clamped every vertex
  1e-12 (absolute) inside the box, so the rim of a wall-pinned surface ended 1e-12 off its wall
  after every parallel remesh (T6/T7 contact-line deviation exactly 1.0e-12; the saved positions
  then kept it). Within τ_loop = 1e-9 L here, but scale-dependent (a 1e-4 domain would put the
  rim 1e-8 L off the closure and make the set UNDEFINED). Fixed in `remeshing_parallel.c`: a
  declared-open body is clamped into the closed box (as the adapter's own clamp since 16E);
  closed bodies unchanged. After the fix the contact line is exact (T6/T7 1.1e-16; the suite
  tolerance is now 1e-14).
- Weight normalisation: only for triangles of open bodies (`MFAdapterBodyIsOpen`); for an interior
  stencil the cosine kernel is an exact partition of unity (scale 1 to round-off), near a wall the
  force is renormalised onto the existing cells (conservative; fluid force + R ≤ 2.8e-16). Closed
  bodies keep the Stage-15 weights (A/B bit-identical).
- Convergence (T3 hemisphere 16/32/64 cells per unit): R_z −1.80 / −0.45 / −0.11 %, Δp −3.61 /
  −0.97 / −0.25 %, Σ I V vs the wall-closed volume +0.88 / +0.11 / −0.014 %: second order,
  independently reproducing §26.3.

### 27.7 Wall closures
- `PhaseSetMember` / `WallClosure` form the singleton dispersed phase of multiflow with explicit
  σ and closures (§5.1 in multiflow form); declarations are checked (W2–W10: no member, no closure,
  wrong wall, periodic axis, wrong σ (W7 of §3.6.10), winding loops (W10 of §3.6.10), PBox
  adaptation, closure without an open member, volume recovery with a free loop) and every status is
  the library's §3.6 status.
- Wall patches, orientation and multiplicities are the frozen §3.6.2 implementation (16B); the
  indicator uses the consistent Neumann condition `∂I/∂n = G·n` on closure faces (the face then
  contributes nothing), every other wall unchanged.
- Re-classification after every remeshing stops the run on any status other than DEFINED; the
  library transition is now tested (test_stage16e `closure`: DEFINED → UNSUPPORTED when flattened
  (degenerate compact, precedence E2), → UNDEFINED when the rim leaves the wall). Observation: on a
  step without remeshing the set is not re-classified (advection cannot change the class; volume
  recovery checks its own status every call). *[Closed in 16F as C16E-7 (§28.1): the set is now
  re-classified after every advection, remeshing and volume recovery.]*
- Set-level volume recovery (T7, Taylor–Green flow, remeshing every 5 steps): drift 3.1e-11 over 50
  steps, contact line exact.
- **Pinned loops for every wall-closed member:** not stated in §3.6.2 or §5.2, but required by
  them: a wall-closed set needs every loop on its closure to τ_loop, and the only boundary model
  that keeps a loop there is PINNED (CONTACT_LINE is reserved, §12). It is the conservative
  Stage-16 choice because no moving-contact-line law exists; now stated in §12.

### 27.8 IBM/DEM supports
§7.2 describes a PINNED support as "a static wall line, or an IBM/DEM body", but no 16E gate
(§9), no T-E test (§10: E3 is a static ring) and no capability row requires the IBM/DEM variant;
§5.5 keeps IBM bodies closed. It is general/future scope: marked UNSUPPORTED in Stage 16 (§7.2,
§12); `SurfaceTensionBoundary … PINNED IBM|DEM` is rejected with "UNSUPPORTED in Stage 16" (R3).
No IBM/DEM physics was added.

### 27.9 CSF curvature source
multiflow's closed-surface CSF has always taken the curvature from its own IRL paraboloid fit of the
front (VOLUMEFIT/VERTEXFIT), never from MFTL vertex curvature; 16E keeps that fit at the rim with
an open N-ring stencil (`NeighbourhoodOrder ≥ 2`, else rejected, R9) and the IRL fractions. Sign
and convention are those of the closed CSF (the same code); Δp error +4.46 / +2.27 % for 16/32
(implementation run: +0.69 % at 64), the IRL volume vs the wall-closed volume +2.0e-3 / +3.3e-4;
identical on 1 and 4 ranks (1.4e-17); the contact line exact. Both approximations are legitimate;
the existing multiflow formulation is retained (§7.3 annotated). Periodic CSF stays rejected (16D).

### 27.10 Regression (final trees)
| Gate | Result |
|---|---|
| MFTL2 ctest: Release (LAPACK on), Release (LAPACK off), Debug + ASan/UBSan | 85 / 85 each (Stage16A/B/C/E np 1/2/4/8 included; Stage16E 109 checks per rank count, +8 for the closure transition); 0 sanitizer reports; 0 build warnings |
| Adapter unit tests np 1/2/3/4 | 146 / 303 / 463 / 629 checks, 0 failed |
| multiflow A/B vs the accepted 16D final outputs (= Stage 15): FT, FT_Sharp_CSF, FT_Sharp_Integral np 1/4, IBM-MFTL A np 1/4, B np 4, Laplace LAPP0 np 4, marching cubes MC0c np 1/4 | **bit-identical**: 12 of 12 runs, every HDF5 file (`h5diff`), after C16E-1 to C16E-6; 0 build warnings |
| D5 (bubble through the seams, 200 steps, np 8) seam and centred; D6 neck on the seam, corner and centred (np 4); D6W (np 2) | `front_log.txt` and every HDF5 file **identical** to the accepted 16D final runs; D6W stops at the capability re-check as before |
| testcases/FT_Stage16E, group `full` (Release, final trees) | **92 / 92 checks pass** (scalar, pinned tension, wall closures, 27 rejections, 4 allowed combinations; S1/S5/S6 on 1/2/4/8 ranks, periodic seam cases, 64-cell hemisphere) |
| multiflow **genuinely instrumented** ASan/UBSan (np 2): 22 suite cases (moving remeshing scalar patch across the seam with Neumann and Dirichlet, mixed, seam, sliver, skewed Dirichlet, pinned film and free cap without indicator, hemisphere, seam hemisphere, remeshed cap, cap with volume recovery, CSF, wall closures, rejections) and the 16D sanitizer cases SAN_D5S, SAN_D6S3, SAN_D6IS1, SAN_D6W, SAN_OPEN, SAN_OPENX | 0 ASan, 0 UBSan reports in all 28 runs; the checked suite values equal the Release ones |

Pre-existing, not changed: two `-Wformat-overflow` warnings in `printfrontmesh.c`
(`sprintf(name_vtu, name)`, present in HEAD) appear only in the instrumented build.

### 27.11 Observations for 16F (not blocking)
- The pinned-loop reaction of a non-hemispherical cap converges at first order (O-4).
- The Dirichlet floor is a conditioning safeguard that costs accuracy on true slivers (§27.2).
- ~~Wall-closed sets are re-classified after remeshing only (§27.7).~~ Closed in 16F (C16E-7).
- D5/D6 inputs and user files still live in the 16D reviewer's scratch; 16F should version them
  the same way (testcases/FT_Stage16E is the model).
- Nothing blocks 16F.

## 28. 16F — final validation and Stage-16 acceptance (2026-09-25)

**Verdict: 16F ACCEPTED; Stage 16 ACCEPTED / COMPLETE.** 16F was executed as a validation stage:
no model was redesigned and no Stage-16 feature was added. One remaining 16E gap was closed
(C16E-7, §28.1; 16E stays ACCEPTED) and three minimal correctness defects found during the
validation were fixed and retested (F16F-1 to F16F-3, §28.2). Stage 17 is not started. Nothing was
committed or pushed. Reviewer/executor: Claude Code.

### 28.1 C16E-7: the phase set is re-classified after every geometry change
- **Gap.** The 16E review found that a wall-closed set was re-classified only after remeshing. A
  set that was DEFINED must never be assumed to stay DEFINED because no remeshing or topology
  change happened.
- **Fix (multiflow, `generalizedsurfaces.c`).** `CheckFrontPhaseSetValidity(Options, Run, FT,
  after)` classifies the dispersed-phase set whenever a phase-dependent module relies on it
  (wall-closed members; or a periodic front with closed bodies and the indicator, CSF, integral
  surface tension, marching cubes, volume recovery, NOA, generic acceleration or the force
  computation) and stops the run on any other status, naming module, body and reason. It is called
  after the advection (`SolveFronttracking`, before scalar transport, remeshing, volume recovery
  and the spreading / indicator / IRL step), after every remeshing (it replaces the wall-closed-only
  block of `CheckFrontClassTransitions`, now also covering periodic closed sets) and after every
  volume recovery. Read-only; a no-op for closed bodies on a non-periodic front (Stage-15 path).
- **Regressions (testcases/FT_Stage16E):**
  - W11_ADVECTED_TO_UNDEFINED: a pinned hemisphere on its z-min closure, uniform through-flow
    along −z (prescribed on the z boundaries), no remeshing. The run stops at time step 106 with
    "The dispersed-phase set became UNDEFINED after advection (time step 106): degenerate volume",
    followed by the capability messages naming the indicator and the body; the 105 earlier steps
    were DEFINED and step 106 never reached the indicator (its diagnostics are absent).
  - W12_ADVECTED_DEFINED: the same cap in a Taylor–Green flow without remeshing: 100 steps,
    DEFINED after every advection, no false rejection.
  - The library transition is tested in `test_stage16e` (`closure`, 16E review).
- Cost: one classification per step (plus one per remeshing / volume recovery) where needed; D5
  (1 280–3 200 triangles, 8 ranks) 1 066 s vs 1 015 s before (+5 %; §28.12).

### 28.2 Defects found and fixed during 16F
| # | Finding (evidence) | Fix |
|---|---|---|
| F16F-1 | **Unchecked volume / inertia of an open body returned a value in Release builds** (MFTL2). `MFTLObjectVolume` and `MFTLMomentOfInertia` rejected wrapping bodies (abort with a message) but, for a declared-open compact body, relied on the debug-only integrity assertion: a Release build returned a meaningless enclosed volume of an open surface. | Both also abort for declared-open bodies with "the body is an open surface (use MFTLPhaseSetVolume with wall closures)"; the area-based `MFTLSurfaceCenter` stays defined for open surfaces. Internal callers were already guarded by `MFTLObjectHasVolume`; closed bodies unchanged. Tests: Stage16F-unchecked-volume_patch / inertia_patch (and _tube / _ribbon for wrapping bodies). |
| F16F-2 | **The abort message of the unchecked calls could be lost** (MFTL2 + test harness). Only rank 0 printed it while every rank called `MPI_Abort`, and MPICH's launcher can end an aborting job before forwarding its output: the new abort-contract tests failed intermittently under `ctest -j8` (stress test: 2 of 160 single-rank launches lost the message, exit code 1). | Library: the check is collective, so `MPI_Barrier` after the message and before `MPI_Abort`. Tests: the abort cases run as an MPI singleton (no launcher) with the output in a file that is searched (240 of 240 stress runs keep the message; 60 ctest repetitions per configuration pass). |
| F16F-3 | **`run_suite.sh` ignored its group arguments** (multiflow test scripts). The scripts assigned the bash special variable `GROUPS` (the user's group ids; assignments are ignored). | Renamed `SUITE_GROUPS` in both suites. |

The 16F restart test also documents a property (no defect): the native-v2 reader re-establishes
the canonical image gauge (`io.c`), so a state written in the canonical gauge is restored exactly
(counters and positions) on any rank count, and a non-canonical one (e.g. fresh from remeshing)
comes back in its canonical gauge — same positions, class, windings and loops; the cut edges of a
wrapping body may move.

### 28.3 Reproducible validation (A)
Nothing required to reproduce the Stage-16 acceptance remains in a session scratch directory:
- `MFTL2/tests/`: `test_stage16a/b/b_gauge/c/e/f` (ctest, 1/2/4/8 ranks, 1- and 2-slab layouts,
  restart write/read across rank counts, abort contracts), `tools/phase_set_prototype/`.
- `multiflow/testcases/FT_Stage16D/` (new, 16F): the D5/D6/marching-cubes/restart/open-patch case
  files of the accepted 16D review **verbatim**, deterministic generators for every mesh (the
  dumbbell and icosphere meshes and the open patches regenerate the 16D review's files **byte for
  byte**; the capsule of D6W to 2e-15), the per-step invariant diagnostics, the P translation family,
  the restart chains, tolerances and checks (`stage16d.py`, `run_suite.sh`, README), and
  `closed_ab/` (the closed legacy A/B case files and `run_ab.sh`: build, run, h5diff against a
  baseline). Its runs reproduce the accepted 16D final outputs **bit for bit**: D5 seam and
  centred (front_log and every HDF5 file), D6 S1/S3/C1, marching cubes MCPXc/MCP0c (HDF5), the
  restart chains LAPPN/LAPPX and the open patch OPENX.
- `multiflow/testcases/FT_Stage16E/` (16E review, extended in 16F: W11/W12, W13, R3D, L1).
- `multiflow/tools/mftladapter_tests/`.

### 28.4 Invariants (B)
- MFTL2 ctest (97 tests, final trees) in four configurations — Release (LAPACK on), Release (LAPACK
  off), Debug, Debug + ASan/UBSan/LSan: see §28.11. It contains the Stage-16 invariant tests on
  1/2/4/8 ranks and the 1- and 2-slab periodic layouts: twin states LOCAL/REMOTE/BOUNDARY,
  discovery, G1/G2, exactly-once ownership, loops/tags, open/closed class, image counters, winding
  validity, cached wrap flags, continuous/minimum-image coordinates, phase-set statuses, boundary
  constraints, the surface-scalar remapping.
- A2 fuzz at the planned length: `S16_ONLY=fuzz S16_FUZZ_STEPS=1000` (closed sphere and open patch,
  1 000 steps of jitter + remeshing + redistribution, every invariant after every step):
  10 002 checks on each of 1/2/4/8 ranks, 0 failed.
- multiflow, every step of every run of both suites: exactly-once ownership of triangles and
  vertices, twin states including ghosts, wrap-flag / cut-edge consistency (FT_Stage16D
  `front_log`), contact line on its support, wall-closed status, scalar budget (FT_Stage16E):
  0 violations.

### 28.5 Closed-surface legacy regression (C)
- FT, FT_Sharp_CSF, FT_Sharp_Integral (np 1, 4), IBM-MFTL A (np 1, 4), IBM-MFTL B (np 4), Laplace
  LAPP0 (np 4), marching cubes MC0c (np 1, 4) — final trees (after C16E-7 and F16F-1): **12 of 12
  bit-identical** to the accepted 16D final outputs (= Stage 15), every HDF5 file (`h5diff`).
  (F16F-2 and F16F-3 change only an abort path and the test scripts.)
- Remeshing, restart and topology change on the Stage-15 path: the non-periodic-front restart chain
  LAPPNC / LAPPNW / LAPPNR (np 4 → 1/2/8), the D5N non-periodic-front bubble (parity remeshing) and
  MC0c: `front_log` and every HDF5 file bit-identical to the accepted 16D runs.
- `closed_ab/run_ab.sh` reproduces the A/B from repository files (§28.11).
- No closed case could not be bit-identical.

### 28.6 Periodic compact surfaces (D) and topology changes (E)
- **P family** (Laplace droplet D = 0.8 on the x, y, z seams, the xy/xz/yz edges and the xyz corner,
  uniform translation, Frenet + indicator solve, interpolation and spreading, parity remeshing):
  every placement equals the centred run to ≤ 3e-15 in centroid, volume and area, identical
  triangle counts, identical spurious velocity (np 4; P_XYZ also np 1/8).
- **D5** (200 steps through the x seam and the xyz corner and back): seam vs centred on 1/2/4/8 ranks
  |dC|/L ≤ 2.0e-12, dV/V ≤ 4.2e-11, dA/A ≤ 2.9e-11, identical triangle counts, 0 violations; D5N
  (non-periodic front) vs centred 1.5e-11 / 7.5e-10 — the accepted §25.2 values, reproduced exactly.
- **Restart on another rank count:** the periodic corner droplet LAPPX (np 4 → 1/2/8), the open patch
  OPENX, the wrapping tube TUBE: class, counters, winding flags and invariants restored; restarts on
  1/2/8 ranks agree to 0 (area); first restarted step vs continuous ≤ 1.6e-15 (droplet), 2.7e-6
  (tube in a uniform flow: the pre-existing fluid-state restart offset, §25.9).
- **D6** (marching-cubes pinch-off, factor 1): neck on the x seam and across the xyz corner vs
  centred, pole-clustered mesh np 1/4/8 and icosphere mesh np 4: 2 bodies in every placement, end
  volumes ≤ 1.0e-7 (corner) and ≤ 1.4e-8 (neck) of centred (criterion 1e-6), 0 violations
  (reconstruction, breakup, assembly / commit, canonicalisation, re-check of the new bodies).
  **D6W**: the tube created across the seam is stopped at the capability re-check.
- Marching-cubes coalescence of two drops across the x seam vs centred (np 1, 4): same body count,
  volumes ≤ 1.2e-8.
- **Cross-rank coalescence detection** stays a documented pre-existing limitation (§24.4: pairs on
  different ranks are not detected); it is not claimed or tested as supported.

### 28.7 Wrapping / non-contractible surfaces (F)
- `test_stage16f` (np 1/2/4/8): open ribbons winding once along x, y, z, (1,1,0), (1,0,1), (0,1,1)
  and (1,1,1); closed films wrapping xy, yz, zx; a closed tube wrapping x; a compact sphere across the
  xyz corner (stays COMPACT). Wrap flags, cut-edge consistency, boundary loops and their windings,
  twin states and area are invariant under redistribution, a whole-body lattice re-gauge (every
  counter shifted by (2,−1,3), then `MFTLObjectRegaugeImages`), 4 remeshing cycles and the rank
  count; native v2 written on 4 ranks and read on 1/2/8 restores counters and positions exactly
  (canonical gauge), class, windings and loops.
- Checked queries never return a value for a wrapping singleton (ribbons, films: volume and centre
  not DEFINED); the closed tube has a DEFINED set-level volume and an UNDEFINED x-centre (§3.6.10
  C10); the corner sphere's set volume equals `MFTLObjectVolume` to 1e-13.
- The unchecked single-body calls reject what their contract excludes: volume of a wrapping tube,
  inertia of a wrapping ribbon ("the body wraps a periodic direction"), volume and inertia of an open
  patch ("the body is an open surface", F16F-1); the surface centre of an open patch stays defined.
- Merging independently gauged pieces into a compact body: test_stage16b merge (B12); marching-cubes
  assembly across seams: D6 (canonicalisation, no cut edge).

### 28.8 Open-surface remeshing (G)
- test_stage16c (np 1/2/4/8, scales 1e-4 and 1e4): planar patch, perturbed hemisphere (REPRESENTED
  and CONSTRAINED), seam-crossing open patch vs translated, link-condition cases (strip pinching, ear,
  loops one edge apart, triangular hole, corner), S-A4 (boundary edges coarsen) and S-A5 (boundary
  survivor kept bitwise), valence growth, ribbon and film remeshing; the adapter's rank-cut test
  (R16D-2); multiflow OPENX / OPENX06 on 1/2/4/8 ranks (generalized remeshing every 5 steps, 0
  violations) and T6 (pinned rim exact through parallel remeshing, R16D and C16E-6 fixes hold).
- **Limitation (unchanged, erratum E9):** a REPRESENTED boundary curve drifts cumulatively under
  repeated re-sampling (≤ 2.8 % of R after 20 cycles on the hemisphere); CONSTRAINED (exact) is the
  mode for supports and walls.

### 28.9 Surface scalar boundary physics (H)
FT_Stage16E (full group, final binary): zero Neumann on a moving, remeshing patch across the seam
(step error ≤ 3.9e-14, drift ≤ 4.6e-13 on 1/2/4/8 ranks); Dirichlet on the same patch (step budget
1.2e-13, floor never used, max L/δ_eff 6.9); Neumann flux exact (4.2e-16); Dirichlet steady rate
3.0 on symmetric and skewed meshes; transient ≤ 2.5e-4; mixed rate 1.96/1.99; seam = centred to
3.9e-15 on 1/2/4/8 ranks; orientation reversal 2.8e-15; the sliver floor acts on exactly the one
sliver edge (coefficient 10 D) and never on the regular or remeshed meshes.

### 28.10 Pinned surface tension (I) and wall closures (J)
- Frenet without an indicator (single-phase fluid): pinned film (budget ≤ 1.6e-16, R_z(0) vs analytic
  +6e-4, relaxing), free cap (first order in the rim edges, rate 1.10/1.05); hemisphere on a wall
  (R_z −1.80 / −0.45 / −0.11 %, Δp −3.61 / −0.97 / −0.25 % for 16/32/64); reversed orientation and
  seam crossing identical; spread = element force to round-off; the same R on 1 and 4 ranks.
- Wall closures: DEFINED phase, indicator, volume recovery (drift 3.1e-11), CSF/IRL (Δp decreasing,
  ranks identical), re-classification after remeshing (T6 DEFINED every step) and after pure
  advection (W11 stops, W12 continues), invalid/missing closures (W2–W9), a free loop on a wall
  closure (W10 volume recovery, W13 indicator), winding wall loops (W7), wall-closed restart on
  1/2 ranks (L1: first restarted step equal to the continuous run).

### 28.11 Restart / MPI / sanitizers (L, M) and the gate
| Gate | Result |
|---|---|
| MFTL2 ctest, 97 tests (incl. Stage16F-np1/2/4/8, restart write np 4 → read np 1/2/8, 4 abort contracts): Release (LAPACK on), Release (LAPACK off), Debug, Debug + ASan/UBSan/LSan | **97 / 97 in each**; 0 sanitizer reports; 0 build warnings |
| A2 fuzz, 1 000 steps, np 1/2/4/8 | 10 002 checks each, 0 failed |
| Adapter unit tests np 1/2/3/4 | 146 / 303 / 463 / 629 checks, 0 failed |
| multiflow closed A/B (12 runs) | bit-identical to 16D / Stage 15 |
| FT_Stage16D quick + full + reject (Release) | all checks pass (D5 np 1/2/4/8, D5N, D6 both meshes, P family, marching cubes, restart chains, open patches, 15 rejection/allowed rows); bit-identical to the accepted 16D outputs where they exist |
| FT_Stage16E full (Release) | 99 / 99 |
| multiflow ASan/UBSan (Debug, `-fsanitize=address,undefined` on every compile and link line, sanitizer calls in the multiflow objects, libasan/libubsan linked; MFTL2 archive instrumented): FT_Stage16D quick incl. a 10-step D5 pair, all rejections (43 runs), FT_Stage16E quick (68 runs), all on 2 ranks | 0 ASan, 0 UBSan reports; 43/43 and 76/76 checks pass |
| LSan | active in the MFTL2 ASan ctest (MPI suppressions, `tests/lsan.supp`); multiflow runs use `detect_leaks=0` (MPI/PETSc runtime allocations at exit are not attributable) |
| `closed_ab/run_ab.sh` from repository files (fresh builds, final library) vs the accepted outputs | **12 / 12 bit-identical** (every HDF5 file); an open IBM mesh is rejected at the read (§5.5) |

Build warnings: 0 in every MFTL2 configuration and every Release multiflow build. The two
pre-existing `-Wformat-overflow` warnings in `printfrontmesh.c` (`sprintf(name_vtu, name)`, present
in HEAD) appear only in the instrumented multiflow builds; recorded, not changed (Stage 17).

### 28.12 Performance sanity (N) — not Stage 18
- No new path has super-linear cost in the number of triangles; the per-step additions are O(N)
  local work plus collectives.
- The C16E-7 re-classification (phase-set triangle replication, one `Allgatherv` of the members'
  triangles per classification) costs about 5 % on D5 (1 066 s vs 1 015 s, the same 8-rank pair
  under the same load); it runs only where a phase-dependent module relies on the set.
- Known compromises (documented, not optimised here; Stage-18 candidates): rank-0 image
  canonicalisation at assembly events (O(E) gather); phase-set triangle replication on every rank;
  boundary loops by `Allgatherv` of O(boundary) records; the always-executed global fallback of the
  discovery protocol; the pinned-support and boundary-tag `Allgather`s at set-up; the §11 targets
  (periodic ≤ 10 %, open ≤ 5 %) are not measured systematically.


### 28.13 Final capability-matrix audit (§5.2 against the code)
Every row was read in `generalizedsurfaces.c` (`FTCheckCapabilities`, `FTOpenModuleRule`,
`FTCheckBoundaryDeclarations`, `CheckFrontPhaseSetValidity`) and the adapter guards, and exercised
by at least one run; a rejection names the module, the body and its class, and the missing
condition, before the first step (the single-fluid guard of Frenet without the indicator is a run
property and names module and phase instead of a body).

| Row | Closed P0 | Closed periodic | Open | Code = matrix? / evidence |
|---|---|---|---|---|
| Advection | yes | yes | yes; PINNED loops held on their support | yes: D5, P family, OPENX; pinned contact line ≤ 1e-16 (T1–T7) |
| Spreading / interpolation | yes | yes (periodic stencil) | yes (weights normalised at walls) | yes: P family (1e-15), D3/D4 (16D), T1 budget 1e-16 |
| Frenet surface tension | yes | yes (compact; a wrapping body only surface-wise) | only with PINNED STATIC on every loop; no indicator needed (single phase) | yes: A/B, D5, T1/T2/T3, R1/R17 (no model), R2/R3/R3D (CONTACT_LINE, IBM, DEM), R16/R_T_ST (unequal phases without indicator) |
| CSF / sharp CSF | yes | **stricter**: UNSUPPORTED on periodic fronts (IRL paths not converted, 16D) | PINNED + wall closure + non-periodic + order ≥ 2 | yes: A/B, R_P_CSF, R10, R9, R11, T8, A3 |
| Integral surface tension | yes | **stricter**: UNSUPPORTED on periodic fronts | REJECT | yes: A/B, R_P_INT, R8 |
| Indicator (Poisson) / IRL fractions | yes | DEFINED compact sets; **stricter**: wrapping bodies rejected even when DEFINED, IRL UNSUPPORTED periodic | wall-closed member with pinned loops | yes: D5, R_T_IND, R_P_IRL, W2/W3/W13, T3; re-classified after advection / remeshing / recovery (C16E-7: W11/W12) |
| Marching cubes, breakup / coalescence | yes | DEFINED sets; factor 1 only; capability re-check of new bodies | REJECT | yes: MC0c A/B, D6, D6W, MCPXc, R13; cross-rank coalescence detection: pre-existing limitation (§24.4), documented, not supported |
| Volume recovery | yes | set DEFINED (iterated after periodic MC) | wall-closed, pinned | yes: T7, W10, R14 |
| Surface scalar transport | yes | yes (16E); overlap-mesh remapping (VCS, MC) UNSUPPORTED periodic | NEUMANN/DIRICHLET per loop; ROBIN reserved | yes: S1–S7, R4–R7, R15; build without FT_SCALAR_TRANSPORT rejected |
| Surface viscosity | yes | **stricter**: UNSUPPORTED periodic | REJECT | yes: R_P_VISC, R12 |
| NOA advection, generic acceleration, force computation | yes | **stricter**: UNSUPPORTED periodic | REJECT (need the indicator of a closed front) | yes: R_P_NOA/GACC/FORC, R_O_NOA/GACC/FORC |
| Divergence-free interpolation | yes | **stricter**: UNSUPPORTED periodic | allowed | yes: R_P_DIVF, R_O_DIVF |
| Remeshing | parity | parity default; generalized by opt-in | generalized | yes: D9 (16D), test_stage16c/f, A/B bit-identical |
| IBM markers (static, GTS) | yes | not applicable: IBM decompositions stay non-periodic | open IBM mesh rejected at the read (§5.5; `IBMMassProperties` not implemented) | yes: IBMA/IBMB A/B; closed_ab IBMOPEN |
| MoveIBM | yes | not applicable | REJECT (never open) | yes (IBM bodies never open) |
| Point-inside queries | yes | DEFINED sets | REJECT (`PointQueryCreate` requires a volume) | yes: adapter tests |

"Stricter" rows are the implementation limits recorded in 16D (§24.5): the matrix allows them in
principle, this implementation rejects them explicitly (never an approximation). No row is laxer
than the matrix.


### 28.14 Requirement classification
Legend: **IT** = IMPLEMENTED + TESTED (evidence), **EU** = EXPLICITLY UNSUPPORTED (rejected before
stepping, never approximated), **DEF** = DEFERRED (named later stage and reason).

| Requirement (plans/16) | Status | Evidence / reason |
|---|---|---|
| §2 three twin states, predicates, `twin_gid` reader audit, CI grep | IT | test_stage16a (A1, A7), TwinGidReaderAudit ctest |
| §2.2 operator maintenance, copy/commit ASSEMBLE/CUT | IT | test_stage16a copy_commit, boundary_operators |
| §2.3 quiescent validation | IT | every Stage-16 test; multiflow per-step `twin` column (D5, P, D6, restarts: 0 violations) |
| §2.4 discovery (global/owned modes, stale positions, dead gids) | IT | test_stage16a discovery (A3), soup_before_ownership (A6) |
| §2.5 ghost completion G1/G2 | IT | test_stage16a; A2 fuzz 1 000 steps np 1/2/4/8 (10 002 checks each) |
| §2.6 tags, loops, manifoldness, builder rejections | IT | test_stage16a (A4), test_stage16c, test_stage16f |
| §3.1–3.2 image counters, displacement rule, lifecycle, precision | IT | test_stage16b lifecycle (B8), large_counters (B10) |
| §3.3–3.4 homological winding, canonicalisation, per-axis flags, diagonal winding | IT | test_stage16b ribbons/film/merge (B3, B4, B12, B13), test_stage16f (x, y, z, (1,1,0), (1,0,1), (0,1,1), (1,1,1) ribbons; xy/yz/zx films; tube; corner sphere) under redistribution, lattice re-gauge, remeshing, rank count and restart |
| §3.5 reference-point evaluation, no extent restriction | IT | test_stage16b (B2, B10), seam sphere |
| §3.6 phase sets (every case of §3.6.10 + review cases) | IT | test_stage16b model cases on 1/2/4/8 ranks and x/y rotations, test_stage16b_gauge |
| §3.6 statuses UNDEFINED / UNSUPPORTED returned, never a value | IT | B11/B14–B16, test_stage16f checked queries, unchecked-call abort contracts |
| §4.1 local-frame kernels, boundary geometry | IT | test_stage16c, test_stage16e |
| §4.2 curvature flags, extrapolated curvature, turning angle, Gauss–Bonnet | IT | test_stage16c gb (C2) |
| §4.3 boundary curve REPRESENTED / CONSTRAINED, corners | IT (limitation) | test_stage16c cycles (C4), 16E T6 (constrained, exact); REPRESENTED drifts cumulatively (erratum E9, §23): limitation, CONSTRAINED is exact |
| §4.4 augmented link condition, boundary split/flip/smoothing, serial + parallel, S-A4/S-A5, valence growth | IT | test_stage16c link (C3), sa5, valence_growth, seam (C5), scales (C6), rank independence (C7); adapter rank-cut test (R16D-2) |
| §4.4.4 remesh-mode selection, class-guarded recovery | IT | test_stage16c vcs_and_class, film; D9 |
| §5.1 declarations (`PeriodicFronts`, `OpenSurface`, `BodyMesh`, `RemeshMode`, `PhaseSetMember`, `WallClosure`) | IT | 16D/16E suites |
| §5.1 explicit boundary-edge / loop lists | library IT; multiflow input EU | `MFTL_BoundaryDecl` explicit edge lists with tags: test_stage16a (A6); multiflow provides only `OpenSurface <body> BoundaryFromMesh` (§24.5) and no input syntax for explicit lists; no gate needs it |
| §5.2 capability matrix | IT | §28.13 |
| §5.3 adapter class checks, accessors | IT | adapter tests (146/303/463/629 checks) |
| §5.4 periodic FTDomain, stencils, ghosts, output, restart v2 | IT | P family, D5, restart chains, test_stage16b/f restart |
| §5.5 IBM: open IBM bodies only with force coupling or explicit mass properties | EU | open IBM mesh rejected at the read; `IBMMassProperties` not implemented; IBM decompositions non-periodic |
| §5.6 raw-position conversion | IT for every module allowed on periodic fronts; the others EU (§24.5) | P family, D5 |
| §6 scalar BCs (NEUMANN, DIRICHLET with the corrected normal distance and sliver floor) | IT | FT_Stage16E S1–S7, S1D, S3K, 1/2/4/8 ranks |
| §6 ROBIN | EU (reserved) | R7 |
| §7.2 PINNED STATIC, reaction, momentum budget | IT | T1–T7 |
| §7.2 PINNED on IBM/DEM supports | EU (Stage 16) | R3, R3D (§27.8) |
| §7.2 CONTACT_LINE | EU (reserved; later stage) | R2 |
| §7.3 CSF on a wall closure | IT (IRL curvature, §27.9) | T8, A3 |
| §8 boundary / seam / rim distinction | IT | test_stage16a/c/e (seam-crossing open patch, loops across ranks) |
| §10 A1–A7 | IT | test_stage16a (A2 fuzz also at 1 000 steps), test_stage16b restart (A5), test_stage16f restart |
| §10 B1–B16 | IT | test_stage16b/b_gauge/c/f |
| §10 C1–C7 | IT | test_stage16c |
| §10 D1 capability matrix | IT | §28.13 (all rejection rows) |
| §10 D2 mixed open + closed | IT | adapter test D2 |
| §10 D3 seam-crossing spreading | IT | P family (x, y, z, xy, xz, yz, xyz: 1e-15), 16D D3 |
| §10 D4 droplet across the xyz corner vs centred (static and oscillating) | IT (static, translating); oscillating variant not run separately | 16D D3/D4 (static, 20 steps), P_XYZ translating droplet 40 steps identical to 1e-15; an oscillation would be compared on the same identical fields |
| §10 D5 | IT | FT_Stage16D D5 (bit-identical to the accepted 16D run), full: 1/2/4/8 ranks, D5N |
| §10 D6 | IT | FT_Stage16D D6 S1/S3/C1 (bit-identical to 16D), icosphere meshes (full), D6W |
| §10 D7 LeVeque open patch | IT | adapter test (16D) |
| §10 D8 restart on another rank count | IT | FT_Stage16D LAPPX, LAPPN, OPENX, TUBE chains (np 4 → 1/2/8); FT_Stage16E L1 (wall-closed, np 4 → 1/2); test_stage16b/f |
| §10 D9 remesh-mode selection | IT | adapter test, 16D |
| §10 E1–E4 | IT | FT_Stage16E |
| §11 performance targets (periodic ≤ 10 %, open ≤ 5 %, closed unchanged) | closed unchanged: IT (A/B bit-identical, same code path); the others DEF → Stage 18 (performance) | sanity measurements §28.12; the documented collective paths are Stage-18 candidates |
| §12 list (contact lines, free liquid edges, open-surface viscosity / integral ST, UNDEFINED/UNSUPPORTED phase quantities, open mass properties, non-manifold / self-intersecting meshes, `mftl1_criteria` on open bodies) | EU | R-rows of both suites; builder rejections (A4); embeddedness assumed, not checked (§3.6.1) |
| 16D implementation limits (CSF/integral/IRL/viscosity/NOA/GACC/forces/div-free on periodic fronts; wrapping closed bodies with phase modules; periodic MC factor > 1; scalar remapping through the overlap mesh on periodic fronts) | EU | R_P_*, R_T_IND, R15; §24.5, R16D-4 |
| Cross-rank coalescence detection | pre-existing limitation, not supported (§24.4) | documented; candidate for a later stage (not Stage-16 scope) |
| Documentation for the manual (Stage 17) | DEF → Stage 17 | this plan, the suite READMEs, the adapter header |


### 28.15 Stage-16 final summary
**Supported surface classes.** One surface model for closed and open triangulated bodies: closed
bodies (non-periodic, periodic compact, periodic wrapping) and open bodies with genuine, tagged
boundary loops, including their combination (an open surface crossing seams, split over ranks).
Every half-edge is LOCAL, REMOTE (MPI rim) or BOUNDARY (tagged); a boundary, a seam or a rim is
never inferred from coordinates; open bodies are declared (`OpenSurface … BoundaryFromMesh`).

**Periodicity.** Per-vertex image counters with a displacement rule, local-frame kernels, minimum-
image geometry, no object-extent restriction, reference-point evaluation (precision with counters
≈ 1e6), periodic spreading / interpolation / indicator / marching cubes in multiflow
(`PeriodicFronts`), translation-exact to round-off through every seam, edge and corner.

**Wrapping topology.** Homological winding from cycle sums of edge lattice shifts, canonicalisation
after every assembly event, cached per-axis wrap flags; ribbons, films, tubes and diagonal
windings keep their class through redistribution, remeshing, re-gauging, restart and rank count.
Closed-phase quantities come from phase-boundary sets with the three statuses DEFINED / UNDEFINED /
UNSUPPORTED (Φ(S) = 0, consistent orientation, wall closures, canonical relative gauge); the old
single-body calls abort outside their contract.

**Remeshing.** Closed bodies keep the Stage-15 parity remesher (bit-identical); open bodies (and
closed bodies by opt-in) use the generalized remesher with the augmented link condition, boundary
curve modes (REPRESENTED, CONSTRAINED), corners, boundary split/collapse/flip/smoothing rules,
serial and parallel (two-pass shift scheme).

**multiflow integration.** Explicit declarations, an enforced capability matrix (checked at start-up,
after topology changes, and — for sets that phase-dependent modules rely on — after every advection,
remeshing and volume recovery), class-checked adapter queries, periodic infrastructure, native v2
restart on any rank count.

**Open-surface physics.** Surface-scalar NEUMANN / DIRICHLET conditions per loop (conservative,
second/third-order Dirichlet convergence on arbitrary boundary triangles, sliver floor 0.1 L_e);
PINNED STATIC surface tension with the support reaction and an exact momentum budget, with or
without a phase indicator (single-phase interpretation); wall-closed phase regions with the Poisson
indicator, volume recovery and CSF / IRL.

**Explicit limitations (all rejected before stepping, never approximated).** Contact-line
dynamics (CONTACT_LINE reserved) and hence free loops on wall closures; ROBIN; PINNED on IBM/DEM
supports; free liquid edges; surface viscosity, integral surface tension, marching cubes, NOA,
generic acceleration and force computation on open surfaces; mass properties of open surfaces and
open IBM bodies; CSF, integral surface tension, IRL, surface viscosity, NOA, generic acceleration,
force computation and divergence-free interpolation on periodic fronts; phase-dependent modules on
closed bodies that wrap a periodic direction; periodic marching cubes with a resolution factor > 1;
surface-scalar remapping through the overlap mesh on periodic fronts; every UNDEFINED / UNSUPPORTED
phase-set case of §3.6.9; non-manifold / non-orientable meshes (self-intersection assumed absent).
Known limitations that are not rejections: cumulative drift of a REPRESENTED boundary curve
(E9), first-order convergence of the pinned reaction for a non-hemispherical cap (O-4), cross-rank
coalescence detection (pre-existing), the fluid-state restart offset (§25.9).

**Validation evidence.** MFTL2 ctest (97 tests × 4 configurations incl. ASan/UBSan/LSan), the 1 000-step
invariant fuzz on 1/2/4/8 ranks, the adapter tests, the versioned multiflow suites
`testcases/FT_Stage16D` and `testcases/FT_Stage16E` (Release and genuinely instrumented), and the
closed legacy A/B, bit-identical to Stage 15.

**Deferred:** the manual and input documentation (Stage 17); performance work on the documented
collective paths and the §11 targets (Stage 18).
