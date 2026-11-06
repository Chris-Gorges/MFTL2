# Remeshing a front under the LeVeque deformation: setup, methods and results

Notes for a paper. Every simulation and comparison made for the remeshing work in MFTL2
(October–November 2026), with the setup needed to describe it and the numbers needed to
cite it. `REPORT.md` is the figure-rich study report; `docs/analysis.md` §14 records the
design decisions; this file collects both, plus the follow-up runs, in paper order.

**Status of the code.** The library has two sizing modes, uniform and
curvature-adaptive, with MFTL's sweep order, and two options measured in section 9:
paraboloid placement of split and collapse vertices (`split_placement` and
`collapse_placement`, separate switches, both midpoint by default) and the Phong
back-projection of smoothed vertices onto the kernel call's input (`projection`, once
per call after the last smoothing sweep). A third, the volume-preserving collapse
(`collapse_preserves_volume`, off by default), is measured in section 10.6. Section 10
also measures two corrections that live outside the remesher: the Newton volume
correction and TSUR3D undulation removal.
Everything else measured here (Jiao et al.'s anisotropic mode, quadric-error placement,
Jiao's sweep order, length-sorted sweeps, the experiment's other back-projections, Phong
midpoints, Delaunay flips) was removed or never merged. Every run can still be reproduced
from the patches in `archive/` (section 13).

**Conventions.** "± sd" is the standard deviation over 4 runs (the base run and three
rotated replicas, section 1). Rows without ± are single runs. Percentages of volume are
`V/V(0) − 1`. "Refinement r" is the icosphere level of the initial mesh.

---

## 1. Test problem

**Flow.** The 3D deformation field of LeVeque (1996) in the form of Enright et al. (2002),
on the unit cube:

```
u(x, t) = cos(πt/T) · ( 2 sin²(πx) sin(2πy) sin(2πz),
                        − sin²(πy) sin(2πx) sin(2πz),
                        − sin²(πz) sin(2πx) sin(2πy) )
```

with T = 3. The flow stretches the sphere into a thin sheet up to t = T/2 and reverses
exactly, so the exact solution at t = T is the initial sphere.

**Interface.** Sphere of radius R0 = 0.15 centred at (0.35, 0.35, 0.35), triangulated
by icosahedron subdivision:

| refinement | 3 | 4 | 5 |
|---|---|---|---|
| initial triangles | 1280 | 5120 | 20480 |
| initial vertices | 642 | 2562 | 10242 |

(Jiao et al. 2010 used 46 472 initial triangles.)

**Time integration.** Classical RK4 on the vertex positions with the exact velocity,
dt = 0.015 (Jiao et al.'s step; 200 steps; dt divides T/4). One MPI rank. No flow solver:
the velocity is analytic, so every error measured is the remesher's (section 4).

**Replicas.** Each configuration of the main study also runs on the initial sphere
rotated about its centre by 7°, 13° and 29°. The exact solution is the same; the
alignment of the mesh with the flow is not. The spread over the 4 runs is the run-to-run
noise of the remeshing decisions.

**Remeshing schedule.** Every step, unless stated otherwise (Jiao's protocol, section 6.2).

---

## 2. The remesher common to all variants

MFTL2's isotropic remesher (Botsch & Kobbelt 2004 structure). One call of
`MFTLRemeshObject` performs up to `iterations` = 3 iterations, each one:

1. **DETECT** features (disabled here: feature angle 180°, the right setting for a smooth
   closed interface; see `docs/analysis.md` §4).
2. **SPLIT** every edge the sizing calls too long, at its midpoint (pool order, each edge
   once per sweep).
3. **COLLAPSE** every edge the sizing calls too short, the origin into the target, the
   survivor moved to the edge midpoint. Approval: link condition, valence in [3, max] and
   wing valence ≥ 4, a stretch clause (no new edge the split sweep would split), and a
   fold-over clause (no surviving face turns by more than about 78°, cos = 0.2; relative
   degeneracy test).
4. **COMPACT** (remove dead entities, restore the ownership prefix).
5. **FLIP** every edge whose flip strictly lowers the squared valence deviation of its
   four vertices (target valence 6), refused for a non-convex quad (same 78° test).
6. **DETECT** again, then **SMOOTH**: 3 Jacobi sweeps of tangential relaxation. Each free
   vertex moves towards the area-weighted centroid of its incident triangles' centroids,
   minus the component along the area-weighted vertex normal. Step scale 1.0 (full Jacobi
   step), capped at 0.4 of the shortest incident edge. Normals recomputed every sweep.

The iteration stops early if it performs no operation. The target length L is the
initial mesh's mean edge length; the uniform band is [4/5 L, 4/3 L].

**Volume behaviour of the operators**, needed for the follow-up. A midpoint split is
exactly volume-neutral, because the new vertex lies on the edge. A collapse to the
midpoint replaces two surface points by one chord point. On a convex surface it loses
volume. A valence flip away from the convex diagonal cuts off the tetrahedron spanned by
the quad and loses volume too. A move perpendicular to the area-weighted vertex normal
is volume-neutral to first order. To second order it gains volume on a convex surface,
because the tangent plane lies outside it.

---

## 3. The variants compared

### 3.1 Sizing modes

**Uniform** (Botsch & Kobbelt 2004). Split above 4/3 L, collapse below 4/5 L, stretch
clause at 4/3 L.

**Curvature-adaptive** (Dunyach et al. 2013), as implemented:

- Per-vertex target `L(v) = sqrt(6ε/κ − 3ε²)`, clamped to [L_min, L_max]; edge target
  min(L(a), L(b)); split above 4/3 L(e), collapse below 4/5 L(e), stretch at 4/3 of the
  merged and ring targets.
- κ = H + sqrt(max(H² − K, 0)) with Meyer et al.'s operators: cotangent Laplacian
  H = ½‖Δx‖, angle defect K = (2π − Σθ)/A_mixed, mixed Voronoi areas with the obtuse cases.
- One Jacobi smoothing pass of the curvature over the one-ring before the target is
  taken. This is in pmp-library, not in the paper; it is needed for a fixed point
  (section 8.1).
- The field is computed once per remesh call and **carried**: a split interpolates it to
  the new vertex, a collapse moves it with the survivor, compaction permutes it.
- Feature and rim vertices take the mean target of their non-feature neighbours.
- Smoothing target `Σ_t |t| ρ_t b_t / Σ_t |t| ρ_t` with the ODT density
  ρ_t = 1/L̄_t² (L̄_t the mean corner target). This is a deliberate deviation from the
  printed Eq. 6 (weight |t| L(b_t)), which pulls vertices towards the coarse side.
- Parameters in this study: ε = 0.02 L, L_min = L/4, L_max = L.

**Anisotropic** (Jiao et al. 2010), as implemented (removed after the study):

- Vertex quadric A_v = Σ_f A_f n_f n_fᵀ; eigenvalues λ1 ≥ λ2 ≥ λ3 by deterministic Jacobi;
  metric diag(s(λ2), s(λ3)) in the frame (e2, e3), s(λ) = min(max(λ/λ1, ψ_l), ψ_u),
  ψ_l = 0.005, ψ_u = 0.07 (aspect ratio ≤ sqrt(ψ_u/ψ_l) = 3.74).
- Split (§5.1): absolute longness (longer than L = 1.5 l and the longest edge of its
  triangles) OR relative longness (longer than l, opposite an angle above θ_l = 145°, no
  edge of its triangles under s = l·sqrt(ψ_l/ψ_u)). Desired length l = L.
- Collapse (§5.2): an angle below θ_s = 15° in a triangle smaller than l; shorter than
  r = 0.25 times the longest edge of its triangles; the shortest edge of triangles all
  below S = R·l (R = 0.45); or, in triangles below l, shorter than R times the longest edge
  in physical AND normalised space. Merit clauses as in section 2, stretch at L.
- Flip (§4.3): Delaunay in the normalised space of the four vertices' summed quadrics,
  sin(α+β) < −10⁻⁴; refused when a face normal turns by more than 45° (the paper's
  limit), and when a new triangle would face against the triangle across a quad rim edge.
- Smoothing (§4.2): each face's centroid offset mapped by T sqrt(M) Tᵀ, averaged with the
  face areas over Σ|t|σ2 (the larger scale), so no direction's gain exceeds one; face
  quadric weights 1 / 10⁻⁶ / 10⁻¹² for free / ridge / corner vertices; a step that would
  invert an incident face is halved, up to four times.
- Placement: quadric-error placement (below), on by default in this mode.
- Kept from MFTL: the sweep order, the uniform curve smoothing of ridges and rims, the
  vertex-normal tangent projection.

### 3.2 Vertex placement for split and collapse

- **Chord midpoint**: the default (all modes except anisotropic).
- **Quadric-error placement (QEM)**: the midpoint moved along the edge normal to the
  minimum of the quadric error of both endpoints' faces,
  t* = −Σ w_f (n_f·n̄)(n_f·(m − x_f)) / Σ w_f (n_f·n̄)², capped at a quarter of the edge;
  a split falls back to the midpoint if the point would fold a new triangle (Jiao §5.1–5.2,
  after Garland & Heckbert 1997). Library option `qem_placement`; tested on/off in every
  mode.
- **Phong midpoint** (follow-up experiment): the point of the Phong curve over the edge
  midpoint, `m − ¼[((m−a)·n_a) n_a + ((m−b)·n_b) n_b]` with the current area-weighted
  vertex normals (Phong tessellation with α = ½ restricted to the edge). Used for both
  split and collapse; a split falls back to the midpoint if it would fold.

### 3.3 Sweep orders (main study, section 6)

- **MFTL**: DETECT, SPLIT, COLLAPSE, COMPACT, FLIP, DETECT, SMOOTH (section 2).
- **Jiao (§5.4)**: DETECT, SMOOTH, COLLAPSE, COMPACT, FLIP, DETECT, SPLIT, COMPACT, FLIP,
  DETECT, SMOOTH, with the operator marks cleared between stages. It smooths twice per
  iteration, so a control with equal smoothing work was also run (MFTL with 2 sweeps
  against Jiao with 2 × 1).
- **Length-sorted**: split candidates longest first, collapse candidates shortest first
  (length over target under curvature sizing), each re-approved when its turn comes.
- **Jiao's protocol** (refinement 5): full adaptation every 4 steps, smoothing every
  step, Jiao order, sorted sweeps. Their denoising term is not reproduced: the paper does
  not specify it (§6.2.2: "a smoothing term similar to that in Fig. 4", which refers to
  Ohtake et al. 2002).

### 3.4 Follow-up experiments (section 7)

Instrumented build (`followup_experiment.patch`), uniform and curvature sizing unless
stated:

- **Volume budget**: the enclosed volume before and after each sweep of every remesh
  call, summed over the period. The remainder of the total change is advection.
- **QEM placement** on in uniform and curvature sizing, off in the anisotropic mode.
- **Delaunay flips** instead of valence flips: Jiao's criterion with an isotropic metric
  (ψ_l = ψ_u, the quad's tangent plane), or with the 3D angles opposite the edge
  (sin(α+β) < −10⁻⁴ from unnormalised dot and cross products).
- **Projection after smoothing** (split and collapse unchanged, midpoints). After each
  vertex's tangential move to q, within the Jacobi sweep:
  - *flat, stored input*: a copy of the mesh as handed to this remesh call, binned in a
    uniform grid; q moves to its closest point on the input's flat triangles. This is
    Botsch & Kobbelt's back-projection with the call's input as the reference.
  - *Phong, stored input*: the same closest point p (triangle with corners x_i,
    barycentrics u_i, the input's angle-weighted vertex normals n_i), then lifted onto
    the Phong surface (Boubekeur & Alexa 2008, α = ½):
    `p ← p − ½ Σ_i u_i ((p − x_i)·n_i) n_i`. The reference is fixed for the whole call
    (3 iterations × 3 sweeps).
  - *Phong, local fan (no stored input)*: the closest point on the flat triangles of the
    vertex's own current fan, then the same Phong lift with the current area-weighted
    normals. The reference is rebuilt from the current mesh at every sweep.
- **Phong midpoints** for split and collapse (section 3.2), alone and with the local fan
  projection ("local Phong").

**Superseded runs.** The first local-Phong runs (`*+localphong-flatpre_r4`) used a
patch in which `MFTL_EXP_PROJECT=3` also built the stored input and projected onto its
flat triangles before the fan's Phong lift. They are kept in `results/`, but no
conclusion rests on them.

### 3.5 The library's placement and projection (section 9)

**Paraboloid placement** (`MFTL_RemeshParameters::split_placement` and `::collapse_placement`,
`src/remeshing/remesh_fit.c`; the section-9 runs set both), for a split, and for a
collapse between two free vertices:

- frame at the minimum-image edge midpoint m; z = normalised sum of the endpoints'
  vertex normals from Max's weights, Σ (e1 × e2)/(|e1|²|e2|²); x = the edge projected
  into the tangent plane; y = z × x;
- samples: the vertices of the edge's two-ring (20–30); coordinates divided by h, the
  mean length of the edges at both endpoints; weight Wendland C2, (1 − q)⁴(1 + 4q),
  q = |x − m| / (ρh), ρ = `fit_radius` (3 or 4 here); a sample whose normal is more than
  60° from z is dropped;
- model z = c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y²; rows √w (1, x, y, x², xy, y²) = √w z,
  and, with normals, √w (0, 1, 0, 2x, y, 0) = √w (−nx/nz), √w (0, 0, 1, 0, x, 2y) = √w (−ny/nz)
  for owned samples with a complete fan off any feature (a ghost sample contributes its
  position only);
- solved by Householder QR with grouped column pivoting ({1, x, y} then {x², xy, y²}),
  relative rank tolerance 10⁻³; dropped curvature columns get coefficient 0;
- new vertex m + h c0 z; back to the midpoint if c0 is undetermined, fewer than 6
  samples remain, |h c0| exceeds a quarter of the edge, or (split) a new triangle would
  turn past the fold threshold;
- a collapse is fitted only if its midpoint passes the stretch and fold clauses, then
  judged again at the fitted point; its execution reuses the approval's fit.

**Phong back-projection** (`::projection` = PHONG, `src/remeshing/remesh_project.c`).
This is the GLOBAL-reference form: the reference is a copy of the whole mesh as the
remesh kernel call received it, i.e. the advected front, not the local fan of section
7 (which rebuilt its surface from the current mesh and lost volume). Only the search is
local. Once per call (after its first feature detection) the mesh is copied: positions,
angle-weighted normals, corners, edge neighbours, sorted vertex global ids. After the
call's LAST smoothing sweep (last sweep of the last iteration) every moved free vertex is
located in the copy (its own global id, or a neighbour's for a vertex a split created,
then a walk over edge neighbours), projected onto the closest point of the flat copy and
lifted by −½ Σ u_i ((p − x_i)·n_i) n_i. Feature, corner, boundary and ghost corners get no
lift. A projection bridging more than half the vertex's shortest edge is not made. The
first version projected after every sweep (9 per call here); section 9.5 compares the
schedules.

**Reference geometry for the Phong surface.** On a sphere with exact vertex normals,
Phong tessellation with α = ½ is fourth-order accurate. Measured distance to the sphere:
4.7e-5 → 3.05e-6 → 1.9e-7 as h halves, against second order for the flat interpolant.

---

## 4. Exact reference and metrics

**Exact surface.** The velocity is a fixed field times cos(πt/T), so the flow map to time
t is the autonomous flow over the pseudo-time τ = (T/π) sin(πt/T). Every vertex is traced
back to t = 0 by RK4 (1000 steps per unit pseudo-time), together with its first and
second variational equations. That gives the exact level set φ(x) = |X0(x) − c| − R0,
its gradient and its Hessian, and from them:

- the signed distance d = φ/|∇φ|;
- the exact normal;
- the exact mean curvature H at the foot point x − d n.

The reference is converged to 1e-10 R0. Without remeshing the advected vertices land
within 3.7e-6 R0 of it, so every error above that is the remesher's.

**Metrics** (`metrics.py`; A_v are Meyer mixed areas, ‖f‖²_A = Σ_v A_v f_v²):

| symbol | definition | measures |
|---|---|---|
| dV | V/V(0) − 1, V(0) the initial discrete volume | conservation |
| dA | A/A(0) − 1 | area error |
| d_rms, d_mean | area-weighted RMS and mean of d over the vertices, in % of R0 | position error, bias |
| normal error | area-weighted RMS angle between face normal and exact normal at the centroid | orientation |
| E_H | ‖H_h − H‖_A / ‖H‖_A, H_h the cotangent-Laplacian mean curvature (what a front tracker uses for surface tension) | curvature accuracy |
| N_H | ‖H_h − ⟨H_h⟩_ring‖_A / ‖H‖_A, ⟨·⟩_ring the area-weighted mean over the one-ring neighbours | curvature noise; needs no reference except the scale |
| E_H on the projected mesh | E_H with every vertex moved onto the exact surface (x − d n) | the triangulation's share of E_H; the rest is roughness |
| faces against the exact normal | faces whose normal has a negative dot product with the exact one | gross defects |
| q | 4√3 A / Σ l², min and mean | triangle quality |

Snapshots at t = T/4, T/2, 3T/4, T.

---

## 5. Results A: three modes, three resolutions (MFTL order)

### 5.1 Volume, dV in %

| mode | r | T/4 | T/2 | 3T/4 | **T** | dA at T [%] |
|---|---|---|---|---|---|---|
| uniform | 3 | −4.95 ± 0.05 | −15.87 ± 0.82 | −21.01 ± 1.32 | **−23.16 ± 1.43** | +19.85 ± 1.89 |
| uniform | 4 | −1.46 ± 0.01 | −2.64 ± 0.10 | −3.09 ± 0.17 | **−3.90 ± 0.14** | −1.58 ± 0.22 |
| uniform | 5 | −0.41 ± 0.00 | −0.75 ± 0.00 | −0.89 ± 0.01 | **−1.17 ± 0.01** | −0.65 ± 0.01 |
| curvature | 3 | −2.29 ± 0.05 | −5.60 ± 0.07 | −10.42 ± 0.18 | **−11.66 ± 0.15** | +4.03 ± 0.33 |
| curvature | 4 | −0.94 ± 0.02 | −1.62 ± 0.02 | −2.43 ± 0.02 | **−3.15 ± 0.02** | −1.29 ± 0.02 |
| curvature | 5 | −0.33 ± 0.00 | −0.54 ± 0.00 | −0.72 ± 0.00 | **−0.97 ± 0.01** | −0.55 ± 0.01 |
| anisotropic | 3 | +0.80 ± 0.02 | +0.38 ± 0.09 | +0.51 ± 0.19 | **+0.34 ± 0.33** | +12.61 ± 2.27 |
| anisotropic | 4 | +0.10 ± 0.00 | +0.16 ± 0.01 | +0.26 ± 0.01 | **+0.33 ± 0.01** | +1.65 ± 0.12 |
| anisotropic | 5 | −0.01 ± 0.00 | +0.00 ± 0.00 | +0.03 ± 0.00 | **+0.06 ± 0.00** | +0.36 ± 0.01 |
| no remeshing | 3 | −11.23 | −35.40 | −11.23 | +0.00 | +0.00 |
| no remeshing | 4 | −3.09 | −14.69 | −3.09 | +0.00 | +0.00 |
| no remeshing | 5 | −0.79 | −4.87 | −0.79 | +0.00 | +0.00 |

Without remeshing, the intermediate values are the chord error of triangles stretched
far beyond the curvature; they vanish at T. The remeshed runs lose volume monotonically.

### 5.2 Accuracy and smoothness at T/2 (maximum deformation)

| mode | r | triangles | d_rms [% R0] | d_mean [% R0] | normal err. [°] | E_H | N_H | E_H projected |
|---|---|---|---|---|---|---|---|---|
| uniform | 3 | 3495 | 2.940 | −0.169 | 27.50 | 1.61 ± 0.44 | 0.83 ± 0.36 | 1.486 |
| uniform | 4 | 18381 | 0.546 | −0.132 | 7.16 | 0.92 ± 0.03 | 0.62 ± 0.02 | 0.829 |
| uniform | 5 | 75315 | 0.134 | −0.047 | 2.70 | 0.56 ± 0.01 | 0.44 ± 0.01 | 0.460 |
| curvature | 3 | 15422 | 1.380 | −0.476 | 7.18 | 1.10 ± 0.12 | 0.25 ± 0.22 | 1.830 |
| curvature | 4 | 38202 | 0.297 | −0.135 | 2.38 | 0.73 ± 0.01 | 0.54 ± 0.00 | 0.574 |
| curvature | 5 | 106778 | 0.087 | −0.042 | 1.30 | 0.46 ± 0.01 | 0.47 ± 0.01 | 0.208 |
| anisotropic | 3 | 5698 | 1.375 | +0.120 | 11.89 | 0.95 ± 0.03 | 0.63 ± 0.03 | 0.817 |
| anisotropic | 4 | 21138 | 0.114 | +0.024 | 3.18 | 0.53 ± 0.02 | 0.41 ± 0.01 | 0.408 |
| anisotropic | 5 | 80602 | 0.020 | +0.004 | 1.11 | 0.30 ± 0.01 | 0.28 ± 0.00 | 0.216 |

At T/2 the projected-mesh curvature error is close to the full one: the error at maximum
deformation is resolution of the sheet's rim, not roughness.

### 5.3 Accuracy and smoothness at T

| mode | r | triangles | d_rms [% R0] | d_mean [% R0] | normal err. [°] | E_H | N_H | E_H projected |
|---|---|---|---|---|---|---|---|---|
| uniform | 3 | 1486 | 26.683 | −15.940 | 52.43 | 1.53 ± 0.07 | 1.43 ± 0.10 | 2.669 |
| uniform | 4 | 4818 | 2.167 | −1.314 | 8.51 | 1.61 ± 0.20 | 1.78 ± 0.19 | 0.017 |
| uniform | 5 | 19561 | 0.629 | −0.390 | 3.00 | 1.29 ± 0.06 | 1.51 ± 0.07 | 0.002 |
| curvature | 3 | 6668 | 11.403 | −5.528 | 30.28 | 3.75 ± 0.14 | 3.01 ± 0.12 | 0.725 |
| curvature | 4 | 9610 | 1.967 | −1.076 | 7.70 | 2.72 ± 0.14 | 2.57 ± 0.18 | 0.005 |
| curvature | 5 | 21906 | 0.542 | −0.323 | 2.52 | 1.44 ± 0.14 | 1.58 ± 0.13 | 0.002 |
| anisotropic | 3 | 4562 | 8.141 | +1.054 | 30.39 | 6.81 ± 2.49 | 5.48 ± 0.06 | 0.961 |
| anisotropic | 4 | 16145 | 0.673 | +0.088 | 9.88 | 5.92 ± 0.15 | 6.86 ± 0.14 | 0.134 |
| anisotropic | 5 | 59259 | 0.076 | +0.009 | 4.63 | 7.08 ± 0.12 | 8.48 ± 0.15 | 0.041 |

At T the projected-mesh E_H is 0.002 (uniform and curvature, refinement 5) and 0.04
(anisotropic). The triangulations are fine; the curvature error at T is vertex-scale
roughness. The isotropic modes' vertices lie systematically inside the surface
(d_mean < 0).

### 5.4 Mesh quality and cost

| mode | r | peak triangles | triangles at T | min q (worst of T/4..T) | mean q at T | faces against exact normal (max) | ms/step |
|---|---|---|---|---|---|---|---|
| uniform | 3 | 3905 | 1486 | 0.083 | 0.980 | 219.2 | 4 |
| uniform | 4 | 18386 | 4818 | 0.639 | 0.977 | 14.5 | 18 |
| uniform | 5 | 75320 | 19561 | 0.653 | 0.975 | 0.0 | 78 |
| curvature | 3 | 15491 | 6668 | 0.517 | 0.969 | 538.2 | 19 |
| curvature | 4 | 38348 | 9610 | 0.528 | 0.973 | 0.0 | 41 |
| curvature | 5 | 106982 | 21906 | 0.544 | 0.974 | 0.0 | 114 |
| anisotropic | 3 | 6112 | 4562 | 0.118 | 0.907 | 208.0 | 25 |
| anisotropic | 4 | 22032 | 16145 | 0.295 | 0.892 | 23.8 | 91 |
| anisotropic | 5 | 82594 | 59259 | 0.313 | 0.911 | 0.0 | 338 |

Timing: each configuration alone, best of 2, no snapshots, Release build, Apple M4,
one rank (`time_runs.py`). Refinement 3 is under-resolved for every mode: the sheet at
T/2 is thinner than an edge and no mode returns a sphere.

### 5.5 Against Jiao et al.'s published numbers (anisotropic, refinement 5)

| | Jiao et al. 2010 (§6.2.2) | this implementation |
|---|---|---|
| initial triangles | 46 472 | 20 480 |
| area at T/2 | ×4.12 | ×4.12 |
| triangles at T/2 | ×4.2 (197 292) | ×3.9 (80 434) |
| volume error at T | < 0.1 % | +0.06 % (every step), +0.08 % (their protocol) |
| area error at T | < 0.1 % | +0.36 % (every step) |

---

## 6. Results B: sweep orders (refinement 4) and Jiao's protocol (refinement 5)

### 6.1 Sweep order and sorting

| mode | order | dV at T/2 [%] | dV at T [%] | d_rms at T | E_H at T | N_H at T | min q | triangles T/2 / T | ms/step |
|---|---|---|---|---|---|---|---|---|---|
| uniform | MFTL | −2.64 ± 0.10 | −3.90 ± 0.14 | 2.17 | 1.61 ± 0.20 | 1.78 ± 0.19 | 0.639 | 18381 / 4818 | 18 |
| uniform | MFTL, sorted | −2.57 ± 0.03 | −3.79 ± 0.06 | 2.12 | 1.58 ± 0.09 | 1.75 ± 0.06 | 0.715 | 18400 / 4833 | 18 |
| uniform | Jiao | −4.25 ± 0.51 | −9.46 ± 1.10 | 16.52 | 3.03 ± 0.35 | 2.51 ± 0.14 | 0.295 | 17892 / 5712 | 33 |
| uniform | Jiao, sorted | −4.61 ± 0.66 | −10.33 ± 1.47 | 20.83 | 3.17 ± 0.63 | 2.56 ± 0.23 | 0.102 | 17696 / 6055 | 32 |
| uniform | MFTL, 2 smoothing sweeps | −3.01 ± 0.14 | −4.54 ± 0.17 | 2.54 | 1.72 ± 0.12 | 1.86 ± 0.11 | 0.710 | 18344 / 4842 | 17 |
| uniform | Jiao, 2 × 1 smoothing sweeps | −4.88 ± 0.42 | −9.97 ± 0.95 | 12.07 | 2.85 ± 0.13 | 2.46 ± 0.07 | 0.212 | 18092 / 5368 | 27 |
| curvature | MFTL | −1.62 ± 0.02 | −3.15 ± 0.02 | 1.97 | 2.72 ± 0.14 | 2.57 ± 0.18 | 0.528 | 38202 / 9610 | 41 |
| curvature | MFTL, sorted | −1.56 ± 0.03 | −3.00 ± 0.04 | 1.86 | 3.07 ± 0.24 | 2.94 ± 0.21 | 0.582 | 38301 / 9880 | 41 |
| curvature | Jiao | −2.19 ± 0.08 | −5.16 ± 0.13 | 3.61 | 3.95 ± 0.14 | 3.35 ± 0.09 | 0.422 | 40529 / 12605 | 79 |
| curvature | Jiao, sorted | −2.03 ± 0.03 | −4.52 ± 0.06 | 3.06 | 4.49 ± 0.12 | 4.11 ± 0.09 | 0.531 | 40402 / 13449 | 79 |
| curvature | MFTL, 2 smoothing sweeps | −1.76 ± 0.03 | −3.35 ± 0.05 | 2.04 | 2.85 ± 0.31 | 2.71 ± 0.32 | 0.482 | 38164 / 9831 | 39 |
| curvature | Jiao, 2 × 1 smoothing sweeps | −2.45 ± 0.05 | −5.27 ± 0.08 | 3.27 | 3.69 ± 0.18 | 3.17 ± 0.19 | 0.123 | 39759 / 11724 | 64 |
| anisotropic | MFTL | +0.16 ± 0.01 | +0.33 ± 0.01 | 0.67 | 5.92 ± 0.15 | 6.86 ± 0.14 | 0.295 | 21138 / 16145 | 91 |
| anisotropic | MFTL, sorted | +0.16 ± 0.01 | +0.33 ± 0.01 | 0.45 | 5.95 ± 0.11 | 6.91 ± 0.12 | 0.313 | 21089 / 16202 | 90 |
| anisotropic | Jiao | +0.27 ± 0.02 | +0.52 ± 0.03 | 0.69 | 6.42 ± 0.13 | 7.31 ± 0.21 | 0.290 | 21418 / 17858 | 183 |
| anisotropic | Jiao, sorted | +0.27 ± 0.01 | +0.52 ± 0.02 | 0.65 | 6.51 ± 0.03 | 7.46 ± 0.11 | 0.321 | 21346 / 17764 | 184 |
| anisotropic | MFTL, 2 smoothing sweeps | +0.15 ± 0.01 | +0.31 ± 0.01 | 0.48 | 5.96 ± 0.11 | 6.97 ± 0.04 | 0.322 | 21452 / 16082 | 79 |
| anisotropic | Jiao, 2 × 1 smoothing sweeps | +0.17 ± 0.01 | +0.37 ± 0.02 | 0.46 | 6.14 ± 0.15 | 7.08 ± 0.16 | 0.301 | 21730 / 17080 | 135 |

- Jiao's order loses more volume in every mode, and still does at equal smoothing work
  (uniform −10.0 % against −4.5 %, curvature −5.3 % against −3.4 %). It costs 1.8–2.0
  times as much. In the uniform mode the extra loss accrues mostly between 0.45 T and
  0.7 T. A plausible mechanism is that the order smooths the freshly advected mesh before
  repairing it; this was not isolated.
- Sorting moves the volume by at most 0.15 pp in MFTL's order, the size of the
  run-to-run spread. Curvature sizing gains 0.15 pp (MFTL order) and 0.6 pp (Jiao order)
  of volume at the cost of curvature noise.

### 6.2 Jiao et al.'s protocol (refinement 5, single runs; every-step rows are 4-run means)

| mode | protocol | dV at T/2 [%] | dV at T [%] | d_rms at T | E_H at T | N_H at T | triangles at T | ms/step |
|---|---|---|---|---|---|---|---|---|
| uniform | every step, MFTL order | −0.750 | −1.175 | 0.629 | 1.29 | 1.51 | 19561 | 78 |
| uniform | Jiao's protocol | −0.911 | −1.441 | 0.770 | 1.62 | 1.90 | 20410 | 48 |
| curvature | every step, MFTL order | −0.542 | −0.969 | 0.542 | 1.44 | 1.58 | 21906 | 114 |
| curvature | Jiao's protocol | −0.612 | −1.064 | 0.609 | 3.07 | 3.32 | 28892 | 71 |
| anisotropic | every step, MFTL order | +0.003 | +0.058 | 0.076 | 7.08 | 8.48 | 59259 | 338 |
| anisotropic | Jiao's protocol | +0.016 | +0.077 | 0.090 | 8.19 | 9.92 | 60018 | 211 |

**Correctness note.** The study found a bug in the first implementation of Jiao's order:
a split after the last compaction broke the ownership prefix in a distributed run
(correct at np = 1). A two-rank conservation test caught it, and all Jiao-order data
above were produced after the fix.

---

## 7. Results C: follow-up — where the accuracy comes from, and back-projection

Refinement 4: 4-run means where ± is shown, single runs otherwise; refinement 5: single
runs except the plain modes (4-run means). The ms/step of this section come from runs
made four at a time with snapshots, and the projection variants use an experimental
grid search, so they are **not** cost measurements (section 12).

### 7.1 Volume budget over the period (base run of each variant), % of V(0)

| variant | r | split | collapse | flip | smooth | advection | total |
|---|---|---|---|---|---|---|---|
| uniform | 4 | 0.000 | −2.736 | −1.180 | +0.167 | −0.053 | −3.802 |
| curvature | 4 | 0.000 | −1.963 | −1.246 | +0.132 | −0.082 | −3.158 |
| anisotropic | 4 | +0.277 | −0.015 | +0.063 | +0.099 | −0.076 | +0.348 |
| anisotropic, QEM off | 4 | 0.000 | −0.056 | −0.079 | +0.099 | −0.074 | −0.110 |
| uniform + QEM | 4 | +1.437 | −0.293 | −0.410 | +0.172 | −0.049 | +0.857 |
| curvature + QEM | 4 | +1.298 | −0.028 | −0.321 | +0.094 | −0.105 | +0.937 |
| uniform + Delaunay flips | 4 | 0.000 | −2.333 | −1.562 | −0.012 | −0.053 | −3.960 |
| curvature + Delaunay flips | 4 | 0.000 | −1.844 | −0.229 | +0.020 | −0.075 | −2.128 |
| curvature + Delaunay flips (3D angles) | 4 | 0.000 | −1.812 | −0.216 | +0.022 | −0.077 | −2.084 |
| curvature + Delaunay + QEM | 4 | +1.267 | −0.029 | +0.050 | −0.009 | −0.106 | +1.174 |
| uniform + flat projection | 4 | 0.000 | −3.673 | −1.104 | −21.159 | −0.052 | −25.989 |
| curvature + flat projection | 4 | 0.000 | −2.031 | −1.241 | −9.279 | −0.065 | −12.616 |
| uniform + Phong projection | 4 | 0.000 | −2.907 | −1.360 | +2.905 | −0.052 | −1.414 |
| curvature + Phong projection | 4 | 0.000 | −1.752 | −1.118 | +2.827 | −0.045 | −0.088 |
| uniform + QEM + Phong projection | 4 | +1.425 | −0.205 | −0.533 | −2.055 | −0.049 | −1.418 |
| curvature + QEM + Phong projection | 4 | +1.348 | −0.018 | −0.333 | −1.045 | −0.043 | −0.092 |
| curvature + Delaunay + QEM + Phong projection | 4 | +1.239 | −0.017 | +0.015 | −1.302 | −0.040 | −0.104 |
| uniform + Phong midpoints | 4 | +0.599 | −1.574 | −0.833 | +0.163 | −0.052 | −1.698 |
| curvature + Phong midpoints | 4 | +0.606 | −1.073 | −0.830 | +0.119 | −0.056 | −1.233 |
| uniform + local fan projection | 4 | 0.000 | −2.925 | −1.298 | −2.856 | −0.053 | −7.132 |
| curvature + local fan projection | 4 | 0.000 | −1.952 | −1.217 | −1.042 | −0.049 | −4.260 |
| uniform + local Phong (midpoints + fan) | 4 | +0.600 | −1.632 | −0.930 | −2.673 | −0.051 | −4.686 |
| curvature + local Phong (midpoints + fan) | 4 | +0.601 | −1.038 | −0.786 | −0.878 | −0.046 | −2.147 |
| uniform | 5 | 0.000 | −0.825 | −0.382 | +0.044 | −0.010 | −1.172 |
| curvature | 5 | 0.000 | −0.622 | −0.378 | +0.035 | −0.012 | −0.977 |
| anisotropic | 5 | +0.061 | 0.000 | −0.016 | +0.032 | −0.019 | +0.058 |
| anisotropic, QEM off | 5 | 0.000 | −0.006 | −0.048 | +0.032 | −0.019 | −0.041 |
| uniform + QEM | 5 | +0.482 | −0.025 | −0.130 | +0.043 | −0.010 | +0.359 |
| curvature + QEM | 5 | +0.420 | −0.003 | −0.125 | +0.030 | −0.019 | +0.303 |
| uniform + Delaunay flips | 5 | 0.000 | −0.664 | −0.166 | −0.003 | −0.010 | −0.843 |
| curvature + Delaunay flips | 5 | 0.000 | −0.518 | −0.056 | −0.001 | −0.012 | −0.586 |
| curvature + Delaunay flips (3D angles) | 5 | 0.000 | −0.525 | −0.054 | 0.000 | −0.011 | −0.591 |
| curvature + Delaunay + QEM | 5 | +0.379 | −0.003 | +0.023 | −0.005 | −0.022 | +0.373 |
| uniform + flat projection | 5 | 0.000 | −0.858 | −0.391 | −5.125 | −0.010 | −6.384 |
| curvature + flat projection | 5 | 0.000 | −0.603 | −0.359 | −3.341 | −0.012 | −4.315 |
| uniform + Phong projection | 5 | 0.000 | −0.858 | −0.401 | +1.094 | −0.010 | −0.175 |
| curvature + Phong projection | 5 | 0.000 | −0.611 | −0.356 | +0.948 | −0.011 | −0.029 |
| uniform + QEM + Phong projection | 5 | +0.481 | −0.018 | −0.135 | −0.489 | −0.010 | −0.172 |
| curvature + QEM + Phong projection | 5 | +0.417 | −0.001 | −0.117 | −0.317 | −0.011 | −0.030 |
| curvature + Delaunay + QEM + Phong projection | 5 | +0.355 | −0.001 | +0.009 | −0.384 | −0.011 | −0.033 |
| uniform + Phong midpoints | 5 | +0.206 | −0.485 | −0.286 | +0.042 | −0.010 | −0.532 |
| curvature + Phong midpoints | 5 | +0.180 | −0.359 | −0.261 | +0.033 | −0.011 | −0.419 |
| uniform + local fan projection | 5 | 0.000 | −0.838 | −0.393 | −0.502 | −0.009 | −1.743 |
| curvature + local fan projection | 5 | 0.000 | −0.616 | −0.368 | −0.302 | −0.012 | −1.297 |
| uniform + local Phong (midpoints + fan) | 5 | +0.204 | −0.486 | −0.290 | −0.440 | −0.010 | −1.022 |
| curvature + local Phong (midpoints + fan) | 5 | +0.181 | −0.359 | −0.256 | −0.253 | −0.011 | −0.698 |

### 7.2 Operations over the period (base runs: splits / collapses / flips)

| variant | r = 4 | r = 5 |
|---|---|---|
| uniform | 19 984 / 20 154 / 39 998 | 93 082 / 93 570 / 183 795 |
| curvature | 105 675 / 103 446 / 212 745 | 160 668 / 160 092 / 323 155 |
| anisotropic | 8 823 / 3 315 / 26 342 | 31 301 / 11 945 / 116 667 |
| anisotropic, QEM off | 8 739 / 3 179 / 26 391 | 31 363 / 12 211 / 117 295 |
| curvature + QEM | 88 898 / 83 574 / 181 389 | 172 551 / 166 092 / 347 670 |
| curvature + Delaunay flips | 104 946 / 103 418 / 48 205 | 155 537 / 155 258 / 62 345 |
| curvature + Phong projection | 77 351 / 77 409 / 155 182 | 142 166 / 142 573 / 284 150 |

In uniform sizing at refinement 5, 42k of the 94k collapses happen in the first half,
while the surface is still stretching.

### 7.3 Accuracy at T

| variant | r | dV [%] | d_rms [% R0] | E_H | N_H | E_H at T/2 | triangles |
|---|---|---|---|---|---|---|---|
| uniform | 4 | −3.90 ± 0.14 | 2.167 ± 0.135 | 1.61 ± 0.20 | 1.78 ± 0.19 | 0.92 ± 0.03 | 4818 |
| uniform + QEM | 4 | +0.84 ± 0.02 | 0.893 ± 0.035 | 1.99 ± 0.09 | 2.42 ± 0.10 | 0.66 ± 0.05 | 4964 |
| uniform + Delaunay flips | 4 | −4.09 ± 0.12 | 2.606 ± 0.105 | 1.51 ± 0.15 | 1.71 ± 0.18 | 0.91 ± 0.02 | 4758 |
| uniform + flat projection | 4 | −25.58 ± 0.57 | 13.481 ± 0.711 | 1.63 ± 0.20 | 0.84 ± 0.14 | 1.58 ± 0.31 | 4281 |
| uniform + Phong projection | 4 | −1.45 ± 0.02 | 1.392 ± 0.031 | 0.44 ± 0.04 | 0.26 ± 0.04 | 0.80 ± 0.00 | 4845 |
| uniform + QEM + Phong projection | 4 | −1.41 ± 0.01 | 1.354 ± 0.021 | 0.45 ± 0.03 | 0.27 ± 0.03 | 0.80 ± 0.02 | 4882 |
| uniform + Phong midpoints | 4 | −1.70 | 0.996 | 1.07 | 1.24 | 0.73 | 4874 |
| uniform + local fan projection | 4 | −7.13 | 4.016 | 0.77 | 0.42 | 1.17 | 4716 |
| uniform + local Phong | 4 | −4.69 | 2.867 | 0.64 | 0.34 | 1.00 | 4822 |
| curvature | 4 | −3.15 ± 0.02 | 1.967 ± 0.009 | 2.72 ± 0.14 | 2.57 ± 0.18 | 0.73 ± 0.01 | 9610 |
| curvature + QEM | 4 | +0.94 ± 0.00 | 0.656 ± 0.016 | 7.33 ± 0.24 | 7.32 ± 0.19 | 0.44 ± 0.00 | 16312 |
| curvature + Delaunay flips | 4 | −2.11 ± 0.02 | 1.352 ± 0.015 | 2.24 ± 0.12 | 2.08 ± 0.10 | 0.60 ± 0.01 | 8346 |
| curvature + Delaunay flips (3D angles) | 4 | −2.09 ± 0.01 | 1.333 ± 0.017 | 2.28 ± 0.19 | 2.12 ± 0.17 | 0.58 ± 0.01 | 8580 |
| curvature + Delaunay + QEM | 4 | +1.17 ± 0.01 | 0.819 ± 0.015 | 10.11 ± 0.34 | 10.79 ± 0.37 | 0.44 ± 0.00 | 20784 |
| curvature + flat projection | 4 | −12.51 ± 0.09 | 8.200 ± 0.049 | 1.70 ± 0.11 | 0.58 ± 0.05 | 1.01 ± 0.01 | 7243 |
| **curvature + Phong projection** | 4 | **−0.09 ± 0.00** | **0.136 ± 0.001** | **0.15 ± 0.01** | **0.08 ± 0.00** | 0.31 ± 0.01 | 5018 |
| curvature + QEM + Phong projection | 4 | −0.09 ± 0.00 | 0.132 ± 0.002 | 0.14 ± 0.01 | 0.07 ± 0.00 | 0.30 ± 0.01 | 5001 |
| curvature + Delaunay + QEM + Phong projection | 4 | −0.11 ± 0.00 | 0.147 ± 0.002 | 0.20 ± 0.01 | 0.09 ± 0.01 | 0.31 ± 0.01 | 5014 |
| curvature + Phong midpoints | 4 | −1.23 | 0.830 | 1.08 | 0.96 | 0.45 | 6182 |
| curvature + local fan projection | 4 | −4.26 | 2.395 | 0.82 | 0.27 | 0.67 | 5706 |
| curvature + local Phong | 4 | −2.15 | 1.202 | 0.27 | 0.13 | 0.48 | 5054 |
| anisotropic | 4 | +0.33 ± 0.01 | 0.673 ± 0.476 | 5.92 ± 0.15 | 6.86 ± 0.14 | 0.53 ± 0.02 | 16145 |
| anisotropic, QEM off | 4 | −0.12 ± 0.01 | 0.423 ± 0.037 | 5.16 ± 0.18 | 6.00 ± 0.19 | 0.54 ± 0.02 | 16306 |
| uniform | 5 | −1.17 ± 0.01 | 0.629 ± 0.011 | 1.29 ± 0.06 | 1.51 ± 0.07 | 0.56 ± 0.01 | 19561 |
| uniform + QEM | 5 | +0.36 | 0.243 | 2.05 | 2.53 | 0.45 | 19876 |
| uniform + Delaunay flips | 5 | −0.84 | 0.510 | 1.35 | 1.60 | 0.62 | 19404 |
| uniform + flat projection | 5 | −6.38 | 3.181 | 0.72 | 0.36 | 0.92 | 18974 |
| uniform + Phong projection | 5 | −0.18 | 0.242 | 0.16 | 0.07 | 0.48 | 19730 |
| uniform + QEM + Phong projection | 5 | −0.17 | 0.237 | 0.16 | 0.08 | 0.47 | 19744 |
| uniform + Phong midpoints | 5 | −0.53 | 0.291 | 0.72 | 0.86 | 0.45 | 19632 |
| uniform + local fan projection | 5 | −1.74 | 0.959 | 0.36 | 0.23 | 0.64 | 19502 |
| uniform + local Phong | 5 | −1.02 | 0.617 | 0.25 | 0.13 | 0.58 | 19668 |
| curvature | 5 | −0.97 ± 0.01 | 0.542 ± 0.002 | 1.44 ± 0.14 | 1.58 ± 0.13 | 0.46 ± 0.01 | 21906 |
| curvature + QEM | 5 | +0.30 | 0.194 | 4.90 | 4.85 | 0.43 | 33398 |
| curvature + Delaunay flips | 5 | −0.59 | 0.339 | 1.24 | 1.36 | 0.38 | 21038 |
| curvature + Delaunay flips (3D angles) | 5 | −0.59 | 0.343 | 1.64 | 1.65 | 0.37 | 21770 |
| curvature + Delaunay + QEM | 5 | +0.37 | 0.234 | 8.60 | 8.78 | 0.42 | 47288 |
| curvature + flat projection | 5 | −4.31 | 2.431 | 0.73 | 0.32 | 0.46 | 19872 |
| **curvature + Phong projection** | 5 | **−0.03** | **0.020** | **0.04** | **0.02** | 0.16 | 19666 |
| curvature + QEM + Phong projection | 5 | −0.03 | 0.019 | 0.04 | 0.02 | 0.15 | 19708 |
| curvature + Delaunay + QEM + Phong projection | 5 | −0.03 | 0.021 | 0.05 | 0.03 | 0.16 | 19488 |
| curvature + Phong midpoints | 5 | −0.42 | 0.238 | 0.52 | 0.60 | 0.24 | 19862 |
| curvature + local fan projection | 5 | −1.30 | 0.691 | 0.32 | 0.21 | 0.34 | 19666 |
| curvature + local Phong | 5 | −0.70 | 0.367 | 0.17 | 0.11 | 0.24 | 19610 |
| anisotropic | 5 | +0.06 ± 0.00 | 0.076 ± 0.001 | 7.08 ± 0.12 | 8.48 ± 0.15 | 0.30 ± 0.01 | 59259 |
| anisotropic, QEM off | 5 | −0.04 | 0.092 | 5.93 | 7.02 | 0.29 | 58784 |

These are the EXPERIMENT's projections. The library implements the stored-input Phong
projection ("Phong projection" here) in section 9; the flat and the local-fan variants were
not taken.

### 7.4 Reading the follow-up

1. **The loss of the isotropic modes is collapses and flips, nothing else.** Split is
   exactly neutral, and plain tangential smoothing gains +0.1 to +0.2 %. A stretching
   flow makes isotropic sizing split and collapse every step.
2. **The anisotropic mode's volume comes from collapsing rarely**, about a seventh as
   often as uniform sizing at refinement 5, not from its metric or its placement.
   Without QEM it still ends at −0.11 % (r4) and −0.04 % (r5). It keeps three times the
   triangles, which also lowers its position error.
3. **QEM placement is a trade**: volume and position error improve (r4 curvature: +0.94 %
   and d_rms 0.66), curvature noise rises 1.4–3 times (N_H 7.3 against 2.6), and
   curvature sizing then refines the noise (16k against 9.6k triangles).
4. **Delaunay flips help curvature sizing on every measure**, with 4.4 times fewer flips.
   The flip loss falls from −1.25 to −0.23 %. The tangent-plane and the 3D-angle
   criterion are equivalent. For uniform sizing the result is mixed (worse at r4,
   better at r5).
5. **Projection onto the flat input is rejected.** It smooths by shrinking: the
   smoothing sweep becomes a sink of −9.3 % (curvature, r4) and −21.2 % (uniform, r4).
   In front tracking the reference is the previous step's mesh at the same resolution,
   and every call moves the smoothed vertices onto its chords.
6. **Projection onto the Phong surface of the stored input is the best variant
   measured.** The smoothing sweep restores what the call's collapses and flips removed
   (+2.83 % against −1.75 % and −1.12 %). Curvature sizing at r5 reaches −0.03 %, d_rms
   0.020 % R0, E_H 0.04, N_H 0.02 with 19.7k triangles. That beats the anisotropic mode
   on every measure with a third of its triangles. QEM and Delaunay flips add nothing
   on top of it.
7. **A local reconstruction is not a substitute.**
   - *Phong midpoints alone* halve the loss (r4 curvature −1.23 %, r5 −0.42 %) and lower
     the noise moderately: the split sweep gains +0.6 % and the collapse loss falls from
     −1.96 to −1.07 %.
   - *Local fan projection alone* removes most of the noise (N_H 0.27 / 0.21) but loses
     more volume than no projection (r4 −4.26 %, r5 −1.30 %). Its smoothing sweep is a
     sink.
   - *Both* give −2.15 / −0.70 %.

   The local reference is rebuilt from the samples being smoothed, nine times per call.
   That acts like a curvature flow and cannot restore what the call removed. The stored
   input stays fixed for the whole call. This mechanism is an interpretation; it was not
   isolated further.

---

## 8. Results D: development measurements (`docs/analysis.md` §14)

### 8.1 The curvature field needs one smoothing pass for a fixed point

Ellipsoid a = 0.7, b = c = 0.3, refinement 3, L_max = 0.08, ε = 0.01 L_max,
L_min = 0.3 L_max; `MFTLRemeshObject` called repeatedly:

| call | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unsmoothed: triangles | 4578 | 5092 | 5040 | 4952 | 4958 | 4938 | 4924 | 4922 |
| one Jacobi pass: triangles | 4600 | 4518 | 4444 | 4430 | 4418 | 4418 | 4418 | 4418 |
| one Jacobi pass: split/collapse/flip | 3321/1661/3568 | 344/385/726 | 33/70/90 | 3/10/10 | 1/7/4 | 0/0/0 | 0/0/0 | 0/0/0 |

The edge rule takes the minimum of two noisy estimates, which is biased low. Without the
pass every call refines further. The mean κ_max at the waist against the exact 1/b:
1.028 on the icosphere, 1.084 after unsmoothed remeshing, 1.061 with the pass.

### 8.2 Distributed fixed point under curvature sizing

Same ellipsoid, spanning driver, each run iterated to its fixed point:

| np | 1 | 2 | 4 | 8 | 27 |
|---|---:|---:|---:|---:|---:|
| triangles | 4418 | 4476 | 4592 | 4682 | 4594 |
| against one rank | — | +1.3 % | +3.9 % | +6.0 % | +4.0 % |
| tip/waist edge ratio (analytic 0.654) | 0.703 | 0.738 | 0.713 | 0.694 | 0.711 |

The cause is that the field is recomputed after every migration. The anisotropic mode
reached 0 / 0 / 1.3 / 1.9 %.

### 8.3 Ablation of the first anisotropic implementation (refinement 3, 150 steps)

| replaced by its uniform counterpart | V/V0 at T |
|---|---:|
| nothing | 0.609 |
| QEM placement off | 0.563 |
| **smoothing** | **0.995** |
| flip → valence | 0.585 |
| split/collapse → uniform band | 0.854 |
| (uniform mode) | 0.745 |

The first smoother normalisation (Σ|t| sqrt(σ2σ3)) had a gain of up to 1.93 across the
curved direction. With Σ|t|σ2 the period ends at 0.945, and with the paper's 45° flip
limit at 0.9997. Static and 12-step tests showed nothing. The lesson for a paper:
**smoother weightings must be validated under advection.**

### 8.4 Iterations per spanning placement (refinement 3, Release, `-budget 4`)

| sizing | np = 1 ms/step | V/V0 | np = 4 ms/step | V/V0 |
|---|---:|---:|---:|---:|
| uniform | 3.6 | 0.745 | 6.7 | 0.762 |
| curvature | 17.7 | 0.903 | 45.3 | 0.717 |
| curvature, 1 iteration | — | — | 18.8 | 0.881 |
| curvature, `-budget 2`, 1 iteration | — | — | 11.4 | 0.921 |
| anisotropic | 24.2 | 1.004 | 63.1 | 0.998 |

Every extra pass over a curvature-sized mesh is real splitting and collapsing, so it
costs volume. One iteration per placement brings the distributed run back to the
serial answer.

### 8.5 Side check: which surface a paraboloid fit reproduces

`checks/facet_vs_vertex_paraboloid.py`. On a unit icosphere (vertices on the sphere), a
weighted least-squares paraboloid (Wendland C2 weights, radius 2.5 h) is fitted in the
frame of an edge midpoint with the radial normal, and evaluated at the midpoint. The
heights are relative to the chord midpoint; "sphere" is the true sagitta.

| refinement | h | sphere | fit to the vertices | fit to the facet integrals (Jibben-style) |
|---|---|---|---|---|
| 2 | 0.276 | 9.56e-3 | 1.034e-2 (1.08) | 1.2e-5 (0.00) |
| 3 | 0.138 | 2.39e-3 | 2.444e-3 (1.02) | −2.3e-4 (−0.10) |
| 4 | 0.069 | 5.99e-4 | 6.017e-4 (1.005) | −7.2e-5 (−0.12) |

A fit that matches each facet's integrated height reproduces the volume-equivalent
surface of the facets, which on an inscribed mesh lies a mean sagitta below the
vertices. Placing a split vertex on it is therefore close to placing it at the chord
midpoint. A fit to the vertices reproduces the surface the vertices sample.

---

## 9. Results E: paraboloid placement and Phong projection in the library

Library build of the two options (section 3.5), the same LeVeque setup, refinements 4 and
5, uniform and curvature sizing, four orientations each (128 runs; data in
`results/leveque_remeshing_placement/`, script `placement_study.py`).

### 9.1 Accuracy and smoothness at T (mean ± sd over 4 orientations)

"Once per call" is the library. The "every sweep" rows are the first version and the
"each iteration" row is an intermediate schedule (section 9.5).

| sizing | r | variant | V/V(0) − 1 [%] | d_rms [% R0] | E_H | N_H | E_H at T/2 | triangles at T |
|---|---|---|---|---|---|---|---|---|
| uniform | 4 | midpoint, no projection | −3.899 ± 0.142 | 2.167 ± 0.135 | 1.61 ± 0.20 | 1.782 ± 0.188 | 0.92 ± 0.03 | 4818 |
| uniform | 4 | paraboloid (positions), radius 3 | −1.935 ± 0.069 | 1.402 ± 0.044 | 1.09 ± 0.14 | 1.028 ± 0.180 | 0.74 ± 0.02 | 4874 |
| uniform | 4 | paraboloid (positions), radius 4 | −1.844 ± 0.080 | 1.218 ± 0.115 | 0.94 ± 0.20 | 0.965 ± 0.177 | 0.75 ± 0.01 | 4866 |
| uniform | 4 | paraboloid (+ normals), radius 3 | −1.082 ± 0.034 | 0.976 ± 0.111 | 0.88 ± 0.09 | 0.890 ± 0.076 | 0.79 ± 0.14 | 4882 |
| uniform | 4 | paraboloid (+ normals), radius 4 | −0.846 ± 0.051 | 1.087 ± 0.136 | 0.99 ± 0.12 | 1.010 ± 0.110 | 0.75 ± 0.05 | 4901 |
| uniform | 4 | Phong projection (once per call, the library) | −1.602 ± 0.032 | 1.551 ± 0.059 | 0.55 ± 0.08 | 0.331 ± 0.067 | 0.84 ± 0.03 | 4879 |
| uniform | 4 | Phong projection + paraboloid (positions) | −1.490 ± 0.014 | 1.413 ± 0.017 | 0.47 ± 0.03 | 0.299 ± 0.025 | 0.84 ± 0.02 | 4866 |
| uniform | 4 | Phong projection + paraboloid (+ normals) | −1.544 ± 0.025 | 1.483 ± 0.059 | 0.51 ± 0.06 | 0.314 ± 0.050 | 0.84 ± 0.01 | 4879 |
| uniform | 4 | Phong projection after every sweep (experiment) | −1.486 ± 0.036 | 1.441 ± 0.065 | 0.49 ± 0.06 | 0.311 ± 0.034 | 0.84 ± 0.03 | 4866 |
| uniform | 4 | projection every sweep + paraboloid (positions) | −1.445 ± 0.020 | 1.394 ± 0.016 | 0.45 ± 0.01 | 0.256 ± 0.006 | 0.81 ± 0.01 | 4859 |
| uniform | 4 | projection every sweep + paraboloid (+ normals) | −1.451 ± 0.012 | 1.403 ± 0.005 | 0.46 ± 0.01 | 0.267 ± 0.021 | 0.82 ± 0.02 | 4878 |
| uniform | 4 | Phong projection after each iteration (experiment) | −1.534 ± 0.024 | 1.482 ± 0.024 | 0.51 ± 0.02 | 0.314 ± 0.017 | 0.85 ± 0.01 | 4892 |
| uniform | 5 | midpoint, no projection | −1.175 ± 0.010 | 0.629 ± 0.011 | 1.29 ± 0.06 | 1.515 ± 0.073 | 0.56 ± 0.01 | 19561 |
| uniform | 5 | paraboloid (positions), radius 3 | −0.489 ± 0.003 | 0.280 ± 0.004 | 0.51 ± 0.02 | 0.560 ± 0.036 | 0.46 ± 0.02 | 19679 |
| uniform | 5 | paraboloid (positions), radius 4 | −0.481 ± 0.003 | 0.281 ± 0.001 | 0.47 ± 0.03 | 0.502 ± 0.025 | 0.46 ± 0.01 | 19704 |
| uniform | 5 | paraboloid (+ normals), radius 3 | −0.381 ± 0.006 | 0.236 ± 0.004 | 0.44 ± 0.03 | 0.483 ± 0.040 | 0.44 ± 0.01 | 19649 |
| uniform | 5 | paraboloid (+ normals), radius 4 | −0.303 ± 0.005 | 0.228 ± 0.005 | 0.44 ± 0.03 | 0.474 ± 0.022 | 0.46 ± 0.01 | 19668 |
| uniform | 5 | Phong projection (once per call, the library) | −0.182 ± 0.002 | 0.247 ± 0.002 | 0.17 ± 0.02 | 0.082 ± 0.014 | 0.48 ± 0.01 | 19693 |
| uniform | 5 | Phong projection + paraboloid (positions) | −0.178 ± 0.003 | 0.244 ± 0.003 | 0.16 ± 0.01 | 0.073 ± 0.006 | 0.47 ± 0.00 | 19744 |
| uniform | 5 | Phong projection + paraboloid (+ normals) | −0.177 ± 0.001 | 0.242 ± 0.001 | 0.16 ± 0.02 | 0.076 ± 0.015 | 0.47 ± 0.00 | 19677 |
| uniform | 5 | Phong projection after every sweep (experiment) | −0.176 ± 0.002 | 0.244 ± 0.003 | 0.17 ± 0.01 | 0.079 ± 0.006 | 0.47 ± 0.00 | 19639 |
| uniform | 5 | projection every sweep + paraboloid (positions) | −0.173 ± 0.004 | 0.240 ± 0.007 | 0.17 ± 0.01 | 0.079 ± 0.004 | 0.47 ± 0.01 | 19726 |
| uniform | 5 | projection every sweep + paraboloid (+ normals) | −0.175 ± 0.003 | 0.242 ± 0.004 | 0.16 ± 0.00 | 0.070 ± 0.004 | 0.48 ± 0.01 | 19692 |
| uniform | 5 | Phong projection after each iteration (experiment) | −0.180 ± 0.005 | 0.247 ± 0.008 | 0.17 ± 0.02 | 0.078 ± 0.012 | 0.48 ± 0.01 | 19741 |
| curvature | 4 | midpoint, no projection | −3.147 ± 0.020 | 1.967 ± 0.009 | 2.72 ± 0.14 | 2.567 ± 0.176 | 0.73 ± 0.01 | 9610 |
| curvature | 4 | paraboloid (positions), radius 3 | −0.561 ± 0.010 | 0.648 ± 0.010 | 0.73 ± 0.10 | 0.582 ± 0.101 | 0.38 ± 0.01 | 5505 |
| curvature | 4 | paraboloid (positions), radius 4 | −0.738 ± 0.010 | 0.658 ± 0.008 | 0.40 ± 0.01 | 0.405 ± 0.005 | 0.42 ± 0.01 | 5234 |
| curvature | 4 | paraboloid (+ normals), radius 3 | +0.013 ± 0.009 | 0.774 ± 0.044 | 1.22 ± 0.28 | 0.899 ± 0.198 | 0.42 ± 0.01 | 5667 |
| curvature | 4 | paraboloid (+ normals), radius 4 | +0.241 ± 0.013 | 0.835 ± 0.040 | 1.03 ± 0.07 | 0.755 ± 0.066 | 0.46 ± 0.01 | 5621 |
| curvature | 4 | Phong projection (once per call, the library) | −0.093 ± 0.001 | 0.138 ± 0.002 | 0.14 ± 0.01 | 0.075 ± 0.003 | 0.32 ± 0.00 | 5005 |
| curvature | 4 | Phong projection + paraboloid (positions) | −0.092 ± 0.003 | 0.136 ± 0.001 | 0.15 ± 0.00 | 0.075 ± 0.004 | 0.31 ± 0.00 | 5016 |
| curvature | 4 | Phong projection + paraboloid (+ normals) | −0.087 ± 0.002 | 0.139 ± 0.004 | 0.16 ± 0.01 | 0.080 ± 0.004 | 0.31 ± 0.01 | 5035 |
| curvature | 4 | Phong projection after every sweep (experiment) | −0.089 ± 0.002 | 0.137 ± 0.002 | 0.15 ± 0.01 | 0.075 ± 0.003 | 0.31 ± 0.00 | 5017 |
| curvature | 4 | projection every sweep + paraboloid (positions) | −0.089 ± 0.002 | 0.135 ± 0.001 | 0.15 ± 0.00 | 0.079 ± 0.004 | 0.31 ± 0.01 | 5019 |
| curvature | 4 | projection every sweep + paraboloid (+ normals) | −0.088 ± 0.002 | 0.135 ± 0.002 | 0.15 ± 0.01 | 0.075 ± 0.006 | 0.30 ± 0.00 | 5020 |
| curvature | 4 | Phong projection after each iteration (experiment) | −0.091 ± 0.004 | 0.136 ± 0.002 | 0.14 ± 0.00 | 0.073 ± 0.001 | 0.31 ± 0.01 | 5019 |
| curvature | 5 | midpoint, no projection | −0.969 ± 0.006 | 0.542 ± 0.002 | 1.44 ± 0.14 | 1.579 ± 0.126 | 0.46 ± 0.01 | 21906 |
| curvature | 5 | paraboloid (positions), radius 3 | −0.320 ± 0.003 | 0.201 ± 0.002 | 0.34 ± 0.01 | 0.377 ± 0.017 | 0.21 ± 0.00 | 19748 |
| curvature | 5 | paraboloid (positions), radius 4 | −0.338 ± 0.003 | 0.203 ± 0.001 | 0.30 ± 0.01 | 0.330 ± 0.005 | 0.22 ± 0.00 | 19714 |
| curvature | 5 | paraboloid (+ normals), radius 3 | −0.222 ± 0.004 | 0.191 ± 0.001 | 0.32 ± 0.02 | 0.356 ± 0.020 | 0.32 ± 0.01 | 19719 |
| curvature | 5 | paraboloid (+ normals), radius 4 | −0.164 ± 0.003 | 0.187 ± 0.001 | 0.31 ± 0.02 | 0.331 ± 0.024 | 0.36 ± 0.00 | 19777 |
| curvature | 5 | Phong projection (once per call, the library) | −0.031 ± 0.000 | 0.021 ± 0.000 | 0.04 ± 0.00 | 0.025 ± 0.002 | 0.16 ± 0.00 | 19776 |
| curvature | 5 | Phong projection + paraboloid (positions) | −0.030 ± 0.000 | 0.020 ± 0.000 | 0.04 ± 0.00 | 0.022 ± 0.001 | 0.16 ± 0.00 | 19730 |
| curvature | 5 | Phong projection + paraboloid (+ normals) | −0.030 ± 0.000 | 0.020 ± 0.000 | 0.04 ± 0.00 | 0.024 ± 0.002 | 0.16 ± 0.00 | 19779 |
| curvature | 5 | Phong projection after every sweep (experiment) | −0.030 ± 0.000 | 0.020 ± 0.000 | 0.04 ± 0.00 | 0.023 ± 0.001 | 0.16 ± 0.00 | 19769 |
| curvature | 5 | projection every sweep + paraboloid (positions) | −0.030 ± 0.000 | 0.020 ± 0.000 | 0.04 ± 0.00 | 0.022 ± 0.001 | 0.16 ± 0.00 | 19789 |
| curvature | 5 | projection every sweep + paraboloid (+ normals) | −0.030 ± 0.000 | 0.020 ± 0.000 | 0.04 ± 0.00 | 0.022 ± 0.002 | 0.16 ± 0.01 | 19738 |
| curvature | 5 | Phong projection after each iteration (experiment) | −0.030 ± 0.000 | 0.020 ± 0.001 | 0.04 ± 0.00 | 0.024 ± 0.001 | 0.16 ± 0.01 | 19747 |

### 9.2 Cost (base orientation, each configuration alone, best of 2; Apple M4, np = 1)

| sizing | r | variant | ms/step | remesh ns per triangle-step | fit: share of remesh, ns per fit, fits per step | projection: share, ns per move | reference: share, largest kB |
|---|---|---|---|---|---|---|---|
| uniform | 4 | midpoint, no projection | 17.7 | 1427 | − | − | − |
| uniform | 4 | paraboloid (positions), radius 3 | 18.3 | 1465 | 2.7 %, 2317, 206 | − | − |
| uniform | 4 | paraboloid (positions), radius 4 | 18.4 | 1480 | 2.7 %, 2321, 205 | − | − |
| uniform | 4 | paraboloid (+ normals), radius 3 | 18.5 | 1477 | 3.7 %, 3066, 208 | − | − |
| uniform | 4 | paraboloid (+ normals), radius 4 | 18.2 | 1458 | 3.7 %, 3077, 206 | − | − |
| uniform | 4 | Phong projection (once per call, the library) | 19.7 | 1584 | − | 3.5 %, 111 | 2.8 %, 969 |
| uniform | 4 | Phong projection + paraboloid (positions) | 19.6 | 1586 | 2.5 %, 2326, 205 | 3.4 %, 108 | 2.8 %, 972 |
| uniform | 4 | Phong projection + paraboloid (+ normals) | 19.6 | 1576 | 3.4 %, 3074, 208 | 3.4 %, 107 | 2.8 %, 975 |
| uniform | 4 | Phong projection after every sweep (experiment) | 21.2 | 1719 | − | 13.3 %, 51 | 2.5 %, 972 |
| uniform | 4 | projection every sweep + paraboloid (positions) | 21.6 | 1751 | 2.3 %, 2328, 203 | 12.9 %, 51 | 2.5 %, 976 |
| uniform | 4 | projection every sweep + paraboloid (+ normals) | 21.8 | 1760 | 3.1 %, 3101, 206 | 12.8 %, 51 | 2.5 %, 979 |
| uniform | 4 | Phong projection after each iteration (experiment) | 19.9 | 1605 | − | 9.5 %, 104 | 2.7 %, 973 |
| uniform | 5 | midpoint, no projection | 77.6 | 1537 | − | − | − |
| uniform | 5 | paraboloid (positions), radius 3 | 79.8 | 1577 | 2.9 %, 2326, 940 | − | − |
| uniform | 5 | paraboloid (positions), radius 4 | 79.9 | 1581 | 2.9 %, 2336, 945 | − | − |
| uniform | 5 | paraboloid (+ normals), radius 3 | 81.0 | 1602 | 3.8 %, 3123, 943 | − | − |
| uniform | 5 | paraboloid (+ normals), radius 4 | 80.9 | 1600 | 3.9 %, 3130, 947 | − | − |
| uniform | 5 | Phong projection (once per call, the library) | 85.1 | 1685 | − | 3.7 %, 126 | 2.7 %, 3992 |
| uniform | 5 | Phong projection + paraboloid (positions) | 86.8 | 1725 | 2.7 %, 2370, 942 | 3.7 %, 127 | 2.7 %, 3988 |
| uniform | 5 | Phong projection + paraboloid (+ normals) | 87.0 | 1726 | 3.6 %, 3165, 944 | 3.6 %, 125 | 2.7 %, 3992 |
| uniform | 5 | Phong projection after every sweep (experiment) | 92.1 | 1837 | − | 13.5 %, 55 | 2.5 %, 3981 |
| uniform | 5 | projection every sweep + paraboloid (positions) | 94.8 | 1887 | 2.4 %, 2336, 947 | 13.1 %, 55 | 2.4 %, 3999 |
| uniform | 5 | projection every sweep + paraboloid (+ normals) | 94.8 | 1891 | 3.3 %, 3159, 949 | 13.0 %, 55 | 2.4 %, 3989 |
| uniform | 5 | Phong projection after each iteration (experiment) | 89.8 | 1789 | − | 10.2 %, 122 | 2.6 %, 3980 |
| curvature | 4 | midpoint, no projection | 40.8 | 1788 | − | − | − |
| curvature | 4 | paraboloid (positions), radius 3 | 39.5 | 1761 | 4.1 %, 2345, 655 | − | − |
| curvature | 4 | paraboloid (positions), radius 4 | 39.4 | 1757 | 4.3 %, 2389, 671 | − | − |
| curvature | 4 | paraboloid (+ normals), radius 3 | 40.7 | 1778 | 4.7 %, 3112, 585 | − | − |
| curvature | 4 | paraboloid (+ normals), radius 4 | 41.2 | 1784 | 4.8 %, 3169, 594 | − | − |
| curvature | 4 | Phong projection (once per call, the library) | 40.1 | 1782 | − | 3.3 %, 118 | 2.6 %, 1997 |
| curvature | 4 | Phong projection + paraboloid (positions) | 42.5 | 1891 | 4.5 %, 2408, 765 | 3.2 %, 121 | 2.5 %, 1996 |
| curvature | 4 | Phong projection + paraboloid (+ normals) | 43.1 | 1916 | 5.8 %, 3121, 760 | 3.2 %, 121 | 2.4 %, 2001 |
| curvature | 4 | Phong projection after every sweep (experiment) | 44.3 | 1981 | − | 12.6 %, 55 | 2.4 %, 1992 |
| curvature | 4 | projection every sweep + paraboloid (positions) | 46.3 | 2068 | 4.2 %, 2386, 775 | 12.1 %, 55 | 2.2 %, 1999 |
| curvature | 4 | projection every sweep + paraboloid (+ normals) | 46.9 | 2095 | 5.4 %, 3130, 779 | 11.9 %, 55 | 2.2 %, 1998 |
| curvature | 4 | Phong projection after each iteration (experiment) | 42.6 | 1894 | − | 9.1 %, 113 | 2.4 %, 2000 |
| curvature | 5 | midpoint, no projection | 113.1 | 1693 | − | − | − |
| curvature | 5 | paraboloid (positions), radius 3 | 114.0 | 1727 | 3.0 %, 2382, 1393 | − | − |
| curvature | 5 | paraboloid (positions), radius 4 | 113.7 | 1725 | 3.0 %, 2377, 1393 | − | − |
| curvature | 5 | paraboloid (+ normals), radius 3 | 115.5 | 1742 | 4.0 %, 3156, 1381 | − | − |
| curvature | 5 | paraboloid (+ normals), radius 4 | 116.2 | 1740 | 4.0 %, 3175, 1387 | − | − |
| curvature | 5 | Phong projection (once per call, the library) | 117.3 | 1784 | − | 3.6 %, 133 | 2.6 %, 5553 |
| curvature | 5 | Phong projection + paraboloid (positions) | 122.3 | 1864 | 2.9 %, 2426, 1418 | 3.5 %, 134 | 2.5 %, 5549 |
| curvature | 5 | Phong projection + paraboloid (+ normals) | 123.7 | 1883 | 3.8 %, 3171, 1427 | 3.4 %, 131 | 2.5 %, 5557 |
| curvature | 5 | Phong projection after every sweep (experiment) | 130.6 | 1992 | − | 13.3 %, 60 | 2.3 %, 5564 |
| curvature | 5 | projection every sweep + paraboloid (positions) | 133.9 | 2046 | 2.6 %, 2376, 1432 | 12.9 %, 60 | 2.2 %, 5561 |
| curvature | 5 | projection every sweep + paraboloid (+ normals) | 135.1 | 2063 | 3.5 %, 3158, 1426 | 12.8 %, 60 | 2.2 %, 5562 |
| curvature | 5 | Phong projection after each iteration (experiment) | 125.4 | 1910 | − | 9.8 %, 128 | 2.4 %, 5562 |

Fits per step count every evaluated fit (splits and collapse approvals). The projection
shares are of the remesh time. Projected moves over the period: once per call 1.2M
(uniform, r4) to 6.2M (curvature, r5); after every sweep 10.6M to 55.5M. A projection
once per call is a cold start (binary search on the global id and a walk from the
vertex's own reference fan), 107–134 ns; after every sweep most start from the previous
sweep's triangle, 51–60 ns. The largest reference, 5.6 MB, belonged to a 105k-triangle
mesh: 53 B per triangle, against the pool's 140 B per triangle plus 64 B per vertex
(about 172 B per triangle), held only during a kernel call. The smoother's projection
state adds 16 B per vertex for the same time.

Counts over the 128 runs of the first batch: 15,486,270 fits, 9 rank-deficient, 1,880
fallbacks to the midpoint. Missed projections (no reference triangle within half the
shortest edge): 0.002–0.007 % of the moves.

### 9.3 Exact sphere: placement order and the solver

Unit icosphere, every edge split once (`test_split_points_land_on_the_sphere`): mean
|1 − r| of the new vertices

| refinement | midpoint | paraboloid (positions, radius 3) | + normals |
|---|---|---|---|
| 2 | 1.120e-2 | 2.009e-3 | 2.633e-3 |
| 3 | 2.830e-3 | 1.251e-4 | 1.592e-4 |
| 4 | 7.085e-4 | 7.928e-6 | 1.002e-5 |

The midpoint is second-order (the sagitta); both fits are fourth-order (×16 per halving).

Jittered unit sphere (vertices moved tangentially by up to 0.2 of an edge, back onto the
sphere), one call of three smoothing sweeps (`test_projected_smoothing_stays_on_the_sphere`):
mean |1 − r| tangential 2.33e-3 / 5.37e-4 / 1.30e-4 at refinement 2 / 3 / 4, projected
(angle-weighted reference normals) 9.00e-4 / 1.81e-4 / 4.21e-5: second-order, three times
smaller. With Max-weighted reference normals 2.28e-4 / 1.48e-5 / 9.0e-7, fourth-order,
but over the LeVeque period those were worse at refinement 4 (curvature sizing: d_rms
0.194 against 0.136 % R0, E_H 0.29 against 0.15) and equal at 5, so the library uses angle
weights for the reference and Max's for the fit (for the fit, angle weights measured worse:
N_H 0.555 against 0.49, curvature, r4).

The solver against LAPACK `dgelsy` (`studies/surface_fit/`), the placement's own systems on
an icosphere of radius R:

```
dgelsy optimal workspace: 242 doubles

1. The library's problems (two-ring, scaled, Wendland radius 3), icosphere R = 1
ref  rows       edges     m |    QRCP ns  dgelsy ns   ratio | max |dx|/|x|  rank diff |     QRCP err   dgelsy err
3    pos         1920  23.8 |      381.6     1307.7    3.43 |     1.47e-15          0 |    5.075e-02    5.075e-02   (0)
3    pos+norm    1920  71.3 |     1209.4     2431.2    2.01 |     2.08e-15          0 |    6.339e-02    6.339e-02   (0)
4    pos         7680  23.9 |      410.1     1383.2    3.37 |     1.52e-15          0 |    1.267e-02    1.267e-02   (0)
4    pos+norm    7680  71.8 |     1289.4     2533.8    1.97 |     2.31e-15          0 |    1.552e-02    1.552e-02   (0)
5    pos        30720  24.0 |      524.8     1461.0    2.78 |     1.76e-15          0 |    3.172e-03    3.172e-03   (0)
5    pos+norm   30720  72.0 |     1323.0     2590.3    1.96 |     2.25e-15          0 |    3.861e-03    3.861e-03   (0)
   err: mean |z0 - z_sphere| / z_sphere at the edge midpoint (the sagitta's relative error)

1b. Mean relative sagitta error on the exact sphere against the Wendland radius (QRCP)
ref  | radius 2  pos / +norm   | radius 2.5              | radius 3                | radius 4                | mftl 8 pts, scaled
3    |  1.359e-02  1.314e-02 |  2.481e-02  3.129e-02 |  5.075e-02  6.339e-02 |  8.871e-02  1.355e-01 |  1.298e-02
4    |  3.361e-03  3.238e-03 |  6.124e-03  7.663e-03 |  1.267e-02  1.552e-02 |  2.205e-02  3.314e-02 |  3.220e-03
5    |  8.375e-04  8.063e-04 |  1.524e-03  1.904e-03 |  3.172e-03  3.861e-03 |  5.501e-03  8.247e-03 |  8.031e-04

2. Placement error against the sphere's sagitta at refinement 4, sphere radius R
R        | mftl: 8 pts, unscaled, dgelsy | mftl's 8 pts, scaled, QRCP   | scaled two-ring, dgelsy      | scaled two-ring, QRCP       
         |  mean err   max err   rank<6 |  mean err   max err   rank<6 |  mean err   max err   rank<6 |  mean err   max err   rank<6
1e+00    |  3.22e-03  3.82e-03        0 |  3.22e-03  3.82e-03        0 |  1.27e-02  1.52e-02        0 |  1.27e-02  1.52e-02        0
1e-02    |  3.22e-03  3.82e-03        0 |  3.22e-03  3.82e-03        0 |  1.27e-02  1.52e-02        0 |  1.27e-02  1.52e-02        0
1e-03    |  3.22e-03  3.82e-03        0 |  3.22e-03  3.82e-03        0 |  1.27e-02  1.52e-02        0 |  1.27e-02  1.52e-02        0
1e-04    |  4.51e+00  4.95e+00     7680 |  3.22e-03  3.82e-03        0 |  1.27e-02  1.52e-02        0 |  1.27e-02  1.52e-02        0
1e-05    |  4.51e+00  4.95e+00     7680 |  3.22e-03  3.82e-03        0 |  1.27e-02  1.52e-02        0 |  1.27e-02  1.52e-02        0
1e-06    |  4.51e+00  4.95e+00     7680 |  3.22e-03  3.82e-03        0 |  1.27e-02  1.52e-02        0 |  1.27e-02  1.52e-02        0
   (the columns are ordered 1, x, y, x^2, xy, y^2 in every form; mftl's own order only permutes them)

3. Rank-deficient stencils (scaled coordinates)
   (a) ring of 8 at z = 1:   QRCP rank 5  c0 = 1.000000  (c3, c4, c5) = (-0.000, 0.000, 0.000)
                             dgelsy rank 5  c0 = 0.666667  (c3, c4, c5) = (0.333, -0.000, 0.333)
   (b) strip along x, z = 0.2 + 0.1x - 0.3x^2:  QRCP rank 3 c0 = 0.200000   dgelsy rank 3 c0 = 0.200000
```

### 9.4 Distributed run

np = 4, refinement 4, curvature sizing, `-budget 4 -iterations 1`, full period, base
orientation, the final library:

| | V/V(0) − 1 | off the exact sphere at T: rms / mean / max [% R0] |
|---|---|---|
| midpoint, no projection | −3.540 % | 4.27 / −2.81 / 18.2 |
| Phong projection (once per call) | −0.111 % | 0.56 / +0.04 / 5.4 |
| projection + paraboloid (positions) | −0.115 % | 0.48 / +0.02 / 4.9 |

Euler 2 and no inverted triangle in each. (The first version, projecting after every
sweep, gave −0.114 % with both options.) On rank 0, 17,692 of 51,332 fits (34 %) had a ghost among their
two-ring samples; none can have one in an endpoint's one-ring (the freeze rule keeps it
owned and unreplicated).

### 9.5 When to project: every sweep, each iteration, once per call

A remesh call here makes 3 iterations of 3 smoothing sweeps. Projecting after every sweep
(9 per call), after the last sweep of each iteration (3) and once after the call's last
sweep (1) were run on the same 4 orientations at both refinements
(`archive/projection_schedule_experiment.patch`, `MFTL_EXP_PROJSWEEP`; the once-per-call
build is bit-identical to the library's final form, verified on two runs). From 9.1–9.2:

| sizing | r | every sweep: dV, N_H, projection share | each iteration | once per call (library) |
|---|---|---|---|---|
| uniform | 4 | −1.49 ± 0.04 %, 0.31, 13.3 % | −1.53 ± 0.02 %, 0.31, 9.5 % | −1.60 ± 0.03 %, 0.33, 3.5 % |
| uniform | 5 | −0.176 %, 0.079, 13.5 % | −0.180 %, 0.078, 10.2 % | −0.182 %, 0.082, 3.7 % |
| curvature | 4 | −0.089 %, 0.075, 12.6 % | −0.091 %, 0.073, 9.1 % | −0.093 %, 0.075, 3.3 % |
| curvature | 5 | −0.030 %, 0.023, 13.3 % | −0.030 %, 0.024, 9.8 % | −0.031 %, 0.025, 3.6 % |

Once per call is within the spread of every sweep except under-resolved uniform sizing at
refinement 4, where the paraboloid placement closes the gap (−1.49 %). The remesh cost per
triangle-step over the baseline falls from +18–20 % to +5–10 %. The library projects once
per call.

### 9.6 Ghost normals for the fit

np = 4, refinement 4, budget 4, one iteration per placement, paraboloid placement with
normals (radius 3), no projection, 4 orientations; distance to the exact sphere at T as a
vertex average (the example's distributed diagnostic; it is not the area-weighted d_rms of
section 4). Mode 0 is the library: a ghost sample contributes its position only. Mode 1
gives every ghost its owner's Max-weighted normal (one `MFTLUpdateGhostVertexNormals` per
placement, kept by global id) for the filter and the slope rows. Mode 2 drops ghost
samples (`archive/ghost_normals_experiment.patch`, `MFTL_EXP_GHOSTNRM`).

| sizing | run | V/V(0) − 1 [%] | rms off the sphere [% R0] | rank-0 fits, with ghosts |
|---|---|---|---|---|
| curvature | np = 1 | −0.171 ± 0.011 | 0.92 ± 0.08 | 113k, 0 |
| curvature | np = 4, ghost positions only | −0.005 ± 0.019 | 1.39 ± 0.20 | 40.0k, 11.9k |
| curvature | np = 4, + owners' ghost normals | −0.002 ± 0.014 | 1.47 ± 0.25 | 40.0k, 11.9k |
| curvature | np = 4, no ghost samples | −0.021 ± 0.014 | 1.40 ± 0.07 | 39.7k, 0 |
| uniform | np = 1 | −1.256 ± 0.042 | 1.00 ± 0.03 | 47.4k, 0 |
| uniform | np = 4, ghost positions only | −1.081 ± 0.021 | 0.95 ± 0.09 | 13.0k, 3.7k |
| uniform | np = 4, + owners' ghost normals | −1.018 ± 0.055 | 0.91 ± 0.05 | 12.8k, 3.6k |
| uniform | np = 4, no ghost samples | −1.060 ± 0.041 | 0.99 ± 0.17 | 12.8k, 0 |

The three modes agree within the spread. About 30 % of rank 0's fits reach a ghost, all
in the two-ring, where the Wendland weights are small (about 0.05 at radius 3); an
endpoint's one-ring is never a ghost (freeze rule). The difference between np = 1 and
np = 4 is the spanning driver's (four placements, the curvature field recomputed per
placement), not the ghosts'. Not taken.

### 9.7 Mesh quality

Mean triangle quality q = 4√3 A / Σ l² at T and the worst triangle over T/4..T (means over
4 orientations):

| sizing | r | baseline | projection once per call | projection every sweep | paraboloid (positions) |
|---|---|---|---|---|---|
| uniform | 4 | 0.9775 / 0.639 | 0.9767 / 0.716 | 0.9771 / 0.748 | 0.9776 / 0.730 |
| uniform | 5 | 0.9748 / 0.653 | 0.9756 / 0.627 | 0.9753 / 0.673 | 0.9755 / 0.664 |
| curvature | 4 | 0.9729 / 0.528 | 0.9759 / 0.590 | 0.9765 / 0.522 | 0.9758 / 0.598 |
| curvature | 5 | 0.9738 / 0.544 | 0.9751 / 0.564 | 0.9753 / 0.585 | 0.9745 / 0.591 |

The projection does not undo the smoother's work: the means are equal or better, the
worst triangles within the spread (sd 0.02–0.11).

### 9.8 Which projection, and what it is physically

The library projects onto the GLOBAL reference, the mesh as the remesh call received it,
with a local search. That surface is the advected front: the remesh call follows the
advection step. Remeshing is meant to change the front's discretisation, not its
position, so every normal displacement the remesher makes (collapse to a chord, a flip
off the convex diagonal, a tangent-plane step leaving a curved surface, a placement off
the surface) is a discretisation error, and the projection takes it back to the advected
front as its vertices and normals describe it (the Phong interpolant: second-order with
angle-weighted normals on an irregular mesh, fourth-order on a sphere with exact normals).
Three consequences:

- What the advection put there stays: noise from an interpolated velocity included. The
  projection is not a smoother of the interface.
- A normal motion made ON PURPOSE inside the remesh call (a TSUR3D-style denoising, a
  volume correction) would be taken back too. It has to run outside the remesher, or
  produce the reference.
- The enclosed volume is not conserved exactly: it follows the advected front, up to the
  difference between an inscribed polyhedron at the old and the new resolution, and
  nothing accumulates from call to call.

### 9.9 Reading section 9

1. The library's projection reproduces the experiment of section 7.
2. The paraboloid placement alone removes 50–82 % of the volume loss and 42–77 % of the
   curvature noise (positions). The normals' slopes help volume and position everywhere,
   and noise except under curvature sizing at refinement 4.
3. With the projection on, the placement adds nothing beyond the spread, except under
   uniform sizing at refinement 4 (−1.49 against −1.60 %).
4. The Wendland radius trades accuracy on smooth data (smaller is better, 9.3) against
   noise rejection on the front (4 smoother than 3).
5. The projection, once per call, costs 3.3–3.7 % of the remesh plus 2.4–2.8 % for the
   copies, +5–10 % per triangle-step (after every sweep: 12.6–13.5 %, +18–20 %); a fit
   2.3–3.1 µs, 2.5–5.8 % of the remesh.
6. The rank rule practically never fires (9 in 15.5 million); the solver's speed (2–3.4×
   `dgelsy`) and the scaling are what matter.
7. Projecting once per call is as good as after every sweep (9.5); ghost normals for the
   fit buy nothing (9.6); the projection leaves mesh quality alone (9.7).

---

## 10. Results F: front corrections outside the remesher

Two volume-conserving corrections a front-tracking solver makes between flow steps, in the
library as `src/correction/` (`MFTLEnsembleCorrectVolumes`, `MFTLEnsembleRemoveUndulations`).
Both move vertices along the surface normal. Both are deliberately separate calls in the
solver's loop and never part of a remesh: the Phong back-projection of section 9 takes
back every normal displacement made inside its call.

### 10.1 Methods

**Volume correction.** Each owned vertex of a body moves by d u, with u the area-weighted
unit vertex normal (a rim vertex keeps only its in-wall components), and

  d = (V_ref − V) / Σ_t S_t · (u_0 + u_1 + u_2)/3,

the exact first variation of the volume of the field actually applied (S_t the area
vector), iterated as Newton's method a fixed number of times (default 3, so every rank
runs the same collectives). The reference volume is per body, measured once
(`MFTLEnsembleSetReferenceVolumes`) and carried by the library. mftl's `VolumeRecovery` is
the same field with one step of d = ΔV/A, with the area in place of the first variation.

**TSUR3D** (de Sousa et al. 2004, mftl `tsur3d.c`). Per edge a-b, n is the normalised sum
of the two fans' area-weighted normals, computed with both ends moved to their one-ring
centroids. About the edge midpoint the ends sit at h_a = −h_b. Both move along n to

  h = (h_a P_a + h_b P_b) / (P_a + P_b),

with P the fans' areas projected normal to n. Moving an end along n changes the volume by
its displacement times P/3, and no cross term arises because both moves are parallel. So
each operation conserves the volume exactly. The published h = (V_1 + V_2)/(V_1/h_1 +
V_2/h_2) is this with V_i = h_i P_i/3; written out, its 0/0 on an already-flat edge
disappears, and h is a convex combination, so neither end moves beyond the edge's kink.

Further changes against mftl:

- fans folded against n are refused, and so is any move that would turn a triangle over;
- minimum-image geometry replaces mftl's clamp into the domain, which broke the
  conservation;
- feature, corner and rim vertices stay fixed;
- a conservative surface density keeps its amount per triangle, applied once per call;
- a threshold on the kink angle (`min_angle`), whose library default became 15° after
  this study (the study's "every edge" rows set it to 0);
- the distributed form needs no gather onto one rank: rank-private edges first, then
  colour rounds in which no two neighbouring ranks are active, each round followed by a
  ghost exchange.

**Additional cases.**

- **"noise 0.05"** adds eps · dt · ξ to every vertex every step, with ξ uniform in
  [−1, 1]³ and hashed from (vertex id, step), and eps = 0.05 against a peak LeVeque speed
  of about 2. It is a model of the sub-grid fluctuations an interpolated velocity puts on
  markers: white in space and time, a worst case for grid-scale undulations.
- **"rotation"** turns the sphere rigidly once per period about its own centre. The mesh
  resolves the exact surface (the sphere) at all times, and every normal motion is error.

Setup otherwise as section 9 (uniform sizing, dt = 0.015, 200 steps, remeshed every step,
one rank, four orientations). Data: `results/leveque_remeshing_correction/`. The
correction runs after the remesh ("end"), or also directly after the advection ("both").

### 10.2 Volume correction (at T, mean ± sd over 4 orientations)

| r | variant | dV [%] | d_rms [% R0] | E_H | N_H | E_H at T/2 | share of the step |
|---|---|---|---|---|---|---|---|
| 4 | none | −3.90 ± 0.14 | 2.20 ± 0.14 | 1.61 ± 0.20 | 1.78 ± 0.19 | 0.92 | -- |
| 4 | Newton × 3, end | 0 | 2.07 ± 0.15 | 1.87 ± 0.12 | 1.94 ± 0.10 | 0.83 | 4.0 % |
| 4 | Newton × 3, both | 0 | 2.06 ± 0.22 | 1.88 ± 0.19 | 1.91 ± 0.12 | 0.83 | 7.5 % |
| 4 | Newton × 1, end | 0 | 2.07 ± 0.15 | 1.87 ± 0.12 | 1.94 ± 0.10 | 0.83 | 2.3 % |
| 4 | mftl area step × 1, end | −0.0002 | 2.01 ± 0.15 | 1.79 ± 0.18 | 1.86 ± 0.16 | 0.83 | 2.3 % |
| 4 | Phong projection | −1.60 ± 0.03 | 1.55 ± 0.06 | 0.55 ± 0.08 | 0.33 ± 0.07 | 0.84 | -- |
| 4 | projection + volume, end | 0 | 1.81 ± 0.07 | 1.05 ± 0.06 | 0.72 ± 0.06 | 0.80 | 3.8 % |
| 4 | projection + volume, both | 0 | 1.80 ± 0.09 | 1.03 ± 0.06 | 0.71 ± 0.05 | 0.80 | 7.1 % |
| 5 | none | −1.17 ± 0.01 | 0.641 ± 0.011 | 1.29 ± 0.06 | 1.52 ± 0.07 | 0.56 | -- |
| 5 | Newton × 3, end | 0 | 0.547 ± 0.008 | 1.36 ± 0.01 | 1.50 ± 0.01 | 0.53 | 3.9 % |
| 5 | Newton × 3, both | 0 | 0.547 ± 0.005 | 1.37 ± 0.02 | 1.52 ± 0.05 | 0.53 | 7.4 % |
| 5 | Newton × 1, end | 0 | 0.547 ± 0.008 | 1.36 ± 0.01 | 1.50 ± 0.01 | 0.53 | 2.2 % |
| 5 | mftl area step × 1, end | 0.0000 | 0.545 ± 0.005 | 1.37 ± 0.01 | 1.52 ± 0.03 | 0.53 | 2.2 % |
| 5 | Phong projection | −0.18 ± 0.00 | 0.247 ± 0.002 | 0.17 ± 0.02 | 0.082 ± 0.014 | 0.48 | -- |
| 5 | projection + volume, end | 0 | 0.278 ± 0.004 | 0.35 ± 0.01 | 0.207 ± 0.014 | 0.47 | 3.7 % |
| 5 | projection + volume, both | 0 | 0.277 ± 0.005 | 0.35 ± 0.01 | 0.210 ± 0.006 | 0.47 | 7.0 % |

The diagnostic for the projection rows, r5, one orientation, at T:

| correction applied | E_H | N_H |
|---|---|---|
| never | 0.170 | 0.082 |
| always | 0.364 | 0.207 |
| only after 3T/4 | 0.181 | 0.098 |
| only before 3T/4 | 0.349 | 0.225 |

### 10.3 TSUR3D (at T)

| r | case | none | every edge | kinks > 5° | kinks > 15° | every edge + volume |
|---|---|---|---|---|---|---|
| 4 | rotation, d_rms [% R0] | 0.006 | 0.093 | -- | -- | -- |
| 4 | rotation + noise, d_rms | 2.80 ± 0.02 | 0.58 ± 0.04 | 0.76 ± 0.01 | 1.13 ± 0.05 | 0.52 ± 0.04 |
| 5 | rotation, d_rms | 0.002 | 0.013 | -- | -- | -- |
| 5 | rotation + noise, d_rms | 1.60 ± 0.02 | 0.51 ± 0.02 | 0.60 ± 0.03 | 0.80 ± 0.02 | 0.47 ± 0.02 |
| 4 | LeVeque, d_rms / E_H | 2.20 / 1.61 | 56.7 / 5.37 | -- | 2.70 / 1.26 | -- |
| 5 | LeVeque, d_rms / E_H | 0.64 / 1.29 | 4.24 / 1.27 | -- | 0.66 / 1.23 | -- |
| 4 | LeVeque + noise, d_rms / E_H | 6.20 / 6.97 | 60.8 / 5.97 | -- | 5.48 / 2.17 | (15° + volume) 7.31 / 2.41 |
| 5 | LeVeque + noise, d_rms / E_H | 3.67 / 14.2 | 5.20 / 1.88 | -- | 2.66 / 4.37 | (15° + volume) 3.15 / 4.42 |

Every TSUR3D call conserved the volume to 1e-15, measured inside the LeVeque loop. Cost at
r5, each configuration alone: one sweep is 14.7 % of the step (np = 1) and 4.4 % at
np = 4; with the 15° threshold, 9.0 % and 2.6 %.

### 10.4 Distributed TSUR3D, cost, and the default

**Distributed.** A jittered sphere (refinement 4, radius 0.6, noise 1 % of R, three
sweeps of every edge) on a periodic box cut at np = 1 … 27 (`test_correction`):

| np | 1 | 2 | 4 | 8 | 27 |
|---|---|---|---|---|---|
| volume change | 1.5e-15 | 1.1e-15 | 2.5e-16 | 1.2e-16 | 1.2e-16 |
| moves (of them in colour rounds) | 23 040 (0) | 22 470 (1 962) | 21 906 (3 576) | 21 348 (5 220) | 20 358 (7 833) |
| colour rounds | 0 | 6 | 12 | 24 | 72 |
| cross-rank edges per sweep (never moved) | 0 | 190 | 378 | 564 | 894 |
| residual radial rms against one rank | 1 | 1.017 | 1.029 | 1.048 | 1.101 |

- **Exactness:** the conservative surface density's total amount is conserved to
  2e-16, and every ghost position and ghost density equals its owner's to the bit.
- **What it costs:** edges whose ends have different owners are never moved, and the
  residual noise grows with the cut planes' share of the surface.

**Cost** (refinement 5, LeVeque, each configuration alone, best of 2, Release, Apple M4;
np = 4 with `-budget 4`):

| | np = 1, share of the step | np = 4, ms/step and share |
|---|---|---|
| volume correction, Newton × 3, once per step | 4.1 % (≈ 4.9 ms) | 3.9 ms, 0.9 % |
| volume correction, Newton × 1 | 2.3 % | -- |
| volume correction, Newton × 3, twice per step | 7.7 % | -- |
| TSUR3D, one sweep, every edge | 14.7 % (≈ 16.7 ms) | 19.7 ms, 4.4 % |
| TSUR3D, one sweep, kinks > 15° | 9.0 % (≈ 10.3 ms) | 11.6 ms, 2.6 % |

- **TSUR3D at np = 4** is no faster in absolute time than on one rank. Its four colour
  rounds (2×2×1) each end with a ghost exchange, and on a mesh of this size the
  exchanges are the cost.

**The default.** After this study the library's `min_angle` default became **15°**. The
published every-edge method (0) destroyed the LeVeque filament, and 15° was neutral in
position and better in curvature there. A clean resolved icosphere is a fixed point of
the default to the bit, and noise is removed down to the threshold's own level, about
tan 15° · ℓ/2 of height (0.41× of a 5 % jitter in five sweeps,
`test_the_default_spares_a_resolved_surface_and_still_removes_noise`). Where the mesh
resolves the interface and the noise matters most, 0 or a few degrees removes more: on
the rotating sphere at r5, 0.51 % R0 at 0°, 0.60 % at 5° and 0.80 % at 15°.

### 10.5 Reading section 10

1. **Volume correction: once per step, after the remesh, is enough.** Correcting after the
   advection too gives the same volume (exact either way) and the same shape within the
   spread, at twice the cost. The correction is a uniform offset and the errors it
   removes add at first order.
2. **In a loop, one Newton step equals three to every digit, and mftl's area step equals
   Newton within the spread.** The per-step error (≤ 6.7e-4 of V) is small enough that
   the step rule does not matter. The exact first variation and the iteration matter
   for large deficits: from 8.7 %, one step leaves 2.8e-3 and three leave 2.3e-12. On
   a coarse satellite after a topology change, the area step converges at 0.03 per
   step. The exact form is the right default for a library that also serves those cases.
3. **A global correction misplaces locally lost volume, and the flow carries the
   misplacement.** Without projection it improves the position (bias −0.39 → +0.005 %
   R0 at r5) and leaves the curvature alone. With projection it doubles the curvature
   roughness at T. The diagnostic pins the cause on the offsets applied while the
   filament is stretched: they are uniform in the stretched configuration, while the
   loss happened at collapses on the filament. On compression a normal displacement
   scales with the local stretch, so a uniform offset becomes a bumpy one.
4. **TSUR3D removes noise where the mesh resolves the interface** (rotation + noise:
   2.80 → 0.58 % R0 at r4, 1.60 → 0.51 % at r5, minimum quality 0.81 → 0.93). On a
   resolved sphere it is third order (0.093 → 0.013 % from r4 to r5).
5. **Where the mesh does not resolve the interface, TSUR3D is a smoother and cannot be
   told apart from one.** Every edge every step destroys the LeVeque filament (56.7 %
   R0 at r4). With a 15° threshold it is neutral on position and improves the curvature.
6. **For surface tension the curvature counts.** With noise on LeVeque at r5, E_H falls
   from 14.2 to 1.9 (every edge) and 4.4 (15°). The 15° threshold also improves the
   position (3.67 → 2.66 % R0), and every edge worsens it (5.20 %).

### 10.6 Where the remesher loses volume, and three local fixes

**Method.** An instrumented build (`archive/volume_budget_experiment.patch`) measures two
things:

- the enclosed volume before and after each sweep of every remesh call;
- the EXACT local volume change of every collapse and flip, binned by the local bending
  angle. The bending angle is the largest angle between a patch triangle's normal and
  the patch normal; for a flip, the angle between the quad's two normals.

The two agree: the per-operation sums reproduce the per-sweep totals, and the midpoint
run reproduces section 7.1 to five digits. Three fixes, each exact by construction:

- **volume-preserving collapse (vpc).** The collapse's volume change is linear in the
  survivor's position (each surviving patch triangle holds it once), so the chosen
  target -- midpoint or paraboloid -- is moved along the gradient of that change until
  the change is zero. This is Lindstrom & Turk's (1998) volume constraint.
- **flip guard.** Refuse a flip whose volume change exceeds τ A_quad |e|/3, τ = 0.05: a
  fold of about 11° across the edge.
- **compensated flip.** After the flip, shift its four vertices by ONE vector along the
  quad normal, sized to restore the volume. A common shift changes every determinant
  linearly, so this is exact.

Uniform sizing, four orientations, otherwise as 10.1. Data:
`results/leveque_remeshing_volume_ops/` (`volume_ops_study.py`).

**Budget over the period (% of V0) and accuracy at T:**

| r | config | fix | split | collapse | flip | smooth | dV [%] | d_rms [% R0] | E_H | N_H |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | midpoint | none | 0 | −2.94 | −1.09 | +0.18 | −3.90 ± 0.14 | 2.17 ± 0.14 | 1.61 | 1.78 |
| 4 | midpoint | vpc | 0 | 0 | −1.35 | +0.18 | −1.22 ± 0.06 | 0.89 ± 0.01 | 1.78 | 2.15 |
| 4 | midpoint | flip guard | 0 | −3.34 | −0.49 | +0.15 | −3.74 ± 0.04 | 2.07 ± 0.04 | 1.31 | 1.48 |
| 4 | midpoint | compensated flip | 0 | −3.11 | 0 | +0.19 | −2.97 ± 0.02 | 2.03 ± 0.15 | 1.66 | 1.78 |
| 4 | midpoint | vpc + compensated flip | 0 | 0 | 0 | +0.18 | +0.13 ± 0.01 | 0.60 ± 0.09 | 1.72 | 2.09 |
| 4 | projection | none | 0 | −2.90 | −1.28 | +2.62 | −1.60 ± 0.03 | 1.55 ± 0.06 | 0.55 | 0.33 |
| 4 | projection | vpc | 0 | 0 | −1.50 | +0.11 | −1.44 ± 0.01 | 1.34 ± 0.01 | 0.38 | 0.22 |
| 4 | projection | flip guard | 0 | −3.47 | −0.55 | +2.48 | −1.59 ± 0.03 | 1.51 ± 0.05 | 0.54 | 0.34 |
| 4 | projection | compensated flip | 0 | −2.89 | 0 | +1.37 | −1.56 ± 0.02 | 1.51 ± 0.02 | 0.55 | 0.35 |
| 4 | projection | vpc + compensated flip | 0 | 0 | 0 | −1.36 | −1.41 ± 0.04 | 1.33 ± 0.04 | 0.39 | 0.22 |
| 4 | proj. + paraboloid | none | +0.68 | −1.64 | −0.94 | +0.46 | −1.49 ± 0.01 | 1.41 ± 0.02 | 0.47 | 0.30 |
| 4 | proj. + paraboloid | vpc | +0.69 | 0 | −1.04 | −1.00 | −1.41 ± 0.05 | 1.32 ± 0.05 | 0.38 | 0.21 |
| 5 | midpoint | none | 0 | −0.83 | −0.38 | +0.04 | −1.18 ± 0.01 | 0.63 ± 0.01 | 1.29 | 1.52 |
| 5 | midpoint | vpc | 0 | 0 | −0.45 | +0.05 | −0.41 ± 0.01 | 0.26 ± 0.00 | 1.78 | 2.18 |
| 5 | midpoint | flip guard | 0 | −0.92 | −0.24 | +0.04 | −1.13 ± 0.01 | 0.61 ± 0.01 | 1.20 | 1.41 |
| 5 | midpoint | compensated flip | 0 | −0.84 | 0 | +0.05 | −0.80 ± 0.01 | 0.44 ± 0.01 | 1.23 | 1.47 |
| 5 | midpoint | vpc + compensated flip | 0 | 0 | 0 | +0.05 | +0.035 ± 0.001 | 0.120 ± 0.005 | 1.54 | 1.91 |
| 5 | projection | none | 0 | −0.84 | −0.42 | +1.08 | −0.182 ± 0.002 | 0.247 ± 0.002 | 0.17 | 0.082 |
| 5 | projection | vpc | 0 | 0 | −0.46 | +0.30 | −0.175 ± 0.003 | 0.236 ± 0.004 | 0.15 | 0.066 |
| 5 | projection | vpc + compensated flip | 0 | 0 | 0 | −0.16 | −0.173 ± 0.001 | 0.236 ± 0.003 | 0.15 | 0.066 |
| 5 | proj. + paraboloid | none | +0.23 | −0.47 | −0.28 | +0.35 | −0.178 ± 0.003 | 0.244 ± 0.003 | 0.16 | 0.073 |
| 5 | proj. + paraboloid | vpc | +0.23 | 0 | −0.31 | −0.08 | −0.174 ± 0.000 | 0.236 ± 0.002 | 0.16 | 0.068 |

(The flip-guard and compensated-flip rows with the paraboloid, and the rest of r5, are in
`results/leveque_remeshing_volume_ops/tables.md`; they change nothing below.) The
volume-preserving collapse costs +0.5–1.0 % of the step (in the library; below).

**Where the loss is** (base orientation, midpoint, by the local bending angle):

| r | angle | collapses | their dV [%] | flips | their dV [%] |
|---|---|---|---|---|---|
| 4 | < 5° | 10 669 | −0.16 | 30 739 | −0.23 |
| 4 | 5–10° | 4 829 | −0.67 | 4 816 | −0.30 |
| 4 | 10–20° | 2 569 | −0.65 | 2 574 | −0.30 |
| 4 | 20–40° | 1 358 | −0.61 | 1 288 | −0.23 |
| 4 | > 40° | 729 | −0.65 | 581 | −0.11 |
| 5 | < 5° | 74 008 | −0.27 | 166 515 | −0.17 |
| 5 | 5–10° | 12 334 | −0.21 | 11 149 | −0.09 |
| 5 | > 10° | 7 228 | −0.34 | 6 131 | −0.13 |

**Operations per half period** (r4, base): the first half (area 5k → 18k triangles) has
15 417 splits, 8 777 collapses and 27 776 flips; the second half has 4 567, 11 377 and
12 222. The flow stretches the filament along one direction and compresses it across the
other, so the same material is split in one direction and collapsed in the other
throughout.

**Reading.**

1. **The loss is made by collapses and flips, never by splits.** A midpoint split leaves
   the piecewise-linear surface unchanged; a paraboloid split ADDS volume. A collapse
   or flip replaces a bent patch by a flatter one, and loses roughly in proportion to
   the square of the bending. At r4, 94 % of the collapse loss comes from the 47 % of
   collapses whose patch bends by more than 5° -- the filament's under-resolved rims.
   At r5 the bending is smaller everywhere, and the loss spreads over the flat
   majority.
2. **The volume-preserving collapse is the lever.** Without projection it removes the
   collapse loss exactly. It takes the volume error from −3.90 to −1.22 % (r4) and
   −1.18 to −0.41 % (r5), and the position error from 2.17 to 0.89 and 0.63 to 0.26 %
   R0. But it raises the curvature noise (N_H 1.52 → 2.18 at r5): it restores a whole
   patch's deficit at one vertex, a point bump. **With the Phong projection it improves
   every metric** (r4: d_rms 1.55 → 1.34, E_H 0.55 → 0.38, N_H 0.33 → 0.22; r5: 0.247 →
   0.236, 0.17 → 0.15, 0.082 → 0.066), because the smoothing and projection that follow
   remove the bump. With projection plus paraboloid it improves r4 the same way and is
   neutral at r5.
3. **A flip guard is the wrong tool.** It refused half the flips (20 673 at r4) and cut
   the flip loss by 55–60 %, but the worse valences made the collapses lose more: net
   −3.74 against −3.90 %. It improved the curvature without projection (E_H 1.61 →
   1.31 at r4) and nothing with it.
4. **A compensated flip is exact, and helps only without projection** (r5: −1.18 →
   −0.80 %, 0.63 → 0.44 % R0). With projection it buys nothing. Its shifts are normal
   moves made inside the remesh call, and the call's back-projection takes them back:
   the smoothing budget falls from +1.37 to −1.36 % at r4 when both flips and collapses
   are exact. The same reason puts any LOCAL restoration of the remaining loss after
   the call, not inside it.
5. **Volume-preserving collapse plus compensated flip without projection is nearly
   exact** (+0.035 % at r5, with only the smoothing and advection left) and has the
   best position (0.120 % R0 at r5, five times the base), but the curvature noise of
   item 2.

**In the library.** The volume-preserving collapse is
`MFTL_RemeshParameters::collapse_preserves_volume` (off by default; example flag
`-vpcollapse 1`), applied to the point `collapse_placement` chose. Split and collapse
placement are separate switches (`split_placement`, `collapse_placement`), both midpoint
by default. The flip variants are not in the library. Re-run from the library tree
(fix `vpc_lib`, 24 runs), every row above with vpc is reproduced in every metric, every
orientation and the triangle count. Its cost, serial Release, one projected run at a
time, median of three: 19.02 → 19.22 ms/step at r4 (+1.0 %), 83.45 → 83.89 at r5
(+0.5 %).

---

## 11. Claims a paper can make, with their support

1. **The volume loss of an isotropic remesher on a convex front is made by collapses and
   valence flips. Splits and tangential smoothing do not lose volume** (7.1, all
   isotropic rows). It is cumulative: each call's error becomes the next call's input.
2. **A remesher that tolerates stretched triangles conserves volume because it collapses
   less, not because of its metric**: the anisotropic mode without QEM, 7.1–7.2.
3. **Higher-order vertex placement in split/collapse alone halves the loss but does not
   remove it.** Phong midpoints: 7.3, 7.4 item 7. QEM placement goes further on volume
   and position, but at 1.4–3 times the curvature noise (7.4 item 3).
4. **Back-projection onto the flat input shrinks the front** (−12.5 to −25.6 % at r4). In
   graphics the reference is a fixed, finer original mesh; in front tracking it is the
   previous step's mesh at the same resolution.
5. **Back-projection onto a smooth (Phong) reconstruction of the call's input removes
   both the volume drift and the curvature noise.** Curvature sizing at r5: −0.03 %,
   d_rms 0.02 % R0, N_H 0.02, with 19.7k triangles, against the anisotropic mode's
   +0.06 %, 0.076, 8.48 with 59k triangles.
6. **The reference must be fixed for the call.** A reconstruction from the current mesh
   lowers the noise but loses more volume than no projection (7.4 item 7).
7. **Jiao's sweep order is worse than split→collapse→flip→smooth in every mode, also at
   equal smoothing work, and length sorting is neutral** (6.1).
8. **Curvature-adaptive sizing as published needs two corrections**: one smoothing pass
   of the curvature for a fixed point (8.1), and the ODT density in the smoother instead
   of the printed |t| L(b) weight (section 3.1; asserted by a sign test).
9. **The library form of the Phong back-projection, once per remesh call, keeps its
   accuracy at 5–10 % of the remesh cost** and needs no search structure and no
   communication (9.2, 9.4, 9.5); it leaves mesh quality alone (9.7).
10. **A vertex-based weighted paraboloid placement in scaled coordinates removes half to
    four fifths of the volume loss on its own and is fourth-order on a sphere; a
    facet-integral fit lands at the chord** (9.1, 9.3, 8.5), and an unscaled one fails at
    small length scales.
11. **A normal correction belongs outside the remesher, and a global volume correction is
    exact but not free.** Once per step suffices; one Newton step suffices in a loop. On a
    deforming flow a uniform offset misplaces locally lost volume, and with a
    well-conserving remesher (Phong projection) it doubles the curvature roughness at the
    end of the LeVeque period (10.2, 10.5 items 1–3).
12. **TSUR3D's volume-exact edge flattening removes grid-scale velocity noise on a resolved
    interface (×3–5), and on an under-resolved one it smooths real geometry as well.** A
    kink threshold is the practical switch: 15° turned it from destructive into neutral
    in position and beneficial in curvature on LeVeque (10.3, 10.5 items 4–6).

13. **An isotropic remesher's volume loss on a deforming front is made by collapses and
    flips of bent patches, and a volume-preserving collapse target removes the larger
    part exactly, for +0.5–1.0 % of the step.** With the Phong projection it improves
    position, curvature and noise together. A flip guard does not help. A compensated
    flip helps only without back-projection, which takes its normal shifts back (10.6).
    The library ships the collapse as an option (`collapse_preserves_volume`), not the
    flip variants.

---

## 12. Limitations

- One flow (LeVeque/Enright) with an exact, smooth velocity. Noise that a real flow
  solver's interpolated velocity introduces is not tested. No projection onto the input
  addresses it.
- One rank for everything in sections 5–7. Section 8.2 is the only distributed
  measurement of the sizing.
- The follow-up variants at refinement 5, the Phong-midpoint and local variants at both
  refinements, and the protocol rows are single runs.
- The back-projection's cost is not measured. The experiment searches a uniform grid
  over a copy of the input (20–40 times the plain pass). A production version would
  walk locally from the vertex's previous input triangle.
- The Phong surface uses the input's angle-weighted vertex normals with α = ½. Other
  normals and α, and other reconstructions (PN triangles, local polynomial fits), were
  not tried.
- Timings come from a laptop (Apple M4, 10 cores, of which 4 performance). The
  follow-up's ms/step are not clean timings.
- The local-projection mechanism (7.4 item 7) is an interpretation of the volume
  budget, not an isolated measurement.
- Section 10: the noise model is white in space and time. A real interpolated velocity
  is correlated over the kernel width, so TSUR3D's effect on it is not measured here. The
  distributed TSUR3D was measured for cost (np = 4) and correctness (np ≤ 27, unit
  tests), not for accuracy over the period. Edges with ends on different ranks are never
  moved by it. A local volume correction, putting back volume where the remesher lost
  it, was not tried.
- Section 9: the LeVeque front has no features and no rims, so the placement's and the
  projection's feature, corner and rim rules are exercised only by the unit tests; the
  distributed runs are np = 4 only; the fit radius was scanned at 3 and 4 only (2 to 4 on
  the exact sphere); the projection schedule was compared at 3 iterations of 3 sweeps
  only.

---

## 13. Reproduction

**Software.** MFTL2 at commit `bb3c785` plus the patches below; Apple clang 21.0.0,
Open MPI 5.0.9, macOS 27.0.1, Release build (`python3 build_all.py`). Python with NumPy
and SciPy for the evaluation; ParaView `pvbatch` for figures.

**Trees.**

| what | tree |
|---|---|
| sections 5, 7 (uniform, curvature rows without experiment variants) | the current tree |
| sections 5–7, anisotropic and QEM rows | `bb3c785` + `archive/sizing_with_anisotropic_and_qem.patch` |
| section 6 | the above + `archive/sweep_orders.patch` |
| section 7 experiment variants | `bb3c785` + `archive/sizing_with_anisotropic_and_qem.patch` + `followup_experiment.patch` |
| section 8 | the first archive tree; 8.1–8.2 are the fixed-point loops of `test_the_spanning_driver_under_curvature_sizing` |
| section 9 | the current tree: `placement_study.py run`, `evaluate.py --out ../../results/leveque_remeshing_placement`, `placement_study.py timing`, `placement_study.py tables`; `../surface_fit/run_benchmarks.sh` |
| section 9.5 rows "every sweep", "each iteration" | the current tree + `archive/projection_schedule_experiment.patch` (`MFTL_EXP_PROJSWEEP` unset / 1) |
| section 9.6 | the current tree + `archive/ghost_normals_experiment.patch` (`MFTL_EXP_GHOSTNRM` = 0, 1, 2); the example's "off the exact sphere" line |

`archive/README.md` documents the patches. The stored command lines of the main study
(first line of each `log.txt`) carry `-order mftl`; on the current tree drop it (MFTL's
order is the only one). Run that way, `curvature_r4_mftl`, `uniform_r4_mftl` and
`curvature_r5_mftl` reproduce their stored final rows bit for bit on the current tree. `sweep_orders.patch` was rebuilt from the
session record. Three study runs re-run with it match their stored series in every
column except the wall-clock time.

**Commands** (from `studies/leveque_remeshing/`, each tree built in its own directory and
passed with `--exe`):

```bash
python3 run_campaign.py [--exe B] [--replicas]     # sections 5 (and 6 with the sweep-order tree)
python3 evaluate.py                                 # metrics.json against the exact reference
python3 time_runs.py                                # clean timings (best of 2, alone)
python3 make_figures.py; python3 make_tables.py --template REPORT.template.md --report REPORT.md
python3 followup.py run --exe B --jobs 4            # section 7 (experiment tree)
python3 evaluate.py --out ../../results/leveque_remeshing_followup
python3 followup.py tables
```

Experiment switches (environment, experiment tree only): `MFTL_EXP_BUDGET=1` volume
budget; `MFTL_EXP_PROJECT=1|2|3` flat-stored, Phong-stored, local-fan projection;
`MFTL_EXP_PHONG=1` Phong midpoints; `MFTL_EXP_FLIP=1|2` Delaunay flips (tangent plane,
3D angles). Example options: `-refine`, `-sizing`, `-qem`, `-order`, `-sorted`,
`-smoothevery`, `-every`, `-rotate`, `-snapshots`, `-csv`, `-noremesh`, `-dt`.

Section 10.6: `python3 volume_ops_study.py run --exe E --jobs 4` (E built with
`archive/volume_budget_experiment.patch`; switches `MFTL_EXP_VBUDGET`,
`MFTL_EXP_FLIPGUARD=τ`, `MFTL_EXP_FLIPCOMP`; the volume-preserving collapse is the
library's `-vpcollapse 1`. The stored vpc runs were made with an earlier form of the
patch that switched it by `MFTL_EXP_VPCOLLAPSE=1`; the rebased patch reproduces them),
`python3 volume_ops_study.py run --exe B --only vpc_lib` for the library rows, then
`python3 evaluate.py --out ../../results/leveque_remeshing_volume_ops` and
`python3 volume_ops_study.py tables`.

Section 10: `python3 correction_study.py run --exe B --exp-exe E --jobs 4`, then
`python3 evaluate.py --out ../../results/leveque_remeshing_correction` and
`python3 correction_study.py tables`. E is a build with
`archive/volume_area_step_experiment.patch`, needed only for the mftl-area-step row
(`MFTL_EXP_VOLAREA=1`). Example options: `-volcorrect off|end|both`, `-volsteps N`,
`-tsur3d N`, `-tsur3d_at advect|end`, `-tsur3d_angle DEG`, `-noise EPS`,
`-field leveque|rotation`.

**Data** (untracked): `results/leveque_remeshing/` (main study: `runs/NAME/{log.txt,
series.csv, snap_q0..q4.vtk}`, `metrics.json`, `timing.json`) and
`results/leveque_remeshing_followup/` (`runs/`, `budget/`, `metrics.json`; 225 MB).
Figures: `studies/leveque_remeshing/figures/`. Run names: `MODE_rR_ORDER[@rotK]` and
`VARIANT_rR[@rotK]`.

---

## 14. References

- Boubekeur, T., Alexa, M. (2008). Phong tessellation. ACM Trans. Graph. 27(5), 141.
- Botsch, M., Kobbelt, L. (2004). A remeshing approach to multiresolution modeling.
  Proc. Symposium on Geometry Processing, 185–192.
- Chen, L., Xu, J. (2004). Optimal Delaunay triangulations. J. Comput. Math. 22(2),
  299–308.
- de Sousa, F. S., Mangiavacchi, N., Nonato, L. G., Castelo, A., Tomé, M. F., Ferreira,
  V. G., Cuminato, J. A., McKee, S. (2004). A front-tracking/front-capturing method for
  the simulation of 3D multi-fluid flows with free surfaces. J. Comput. Phys. 198(2),
  469–499, doi:10.1016/j.jcp.2004.01.032. (TSUR3D: the paper's mass-conserving filter of
  small undulations.)
- Dunyach, M., Vanderhaeghe, D., Barthe, L., Botsch, M. (2013). Adaptive remeshing for
  real-time mesh deformation. Eurographics 2013 Short Papers.
- Enright, D., Fedkiw, R., Ferziger, J., Mitchell, I. (2002). A hybrid particle level set
  method for improved interface capturing. J. Comput. Phys. 183, 83–116.
- Garland, M., Heckbert, P. S. (1997). Surface simplification using quadric error
  metrics. Proc. SIGGRAPH, 209–216.
- Jiao, X., Colombi, A., Ni, X., Hart, J. C. (2010). Anisotropic mesh adaptation for
  evolving triangulated surfaces. Engineering with Computers 26, 363–376.
- LeVeque, R. J. (1996). High-resolution conservative algorithms for advection in
  incompressible flow. SIAM J. Numer. Anal. 33(2), 627–665.
- Meyer, M., Desbrun, M., Schröder, P., Barr, A. H. (2003). Discrete
  differential-geometry operators for triangulated 2-manifolds. Visualization and
  Mathematics III, 35–57.
- Ohtake, Y., Belyaev, A. G., Seidel, H.-P. (2002). Mesh smoothing by adaptive and
  anisotropic Gaussian filter. Proc. Vision, Modeling and Visualization, 203–210.
- pmp-library (Sieger, D., Botsch, M.), `Remeshing` (uniform and adaptive), used as the
  reference implementation of Dunyach et al.
