# Why the LeVeque remeshing test ended with a bad mesh {#mftl_page_remesh_quality}

Companion to `src/remeshing/` and `examples/leveque_remeshing_singleproc.c`. Every number here comes from
that example; re-run it rather than trusting the tables.

```bash
build/leveque_remeshing_singleproc -refine 4 -feature 180      # the recommended configuration
build/leveque_remeshing_singleproc -refine 4 -feature 45       # a typical "safe" threshold
```

## The question

A LeVeque run with feature detection at 45° finishes with a visibly bad mesh, and the natural suspicion is
that tangential smoothing mishandles feature edges and feature vertices.

**That suspicion is correct, and the measurements below locate it precisely.** But the mechanism is one
step further back than "smoothing at features is buggy": on this test case *there are no features*. The
sphere is smooth, it stays smooth, and every edge the detector flags is a false positive. What smoothing
does at a feature vertex barely matters — what matters is that the vertex was called a feature vertex at
all, because every classification removes smoothing degrees of freedom, and on this problem smoothing is
the only thing holding the mesh together.

## Summary

| finding | effect on the final mesh |
|---|---|
| 1. Spurious feature detection on a smooth surface | **dominant**: min quality 0.76 → 0.0002, triangles 4784 → 16702 |
| 2. Smoothing is what maintains quality at all | disabling it: min quality → 0, **65 % volume loss** |
| 3. No fold-over guard on collapse | amplifier: harmless alone, 6× worse *once* spurious features exist |
| 4. Ridge vertices moved to the chord midpoint | real, but **invisible on this test** — it needs genuine features |
| 5. Unweighted "triangle-based" smoothing | **not** a quality problem; it doubles volume loss (3.5 % → 6.0 %) |

Findings 4 and 5 are worth stating separately because they contradict the obvious guess. Finding 5
contradicts mine.

## Setup

Sphere of radius 0.15 at (0.35, 0.35, 0.35), 4 subdivisions = 5120 triangles, in the LeVeque
time-reversed vortex with `T = 3`, `dt = 0.01`, 300 steps. The remeshing target length is the mesh's own
initial mean edge, so the remesher's job is to *hold* the starting resolution while the interface area
grows 4× and returns.

`minQ` / `avgQ` are the mean-ratio quality `q = 4√3·A / (a² + b² + c²)`, 1 for equilateral. `folds`
counts triangles whose normal opposes the mean of their three vertices' area-weighted normals — a
**local** test, which it has to be: by `t = T/2` the sphere is a thin doubly-wound filament, and the
usual "does the normal point away from the centroid" test reports 7800 inversions there and 0 at `t = T`,
which is a statement about the shape rather than about the mesh.

## Finding 1 — the feature detector is the cause, and it is dose-dependent

Sweeping only the dihedral threshold, everything else held fixed:

| feature angle | triangles | minQ | avgQ | folds | V/V₀ | A/A₀ |
|---|---|---|---|---|---|---|
| **180° (off)** | **4784** | **0.75705** | **0.97876** | **0** | 0.9648 | 0.9863 |
| 120° | 4884 | 0.00188 | 0.97271 | 22 | 0.9644 | 0.9971 |
| 90° | 4996 | 0.02880 | 0.96545 | 82 | 0.9620 | 0.9969 |
| 60° | 8770 | 0.00110 | 0.66513 | 908 | 0.9579 | 1.0636 |
| 45° (a typical default) | 16702 | 0.00019 | 0.46389 | 1995 | 0.9537 | 1.1873 |
| 30° | 27128 | 0.00022 | 0.35944 | 2845 | 0.9573 | 1.2828 |

Triangle count, `avgQ` and the fold count are monotone across all five thresholds. `minQ` is not, and
should not be read as if it were: it is an extremum over a single triangle, so once *any* sliver has been
frozen the value reports which sliver happened to be worst rather than how much of the mesh is bad. What
it does say is categorical — 0.76 with detection off, and two to four orders of magnitude below that at
every threshold that flags anything.

The damage is not subtle: at 45° the mesh finishes with 3.5× the triangles it should have, an
average quality of 0.46, and 1995 folded triangles. With detection off it returns to 4784 triangles —
93 % of the 5120 it started with — with no folds at all.

Even 120°, which flags only edges whose two faces are nearly back-to-back, produces 22 folds where the
disabled configuration produces none.

Note that only the `feature 180` row is deterministic: it is bit-reproducible run to run, because it never
produces a degenerate triangle. The others are chaotic — a frozen sliver perturbs the trajectory — so
their individual figures move by 10–20 % between builds while the ordering does not.

### The onset, step by step

The interesting part is how little it takes. Steps 7–16 at 45°:

```
 step      t   ntri      minQ     avgQ  folds  feat  corner  ridge   free
    7  0.070   4820   0.84764  0.98011      0     0       0      0   2412
    8  0.080   4780   0.83734  0.97822      0     0       0      0   2392
    9  0.090   4786   0.79744  0.97773      0     0       0      0   2395
   10  0.100   4790   0.04518  0.97740      0     4       2      2   2393   <-- 4 edges
   11  0.110   4780   0.04401  0.97741      0     4       2      2   2388
   12  0.120   4806   0.04308  0.97754      0     4       2      2   2401
   13  0.130   4842   0.04240  0.97840      0     4       2      2   2419
```

At step 9 the mesh is healthy and there are no features. At step 10, **four** edges cross the threshold,
two vertices are classified as corners — and the worst triangle in the mesh goes from 0.797 to 0.045 in
one step and then *stops changing*. 0.0452, 0.0440, 0.0431, 0.0424, 0.0420: it is not degrading, it is
frozen. Meanwhile `avgQ` is still 0.977, so nothing about the aggregate picture suggests a problem.

It is frozen because a constrained vertex is exempt from all four operations at once:

- the smoother gives it 0 or 1 degrees of freedom instead of 2;
- `MFTLCollapseApproved` refuses to retire a corner, so the sliver cannot be collapsed away;
- `MFTLFlipApproved` refuses to flip a feature edge, so it cannot be flipped away;
- and the sliver's own extreme dihedral angles keep it classified as a feature next step.

That last clause is what makes it a ratchet rather than a transient, and the ratchet is visible in the
vertex classes over the whole run:

```
 step   feat  corner  ridge   free   constrained
   10      4       2      2   2393        0 %
   15      8       4      4   2441        0 %
   75    332     203    131   5664        6 %
  150   1568     754    575   8853       13 %
  225   1640     885    407   5559       19 %
  300   9017    4166   2482   1705       80 %
```

Four spurious edges at step 10 become 9017 by step 300, and the fraction of vertices the smoother is
allowed to move freely falls from 100 % to 20 %. The mesh ends up mostly frozen.

### It is not specifically the 0D lock

The obvious follow-up is whether locking (0 DOF) is the problem and the 1D ridge treatment would be
safer. It is not:

| corner rule at 45° | triangles | minQ | avgQ | folds |
|---|---|---|---|---|
| corner 45° (lock where the curve turns sharply) | 16702 | 0.00019 | 0.46389 | 1995 |
| corner 180° (never lock; every 2-feature vertex is a ridge) | 110510 | 0.00012 | 0.45978 | 20639 |
| corner 0° (lock every 2-feature vertex) | 18780 | 0.00000 | 0.47737 | 2257 |

Refusing to lock makes it **six times worse** in triangle count and ten times worse in folds, because a
ridge vertex is then dragged along a "feature curve" that is a spurious zigzag. The conclusion is about the classification, not about which constraint
follows from it: *any* degree of freedom removed on false evidence is harmful, and the only fix is not to
remove it.

## Finding 2 — why removing smoothing degrees of freedom is fatal here

| configuration | triangles | minQ | avgQ | folds | V/V₀ |
|---|---|---|---|---|---|
| full | 4784 | 0.75705 | 0.97876 | 0 | 0.9648 |
| **no smoothing at all** | 5656 | **0.00000** | 0.66785 | 241 | **0.3448** |

Split, collapse and flip on their own cannot maintain this mesh: without smoothing the interface loses
**65 % of its volume** and the worst triangle is fully degenerate. Smoothing is not a polish step on this
problem, it is the mechanism, which is exactly why a classification that switches it off locally is not a
conservative choice.

## Finding 3 — the fold guard is a brake, not a cause

`MFTLCollapseApproved` now rejects any collapse that would turn a surviving triangle's normal by more than
`max_normal_deviation_cos`. Measured with the test and without it:

| configuration | triangles | minQ | avgQ | folds |
|---|---|---|---|---|
| feature off, guard on | 4784 | 0.75705 | 0.97876 | 0 |
| feature off, guard **off** | 4814 | 0.80430 | 0.97687 | 0 |
| feature 45°, guard on | 16702 | 0.00019 | 0.46389 | 1995 |
| feature 45°, guard **off** | **101048** | 0.00000 | 0.21746 | **33887** |

With a correct feature configuration the guard is worth nothing measurable — the two runs differ by
noise, and the unguarded one is nominally *better*. With spurious features it is the difference between
a bad mesh and a runaway: 101 048 triangles against a 5120-triangle target, and 33 887 folds.

So it does not initiate anything. It bounds the damage when something else does, which is the right way
to describe it and not the way I first guessed.

## Finding 4 — the ridge chord-cut, which this test cannot see

The obvious rule for a vertex with exactly two feature edges is to move it **to the midpoint of its two
feature neighbours**: the full Laplacian displacement, at full strength, with no projection. That pulls the
vertex off its curve and onto the chord between its neighbours, so a feature curve erodes every time it is
smoothed — the 2D shrinkage problem that tangential projection exists to solve, reintroduced on a 1D
curve.

`MFTLSmoothTangential` instead projects the displacement onto the ridge **tangent**, which redistributes
the vertex along the curve while leaving the curve where it is.

The difference is a fixed point. `test_a_regular_ridge_is_a_fixed_point` smooths a regular 24-gon feature
ring for 20 sweeps:

```
projection onto the tangent:  max drift 5.551e-16, min radius 1.000000000000
moving to the chord midpoint: would give radius cos(pi/24)^20 = 0.8421
```

A 16 % collapse of a feature curve the caller explicitly asked to preserve, and nothing reports it.

**None of this affects the LeVeque test**, because a run configured correctly for a smooth interface has
no ridge vertices at all. It is a bug that shows up on geometry with genuine features — a cube, a contact
line, a wedge — and `test_remesh_preserves_a_cube` is the end-to-end guard for it.

## Finding 5 — the missing area weighting, which is not a quality bug

A "triangle-based" smoother that averages the incident triangle centroids **unweighted** is the easy
mistake here. Botsch & Kobbelt specify the area-weighted mean `q = Σ Aᵢcᵢ / Σ Aᵢ`, and the weight is the
mechanism that equalises triangle *size*. The expectation is that removing it costs triangle quality. It
does not:

| configuration | triangles | minQ | avgQ | V/V₀ |
|---|---|---|---|---|
| area-weighted (correct) | 4784 | 0.75705 | 0.97876 | **0.9648** |
| umbrella / unweighted | 4718 | 0.81764 | 0.98210 | **0.9400** |

Triangle quality is a wash — marginally *better* unweighted. What the area weighting buys is **volume
conservation**: 3.5 % loss against 6.0 %, so dropping it nearly doubles the error in the quantity a
front-tracking scheme actually cares about. `test_area_weighting_is_actually_applied` pins that the two
modes genuinely differ, since the failure mode is silent — a caller selecting "triangle based" for its
isotropy would simply have received a more expensive umbrella operator.

## What this means for a caller

**Set `feature_angle_threshold` to 180 (detection disabled) for any smooth closed interface** — a droplet,
a bubble, this test sphere. It is a supported production setting, not a debug switch, and it is the
difference between minQ 0.76 and minQ 0.0002. Feature detection is for geometry that genuinely has sharp
edges to preserve, and there the threshold must be chosen from the geometry, not left at a default.

A mesh boundary is treated as a feature at any threshold, so an open surface's rim is still protected
with detection off.

## An unrelated defect found while measuring — since FIXED

`MFTLGetTriangleProperties` (`include/mftl_geometry.h`), and the two kernels in
`src/geometry/triangle_geometry.c` and `src/ensemble/ensemble_update.c`, all used to compute

```c
const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]));
if (area_out) *area_out = 0.5 * mag;
const double inv = 1.0 / mag;
```

The comment claimed this made a degenerate triangle yield zero area. It did not: the clamp was applied to
the **squared** magnitude *before* the square root, so the reported area was floored at
`0.5·sqrt(1e-12) = 5e-7` rather than reaching zero. Three consequences, all silent:

- `quality.min_triangle_area` could never report a genuinely degenerate triangle;
- `q = 4√3·A / (a²+b²+c²)` became unbounded above for small triangles — this example measured an
  *average* quality of 1.023 on a degenerate mesh, for a quantity whose maximum is 1;
- the **normal** of any triangle below that floor was divided by the clamp instead of by its own length,
  so it came out proportionally short rather than unit. A triangle with 1e-6 legs got a normal of length
  1e-6.

The guard only ever existed to protect the reciprocal, so it belongs on the divisor — and as a test
against zero rather than a clamp:

```c
const double mag = MFTL_SQRT(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]); /* exact: 0 for degenerate */
const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;                      /* exact at every scale */
```

A fully degenerate triangle gets a zero normal, which is the same "not computable" sentinel
`normalise_vertex_normals()` already uses.

**Applied**, in all three places, and in `MFTLNormalizeVector` (`mftl_vectoroperations.h`), which had the
same shape.

### Why a test and not a clamp, and what it costs

`1.0 / MFTL_MAX(mag, tiny)` keeps the instruction count identical to the broken version — one max, one
sqrt, one divide — and measures at exact parity (**7.32 → 7.31 ns/tri**, best of six over two rounds at
327 680 triangles). But it only *moves* the floor. With `tiny = MFTL_VERYVERYSMALL` it was caught by
`test_triangle_normals_are_unit_at_every_scale`: a triangle with 1e-8 edges came out with a normal of
length **0.112**.

The test against zero has no cliff. It costs one instruction — `fcmp`+`fcsel` against `fmax` — measured at
**7.33 → 7.41 ns/tri, +1.1 %**, reproducibly across two rounds. That is the whole price of a normal that
is unit at every scale rather than above an arbitrary one, and for a quantity that multiplies surface
tension it is worth paying.

(Single runs of this benchmark scatter by ~25 % on this machine — 7.40 and 10.24 ns/tri were measured from
*identical* source — so minima over repeats are the only usable statistic here.)

### What pins it

- `test_triangle_normals_are_unit_at_every_scale` — 17 triangle scales from 1e-8 to 1e+8, scalene and
  off-axis, checked on the scalar path **and** on both bulk kernels, since
  `MFTLComputeTriangleGeometry` dispatches on `MFTL_DERIVED_FRAMES` and the two carry separate copies of
  the expression.
- `test_normalize_is_unit_across_magnitudes` — 31 input scales from 1e-150 to 1e+150, asserting unit
  length *and* unchanged direction, because those are separate claims and only one of them used to hold.
- `test_normals_of_a_whole_mesh_are_unit` — a real sphere straddling a periodic seam, both kernels, plus
  the per-vertex pass.
- `inv_triangle_normals_are_unit` / `inv_vertex_normals_are_unit` in `test_mesh_invariants.h`, asserted
  after `MFTLEnsembleUpdate` and after a full remeshing pass. Deliberately **not** part of
  `inv_mesh_is_sound`: redistribution is documented as leaving `tris.normal` unspecified, so "normals are
  unit" is a claim about a mesh whose geometry pass has run, not about every valid mesh.
- `test_a_very_small_triangle_is_measured_exactly` — area, unit normal and quality of a triangle a million
  times below the old floor.
- a tightened `test_degenerate_triangle_does_not_produce_nan`, which now asserts an area of exactly 0.0
  instead of merely "small".

`test_triangle_geometry_paths_are_bit_identical` still passes, which it only can if the expression is
character-for-character the same in the walk and stream kernels.
