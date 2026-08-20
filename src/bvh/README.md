`src/bvh/` — Contact detection {#mftl_mod_bvh}
==============================

**Public header:** `include/mftl_bvh.h` · **Layer:** 3 ·
**Module group:** @ref mftl_bvh

Ported from `../MFTL2` and reworked around periodicity and scale.

| File | Job |
|---|---|
| `bvh_build.c` | Morton codes, radix split, bottom-up AABB fit, refit-only fast path |
| `bvh_traverse.c` | dual-tree traversal over an explicit stack, with a batched leaf phase |
| `bvh_predicates.c` | Shewchuk-adaptive `orient3d`, Guigue–Devillers overlap, exact triangle–triangle distance |
| `bvh_internal.h` | float rounding helpers, AABB overlap, and the two module-private build entry points |

@warning `bvh_predicates.c` **must** be compiled with `-ffp-contract=off`
         (set in `CMakeLists.txt`). This is correctness, not tuning — see
         @ref bvh_fma below.

---

## Linear BVH construction {#bvh_build}

Karras' construction. Each primitive gets a 63-bit Morton code from its centroid;
the codes are sorted; the hierarchy is then built by **radix splitting** — an
internal node covering the sorted range $[i, j]$ splits where the highest
differing bit of the codes changes.

The split position is found by a binary search on the common prefix length

$$
  \delta(a,b) = \mathrm{clz}\big(Z_a \oplus Z_b\big)
$$

so the whole tree is a **function of the sorted key array** and needs no top-down
partitioning pass. The two children are separated by a real axis-aligned plane —
the radix point of the Morton curve — rather than by a median that happens to fall
wherever the primitive count does.

### Duplicate codes would be fatal, so the key is 95 bits

A radix split needs every primitive to have a **unique** key. Two primitives
sharing a code have a common prefix of the full key length, the search for the
highest differing bit finds nothing, and the node **fails to split** — an infinite
descent in the recursive form, and an unbounded leaf in the guarded form.

Duplicates are not exotic: two triangles smaller than one Morton cell share a
code, which for a resolved interface mesh is most of the mesh at low bit counts.
MFTL 2's 10 bits per axis put a $1024^3$ grid over the object, and any mesh finer
than that in any region collides.

Extending the key with the primitive's **index** makes it unique by construction
without another sort — the array is already sorted, so indices are ascending, and
comparing $(Z, i)$ lexicographically is exactly what a 95-bit radix tree over
unique keys does. It also bounds the recursion depth at 95 rather than at the
primitive count, which is why the *build* keeps its recursion while the traversal
does not.

### The fit is three linear passes, no recursion, no stack

`MFTLBVHRefit`:

1. **Reverse scan** over `[nodes_used-1 … 0]`, fitting each leaf to its primitives
   and each internal node to its two children. The build allocates a child pair by
   post-incrementing `nodes_used`, so **a child index is always greater than its
   parent's** — walking backwards therefore visits both children before the parent
   that merges them. *That property is the whole reason a refit-only fast path is
   worth having.*
2. The root box now bounds every primitive, so `bounding_radius` and `tolerance`
   come straight off it.
3. **Forward scan** inflating every node by the tolerance.

Pass 3 is separate because the tolerance depends on the bounding radius, which is
only known once pass 1 has finished. Inflating leaves inside pass 1 would need a
radius from a pass that has not run; a preliminary sweep over the *primitives* to
find the box first costs nine doubles per primitive.

It works because **a uniform inflation commutes with the min/max merge**: growing
every leaf by $\tau$ and re-merging gives exactly the same tree as merging first
and growing every node by $\tau$. So the correction is applied once per **node**,
over an array still in cache from pass 1, instead of once per primitive over the
mesh.

Shared by `MFTL_GenerateLBVH` (as its last build step) and `MFTL_RefitLBVH` (as
the entire operation), so the two cannot disagree about what a fitted tree looks
like — asserted by `test_refit_only_matches_full_rebuild`.

### Node layout: 32 bytes exactly

Six `float` for the AABB plus one `uint64_t` payload. The payload's **low bit** is
the leaf flag, and it is the only thing distinguishing the two layouts:

| | bit 0 | bits 1…31 | bits 32…63 |
|---|---|---|---|
| leaf | 1 | primitive count | first slot in `prim_indices` |
| internal | 0 | left child index | — (the right child is left + 1) |

Storing only the left child is what keeps a node 32 bytes rather than 40.

The boxes are `float` because a broad phase does not need `double` precision — but
the conversion **must round outward**: `mftl_bvh_floor_f` rounds the minimum down
and `mftl_bvh_ceil_f` rounds the maximum up, so the float box is a conservative
**superset**, which is the only direction a broad phase may ever be wrong in. The
slack is a scale by $(1 \pm 2^{-23})$ plus a denormal-proof absolute nudge — two
instructions, against a `nextafterf` libm call.

---

## Periodicity is two problems {#bvh_periodic}

They look related and need entirely different fixes.

### Within an object

A triangle straddling a seam has wrapped vertices, so a centroid built from the
raw coordinates lands in the **middle of the domain** and its AABB spans the box.
That is the corruption MFTL 2's `/* ATTENTION: IS THIS CORRECT FOR PERIODIC
DOMAINS? */` was about.

Fixed by building **every** primitive from the seam-consistent `tris.anchor` /
`tris.edge` frame, through `MFTL_BVHPrimVerts` — the one accessor both primitive
kinds go through, so no consumer branches on local-versus-remote.

A straddling object's **root** box legitimately spans the domain and is **not** a
bug — the same accepted cost, for the same reason, as `MFTLObjectReorderMorton`.

### Between objects

Two bodies on opposite faces are neighbours, and no amount of per-triangle
unwrapping can see it: each body is internally consistent and they are genuinely a
domain-length apart in stored coordinates.

Fixed by traversing the pair once per periodic **image** of B — the offsets
$\{-L, 0, +L\}$ on each periodic axis, up to 27 — each admitted or rejected by a
single root-AABB test before any descent.

A non-periodic domain generates exactly one image and pays one root test it was
going to pay anyway. A periodic domain typically admits one image and rejects the
other 26 in 26 compares; the bodies would have to span half the domain for two to
survive.

The shift is added to **B** rather than subtracted from A, because A is the outer
loop of the traversal and its node is already in registers — and a zero shift
costs three adds the compiler hoists out of the leaf case entirely.

---

## The contact envelope {#bvh_envelope}

$$
  \tau = \max\big(\texttt{proximity\_floor},\; \texttt{proximity\_factor}\cdot R\big)
$$

replacing MFTL 2's hardcoded $10^{-4}$ (AABB inflation) and $10^{-6}$ (proximity),
both **absolute** and therefore correct at exactly one length scale: $10^{-4}$ is
a tenth of a 1 mm droplet's radius, and $10^{-6}$ is $10^{-6}$ of a 1 m body's.

$R$ is **half the longest side** of the root box, not half its diagonal, so it
equals $R$ for a sphere and `factor` means what it says.

$R$ is measured over **one rank's partition**, so the distributed pipeline
overwrites it with the body's **global** radius from one `MPI_Allreduce`
(`MFTL_BVHSetGlobalRadius`) — otherwise splitting a droplet over eight ranks would
change the separation at which contact is detected.

@note **A pair's envelope is $\max(\tau_A, \tau_B)$**, so a µm droplet against a
      metre body is judged at the metre body's scale. The maximum is the only safe
      direction for the broad phase — each tree's leaves are inflated by its own
      tolerance, so admitting the larger keeps the narrow-phase threshold inside
      what the traversal can deliver — but it means the envelope of a pair can
      exceed the smaller body's own diameter. The escape: set `proximity_factor`
      to 0 and state the physical gap as `proximity_floor`. Characterised, not
      asserted away, by `test_two_length_scales_share_the_larger_envelope`.

---

## Traversal {#bvh_traverse}

### Why the recursion went

MFTL 2 descended with a four-argument recursive call. An explicit stack buys three
things, and only the third is really about speed:

1. **The hot state stays in registers.** Both objects, both node arrays, the
   callback and the image shift are loop-invariant, and a recursive call has to
   re-establish all of them per node pair.
2. **The "what next" data becomes contiguous.** A pair stack is two ints per entry
   in one array; a call stack is a frame per entry, interleaved with spill slots
   and a return address.
3. **Leaf pairs can be batched.** A leaf–leaf hit does not run the narrow phase
   where it is found; it appends to a fixed-size array of
   $(\text{primA}, \text{primB})$, flushed when full. The traversal therefore hands the narrow
   phase **a contiguous run of independent tests** — the shape a vectorised kernel
   needs. Making that kernel SIMD is then a change to `flush_batch()` alone.

### The descent rule

When both nodes are internal, split the one whose box is **larger**. Splitting the
smaller one makes the pair of boxes converge more slowly, so the traversal visits
more pairs for the same set of leaf tests.

### A distributed contact is reported by the A-side owner, and only that rank

- The BVH is built over **owned** triangles only. Including the redistribution
  halo makes two ranks report the same contact, exactly as writing ghosts into a
  `.vtu` double-counts the surface.
- Only object **B**'s triangles travel through the spatial halo.
- The traversal **skips any leaf pair whose A side is remote**.

Get it wrong in one direction and cross-rank contacts are missed; wrong in the
other and every cross-rank contact force is applied twice.
`test_distributed_contacts_match_the_serial_answer` compares the gathered **set**
against the answer computed on one rank while it still held the whole mesh, which
is the only check that rules out both at once.

---

## The narrow phase {#bvh_narrow}

Guigue & Devillers' triangle–triangle overlap test: a decision tree over
orientation predicates

$$
  \mathrm{orient3d}(\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{d}) =
  \operatorname{sign} \det
  \begin{pmatrix}
    a_x - d_x & a_y - d_y & a_z - d_z \\
    b_x - d_x & b_y - d_y & b_z - d_z \\
    c_x - d_x & c_y - d_y & c_z - d_z
  \end{pmatrix}
$$

evaluated up to twelve times per pair. The coplanar case falls through to a 2D
overlap test in the coordinate plane the normal is most aligned with — the
projection with the largest Jacobian, hence the best conditioned — with nine
segment crossings plus two containment tests, which is a complete cover.

`MFTL_TriTriProximity` runs the overlap test first and falls back to the exact
minimum distance: **nine edge–edge pairs and six point–face projections**, a
complete cover of the possible witness pairs of two disjoint triangles.

### Only orient3d is adaptive, and that is deliberate {#bvh_adaptive}

It is the **only** predicate here whose *sign*, rather than magnitude, decides the
answer. Everything else — distances, areas, interval endpoints — degrades
continuously: a distance wrong in the last ulp gives a contact decision wrong only
for pairs sitting exactly on the tolerance, which is a tie the caller already
accepted by choosing a tolerance. A **sign** that is wrong makes the algorithm
take a branch for a configuration that does not exist, and it can then report a
clean *"no intersection"* for two triangles that cross.

`MFTL_Orient3D` is Shewchuk's: a floating-point estimate plus a relative-error
bound, escalating to exact expansion arithmetic **only** when the estimate falls
inside that bound. On non-degenerate input it is the plain determinant plus one
compare. The error bounds are *measured* by `exactinit()` rather than assumed, so
a target with extended-precision registers or a different rounding mode gets
bounds that are true for **it**.

Measured by `examples/orient3d_predicate_profile.c`, and the answer is not
one-sided:

| | Result |
|---|---|
| cost vs. the plain determinant | **2.00×** |
| cost as a fraction of a full `MFTL_TriTriProximity` | **≤ 6.2 %** |
| misclassifications by the plain form, 20 000 samples, gaps ≥ $10^{-12}$ of body size | **zero** |
| where the plain form does fail | $10^{-14} \ldots 10^{-16}$ |

$10^{-12}$ of the body size is three orders below where a lubrication model works.
It is kept because the cost is small — the narrow phase is dominated by its
fifteen distance sub-tests, `MFTL_TriTriProximity` measuring **176 ns/pair**
against `MFTL_TriTriIntersect`'s **31** — and the failure it removes is silent.

### `-ffp-contract=off` is correctness {#bvh_fma}

The adaptive predicate is built from error-free transformations. `TWO_PRODUCT`
recovers the exact residue of $a\cdot b$ by evaluating $x - a_{hi}b_{hi}$ in a
**second rounding step**.

A fused multiply-add contracts that into one operation with one rounding, the
residue comes out **zero**, and every exactness guarantee in the file silently
evaporates — while the numbers still look plausible.

The file carries `#pragma STDC FP_CONTRACT OFF`, but not every compiler honours
it, so the flag is set in `CMakeLists.txt` as well. **Do not remove it.**

---

## Deferred {#bvh_deferred}

1. **The leaf–leaf narrow phase is batched but not vectorised.** `flush_batch`
   receives a contiguous run of up to 64 independent tests — the shape a SIMD
   kernel wants — and runs them scalar. The useful form is a gather of the *first*
   orientation test across a batch rather than a naive widening, and that is a
   measurement exercise. Start with `MFTL_TriTriProximity`'s fifteen distance
   sub-tests, which is where the time is.
2. **Nothing in this directory has been benchmarked against MFTL 2, or profiled at
   all beyond `orient3d_predicate_profile.c`.** The explicit-stack traversal, the
   refit-only fast path and the batched leaf phase are all argued from structure
   rather than measured. Build/refit/traverse timings versus the recursive
   original are the first thing to measure before optimising any of it.
3. **Contact detection is discrete-time: there is no CCD.** A pair whose relative
   displacement in one step exceeds the sum of their sizes passes through without
   ever being coincident at a sampled instant, and both samples correctly report
   nothing. This bites when the timestep is set by a CFL condition on the *flow*
   rather than by the approach velocity of two surfaces — the normal case. Fixing
   it needs the swept volume of every triangle: a different primitive, a different
   tree, and a per-step velocity the module does not receive. Pinned as a recorded
   property by `test_tunnelling_between_two_timesteps_is_a_known_miss`, which
   **asserts the miss** — if that test starts failing, something has begun
   reporting contacts that are not present at the sampled configuration.
