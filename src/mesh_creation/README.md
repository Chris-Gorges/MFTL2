`src/mesh_creation/` — Mesh construction {#mftl_mod_meshgen}
=======================================

**Public header:** `include/mftl_meshgen.h` · **Layer:** 3 ·
**Module group:** @ref mftl_meshgen

| File | Job |
|---|---|
| `mesh_builder.c` | indexed triangle list → half-edge mesh; the single conversion point |
| `sphere_creation.c` | the icosphere, in a collective and a split local/publish form |
| `marching_cubes.c` | a generated 256-case table, lattice-keyed extraction, level set → mesh |

There is no mesh reader; one would go through `MFTLBuildIndexedMesh`.

---

## One conversion point {#meshgen_one_builder}

`MFTLBuildIndexedMesh` (positions + indices) and `MFTLBuildTopologyFromIndexList`
(indices over an already filled vertex pool) are the only places raw
connectivity becomes an MFTL mesh, so the conventions every module relies on are
established once:

- half-edge $3t+k$ runs from `tri_verts[3t+k]` to `tri_verts[3t+(k+1)%3]`, so
  slot $k$ of `tris.verts` is the **origin** of half-edge $3t+k$;
- global ids are $[0, n)$ per class, half-edge ids $3\,t_\mathrm{gid}+k$, and the
  `next_*_gid` counters end one past the last id;
- twins are linked (`MFTLRebuildTwins`), the boundary is flagged while the mesh
  is complete (`MFTLMarkMeshBoundary`), derived arrays are built;
- `MFTLBuildIndexedMesh` and the generators also compute the triangle geometry
  (area, normal, centre), so a new mesh can be redistributed at once — the first
  redistribution routes by centres. `MFTLBuildTopologyFromIndexList` does not:
  its caller still owns the positions (subdivision projects them afterwards).

Index lists are validated in Release before anything is modified: an index out
of range or a triangle with a repeated corner (a self-edge that could twin with
itself) is rejected. The caller owns the winding: the normal
$(\mathbf{p}_1-\mathbf{p}_0)\times(\mathbf{p}_2-\mathbf{p}_0)$ points to the
side from which the triangle is counter-clockwise.

---

## The icosphere {#meshgen_icosahedron}

The regular icosahedron — vertices $(0, \pm1, \pm\varphi)$ and cyclic
permutations, $\varphi = (1+\sqrt5)/2$, scaled onto the unit sphere — is refined
$r$ times by 1-to-4 midpoint subdivision and then **projected once** onto the
sphere of radius $R$:

```
      v2
     /  \
   m2 -- m1
   / \  / \
 v0 -- m0 -- v1
```

For a closed triangulation Euler's formula $V - E + F = 2$ with $3F = 2E$ gives

$$
  F_r = 20 \cdot 4^{\,r}, \qquad E_r = 30 \cdot 4^{\,r}, \qquad V_r = 10 \cdot 4^{\,r} + 2 .
$$

<!-- figure: icosphere
The icosphere. (a) One face refined 1-to-4. The midpoint $m_k$ lies on edge
$v_k v_{k+1}$; the three corner children and the centre one are emitted in the
parent's rotational order, so every child is counter-clockwise from outside.
The neighbour across $v_0 v_1$ finds $m_0$ through the twin. (b) Refinement
levels $r = 0 \ldots 3$, each subdivided flat and projected once onto the
sphere.
-->

| $r$ | vertices | triangles | min $q$ | mean $q$ | max/min area | $1 - V_h/V$ |
|---|---|---|---|---|---|---|
| 0 | 12 | 20 | 1.000 | 1.000 | 1.00 | 0.395 |
| 1 | 42 | 80 | 0.985 | 0.989 | 1.19 | 0.127 |
| 2 | 162 | 320 | 0.977 | 0.994 | 1.51 | 0.034 |
| 3 | 642 | 1 280 | 0.975 | 0.996 | 1.73 | 8.6e-3 |
| 4 | 2 562 | 5 120 | 0.974 | 0.997 | 1.86 | 2.2e-3 |
| 5 | 10 242 | 20 480 | 0.974 | 0.997 | 1.93 | 5.4e-4 |

($q = 4\sqrt3A/(a^2+b^2+c^2)$ is the mean-ratio quality; $V_h$ the enclosed
volume of the polyhedron.) Because the midpoints are projected only once, the
flat subdivision of each icosahedron face is mapped centrally onto the sphere:
shapes stay nearly equilateral while areas grow towards the face centres by up to
a factor of about 2. The volume error decreases by a factor of 4 per level —
second order in the edge length $h \propto 2^{-r}$. For a target edge length,
$r \approx \log_2(1.05\,R / h)$.

**Outward normals.** The face table is counter-clockwise as seen from outside and
subdivision emits the sub-triangles in their parent's rotational order, so every
normal points outward at every level. This fixes the sign of the enclosed volume,

$$
  V = \frac13 \oint_{\partial\Omega} \mathbf{x}\cdot\hat{\mathbf{n}}\,\mathrm{d}A
    = \frac13 \sum_t A_t\,(\mathbf{c}_t\cdot\hat{\mathbf{n}}_t),
$$

of surface tension and of any indicator function
(`test_triangle_winding_is_outward`, `test_enclosed_volume_is_positive_and_converges`).

**Midpoints without a hash map.** An edge's midpoint is identified by a
half-edge index: `midpoint_map[h]` is written for $h$ and its twin together, so
the second triangle on an edge finds the vertex by one array read. Subdivision
emits an index list and reuses `MFTLBuildTopologyFromIndexList`; old vertices
keep their indices, midpoints are appended.

### Generation is rank-0 serial {#meshgen_collective}

The whole body is built on rank 0 — every other rank holds an empty object — and
`MFTLRedistribute` spreads it. Id counters must be broadcast and global counts
reduced: `MFTLGenerateSphere` does both per body (two collectives each);
`MFTLGenerateSphereLocal` for every body followed by one
`MFTLPublishGeneratedBodies` costs two collectives in total. Consequences:

- rank 0's memory peak is the whole ensemble ($3440 \cdot 4^r + 128$ bytes per
  body); a distributed generator does not exist;
- the opening scatter is the one redistribution that may leave the
  26-neighbourhood of rank 0 (whenever some cell is two or more cells from rank
  0's along an axis, e.g. a non-periodic $3\times3\times3$ or $4\times1\times1$
  grid), and then takes the global route;
- quantities measured on an empty rank (e.g. `MFTLMeanEdgeLength` = 0) must not
  be used as rank-local parameters; the remesher measures its target length
  collectively (@ref mftl_mod_remeshing).

---

## Marching cubes {#meshgen_mc}

### A table generated from one face rule

Instead of the classic hand-tabulated 256 rows, every case is **derived** from a
rule applied independently to each of the cell's six faces. Walking a face's four
corners counter-clockwise as seen from outside the cell, the sign changes at 0, 2
or 4 edges; every crossing where the walk *enters* the negative region is
connected to the next crossing where it *leaves* it. On an ambiguous face
(negative corners on a diagonal) this cuts each negative corner off separately —
the inside never connects diagonally through a face.

The rule depends only on the face's four signs. The neighbouring cell walks the
same face in the opposite direction and obtains the same segments with opposite
orientation, so the surface is **crack-free and consistently oriented across
cells by construction** — the defect of tables assembled from symmetry classes
and complements, which can resolve one ambiguous face differently from its two
sides. Every segment has the negative region on its right, so the loops chained
from the segments are wound with normals pointing from $\phi < 0$ to $\phi > 0$.
A loop that visits one face twice is fanned from an apex with no diagonal to a
crossing on a shared face, so no triangle lies inside a cell face; the tests
assert that such an apex exists for every loop of all 256 cases.

<!-- figure: marching-cubes
Marching cubes from one face rule. (a) Walking a face counter-clockwise from
outside, each crossing that enters the inside is joined to the next one that
leaves it; on the ambiguous face each inside corner is cut off separately. The
neighbouring cell walks the same face the other way and draws the same segments
reversed, so nothing can crack. (b) Case `0b00000111` as the generated table
builds it: five crossings chained into one loop, fanned from an apex with no
in-face diagonal into three triangles whose normals point from $\phi<0$ to
$\phi>0$.
-->

### Lattice-keyed extraction

`MFTLExtractLevelSet` places a vertex on each crossed lattice edge at the linear
interpolant $t = \phi_0/(\phi_0-\phi_1)$ and names it by its **lattice edge**,
`3 * node + axis`, computed from *wrapped* global node indices; its coordinates
are computed from the wrapped node as well. Two ranks extracting neighbouring cell
ranges therefore produce a shared vertex with the same key and bit-identical
coordinates without comparing positions — the basis of the distributed
reconstruction in @ref mftl_mod_topology_change. Values closer to zero than
`snap * min(spacing)` are pushed out to that distance (zero counts as positive),
so no vertex coincides with a lattice node. A cell with an unknown
(`MFTL_SDF_UNKNOWN`) corner is skipped.

`MFTLBuildMeshFromLevelSet` extracts a whole window and builds the mesh; where
the level set leaves the window the surface is open and its rim is flagged as
mesh boundary, and on a periodic axis a window of exactly one period closes it
across the seam.
