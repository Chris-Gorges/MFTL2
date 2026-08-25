MFTL — MultiFlow Triangulation Library {#mainpage}
======================================

MFTL is an HPC C11 library for distributed-memory (MPI) manipulation of closed
and open **triangulated surface meshes**, built for front tracking and
immersed-boundary CFD simulations.

It carries a surface through a flow solver: it partitions the mesh over a
Cartesian process grid, keeps a halo consistent, computes the per-timestep
geometry, remeshes the surface as it deforms, detects contact between bodies,
and writes the result to VTK.

@note Read @ref mftl_page_architecture before changing anything structural.
      Most of the file layout follows from the one idea described there, and a
      change that ignores it will look local and will not be.

---

## Start here

| Page | What it answers |
|---|---|
| @subpage mftl_page_architecture | Why the memory layout is the way it is, and the three invariants every module may assume |
| @subpage mftl_page_building | How to configure, build, test, and generate this manual |
| @subpage mftl_page_timestep | The per-timestep call order, and the three things in it that are silent when wrong |
| @subpage mftl_page_invariants | Properties that must survive any change, each with the test that guards it |
| @subpage mftl_page_modules | The ten implementation modules, each with its own theory page |
| @subpage mftl_page_analysis | The measured picture, and the seven results that contradicted the design |

The reference material is reachable from the navigation bar: **Modules** groups
the API by architectural layer, **Data Structures** lists every struct, and
**API Index** is the alphabetical list of every function, macro and typedef.

---

## What the library does today

- **Ensemble / domain** — MPI topology, Cartesian process grid, periodic bounds
- **Object pools** — the AoS+SoA container, one allocation per entity class
- **Half-edge topology with implicit indexing** — navigation, twin rebuilding,
  boundary marking, integrity checking
- **Derived per-triangle arrays** and the `MFTL_FOREACH_*` hot-loop iterators
- **Morton renumbering** of the pools
- **Periodic-aware geometry**, and the fused per-timestep geometry + quality pass
- **Sphere generation** by icosahedron subdivision
- **Redistribution and ghost synchronisation** over a 26-neighbour graph
  communicator, with a global fallback path
- **VTK output** — a serial debug dump and a parallel writer (ASCII or raw
  binary; one file per rank, or one shared file via MPI-IO)
- **Isotropic remeshing** — split/collapse/flip/smooth, feature detection, and a
  three-subspace tangential smoother
- **Distributed remeshing**, in two strategies chosen by one ratio
- **LBVH and object–object contact detection** — Morton radix-split tree,
  explicit-stack dual traversal, Shewchuk-adaptive orientation predicate, exact
  Guigue–Devillers narrow phase
- **The spatial halo** — the triangles of one body that neighbouring ranks hold
  near this rank's share of another

**Absent:** open-surface generators, GTS I/O, topological ghost-layer growth,
and a mesh *reader* of any format.

---

## The shortest complete program

```c
#include "mftl_core.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();

  int    cuts[3]     = {2, 2, 1};
  double lo[3]       = {0.0, 0.0, 0.0};
  double hi[3]       = {1.0, 1.0, 1.0};
  bool   periodic[3] = {true, true, true};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 4);

  MFTL_Object *drop   = MFTLObjectCreate(0);
  double       ctr[3] = {0.5, 0.5, 0.5};
  MFTLGenerateSphere(drop, ctr, 0.2, 3);      /* collective */
  MFTLEnsembleAddObject(ens, drop);

  MFTLRedistribute(ens);                      /* spread it over the ranks */
  MFTLEnsembleUpdate(ens);                    /* geometry + quality statistics */
  MFTLEnsembleUpdateVertexNormals(ens);       /* exact normals, ghosts included */

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  MFTLWriteVtu(ens, "droplet", &opt);

  MFTLEnsembleDestroy(&ens);                  /* cascades to every object */
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return 0;
}
```

Every call above is **collective** — see @ref mftl_page_timestep for which
functions are and which are deliberately not.

---

## Reading order for the theory

The library's design rests on a small number of decisions that each have a
measured or structural justification. In rough dependency order:

1. **One allocation per entity class** (@ref mftl_memory) — makes "the AoS grew
   but the SoA did not" inexpressible rather than merely unlikely.
2. **Implicit half-edge indexing** (@ref mftl_topology) — half-edge \f$3t+k\f$
   *is* corner \f$k\f$ of triangle \f$t\f$, so `next`, `prev` and `face` are
   arithmetic and a whole family of traversal corruptions cannot occur.
3. **Ownership is a prefix** (@ref mftl_types) — one notion of ownership, carried
   by the index itself.
4. **The halo rule is vertex-complete** (@ref mftl_communication) — a triangle
   travels to the owner of each of its three corners *and* each of its three edge
   neighbours, which is what makes vertex normals communication-free and what
   the parallel remesher's freeze test reads backwards.
5. **Seam-consistent frames** (@ref mftl_geometry) — periodicity is handled once,
   at the point where an edge vector is formed, and never again downstream.

---

## Measured results that contradict the design's own predictions

These are the reason to read @ref mftl_page_analysis rather than the code
comments alone.

| # | Expectation | Measurement |
|---|---|---|
| 1 | The frame kernel is worth ~3× | 1.25–1.57× non-periodic, 1.94–2.00× periodic — the general path got fast, the frames did not get slow |
| 2 | Morton renumbering pays for itself | No measurable time on Apple silicon despite cutting mean edge index distance 160× |
| 3 | Splitting `verts.pos` into three arrays helps | Faster for gathers, slower for scatters, by similar amounts — a wash |
| 4 | The update kernel fails to vectorise because of the nine-way reduction | It does not; 26 % of the pass is one `sqrt` plus one divide, which is irreducible |
| 5 | Quality statistics cost ~5 % on top of geometry | **+90 %** (1.64 → 3.11 ns/triangle) — fusing is still right, but it is not free |
| 6 | MPI-IO makes output faster | It does not; dropping ASCII does. The 23–32× came entirely from raw binary replacing `printf` |
| 7 | The poor final remesh was an operator bug | It was **feature detection on a smooth surface** — set `feature_angle_threshold` to 180 for any smooth closed interface |

---

## Author and licence

Christian Gorges. The radix sort in `src/arrayfunctions/` derives from Cameron
Hart's public-domain-style implementation; its licence notice is preserved in
the source file.
