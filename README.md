# MFTL — MultiFlow Triangulation Library

A HPC C11 library for distributed-memory (MPI) manipulation of closed and open triangulated surface meshes, for
front-tracking and immersed-boundary CFD simulations.

## The idea

A surface mesh is two different things that want opposite memory layouts, so MFTL stores each in the layout it
wants — and stores them in the **same allocation** so they can never disagree about how many entities exist.

```
MFTL_Object
├── MFTL_VertexPool          ── ONE malloc, ONE capacity ──
│   ├── rec[]                AoS   16 B   out_he_idx, global_id, owner_proc, flags
│   ├── pos[]                SoA   3×     authoritative position
│   ├── normal[] area[]      SoA          derived per-vertex geometry
│   └── mean_curv[] gauss_curv[]
│
└── MFTL_TrianglePool        ── ONE malloc, ONE capacity ──
    ├── rec[]                AoS   12 B   global_id, owner_proc, flags
    ├── he[]                 AoS   16 B   vertex_idx, twin_idx, global_id, flags   (3 per triangle)
    ├── area[] normal[] center[]   SoA    derived per-triangle geometry
    └── verts[] neighbors[]          SoA   compact hot-loop connectivity
```

**Topology and identity are AoS** — a remesher edits one entity at a time and touches most of its record when it
does. **Geometry is SoA** — it is read and written by kernels that touch one component of every entity, every
timestep, which is the access pattern AoS is worst at.

Three invariants fall out, and every module is allowed to assume them:

- **Half-edge `3t+k` belongs to triangle `t`**, and `next(3t+k) == 3t+(k+1)%3`. So `he_idx`, `next_idx` and
  `triangle_idx` are not stored — they are arithmetic — and the half-edge array needs no capacity of its own.
- **Ownership is an array prefix**: `[0, n_local)` is owned, `[n_local, n)` are ghosts. No `is_ghost` field.
- **Positions live in the SoA and only there.** The space-filling-curve ordering is applied to the *pool*
  (`MFTLObjectReorderMorton`), not stored as a permutation, so there is exactly one index space.

There are **no hash maps**. Global IDs are dense per entity class, and "the other side of this edge" is already an
index.

## Quick start

```c
#include "mftl_core.h"

MPI_Init(&argc, &argv);

int    cuts[3]  = {1, 1, 1};
double lo[3]    = {-2, -2, -2}, hi[3] = {2, 2, 2};
bool   per[3]   = {false, false, false};

MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 10);

MFTL_Object *obj = MFTLObjectCreate(1);
MFTLEnsembleAddObject(ens, obj);              // links it to the domain; ensemble now owns it

double centre[3] = {0, 0, 0};
MFTLGenerateSphere(obj, centre, 1.0, 5);      // collective
MFTLObjectReorderMorton(obj);                 // local: spatial renumbering
MFTLEnsembleUpdate(ens);                      // collective: geometry + quality, one pass
MFTLEnsembleUpdateVertexNormals(ens);         // collective: normals + one ghost exchange

printf("mean-ratio quality: %f\n", obj->quality.avg_aspect_ratio);

MFTLWriteVtu(ens, "surface", MFTL_NULL);      // collective: raw-binary .vtu per rank + .pvtu

MFTLEnsembleDestroy(&ens);                    // cascades to every object
MPI_Finalize();
```

A hot loop streams compact triangle indices, gathers current positions and forms
minimum-image edges in registers:

```c
MFTL_FOREACH_TRIANGLE_GEOMETRY(obj, obj->tris.n_local,
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    total_area += 0.5 * sqrt(nx*nx + ny*ny + nz*nz);
);
```

The macro unswitches periodicity outside the triangle loop, so the body receives
seam-consistent edge vectors without managing derived position state.

## Build and test

Requires CMake ≥ 3.13, MPI, and a C11 compiler.

```bash
python3 build_all.py              # Release
python3 build_all.py --debug      # Debug + ASan/UBSan  (run the tests here too)
python3 run_local_tests.py        # build + full CTest suite
```

```bash
./build/sphere_creation_serial       # full lifecycle, writes sphere.vtk
./build/layout_benchmark             # interleaved vs padded vs split xyz, four access patterns
./build/leveque_remeshing_singleproc # what feature detection on a smooth surface costs

# the whole timestep, distributed: advection, ghost sync, geometry, redistribution
mpirun -n 4 ./build/parallel_advection_redistribute_list 2 2 1 1 1 1 --spheres 256 --refine 3 --verify

# distributed remeshing: does it track the serial answer, and what does a phase cost
mpirun -n 4 ./build/leveque_remeshing_parallel
mpirun -n 4 ./build/remesh_phase_profile        # where the time goes inside one phase
mpirun -n 8 ./build/remesh_regime_sweep         # where the spanning/dispersed crossover lies

# strong, weak and granularity scaling: one point by hand, then the whole sweep
mpirun -n 4 ./build/parallel_advection_scaling --mode strong --spheres 256 --refine 3
python3 run_scaling.py --preset laptop --all    # sweeps the rank counts, writes plots + report

# ... and dump the surface every 10 steps, as one shared file written with MPI-IO
mpirun -n 4 ./build/parallel_advection_redistribute_list 2 2 1 1 1 1 --spheres 256 --refine 3 \
       --vtk 10 --vtk-mode shared
```

Every benchmark checks that its variants produce the same answer before it reports a time, and prints the negative
results as prominently as the positive ones.

## Documentation

**The documentation build lives in a separate repository, `MFTL_Documentation`**, normally checked out beside
this one. This tree carries the sources it reads — the Doxygen comments in `include/` and `src/**`, the ten
`src/<module>/README.md` theory chapters and the four measurement documents in `docs/` — and none of the
machinery that renders them.

```bash
cd ../MFTL_Documentation
python3 build_mkdocs.py                 # -> build/site/index.html   (the web manual)
python3 build_mkdocs.py --serve --open  # live-reload dev server on :8000
python3 build_docs.py --open            # -> build/docs/html         (Doxygen alone)
```

It finds this checkout on its own when the two sit side by side; otherwise pass `--source-dir`, set
`MFTL_SOURCE_DIR`, or write the path into its `mftl-source.txt`.

`build_mkdocs.py` builds a MkDocs Material site — guides, the ten module theory chapters, the four
measurement documents and a browsable API reference — and mounts the complete Doxygen manual inside it at
`/doxygen/`, so the call graphs, the include graphs and the annotated source browser come with it. No prose is
duplicated to make this work: its `mkdocs/gen_pages.py` reads the same files Doxygen reads and rewrites the
dialect at build time, resolving `@ref`, `#MFTL_Symbol` and `MFTLFunction()` against Doxygen's own
`index.xml`. Both manuals build with **zero warnings**, and that is the acceptance criterion for each — so a
new function here without `@param`/`@return` fails `build_docs.py --strict` there.

## Status

Present: ensemble/domain, object pools, half-edge topology, derived arrays and hot-loop iterators, Morton
renumbering, periodic geometry, sphere generation, the fused per-timestep geometry+quality pass, area-weighted
vertex normals, radix sort, redistribution and ghost synchronisation, VTK output, isotropic remeshing —
serial and **distributed, by two strategies** (`src/remeshing/`) — the LBVH with exact object–object contact
detection (`src/bvh/`), and the spatial halo that makes contact detection work across ranks.

Geometry is split by SCOPE, not by entity: `MFTLCompute*` works on one object and never touches MPI,
`MFTLEnsembleUpdate*` iterates the object list and owns the collective its result needs — a reduction for the
triangle quality statistics, an exchange for the vertex normals.

Absent: open-surface generators, GTS I/O, topological ghost-layer growth, and a mesh reader of any format.

## Output

```c
MFTL_VtkOptions o = MFTLVtkDefaults();   // raw binary, one .vtu per rank + .pvtu, owned entities only
o.layout = MFTL_IO_SINGLE_FILE;          // ... or ONE .vtu, written collectively with MPI-IO
MFTLWriteVtu(ens, "surface", &o);
```

Triangles that straddle a periodic face are unwrapped at write time from current
positions and compact connectivity, so they render at their true size without the mesh in memory being touched. Only owned entities are written, so the merged
dataset contains every triangle exactly once and an integral computed over it in ParaView is right.

Raw binary is 23–32× faster to write than ASCII and about half the size. The shared file is **not** faster on a
laptop — it exists because P+1 files per dump is what stops working first on a parallel filesystem. See
`docs/performance-analysis.md` §8, and re-measure on the target machine before choosing.

See `CLAUDE.md` for the architecture in detail and `docs/performance-analysis.md` for what has actually been
measured — including six results that did not come out the way the design predicted, and the seven measurement
mistakes made while getting them.
