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
│   └── normal[]             SoA          derived per-vertex geometry
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

Lookups are arrays: global IDs are dense per entity class, and "the other side of this edge" is already an index.
The one hash table is the global-id → slot table redistribution uses for arriving entities.

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
MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_AREA);  // collective: normals + one ghost exchange

printf("mean-ratio quality: %f\n", obj->quality.avg_aspect_ratio);

MFTLWriteVtu(ens, "surface", MFTL_NULL);      // collective: raw-binary .vtu per rank + .pvtu

MFTLEnsembleDestroy(&ens);                    // cascades to every object
MPI_Finalize();
```

## Configuring a run

Every runtime option — domain and decomposition, remeshing, topology changes, contact, output — lives in one struct,
`MFTL_Settings`, filled from one settings file per simulation (a subset of TOML) and `section.key=value` overrides
on the command line. Each section embeds the parameter block its module already takes, so a call passes
`&cfg.remesh.params` or `&cfg.output.vtk`; no module reads the settings behind its caller's back.

```toml
# run.toml
[remesh]
every             = 1
target_length     = 0.0        # 0: hold the mesh's starting resolution
parallel_strategy = "auto"

[topology]
every = 5
```

```c
MFTL_Settings cfg;
MFTLSettingsDefaults(&cfg);
MFTLSettingsLoad(&cfg, "run.toml", MPI_COMM_WORLD);       // rank 0 reads, every rank parses the same text
MFTLSettingsLoadArgs(&cfg, argc, argv, MPI_COMM_WORLD);   // remesh.every=2 on the command line
MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&cfg, MPI_COMM_WORLD);
/* ... create the bodies ... */
MFTLSettingsResolve(&cfg, ens);                           // lengths left at 0, measured from the mesh
MFTLRemeshEnsemble(ens, &cfg.remesh.params, &stats);
```

A misspelled key, a wrong type or a value out of range is an error naming the file, the line and the key that was
probably meant. `./build/mftl_settings template` prints every key with its documentation; `./build/mftl_settings
check run.toml --np 64` validates a file before a job is queued. Every key, default and admissible value is in the
manual's Settings reference, generated from the one table that declares them (`src/settings/settings_keys.c`).

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

# a solver-shaped run configured entirely by a settings file, plus command-line overrides
./build/mftl_settings check examples/settings/settings_driven_run.toml --np 4
mpirun -n 4 ./build/settings_driven_run examples/settings/settings_driven_run.toml remesh.every=2 --steps 50

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

- **Read without building anything:** the pre-built manual in the `MFTL2_Documentation` repository —
  `prebuilt/MFTL_Manual.pdf` (the whole manual in one PDF) and `prebuilt/html/index.html` (the web manual for
  offline use, opened straight from disk, with the Doxygen manual — call graphs, source browser — at
  `prebuilt/html/doxygen/index.html`).
- **In this tree:** the Doxygen comments in `include/` and `src/**` and the twelve `src/<module>/README.md` theory
  chapters (they render on GitHub next to the code). The measurement document the source comments cite as
  `docs/analysis.md §N` is kept local and is not part of the repository; the manual leaves its Analysis section out
  when it is absent.

The documentation build lives in [`MFTL2_Documentation`](https://github.com/Chris-Gorges/MFTL2_Documentation)
(private), checked out beside this one (`../MFTL2` and `../MFTL2_Documentation`). One command checks every prerequisite, builds the whole manual — guides, theory
chapters with their figures, API reference, and the Doxygen manual with its call and include graphs — and opens
it in the browser:

```bash
cd ../MFTL2_Documentation
python3 view_manual.py                  # everything, one command; the first run takes a few minutes
python3 view_manual.py --check          # only report what is missing, with the command that installs it
python3 build_mkdocs.py --serve --open  # live-reload while writing prose
python3 build_docs.py --open            # the Doxygen manual alone: call graphs, source browser
python3 export_manual.py                # refresh prebuilt/ (offline HTML + PDF)
```

It finds this checkout on its own when the two sit side by side; otherwise pass `--source-dir`, set
`MFTL_SOURCE_DIR`, or write the path into its `mftl-source.txt`. Requirements: Python ≥ 3.9, Doxygen and Graphviz;
the MkDocs toolchain installs itself into a private virtual environment on first use. Both manuals build
with **zero warnings**, and that is the acceptance criterion — a new function here without `@param`/`@return`
fails `build_docs.py --strict` there.

## Status

Present: ensemble/domain with sparse per-rank body lists, object pools, half-edge topology, derived arrays and
hot-loop iterators, Morton renumbering, periodic geometry (area- or angle-weighted vertex normals, closest point on
a triangle, signed-distance rasterisation), sphere generation and marching cubes, per-body reductions and prefix
sums over each body's holders, the fused per-timestep geometry + per-body quality pass, radix sort, redistribution
and ghost synchronisation, VTK output, isotropic remeshing — serial and **distributed, by two strategies**
(`src/remeshing/`) — the LBVH with exact object–object and self-proximity contact detection (`src/bvh/`), the
spatial halo that makes contact detection work across ranks, **topology changes** (breakup and coalescence,
`src/topology_change/`), and the **run configuration** (every runtime option in one struct, from a settings file and
the command line, `src/settings/`).

Geometry is split by SCOPE, not by entity: `MFTLCompute*` works on one object and never touches MPI,
`MFTLEnsembleUpdate*` iterates the object list and owns the collective its result needs — a reduction for the
triangle quality statistics, an exchange for the vertex normals.

Absent: open-surface generators (other than a level set through `MFTLBuildMeshFromLevelSet`), GTS I/O, topological
ghost-layer growth, continuous-time (swept) contact or topology detection, and a mesh reader of any format.

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
`docs/analysis.md` §3, and re-measure on the target machine before choosing.

See `docs/analysis.md` for what has actually been measured — the decisions each result supports, the designs that
were rejected, the silent failures found and how each is guarded, the open issues, and the measurement traps that
produced plausible wrong answers first.
