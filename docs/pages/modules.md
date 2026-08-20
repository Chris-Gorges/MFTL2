The ten modules {#mftl_page_modules}
===============

[TOC]

**One module = one `src/<module>/` directory = one `include/mftl_<module>.h`.**

Each directory carries a `README.md` that explains the architecture and the
theory of what lives in it. Those files are bound into this manual as full pages,
and they are also the first thing to read in the repository itself.

| Directory | Public header | Theory page | Layer |
|---|---|---|---|
| `src/arrayfunctions/` | @ref mftl_arrayfunctions.h | @subpage mftl_mod_arrayfunctions | 0 |
| `src/object/` | @ref mftl_object.h | @subpage mftl_mod_object | 3 |
| `src/topology/` | @ref mftl_topology.h | @subpage mftl_mod_topology | 2 |
| `src/geometry/` | @ref mftl_geometry.h | @subpage mftl_mod_geometry | 2 |
| `src/mesh_creation/` | @ref mftl_meshgen.h | @subpage mftl_mod_meshgen | 3 |
| `src/ensemble/` | @ref mftl_ensemble.h | @subpage mftl_mod_ensemble | 3 |
| `src/communication/` | @ref mftl_communication.h | @subpage mftl_mod_communication | 3 |
| `src/remeshing/` | @ref mftl_remeshing.h | @subpage mftl_mod_remeshing | 3 |
| `src/bvh/` | @ref mftl_bvh.h | @subpage mftl_mod_bvh | 3 |
| `src/io/` | @ref mftl_io.h | @subpage mftl_mod_io | 3 |

---

# How they depend on each other {#modules_deps}

The dependency graph is acyclic by construction, because a header may only
include headers from a strictly lower layer (@ref arch_layering).

```
                        ┌──────────────┐
                        │ mftl_core.h  │   layer 4, application only
                        └──────┬───────┘
      ┌──────────┬──────────┬──┴───────┬────────────┬───────────┐
   object    ensemble   meshgen   communication    io       remeshing   bvh
      └──────────┴──────────┴──────────┴────────────┴───────────┘   layer 3
                        ┌──────┴───────┐
                   topology         geometry                        layer 2
                        └──────┬───────┘
                          mftl_types.h                              layer 1
                               │
        constants   memory   vectorops   arrayfunctions             layer 0
```

Two module-to-module dependencies at layer 3 are real and worth naming, because
they are the only ones:

- **remeshing → communication.** The distributed drivers migrate entities, and
  they do it through MFTLRedistributeWithOwners() and #MFTL_OwnerMap rather than
  by growing the production ownership rule. All of it is confined to
  `src/remeshing/remesh_comm.c`.
- **bvh → communication.** The distributed contact pipeline needs the spatial
  halo, which `src/communication/spatial_halo.c` supplies as a standalone shadow
  object per body.

Everything else at layer 3 talks only downward.

---

# What each one owns {#modules_owns}

### @ref mftl_mod_arrayfunctions "arrayfunctions"
Stable LSD radix sort over 8-bit digits, for keys and key/value pairs, plus
duplicate removal. The only algorithm in the library with no mesh in it.

### @ref mftl_mod_object "object"
The pool lifecycle. **The only place a pool pointer, a `capacity` or a `block` is
written.** Also the derived-array rebuild and the Morton renumbering.

### @ref mftl_mod_topology "topology"
Twin rebuilding, boundary marking, integrity checking. The navigation itself is
`static inline` in the header, because it is one or two arithmetic ops.

### @ref mftl_mod_geometry "geometry"
Triangle area/normal/centroid and area-weighted vertex normals, both
periodic-aware, both **local**. The collective wrappers live in `ensemble`.

### @ref mftl_mod_meshgen "mesh_creation"
Indexed-list → half-edge conversion, and the icosphere generator built on top of
it.

### @ref mftl_mod_ensemble "ensemble"
The library handle, the MPI topology setup, and the two fused per-timestep
collectives.

### @ref mftl_mod_communication "communication"
Redistribution, ghost synchronisation, the non-blocking exchange engine, and the
spatial halo. Four files, four distinct jobs.

### @ref mftl_mod_remeshing "remeshing"
Split/collapse/flip/smooth, feature detection, compaction, and **two** distributed
schedules around the same serial kernel. The largest module, and the only one
whose MPI is confined to a single file by a build-time check.

### @ref mftl_mod_bvh "bvh"
Linear BVH build and refit, dual-tree traversal, and the exact triangle–triangle
predicates.

### @ref mftl_mod_io "io"
VTK legacy dump, and the parallel VTK XML writer in four combinations of encoding
and file layout.
