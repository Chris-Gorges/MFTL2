# MultiFlow Triangle Library (MFTL) Version 2.0

MFTL2 is a C99 MPI library for distributed triangulated surfaces: closed bodies, surfaces that
cross or wrap periodic boundaries, and open surfaces with tagged physical boundaries. It is the
surface engine of the multiflow front-tracking and immersed-boundary methods.

The full user and developer manual is `doc/MFTL2.tex`: build it with `make -C doc`, or configure
with `-DMFTL_BUILD_MANUAL=ON` and build the target `manual` (see `doc/README.md`; LaTeX is never
needed for the library itself).

## Features

- **Parallel data model**: half-edge meshes in flat arrays, exactly-once ownership with one ghost
  layer, redistribution over an MPI Cartesian or rectilinear decomposition, ghost refresh and
  opaque per-triangle payloads
- **Generalized surfaces**: open bodies with tagged boundary loops; periodic domains with image
  counters, compact and wrapping bodies, and phase-boundary sets whose closed-phase quantities
  carry a DEFINED / UNDEFINED / UNSUPPORTED status
- **Geometry**: volume, surface centre, moment of inertia, vertex normals, curvature
  (mean 2/R, Gaussian 2/R^2 on a sphere), roughness, transforms (translate, scale, rotate)
- **Queries and intersection**: axis-aligned bounding boxes, point-in-object ray casting, a linear
  bounding-volume hierarchy (LBVH), triangle-triangle intersection and overlap volumes
- **Remeshing**: edge split, collapse and flip with fitting and smoothing, volume conservation,
  serial and load-balanced parallel drivers, and boundary-aware remeshing of open surfaces
- **I/O**: GTS, VTK/VTU/PVTU output, native restart meshes (with image counters and boundary tags)
- **Generators**: spheres, cylinders, boxes, cubes and planar patches

## Requirements

- CMake 3.13 or newer and a C99 compiler
- an MPI implementation (C bindings)
- optional: LAPACK and BLAS (`-DMFTL_USE_LAPACK=ON`) for the paraboloid vertex fit; without them
  the remesher uses a barycentric fit
- optional: pdflatex, only to build the manual

## Quick start

    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DMFTL_USE_LAPACK=ON
    cmake --build build-release -j
    cd build-release && ctest -j6 --output-on-failure

Consumers use the exported package: `find_package(MFTL2 CONFIG REQUIRED)` and link
`MFTL2::MFTL2` (see `external_link_test/`). Example programs are in `examples/`; the
`manual_*.c` programs are the listings of the manual.
