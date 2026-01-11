# MultiFlow Triangle Library (MFTL) Version 2.0

A high-performance C library for efficient manipulation and processing of triangulated surface meshes in parallel computing environments. MFTL provides comprehensive tools for geometric operations, mesh remeshing, surface intersection detection, and parallel data management for closed triangular surfaces.

## Features

- **Parallel Processing**: Built on MPI for distributed computing across multiple processors
- **Geometric Operations**: Volume calculations, center of mass, moments of inertia, transformations (scale, translate, rotate)
- **Surface Intersection**: Efficient detection and computation of surface intersections
- **Mesh Remeshing**: Advanced remeshing algorithms including edge collapse, edge flip, edge split, and vertex smoothing
- **I/O Support**: Read/write support for GTS (GNU Triangulated Surface) and VTK formats
- **Bounding Box Operations**: Oriented and unit bounding box generation for spatial queries
- **Data Redistribution**: (Automatic load balancing) and data redistribution across processors
- **Object Generation**: Built-in generators for spheres, cylinders, boxes, and cubes

## Requirements

- **CMake** (version ≥ 3.5)
- **MPI** (Message Passing Interface) installation
  - The environment variable `MPI_HOME` must be set to the MPI installation root
- **LAPACK** and **BLAS** libraries