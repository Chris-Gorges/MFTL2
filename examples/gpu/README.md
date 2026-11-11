# `examples/gpu/`: front tracking on the GPU with CUDA, OpenACC and OpenMP offload

Two MFTL examples with the per-vertex and per-triangle work of every step on the GPU. Each is written in three
programming models, **CUDA**, **OpenACC** and **OpenMP offload**, behind one small C interface per example:

| Example | Physics | What dominates a step |
|---|---|---|
| [`leveque_gpu.c`](leveque_gpu.c) | the parallel LeVeque remeshing test of [`leveque_remeshing_parallel.c`](../leveque_remeshing_parallel.c): one sphere in a reversing vortex, remeshed every step | the remesher, on the host |
| [`spheres_gpu.c`](spheres_gpu.c) | periodic advection of many spheres, as in [`parallel_advection_redistribute_list.c`](../parallel_advection_redistribute_list.c) `--quality --normals`, without remeshing | geometry, on the device |

The question they answer: **for a front tracker's per-step geometry, is CUDA worth its extra code compared with
directives, and when does the GPU pay off at all?**

On this laptop (see [Machine](#machine)), in short:

- **Physics.** Every backend reproduces the host run to round-off, at every step and over the whole run.
- **Many spheres: the GPU pays off.** 99.9 % of the step runs on the device. One GPU is 2.5–3.1× faster than the
  same directive source on 8 CPU threads, 6.4–7.7× faster than the pure-MPI library on 12 cores, and 17–18× faster
  than one core. This is on a laptop GPU that runs FP64 at 1/64 of FP32.
- **LeVeque: it does not.** The remesher stays on the host and takes 92–98 % of the step.
- **CUDA is not faster.** OpenACC and OpenMP are as fast as CUDA, or up to 8 % faster. The kernels are FP64-bound,
  so code generation hardly matters here. Their compute code is the same size; CUDA adds a second language and build,
  hand-written reductions, and no CPU fallback.
- **OpenACC worked best out of the box.** OpenMP matched it only after the nest was written with the `loop`
  construct ([below](#1-openmp-generic-mode-and-the-loop-construct)).

[Verdict](#verdict-cuda-or-directives) · [The architecture advice, checked](#the-architecture-advice-checked) ·
[Pitfalls](#pitfalls-found-on-the-way)

---

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMFTL_ENABLE_GPU=ON
cmake --build build -j
mpirun -n 2 build/leveque_gpu_openacc -verify
mpirun -n 1 build/spheres_gpu_openacc -spheres 1000 -verify
ctest --test-dir build -L gpu
```

**Compilers.** Either one is enough; a backend whose compiler is missing is not built. Without both, only the two
CPU builds (`gcc_serial`, `gcc_omp`) are made, so the examples and their tests also build on a machine without the
NVIDIA tools. CI runs those on every push and compiles the CUDA backends for sm_80 and sm_90 (`CI-CUDA`); no hosted
runner has a GPU.

- **nvcc** builds the CUDA backends through CMake's CUDA language. It targets the GPU in the machine unless
  `-DCMAKE_CUDA_ARCHITECTURES="80;90"` says otherwise.
- **nvc** from the NVIDIA HPC SDK builds the OpenACC and OpenMP backends.
  - CMake looks for it on `PATH`, under `$NVHPC_ROOT`, under `~/opt/nvidia/hpc_sdk` and under
    `/opt/nvidia/hpc_sdk`; `-DMFTL_NVC=/path/to/nvc` overrides the search.
  - `-DMFTL_NVC_GPU=cc90` sets its target; the default is `ccnative`.

**What gets built.** MFTL and the two drivers are built by the ordinary C compiler, like every other example. Only
the backends see nvcc or nvc, so all variants run identical host code. The nvc backends are shared libraries, so nvc
does their device link itself; a production code would simply compile everything with nvc. Nothing in `src/` or
`include/` changes, and the library stays plain C11 + MPI.

Each example `<ex>` (`leveque_gpu`, prefix `lv_`; `spheres_gpu`, prefix `sp_`) comes as seven executables:

| Executable | Backend source | Compiler | Runs on |
|---|---|---|---|
| `<ex>_cuda` | `<prefix>_backend_cuda.cu` | `nvcc` | GPU |
| `<ex>_openacc` | `<prefix>_backend_directives.c` | `nvc -acc=gpu` | GPU |
| `<ex>_openmp` | the same | `nvc -mp=gpu` | GPU |
| `<ex>_openacc_host` | the same | `nvc -acc=multicore` | CPU threads |
| `<ex>_openmp_host` | the same | `nvc -mp=multicore` | CPU threads |
| `<ex>_gcc_omp` | the same | `gcc -fopenmp -DGPU_OMP_HOST_THREADS` | CPU threads, plain `omp parallel for` |
| `<ex>_gcc_serial` | the same | `gcc` | one core; every directive compiles away |

**Running.**

- `-cpu` runs the library on the host without any device. It is the reference.
- With more ranks than GPUs, ranks share a device (`local_rank % n_devices`).
- Threaded variants need `mpirun --bind-to none`, or Open MPI pins each rank to one core.
- Exit status 77 means no device; CTest counts that as a skip.

## One source, six builds: `gpu_pragmas.h`

The directive backends are written once. [`gpu_pragmas.h`](gpu_pragmas.h) maps a handful of macros onto the model
that the compiler flags select. It keys on the compiler's own `_OPENACC` / `_OPENMP`, so no extra `-D` is needed,
except `GPU_OMP_HOST_THREADS` for plain CPU threads. The macros are variadic C99 `_Pragma`, so clause lists pass
through unchanged:

| Macro | OpenACC | OpenMP target | OpenMP threads | serial |
|---|---|---|---|---|
| `GPU_LOOP(...)` | `acc parallel loop default(present) async(1)` | `omp target teams distribute parallel for` | `omp parallel for` | — |
| `GPU_OUTER(...)` | `acc parallel loop gang default(present) async(1)` | `omp target teams loop` | `omp parallel for schedule(static)` | — |
| `GPU_INNER(...)` | `acc loop vector` | `omp loop bind(parallel)` | — (serial in the thread) | — |
| `GPU_ATOMIC` | `acc atomic update` | `omp atomic update` | `omp atomic update` | — |
| `GPU_INNER_ATOMIC` | `acc atomic update` | `omp atomic update` | — (one thread owns the body) | — |
| `GPU_REDUCE_ARRAY(op, a[0:n])` | `reduction(op:a[0:n]) copy(a[0:n])` | `reduction(op:a[0:n])` | the same | — |
| `GPU_ENTER_CREATE` / `GPU_EXIT_DELETE` | `enter data create` / `exit data delete` | `target enter data map(alloc)` / `exit data map(delete)` | — | — |
| `GPU_UPDATE_DEVICE` / `GPU_UPDATE_HOST` | `update device` / `update self`, `async(1)` | `target update to` / `from` | — | — |
| `GPU_WAIT` | `acc wait(1)` | — (synchronous, see [pitfalls](#3-openmp-nowait-hung)) | — | — |
| `GPU_ROUTINE` | `acc routine seq` | — | — | — |
| `GPU_BEGIN_DEVICE` / `GPU_END_DEVICE` | — | `omp declare target` / `end declare target` | — | — |
| `GPU_FN`, `GPU_RESTRICT` | `static inline`, `restrict` | the same | the same | the same (`__host__ __device__`, `__restrict__` under nvcc) |

The per-element maths lives once, in [`lv_kernels.h`](lv_kernels.h) and [`sp_kernels.h`](sp_kernels.h). Every
function there is `GPU_ROUTINE GPU_FN`, and the block is wrapped in `GPU_BEGIN_DEVICE` / `GPU_END_DEVICE`. The CUDA
backends include the same headers. Every loop body is a call into them, so the backends differ only in how they
loop, reduce and move data.

The vertex-normal loop of the sphere example, in the directive backend and in CUDA:

```c
// sp_backend_directives.c: OpenACC, OpenMP offload, OpenMP threads and serial C
GPU_OUTER()
for (int s = 0; s < ns; ++s)                      // one gang / team per body
{
  /* ... restrict pointers to slot s, its counts nv and nt ... */
  GPU_INNER()
  for (int k = 0; k < 3 * nv; ++k) vns[k] = 0.0;  // its vertices over the vector lanes / threads
  GPU_INNER()
  for (int t = 0; t < nt; ++t)
  {
    double f[3];
    sp_face_normal(b, ps, ts, t, f);
    for (int corner = 0; corner < 3; ++corner)
      for (int k = 0; k < 3; ++k)
      {
        GPU_INNER_ATOMIC
        vns[3 * ts[3 * t + corner] + k] += f[k];
      }
  }
  GPU_INNER()
  for (int v = 0; v < nv; ++v) sp_normalise_one(vns, v);
}
GPU_WAIT
```

```cuda
// sp_backend_cuda.cu
__global__ void normals_kernel(__grid_constant__ const sp_box box, const double *__restrict__ pos, ...)
{
  const int s = blockIdx.x;                       // one block per body
  /* ... pointers to slot s, its counts nv and nt ... */
  for (int k = threadIdx.x; k < 3 * nv; k += blockDim.x) vns[k] = 0.0;
  __syncthreads();
  for (int t = threadIdx.x; t < nt; t += blockDim.x)
  {
    double f[3];
    sp_face_normal(&box, ps, ts, t, f);
    for (int corner = 0; corner < 3; ++corner)
      for (int k = 0; k < 3; ++k) atomicAdd(&vns[3 * ts[3 * t + corner] + k], f[k]);
  }
  __syncthreads();
  for (int v = threadIdx.x; v < nv; v += blockDim.x) sp_normalise_one(vns, v);
}
normals_kernel<<<g.S, kBlock>>>(g.box, g.pos, g.tv, g.cnt, g.NVS, g.NTS, g.vn);
```

### Code size

Lines without comments and blank lines:

| File | Code lines | |
|---|---:|---|
| `gpu_pragmas.h` | 106 | the macros, device selection |
| `lv_kernels.h` / `sp_kernels.h` | 109 / 114 | the per-element maths, shared by all backends |
| `lv_backend.h` / `sp_backend.h` | 48 / 65 | the interfaces: no MPI and no MFTL type |
| `lv_backend_directives.c` | 156 | LeVeque: one source, six builds, 20 directives |
| `lv_backend_cuda.cu` | 230 | LeVeque: CUDA, 5 kernels |
| `sp_backend_directives.c` | 426 | spheres: one source, six builds, 44 directives |
| `sp_backend_cuda.cu` | 370 | spheres: CUDA, 6 kernels |
| `leveque_gpu.c` / `spheres_gpu.c` | 628 / 913 | the drivers: MPI, library calls, `-cpu`, `-verify`, `-compare`, timers |

- **The compute code is the same size.** The sphere loops (advect, classify, geometry, normals) take 136 lines as
  directives and 134 as CUDA kernels plus launches. The CUDA lines are of a different kind: warp-shuffle and
  shared-memory block reductions, `__syncthreads()` between phases, launch configuration.
- **The sphere directive backend is longer in its infrastructure.** C has no templates, so the `int` and `double`
  transfer paths are written twice. The mapping lists also need `(void)` casts so that the serial build stays free
  of warnings. That buys five more builds, including the CPU fallback, from the same file.

---

## The architecture advice, checked

The advice suggested a central `_Pragma` header, data parked on the device, `restrict` local pointers, routine
markers, and targeted host–device sync around the remesh with 1.5× capacity headroom. Both examples follow it. A
few details had to change, and some things that mattered more were missing.

| Advice | Status | Notes |
|---|---|---|
| A: central `_Pragma` header | **done**, `gpu_pragmas.h` | Variadic, so `reduction(...)` and other clauses pass through. Keyed on `_OPENACC` / `_OPENMP` instead of `USE_*` flags. Adds data macros, `GPU_WAIT` and `default(present)`. |
| `GPU_PARALLEL_LOOP_3D collapse(3)` | not used | A triangulated front has no 3D index space. `collapse(3)` belongs to the Eulerian grid of an IBM solver. The front needs flat loops (`GPU_LOOP`) and, for many bodies, a two-level nest (`GPU_OUTER` / `GPU_INNER`). |
| `GPU_ROUTINE` = `omp declare target` | **corrected** | Without clauses, `omp declare target` starts a region that `omp end declare target` must close; it does not mark just the next function. Hence `GPU_BEGIN_DEVICE` / `GPU_END_DEVICE` around the kernel headers, and `GPU_ROUTINE` (`acc routine seq`) per function for OpenACC. |
| `GPU_ATOMIC_UPDATE` on the CPU | **refined** | Where a CPU thread owns a whole body, the atomic is useless, and expensive: nvc's multicore atomics cost 3.6× gcc's. `GPU_INNER_ATOMIC` compiles away there. |
| B: persistent device mirror | **done** | LeVeque: the remesher hands back reallocated pools, so the backend remaps only when a pool's block pointer or capacity changed. Device allocations went from **285 to 3** per 150 steps (refine 4, np 1). Spheres: everything is allocated once in `sp_setup`, never again. |
| map the struct, rely on pointer attach | not used | `MFTL_Object` holds many pointers the device never needs. The backends map the arrays they use and pass them as local pointers. The library's structs stay host-only, and nothing depends on attach rules that differ in detail between OpenACC and OpenMP. |
| Rule 1: `restrict` local pointers | **done** | Every loop works on `restrict` locals; CUDA kernel parameters are `__restrict__`, kernel helpers take `GPU_RESTRICT`. Not measured separately; the sphere kernels are FP64-bound. |
| Rule 2: routine markers | **done** | All helpers are `static inline` in one header, marked `GPU_ROUTINE GPU_FN`, in a `GPU_BEGIN_DEVICE` block. |
| Rule 3: sync only what the remesh needs | **done**, with a correction | The remesher reallocates the pools, so an `update` alone is not enough when a block moved. The backend then remaps. Otherwise it updates only the used sections of the arrays, not their capacity. |
| 1.5× capacity headroom | **done** where the example allocates | Spheres: slots are 1.5× the bodies held + 64, and staging buffers grow by 1.5×. LeVeque sizes the device arrays by the library pools' own capacity, which already has headroom. |
| "while only coordinates move, everything stays 100 % on the GPU" | **only on one rank** | With domain decomposition, ghost positions and the redistribution need host data every step, since this MPI is not CUDA-aware. The sphere example therefore moves only the bodies split across ranks, packed into one copy. That is still 2.1 MB/step at np 2. |

**What mattered more than the advice:**

1. **Batching.** A launch costs ~16 µs, a small copy ~30 µs. One kernel per body and phase would cost ~80 ms per
   step for 1000 bodies, against 2.7 ms for the whole step. Hence one launch over all bodies: `GPU_OUTER` over the
   bodies, `GPU_INNER` over their elements.
2. **Few, packed transfers.** Each sphere transfer is one gather or scatter kernel plus one copy, whatever the number
   of bodies.
3. **The owner map.** `MFTLRedistribute()` routes triangles by `tris.center`, which would mean downloading every
   centroid every step. The device computes the owner rank instead, one int per entity. It sends only the bodies in
   which an owner changed, and `MFTLRedistributeWithOwners()` routes by them.
4. **The OpenMP construct.** `teams distribute` + `parallel for` and `teams loop` + `loop` are both correct OpenMP,
   but with nvc one runs up to 2.2× slower than the other ([pitfall 1](#1-openmp-generic-mode-and-the-loop-construct)).

---

## Example 1: LeVeque, remeshing every step

### What runs where, one step

| Where | Call | |
|---|---|---|
| device | `lv_advect` | RK4 of the owned vertices in the analytical velocity field |
| D → H | `lv_get_owned_positions` | the host needs them for the ghost sync |
| host | `MFTLUpdateGhostPositions` | MPI |
| H → D | `lv_put_ghost_positions` | |
| device | `lv_triangle_geometry` | area, normal and centroid of every triangle |
| D → H | `lv_get_triangle_geometry` | redistribution routes triangles by their centroids |
| host | `MFTLRemeshEnsemble` | redistribution and remeshing: pointer-heavy, topology-changing |
| H → D | `lv_attach` | the remeshed mesh; a remap only if a pool moved or grew |
| device | `lv_measure` | vertex normals, then the diagnostics; 7 numbers come back |

The backends read and write the library's own pools, such as `verts.pos` and `tris.area`, through the pointer view
`lv_mesh`. Nothing is repacked.

| Option | |
|---|---|
| `-refine N` | sphere subdivision level, default 4 |
| `-T`, `-dt`, `-steps N` | period (3.0), timestep (0.02), stop after N steps (default: all 150) |
| `-cpu` | the library on the host, no device: the reference |
| `-verify` | check every device result at every step, and the mesh at the end |
| `-csv FILE` | per-step diagnostics, including a hash of the mesh |
| `-compare FILE` | require the trajectory of FILE (a `-csv` of a `-cpu` run at the same rank count) to round-off |
| `-quiet` | summary only |

**All ranks must run the same backend.** Ghost triangles are routed by their centroids, and host and device round
those differently in the last bit.

### Is the physics the same?

Not bit for bit: nvcc and nvc contract `a*b+c` into FMAs, and the device `sin`/`cos` are not glibc's.

**Every step, on the same input (`-verify`).** The driver recomputes each device result with the library, from the
device's own input. It then continues with the device values, so the check does not perturb the run. Maximum over
np 1, 2, 4 and every backend:

| Quantity | Refine 3–5 | Refine 2 | Tolerance |
|---|---:|---:|---:|
| vertex position after RK4 | 1.1e-16 | 1.1e-16 | 1e-13 |
| triangle area, relative | 3.2e-14 | 2.9e-13 | 1e-12 |
| triangle normal | 2.1e-14 | 5.6e-13 | 1e-12 |
| triangle centroid | 1.1e-16 | 1.1e-16 | 1e-13 |
| measure sums (area, volume, quality), relative | 2.2e-14 | 1.4e-14 | 1e-11 |
| counts (edges, edges out of band, inverted triangles) | 0 | 0 | 0 |

At refine 2 the filament degenerates into slivers, whose normal and relative area are ill-conditioned.

**The whole run (`-compare`).** Remeshing is a chain of threshold decisions: split if longer, collapse if shorter,
flip if better. A last-bit difference changes a decision only when a value lies within round-off of its threshold,
but from then on the meshes differ. `-compare` checks a hash of the connectivity and the diagnostics V, A and
quality, to 1e-11, against a `-cpu` run at the same rank count.

- **Refine 4 (the full period) and refine 5, np 1, 2 and 4, every backend:** the same mesh at every step. The
  largest relative difference is 4.3e-13 at refine 4 and 5.3e-14 at refine 5.
- **Refine 3 under-resolves the flow.** The host run itself loses 26 % of the volume. After 85–95 steps a decision
  flips, and V then differs by up to 9e-3.
  - The device is not the cause: the `*_host` builds, which run the directive code on the CPU, flip at the same step.
  - The rank count alone changes the host result by more: 2.4e-2 in V between np 1 and np 2.
  - Hence the default `-refine 4`, where the device run *is* the host run, to round-off, over the whole period.

### Measurements

Offloadable part / wall, ms per step. Median of 3 runs, `OMP_NUM_THREADS = ACC_NUM_CORES = 4`. "Offloadable" is
advect + geometry + measure + both transfers:

| Refine | np | Peak triangles | Host (gcc) | CUDA | OpenACC | OpenMP | Remesh share, host run | Device allocations, CUDA / ACC / OMP |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1 | 18,356 | 1.61 / 21.5 | 1.38 / 21.7 | 1.15 / 21.6 | 1.31 / 21.6 | 92 % | 3 / 3 / 3 (before: ≈ 285 for the directives) |
| 4 | 2 | 18,562 | 0.95 / 17.0 | 1.79 / 18.1 | 1.57 / 17.8 | 1.95 / 18.3 | 94 % | 8 / 20 / 24 |
| 4 | 4 | 18,390 | 0.64 / 15.3 | 2.83 / 18.6 | 2.55 / 18.2 | 2.51 / 17.8 | 95 % | 10 / 62 / 115 |
| 5 | 1 | 51,326 | 3.93 / 57.2 | 1.90 / 56.9 | 1.55 / 56.1 | 1.93 / 55.5 | 93 % | 3 each |
| 5 | 4 | 51,058 | 2.17 / 49.8 | 2.86 / 50.1 | 4.22 / 53.6 | 3.00 / 51.4 | 96 % | 8 / 29 / 18 |
| 6 | 1 | 88,998 | 10.00 / 147.3 | 3.13 / 140.4 | 2.92 / 140.4 | 2.92 / 143.1 | 93 % | 2 each |
| 6 | 4 | 88,806 | 6.59 / 155.0 | 4.99 / 154.9 | 7.68 / 154.5 | 4.90 / 152.8 | 95 % | 6 / 12 / 12 |

- **The offloaded part gets faster with size:** 1.4× at refine 4, 3.4× at refine 6, one rank.
- **Amdahl wins.** The remesher holds 92–96 % of the host step, and 98 % of the device run at refine 6. Even a device
  that cost nothing would make a refine-4 step at most 1.08× faster.
- **A shared GPU loses.** At np 2 and 4 all ranks time-slice one laptop GPU (no MPS here). The device part then grows
  with np while the host's shrinks.
- **Latency, not kernels.** Nsight Systems at refine 4, before the persistent mapping: 76–133 µs of kernels per step,
  against about 1.1 ms of host time in CUDA API calls. Synchronous copies from pageable pool memory reach ~2 GB/s.

---

## Example 2: many spheres, geometry every step

N spheres at random centres in the periodic box [-4,4] × [-2,2] × [-2,2], rigidly translated by v = (1, 0.37, 0.61)
and repartitioned every step. After every step come the triangle geometry, the per-body quality statistics of
`MFTLEnsembleUpdate()` and the area-weighted vertex normals of `MFTLEnsembleUpdateVertexNormals()`. No remeshing. Where
LeVeque is one big body and a remesh, this is many small bodies and almost nothing but geometry.

### What runs where, one step

| Where | Call | |
|---|---|---|
| device | `sp_advect` | every owned vertex, wrapped into the box |
| D → H | owned positions of the split bodies | |
| host | `MFTLUpdateGhostPositions` | MPI |
| H → D | their ghost positions | |
| device | `sp_classify` | the owner rank of every vertex and wrapped centroid: `MFTLGetOwnerProc()` on the device |
| D → H | owners of the bodies in which one changed, positions of the unsplit bodies that leave | |
| host | `MFTLRedistributeWithOwners` | MPI; routes by the device's owners |
| H → D | the bodies that arrived or were rebuilt | |
| device | `sp_geometry` | area, normal, centroid of every triangle, fused with 9 statistics per body |
| host | `MFTLEnsembleReduceBodies` | MPI: the per-body reduction of `MFTLEnsembleUpdate()` |
| device | `sp_normals` | vertex normals |
| D → H, host, H → D | the ghost normal exchange of the split bodies | |

**On one rank only the device lines run**, and nothing crosses PCIe but 9 doubles per body.

- **Slots.** The device holds every body of the rank in fixed slots, each sized to a whole body. Slots number 1.5×
  the bodies held after the setup + 64, or `-slots`. They are allocated once (`sp_setup`), and a body that leaves
  frees its slot for the next. The device positions are *the* positions: the host copy of a body that no other rank
  holds is stale until the redistribution needs it.
- **Transfers.** The driver lists segments (slot, first entity, count, staging offset). One gather or scatter kernel
  and one copy move all of them. CUDA stages through pinned memory; the directive builds through mapped `malloc`
  memory.

| Option | |
|---|---|
| `-spheres N` | bodies (1000) |
| `-refine N` | subdivision level (3: 1280 triangles, 642 vertices per sphere) |
| `-radius r`, `-dt step`, `-steps N` | 0.1, 0.1, 100 |
| `-slots N` | device slots per rank |
| `-cpu` | the library on the host, no device: the reference |
| `-verify` | every step: area and counts conserved, and every device result against the library on the same positions |
| `-save FILE` / `-compare FILE` | the final state (area, counts, coordinate sums, normals, quality range) against a `-cpu` run, to 1e-12 |
| `-quiet` | summary only |

Unlike LeVeque, the final state does not depend on the partition, so any rank count and backend may be compared with
any other.

### Is the physics the same?

Maximum over refine 2–3, np 1, 2, 4, every backend:

| Quantity (`-verify`, every step) | Measured | Tolerance |
|---|---:|---:|
| triangle area, relative | 4e-16 | 1e-13 |
| triangle unit normal | 4e-16 | 1e-13 |
| triangle centroid (minimum image) | 9e-16 | 1e-13 |
| vertex normal, ghosts included | 4e-16 | 1e-13 |
| the 9 statistics per body, relative | 1e-14 | 1e-12 |
| total area, \|A(t) − A(0)\| / A(0) | 3e-15 | 1e-12 |
| entity counts | exact | exact |

- **Final state vs the `-cpu` run:** at most 4.4e-15 relative, at every rank count.
- **The 1e-14 of the statistics** is \|e\|² of the longest edge. A minimum-image difference of O(1) coordinates over
  an edge of h ≈ 0.016 is good to about 2ε/h, which is its conditioning, not a device error.
- **The checks themselves must not add round-off.** At first the device sums were consistently 5e-14 off the
  library's, against 6e-15 on the host. The cause was the *reference*: a plain running sum over 10⁵ similar terms is
  order-dependent to ~1e-14. With Neumaier-compensated sums on both sides (`ksum` in `spheres_gpu.c`), the device
  agrees to 1–8 ulp.

### Measurements

Release build, without `-verify`; ms per step, maximum over ranks. GPU runs are the median of three at 1000 and 4000
spheres and single runs otherwise.

**One rank, one GPU, against the best the CPU can do:**

| | 1000 × refine 3 | 4000 × refine 3 | 1000 × refine 4 | 16000 × refine 3 |
|---|---:|---:|---:|---:|
| triangles | 1.28 M | 5.12 M | 5.12 M | 20.5 M |
| library (`-cpu`), 1 rank | 45.6 | 175.5 | 172.7 | — |
| library (`-cpu`), 12 ranks (MPI) | 17.4 | 70.1 | 72.9 | 294.1 |
| directive source, gcc OpenMP threads, 8 threads | 6.76 | 28.5 | 28.4 | 123.6 |
| directive source, nvc `-acc=multicore`, 8 threads | 12.2 | 45.9 | | |
| directive source, nvc `-mp=multicore`, 8 threads | 11.9 | 46.1 | | |
| **CUDA** | 2.93 | 10.46 | 10.11 | 41.6 |
| **OpenACC** | **2.70** | **9.69** | 9.54 | **40.4** |
| **OpenMP** (`teams loop`) | 2.72 | 9.94 | **9.43** | 43.0 |
| OpenMP (`teams distribute` + `parallel for`) | 3.91 | 14.3 | 12.8 | |
| best GPU vs gcc OpenMP, 8 threads | 2.5× | 2.9× | 3.0× | 3.1× |
| best GPU vs library, 12 ranks | 6.4× | 7.2× | 7.7× | 7.3× |
| device memory per rank | 126 MB | 502 MB | | 2.0 GB |

- **Phases at 1000 spheres, OpenACC:** advect 0.13, geometry 1.84, normals 0.73 ms; transfers 0.07 MB per step.
  CUDA: 0.17, 2.04, 0.71 ms.
- **Linear in the work** from 1000 spheres on, so the GPU is saturated there.
- **The library at 12 ranks** spends 38–40 % of its step in the redistribution (6.6 of 17.4 ms; 117 of 294 ms at
  16000 spheres). The single-rank GPU run has none.
- **CPU threads.** The best CPU variant is the directive source on gcc's OpenMP threads. The nvc multicore builds
  keep the GPU code's atomics in the normals: 8.4 against gcc's 2.1 ms. Their geometry is faster (3.1 vs 4.4 ms).

**Why CUDA is not faster: the kernels are FP64-bound.**

- The geometry kernel issues about 180–250 FP64 instructions per triangle (counted in the SASS).
- This GPU retires 86–124 G FP64 instructions/s. That puts the floor at 1.9–2.7 ms at 1000 spheres; measured:
  1.8–2.0 ms.
- Memory traffic is ~88 GB/s, 23 % of the 384 GB/s the board offers.
- So all three models wait on the same FP64 units, and differences in code generation hardly show.
- Shared-memory tiling or other CUDA-only tuning would not help either.
- On a data-centre GPU (FP64 at 1/2 of FP32) the same kernels would be memory-bound. There the GPU's margin over
  the CPU would be far larger, and memory layout and coalescing would matter more. That is where a CUDA-vs-directives
  difference could appear. Not tested here.

**Several ranks on one GPU** (1000 spheres, np 2):

| | CUDA | OpenACC | OpenMP | gcc OpenMP, 2 × 4 threads | library, 2 ranks |
|---|---:|---:|---:|---:|---:|
| wall | 12.5 | 12.2 | 12.8 | 21.2 | 40.3 |

- The device part stays at ~3 ms. What is added is host work: redistribution 4.7 ms, transfers ~4 ms, owner
  downloads 0.8 ms.
- Per rank and step, 50 bodies are split, 63 change an owner and 75 are re-uploaded: 2.1 MB/step H → D.
- The cause is the test case. Rigid motion of 1.2 radii per step makes every body cross rank boundaries constantly.
  A real flow moves the front far less than a cell per step.
- With one GPU, run one rank per GPU. With several GPUs, each rank has its own, and GPU-aware MPI would remove the
  host stage of the ghost and redistribution traffic. This Open MPI is not CUDA-aware.

---

## Pitfalls found on the way

None of these shows in the source; all of them showed up in a measurement or a profile. NVHPC 26.3, CUDA 13.1,
GCC 15.2.

### 1. OpenMP generic mode and the `loop` construct

- **The problem.** The two-level nest was first written as `omp target teams distribute` + `omp parallel for`. nvc
  compiles this in **generic mode**: one thread per team runs the outer body, wakes the team for each inner loop, and
  parks per-body values in shared memory. The kernels ran 1.3–2.2× slower than OpenACC's, and wall time was 1.3–1.5×
  OpenACC's.
- **The fix.** `omp target teams loop` + `omp loop bind(parallel)` gives the same SPMD kernel as OpenACC's
  gang/vector (`blockIdx` × 128 threads), at the same speed.
- **Why it is still correct.** Consecutive inner `loop`s stay ordered (zero → scatter → normalise), because the
  iterations of a `loop` bound to a parallel region complete before the threads go on (OpenMP 5.1, §2.11.7).
- **Caveat: `atomic` inside a `loop` region is an OpenMP 6.0 feature.** OpenMP 5.x forbids it in `order(concurrent)`
  regions. nvc accepts it; other compilers may not, so the normals scatter is the portability risk of this mapping.

### 2. OpenACC: a reduction array and `default(present)`

A reduction implies `copy` (OpenACC 2.7). Yet nvc lets `default(present)` win for a reduction array and stops with
"not found on device". `GPU_REDUCE_ARRAY` therefore spells out the `copy`.

### 3. OpenMP `nowait` hung

A `target … nowait` never completed, with and without `depend`, at one thread and without MPI. The OpenMP builds are
therefore synchronous. OpenACC's `async(1)` + one `wait(1)` works and saved 0.07–0.3 ms per step in LeVeque's measure
phase. The setup may matter: the backend is an nvc-built shared library called from a gcc-built driver.

### 4. `firstprivate` struct with `-acc=multicore`

A parameter struct (the domain box) passed `firstprivate` computed garbage in the multicore build. All parameters now
live on the device as a mapped one-element array, like any other datum. This needs no special case, and OpenACC's
`default(present)` would otherwise demand a clause for the aggregate anyway.

### 5. Atomics on CPU threads

nvc's multicore atomics cost 3.6× gcc's: the normals of 200 spheres took 11.2 against 3.1 ms on one thread. Where one
thread owns a whole body they are not needed, so `GPU_INNER_ATOMIC` compiles away in the plain-threads build.

### 6. 15–16 threads under WSL2

With 15 or 16 threads every OpenMP runtime collapsed on this 16-core laptop, to seconds per step. MPI at np 16
degraded likewise (40 ms vs 3.9 ms at np 8). For nvc this held even with `OMP_WAIT_POLICY=PASSIVE`; gcc recovered
with passive waiting. The CPU baselines therefore use at most 8 threads and 12 ranks.

### 7. Launch and copy latency

A launch costs ~16 µs, launch + sync ~35 µs, and any copy ~30 µs. Pageable transfers reach 10–12 GB/s at 64 MB and
half that at 1 MB. Everything that is per body must be batched (see [above](#the-architecture-advice-checked)).

---

## Verdict: CUDA or directives?

**Directives, and among them OpenACC; OpenMP only with the `loop` construct.**

| | CUDA | OpenACC | OpenMP offload |
|---|---|---|---|
| time per step, many spheres | 1.03–1.09× OpenACC's | fastest | = OpenACC with `loop`; 1.3–1.5× with `distribute` + `parallel for` |
| speed, LeVeque (offloadable part) | about equal | about equal | about equal |
| compute code | same size, explicit threads, reductions, barriers | loops + clauses | loops + clauses |
| same source on CPU cores | no | yes (`-acc=multicore`) | yes (`-mp=multicore`, or any OpenMP compiler) |
| language and build | C++, separate compiler | C, nvc | C, nvc (in principle any offloading compiler) |
| asynchrony | streams (default) | `async` / `wait`, worked | `nowait` hung with nvc 26.3 |
| traps hit | none in the compute | one (`GPU_REDUCE_ARRAY`) | generic mode; OpenMP 6.0 atomic; `nowait` |

**For the many-spheres case, the GPU pays off:** 2.5–3.1× over the best CPU variant and 7× over the pure-MPI library,
on a laptop GPU with FP64 at 1/64. **For LeVeque it does not**, as long as the remesher runs on the host.

**Where CUDA would earn its cost:**

- a remesher on the device, which needs sorts, scans and compactions (CUB, Thrust) that no directive provides;
- streams and CUDA Graphs to overlap transfers with compute and with MPI, and to batch many small launches;
- shared-memory tiling on memory-bound data-centre GPUs.

Both OpenACC (`host_data use_device`) and OpenMP (`use_device_ptr`) hand device pointers to CUDA code. The natural
design for MFTL is therefore **directives in the library's own loops, via `gpu_pragmas.h`, and CUDA libraries for the
primitives directives cannot express**, not one or the other.

### When the GPU pays off for a front tracker

- **The front is large and mostly geometry**, as in the many-spheres case: 99.9 % of the step offloadable.
- **Remeshing is rare.** Remesh when the mesh quality asks for it, not every step. Between remeshes the mesh stays on
  the device.
- **One GPU per rank.** Several ranks time-slicing one GPU only add redistribution and PCIe cost.
- **GPU-aware MPI.** Ghost and redistribution buffers would go from device memory directly, without the host stage.
- **Coupling to a GPU flow solver.** This is the strongest case: IBM spreading and interpolation must run where the
  Eulerian fields live, and moving the fields would cost far more than moving the interface.

---

## Tests

`ctest -L gpu`: 24 tests with every backend built, about 15 s; 12 with `gcc_serial` alone.

| Example | Case | Tests |
|---|---|---|
| LeVeque | refine 3, 40 steps (inside the round-off window) | `-cpu` reference at np 2; every backend `-verify -compare` at np 2; np 1; np 4 at refine 2, where three ranks start without a triangle; a rejected option |
| spheres | 60 spheres, refine 2, 30 steps | `-cpu -verify` reference at np 2; every backend `-verify -compare` at np 2; np 1 and np 4 against the np-2 reference; 3 spheres at np 4 (ranks without a body); a rejected option |

## Machine

- Intel Core Ultra 7 265H (6 P + 8 E + 2 LP cores).
- NVIDIA RTX PRO 1000 Blackwell Laptop GPU (cc 12.0, 20 SMs, 8 GB, 384 GB/s, FP64 at 1/64 of FP32).
- WSL2, driver 581.15. CUDA 13.1, NVHPC 26.3, GCC 15.2. Open MPI 5.0.10, not CUDA-aware.
- Nsight Compute counters are not accessible under WSL2 here (`ERR_NVGPUCTRPERM`). The FP64 analysis above counts
  SASS instructions instead.
