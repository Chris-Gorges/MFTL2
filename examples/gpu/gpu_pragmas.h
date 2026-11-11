/*
 * @Author: Christian Gorges
 *
 * @brief One spelling for every directive of the GPU examples, so a backend is
 *        written ONCE and the compiler flags alone pick the programming model:
 *
 *   nvc -acc=gpu        OpenACC on the GPU          (_OPENACC)
 *   nvc -acc=multicore  the same OpenACC on CPU cores
 *   nvc -mp=gpu         OpenMP target offload       (_OPENMP)
 *   nvc -mp=multicore   the same target directives on CPU cores
 *   gcc -fopenmp -DGPU_OMP_HOST_THREADS   plain `omp parallel for` threads
 *   gcc                 serial C: every directive expands to nothing
 *   nvcc                CUDA uses only GPU_FN and GPU_RESTRICT from here
 *
 * The macros forward their arguments with C99 _Pragma; variadic, so clause
 * lists with commas (`reduction(+ : a, b)`) pass through unchanged:
 *
 *   GPU_LOOP(clauses)        a flat parallel loop over elements
 *   GPU_OUTER(clauses)       outer loop of a two-level nest: one body per
 *                            gang / team / thread
 *   GPU_INNER(clauses)       inner loop: that body's elements over the vector
 *                            lanes / the team's threads (serial on CPU threads)
 *   GPU_ATOMIC               the next update is atomic
 *   GPU_INNER_ATOMIC         the same inside a GPU_INNER loop: nothing where
 *                            that loop runs serially (CPU threads, serial C),
 *                            since one thread then owns the whole body
 *   GPU_REDUCE_ARRAY(op, a[0 : n])   a reduction into a small host array; a
 *                            clause inside GPU_LOOP. Scalars take the plain
 *                            reduction(op : x) clause.
 *
 *   GPU_ENTER_CREATE(list)   device copies of host arrays, keyed by host address
 *   GPU_EXIT_DELETE(list)
 *   GPU_UPDATE_DEVICE(list)  host -> device, the listed sections only
 *   GPU_UPDATE_HOST(list)    device -> host
 *   GPU_WAIT                 the queued work is complete
 *
 * Every array a loop touches must be mapped beforehand: OpenACC loops carry
 * default(present), so a forgotten array is a runtime error instead of a silent
 * copy per launch (OpenMP has no portable equivalent before 5.1; there, a
 * pointer inside a target region attaches to the mapped copy).
 *
 * OpenACC queues loops and updates on one async queue and waits once
 * (GPU_WAIT); OpenMP runs them synchronously, since a `nowait` target task in
 * these examples never completed with nvc 26.3 (README.md).
 *
 * Parameters a loop shares (a domain box) live on the device as a mapped
 * one-element array, like every other datum. A struct passed by value needs a
 * data clause in OpenACC (default(present) applies to aggregates), and with
 * `firstprivate(box)` nvc 26.3 -acc=multicore computes garbage (README.md).
 */
#ifndef GPU_PRAGMAS_H
#define GPU_PRAGMAS_H

#define GPU_STR(...) #__VA_ARGS__
#define GPU_PRAGMA(...) _Pragma(GPU_STR(__VA_ARGS__))

/* --- Functions called inside the loops ------------------------------------ */
#if defined(__CUDACC__)
#define GPU_FN static inline __host__ __device__
#define GPU_RESTRICT __restrict__
#else
#define GPU_FN static inline
#define GPU_RESTRICT restrict
#endif

#if defined(_OPENACC)
#define GPU_ROUTINE GPU_PRAGMA(acc routine seq)
#else
#define GPU_ROUTINE
#endif

#if defined(_OPENMP) && !defined(GPU_OMP_HOST_THREADS)
#define GPU_BEGIN_DEVICE GPU_PRAGMA(omp declare target)
#define GPU_END_DEVICE GPU_PRAGMA(omp end declare target)
#else
#define GPU_BEGIN_DEVICE
#define GPU_END_DEVICE
#endif

/* --- Loops and data ------------------------------------------------------- */
#if defined(_OPENACC)

#define GPU_MODEL "openacc"
#define GPU_LOOP(...) GPU_PRAGMA(acc parallel loop default(present) async(1) __VA_ARGS__)
#define GPU_OUTER(...) GPU_PRAGMA(acc parallel loop gang default(present) async(1) __VA_ARGS__)
#define GPU_INNER(...) GPU_PRAGMA(acc loop vector __VA_ARGS__)
#define GPU_ATOMIC GPU_PRAGMA(acc atomic update)
#define GPU_INNER_ATOMIC GPU_ATOMIC
/* A reduction implies `copy` (OpenACC 2.7), yet nvc 26.3 lets default(present)
 * win for an array and stops with "not found on device": spell the copy out. */
#define GPU_REDUCE_ARRAY(op, ...) reduction(op : __VA_ARGS__) copy(__VA_ARGS__)
#define GPU_ENTER_CREATE(...) GPU_PRAGMA(acc enter data create(__VA_ARGS__))
#define GPU_EXIT_DELETE(...) GPU_PRAGMA(acc exit data delete(__VA_ARGS__))
#define GPU_UPDATE_DEVICE(...) GPU_PRAGMA(acc update device(__VA_ARGS__) async(1))
#define GPU_UPDATE_HOST(...) GPU_PRAGMA(acc update self(__VA_ARGS__) async(1))
#define GPU_WAIT GPU_PRAGMA(acc wait(1))

#elif defined(_OPENMP) && !defined(GPU_OMP_HOST_THREADS)

#define GPU_MODEL "openmp"
#define GPU_LOOP(...) GPU_PRAGMA(omp target teams distribute parallel for __VA_ARGS__)
/* The two-level nest as `loop`, not `teams distribute` + `parallel for`: nvc
 * compiles the latter in generic mode (one thread per team runs the outer body,
 * wakes the team for each inner loop, parks per-body values in shared memory)
 * -- 1.3x to 2.2x slower kernels than OpenACC. `loop` gives the same blockIdx /
 * threadIdx(128) kernel as gang / vector. Iterations of a `loop` bound to a
 * parallel region complete before the threads go on (OpenMP 5.1, 2.11.7), so
 * consecutive GPU_INNER loops stay ordered. An `atomic` inside a `loop` region
 * is OpenMP 6.0 (5.x allows none in order(concurrent) regions); nvc accepts it. */
#define GPU_OUTER(...) GPU_PRAGMA(omp target teams loop __VA_ARGS__)
#define GPU_INNER(...) GPU_PRAGMA(omp loop bind(parallel) __VA_ARGS__)
#define GPU_ATOMIC GPU_PRAGMA(omp atomic update)
#define GPU_INNER_ATOMIC GPU_ATOMIC
#define GPU_REDUCE_ARRAY(op, ...) reduction(op : __VA_ARGS__)
#define GPU_ENTER_CREATE(...) GPU_PRAGMA(omp target enter data map(alloc : __VA_ARGS__))
#define GPU_EXIT_DELETE(...) GPU_PRAGMA(omp target exit data map(delete : __VA_ARGS__))
#define GPU_UPDATE_DEVICE(...) GPU_PRAGMA(omp target update to(__VA_ARGS__))
#define GPU_UPDATE_HOST(...) GPU_PRAGMA(omp target update from(__VA_ARGS__))
#define GPU_WAIT

#elif defined(_OPENMP)

#define GPU_MODEL "omp-threads"
#define GPU_LOOP(...) GPU_PRAGMA(omp parallel for __VA_ARGS__)
#define GPU_OUTER(...) GPU_PRAGMA(omp parallel for schedule(static))
#define GPU_INNER(...)
#define GPU_ATOMIC GPU_PRAGMA(omp atomic update)
#define GPU_INNER_ATOMIC
#define GPU_REDUCE_ARRAY(op, ...) reduction(op : __VA_ARGS__)
#define GPU_ENTER_CREATE(...)
#define GPU_EXIT_DELETE(...)
#define GPU_UPDATE_DEVICE(...)
#define GPU_UPDATE_HOST(...)
#define GPU_WAIT

#else

#define GPU_MODEL "serial"
#define GPU_LOOP(...)
#define GPU_OUTER(...)
#define GPU_INNER(...)
#define GPU_ATOMIC
#define GPU_INNER_ATOMIC
#define GPU_REDUCE_ARRAY(op, ...)
#define GPU_ENTER_CREATE(...)
#define GPU_EXIT_DELETE(...)
#define GPU_UPDATE_DEVICE(...)
#define GPU_UPDATE_HOST(...)
#define GPU_WAIT

#endif

/* --- Device selection ----------------------------------------------------- */
#if !defined(__CUDACC__)
#if defined(_OPENACC)
#include <openacc.h>
#elif defined(_OPENMP)
#include <omp.h>
#endif

/**
 * @brief Selects device `local_rank % n_devices` and initialises it now, not in
 *        the first timed call.
 * @param require_device Non-zero for the GPU builds: fail instead of quietly
 *                       running on the host. Zero for the CPU builds.
 * @return 0, or 1 when a device is required and there is none.
 */
static inline int gpu_select_device(int local_rank, int require_device)
{
  if (!require_device) return 0;
#if defined(_OPENACC)
  const int n = acc_get_num_devices(acc_device_nvidia);
  if (n == 0) return 1;
  acc_set_device_num(local_rank % n, acc_device_nvidia);
  acc_init(acc_device_nvidia);
  return 0;
#elif defined(_OPENMP) && !defined(GPU_OMP_HOST_THREADS)
  const int n = omp_get_num_devices();
  if (n == 0) return 1;
  omp_set_default_device(local_rank % n);
#pragma omp target
  {
  }
  return 0;
#else
  (void) local_rank;
  return 1;
#endif
}
#endif

#endif
