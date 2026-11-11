# --- The GPU examples: CUDA vs OpenACC vs OpenMP ------------------------------
#
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMFTL_ENABLE_GPU=ON
#     cmake --build build
#     mpirun -n 2 build/leveque_gpu_cuda -verify
#
# One MPI driver per example (gcc, like every other example) linked against one
# backend per executable, so the backends are compared on identical host code:
#
#     leveque_gpu_<variant>    parallel LeVeque remeshing   leveque_gpu.c   backend prefix lv_
#     spheres_gpu_<variant>    periodic advection of many   spheres_gpu.c   backend prefix sp_
#                              spheres, no remeshing
#
#     <variant>      source (+ prefix)      compiler
#     cuda           backend_cuda.cu        nvcc
#     openacc        backend_directives.c   nvc -acc=gpu
#     openmp         backend_directives.c   nvc -mp=gpu
#     openacc_host   backend_directives.c   nvc -acc=multicore   the same source on CPU cores
#     openmp_host    backend_directives.c   nvc -mp=multicore    the same source on CPU cores
#     gcc_omp        backend_directives.c   gcc -fopenmp         plain OpenMP threads
#     gcc_serial     backend_directives.c   gcc                  every directive compiled away
#
# The directive variants are ONE source; gpu_pragmas.h turns the compiler flags
# into the programming model. The nvc backends are built as shared libraries
# with nvc itself: nvc does the device link and records its runtime, and the
# driver and MFTL need not be built with nvc. A production code would simply
# compile everything with nvc.
#
# Opt-in from the top level (MFTL_ENABLE_GPU): the library is and stays plain
# C11 + MPI. See README.md in this directory.

# --- CUDA ----------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # The GPU in this machine unless told otherwise; e.g. -DCMAKE_CUDA_ARCHITECTURES="80;90".
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_VERSION VERSION_LESS 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    # One runtime in the process: the shared libcudart.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

# --- nvc (NVIDIA HPC SDK) ------------------------------------------------------
file(GLOB _mftl_nvc_hints
     "$ENV{NVHPC_ROOT}/compilers/bin"
     "$ENV{HOME}/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin"
     "/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin")
find_program(MFTL_NVC nvc HINTS ${_mftl_nvc_hints} DOC "nvc of the NVIDIA HPC SDK, for the OpenACC and OpenMP backends")
set(MFTL_NVC_GPU "ccnative" CACHE STRING "nvc -gpu= target, e.g. ccnative, cc90 or cc80,cc90")

# Without either, only the CPU builds of the directive source (gcc_serial,
# gcc_omp) are made: what CI and a machine without the NVIDIA tools can run.
if(NOT CMAKE_CUDA_COMPILER AND NOT MFTL_NVC)
    message(STATUS "GPU example: neither nvcc nor nvc found, building the CPU variants only. Put them on PATH, or pass "
                   "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc and/or -DMFTL_NVC=/path/to/nvc")
endif()

find_package(OpenMP COMPONENTS C)

# --- Building blocks -----------------------------------------------------------
# <example>_<variant>: the example's driver objects + one backend.
function(mftl_gpu_executable example variant backend)
    add_executable(${example}_${variant} $<TARGET_OBJECTS:${example}_driver>)
    target_link_libraries(${example}_${variant} PRIVATE MFTL ${backend} m)
    set_target_properties(${example}_${variant} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# lib<prefix>_<variant>.so from <source> with nvc and the given model flags.
function(mftl_nvc_backend example prefix variant label source)
    set(_lib ${CMAKE_BINARY_DIR}/lib/lib${prefix}_${variant}.so)
    add_custom_command(
        OUTPUT ${_lib}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E rm -f ${_lib}
        COMMAND ${MFTL_NVC} -O3 -fPIC -shared ${ARGN} -DGPU_BACKEND_NAME="${label}"
                -I${CMAKE_CURRENT_SOURCE_DIR} -Wl,-soname,lib${prefix}_${variant}.so
                -o ${_lib} ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        DEPENDS ${source} gpu_pragmas.h ${prefix}_backend.h ${prefix}_kernels.h
        COMMENT "nvc ${ARGN}: ${source} -> lib${prefix}_${variant}.so"
        VERBATIM)
    add_custom_target(${prefix}_${variant}_so DEPENDS ${_lib})
    add_library(${prefix}_${variant} SHARED IMPORTED GLOBAL)
    set_target_properties(${prefix}_${variant} PROPERTIES IMPORTED_LOCATION ${_lib} IMPORTED_SONAME lib${prefix}_${variant}.so)
    add_dependencies(${prefix}_${variant} ${prefix}_${variant}_so)
    mftl_gpu_executable(${example} ${variant} ${prefix}_${variant})
endfunction()

# The CUDA backend. It does NOT link MFTL: the library's Debug sanitizer flags are PUBLIC and nvcc does not take them.
function(mftl_cuda_backend example prefix source)
    add_library(${prefix}_cuda STATIC ${source})
    set_target_properties(${prefix}_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        INTERPROCEDURAL_OPTIMIZATION OFF
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    target_compile_options(${prefix}_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    mftl_gpu_executable(${example} cuda ${prefix}_cuda)
endfunction()

# The directive source with the build's own C compiler: OpenMP threads and serial C.
function(mftl_host_backends example prefix source)
    add_library(${prefix}_gcc_serial STATIC ${source})
    target_compile_definitions(${prefix}_gcc_serial PRIVATE GPU_BACKEND_NAME="${CMAKE_C_COMPILER_ID}-serial-host")
    target_compile_options(${prefix}_gcc_serial PRIVATE -Wall -Wextra)
    mftl_gpu_executable(${example} gcc_serial ${prefix}_gcc_serial)
    if(OpenMP_C_FOUND)
        add_library(${prefix}_gcc_omp STATIC ${source})
        target_compile_definitions(${prefix}_gcc_omp PRIVATE GPU_OMP_HOST_THREADS GPU_BACKEND_NAME="${CMAKE_C_COMPILER_ID}-omp-host")
        target_compile_options(${prefix}_gcc_omp PRIVATE -Wall -Wextra)
        target_link_libraries(${prefix}_gcc_omp PUBLIC OpenMP::OpenMP_C)
        mftl_gpu_executable(${example} gcc_omp ${prefix}_gcc_omp)
    endif()
endfunction()

# Every backend of one example.
function(mftl_gpu_example example prefix)
    add_library(${example}_driver OBJECT ${example}.c)
    target_link_libraries(${example}_driver PRIVATE MFTL)
    target_compile_options(${example}_driver PRIVATE -Wall -Wextra)

    if(CMAKE_CUDA_COMPILER)
        mftl_cuda_backend(${example} ${prefix} ${prefix}_backend_cuda.cu)
    endif()
    if(MFTL_NVC)
        mftl_nvc_backend(${example} ${prefix} openacc openacc-gpu ${prefix}_backend_directives.c -acc=gpu -gpu=${MFTL_NVC_GPU} -DGPU_REQUIRE_DEVICE=1)
        mftl_nvc_backend(${example} ${prefix} openmp openmp-gpu ${prefix}_backend_directives.c -mp=gpu -gpu=${MFTL_NVC_GPU} -DGPU_REQUIRE_DEVICE=1)
        mftl_nvc_backend(${example} ${prefix} openacc_host openacc-host ${prefix}_backend_directives.c -acc=multicore)
        mftl_nvc_backend(${example} ${prefix} openmp_host openmp-host ${prefix}_backend_directives.c -mp=multicore)
    endif()
    mftl_host_backends(${example} ${prefix} ${prefix}_backend_directives.c)
endfunction()

mftl_gpu_example(leveque_gpu lv)
mftl_gpu_example(spheres_gpu sp)

set(_mftl_gpu_backends "")
if(CMAKE_CUDA_COMPILER)
    list(APPEND _mftl_gpu_backends "cuda (nvcc ${CMAKE_CUDA_COMPILER_VERSION}, ${CMAKE_CUDA_ARCHITECTURES})")
endif()
if(MFTL_NVC)
    list(APPEND _mftl_gpu_backends "openacc, openmp, openacc_host, openmp_host (${MFTL_NVC}, -gpu=${MFTL_NVC_GPU})")
endif()
list(APPEND _mftl_gpu_backends "gcc_serial")
if(OpenMP_C_FOUND)
    list(APPEND _mftl_gpu_backends "gcc_omp (OpenMP ${OpenMP_C_VERSION})")
endif()

list(JOIN _mftl_gpu_backends "; " _mftl_gpu_list)
message(STATUS "GPU example      : ${_mftl_gpu_list}")
