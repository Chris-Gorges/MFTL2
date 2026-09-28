cmake_minimum_required(VERSION 3.13)

project(MFTL VERSION 6.0 LANGUAGES C)

# --- Options ---
option(ENABLE_SANITIZERS "Enable Address and Undefined sanitizers in Debug mode" ON)
option(BUILD_EXAMPLES    "Build example executables" ON)
option(BUILD_TESTING     "Build the unit test suite" ON)

# -march=native bakes in the ISA of the *build* machine. On a cluster the login
# node routinely differs from the compute nodes, which turns into SIGILL at run
# time, so it must be opted into explicitly. Set MFTL_ARCH_FLAGS instead to
# target a specific microarchitecture (e.g. -march=sapphirerapids).
option(MFTL_ARCH_NATIVE "Compile with -march=native (build machine ISA only!)" OFF)
set(MFTL_ARCH_FLAGS "" CACHE STRING "Extra architecture flags for Release builds")

# The redistribution's per-phase timers. OFF by default because two of them are
# not free: the per-body timeline reads the clock ten times per BODY per call
# (1.5-2 % of the phase at a thousand droplets), and the ownership-change count
# adds a compare to compute_owners' two per-entity loops. The COUNTERS in
# MFTL_RedistributeStats are unconditional either way; only the t_* fields need
# this. See the instrumentation section of src/communication/comm_internal.h.
option(MFTL_COMM_PROFILE "Time the individual phases of MFTLRedistribute()" OFF)

# --- Dependencies ---
find_package(MPI REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Disable compiler-specific extensions (like GNU's gnu11) to guarantee
# strictly standard-compliant C11.
set(CMAKE_C_EXTENSIONS OFF)

# --- Link Time Optimization (LTO/IPO), Release only ---
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(WARNING "Link Time Optimization not supported: ${IPO_ERROR}")
endif()

# --- Library Target ---
# CONFIGURE_DEPENDS makes CMake re-run the glob when the source list changes,
# so a newly added .c under src/ is picked up without a manual reconfigure.
file(GLOB_RECURSE MFTL_SOURCES CONFIGURE_DEPENDS "src/*.c")

add_library(MFTL STATIC ${MFTL_SOURCES})

if(MFTL_COMM_PROFILE)
    target_compile_definitions(MFTL PUBLIC MFTL_COMM_PROFILE)
endif()

# The adaptive orientation predicate in src/bvh/bvh_predicates.c is built from
# error-free transformations: Two_Product recovers the exact residue of a*b by
# evaluating `x - a*b` in a SECOND rounding step. A fused multiply-add contracts
# that into one operation with one rounding, the residue comes out zero, and
# every exactness guarantee in the file silently evaporates -- the predicate
# still returns plausible numbers, just no longer exact ones. The file carries
# `#pragma STDC FP_CONTRACT OFF`, but not every compiler honours it, so the flag
# is set here as well. This is correctness, not tuning: do not remove it.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set_source_files_properties(src/bvh/bvh_predicates.c
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# Usage requirements: anything linking MFTL inherits its headers and MPI.
target_include_directories(MFTL PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
)
target_link_libraries(MFTL PUBLIC MPI::MPI_C)
target_link_libraries(MFTL PUBLIC m)

set_target_properties(MFTL PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# --- Compiler flags (scoped to the library target, not the whole directory) ---
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")

    target_compile_options(MFTL PRIVATE
        -Wall -Wextra
        $<$<CONFIG:Release>:-O3>
    )

    if(MFTL_ARCH_NATIVE)
        target_compile_options(MFTL PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
    if(MFTL_ARCH_FLAGS)
        target_compile_options(MFTL PRIVATE $<$<CONFIG:Release>:${MFTL_ARCH_FLAGS}>)
    endif()

    # Guarantee assertions are stripped in Release (MFTL_ASSERT keys off NDEBUG).
    target_compile_definitions(MFTL PUBLIC $<$<CONFIG:Release>:NDEBUG>)

    if(ENABLE_SANITIZERS)
        # Sanitizers must be PUBLIC: every target linking MFTL has to be built
        # and linked with the same instrumentation, or ASan reports bogus errors.
        target_compile_options(MFTL PUBLIC
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
            $<$<CONFIG:Debug>:-g>
        )
        target_link_options(MFTL PUBLIC $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    else()
        # Debug symbols only (for Valgrind)
        target_compile_options(MFTL PUBLIC $<$<CONFIG:Debug>:-g>)
    endif()

elseif(MSVC)
    target_compile_options(MFTL PRIVATE $<$<CONFIG:Release>:/O2>)
    target_compile_definitions(MFTL PUBLIC $<$<CONFIG:Release>:NDEBUG>)
endif()

# --- Examples ---
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "examples/*.c")
    foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
        get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE MFTL)
    endforeach()
endif()

# --- Testing ---
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Documentation ---------------------------------------------------------
#
# There is none here, on purpose. The Doxygen comments in include/ and src/**,
# the ten src/<module>/README.md theory chapters and the four measurement
# documents in docs/ are the sources; everything that RENDERS them -- the
# Doxyfile, the MkDocs site, the page-generation machinery and their Python
# environment -- lives in the separate MFTL_Documentation repository, which
# reads this tree from wherever it is checked out.
#
#     cd ../MFTL_Documentation
#     python3 build_mkdocs.py --serve --open    # the web manual
#     python3 build_docs.py --strict            # the Doxygen manual, zero-warning
#
# So this file never calls find_package(Doxygen) and there is no `docs` target:
# a documentation toolchain cannot slow down, or fail, a library build it is not
# part of.

# --- Status Output ---
message(STATUS "-------------------------------------------------")
message(STATUS "MFTL Build Configuration")
message(STATUS "-------------------------------------------------")
message(STATUS "Compiler ID      : ${CMAKE_C_COMPILER_ID}")
message(STATUS "Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers       : ${ENABLE_SANITIZERS}")
message(STATUS "LTO (Release)    : ${IPO_SUPPORTED}")
message(STATUS "-march=native    : ${MFTL_ARCH_NATIVE}")
message(STATUS "comm profiling   : ${MFTL_COMM_PROFILE}")
message(STATUS "Extra arch flags : ${MFTL_ARCH_FLAGS}")
message(STATUS "Build Examples   : ${BUILD_EXAMPLES}")
message(STATUS "Build Tests      : ${BUILD_TESTING}")
message(STATUS "Output Directory : ${CMAKE_BINARY_DIR}/lib")
message(STATUS "-------------------------------------------------")
