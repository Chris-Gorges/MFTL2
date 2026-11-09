cmake_minimum_required(VERSION 3.13)

project(MFTL VERSION 2.0 LANGUAGES C)

# Examples and tests are built only when MFTL is the top-level project. Pulled
# into a solver with add_subdirectory(), it contributes the library and nothing
# else -- not even when the solver's own include(CTest) has set BUILD_TESTING.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MFTL_IS_TOP_LEVEL ON)
else()
    set(MFTL_IS_TOP_LEVEL OFF)
endif()

# --- Options ---
option(ENABLE_SANITIZERS "Enable Address and Undefined sanitizers in Debug mode" ON)
option(BUILD_EXAMPLES    "Build example executables" ON)
option(BUILD_TESTING     "Build the unit test suite" ON)
option(MFTL_ENABLE_LTO   "Link-time optimisation for Release builds, where supported" ON)
option(MFTL_WARNINGS_AS_ERRORS "Treat compiler warnings in the library as errors (CI)" OFF)

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
# Off for a symbol audit (tools/check_symbols.sh): GCC's slim LTO objects carry
# no symbol table that plain `nm` can read.
set(IPO_SUPPORTED OFF)
if(MFTL_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

    if(IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(WARNING "Link Time Optimization not supported: ${IPO_ERROR}")
    endif()
endif()

include(GNUInstallDirs)

# --- Library Target ---
# CONFIGURE_DEPENDS makes CMake re-run the glob when the source list changes,
# so a newly added .c under src/ is picked up without a manual reconfigure.
file(GLOB_RECURSE MFTL_SOURCES CONFIGURE_DEPENDS "src/*.c")

add_library(MFTL STATIC ${MFTL_SOURCES})
# The same name a consumer gets from find_package(MFTL), so a solver's
# target_link_libraries(... MFTL::MFTL) works with add_subdirectory() too.
add_library(MFTL::MFTL ALIAS MFTL)

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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
    if(MFTL_WARNINGS_AS_ERRORS)
        target_compile_options(MFTL PRIVATE -Werror)
    endif()

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
        # BUILD_INTERFACE keeps them out of the installed package: a solver that
        # links an installed MFTL in its own Debug build must not inherit ASan.
        target_compile_options(MFTL PUBLIC
            $<BUILD_INTERFACE:$<$<CONFIG:Debug>:-fsanitize=address,undefined>>
            $<BUILD_INTERFACE:$<$<CONFIG:Debug>:-fno-omit-frame-pointer>>
            $<BUILD_INTERFACE:$<$<CONFIG:Debug>:-g>>
        )
        target_link_options(MFTL PUBLIC $<BUILD_INTERFACE:$<$<CONFIG:Debug>:-fsanitize=address,undefined>>)
    else()
        # Debug symbols only (for Valgrind)
        target_compile_options(MFTL PUBLIC $<BUILD_INTERFACE:$<$<CONFIG:Debug>:-g>>)
    endif()

elseif(MSVC)
    target_compile_options(MFTL PRIVATE $<$<CONFIG:Release>:/O2>)
    target_compile_definitions(MFTL PUBLIC $<$<CONFIG:Release>:NDEBUG>)
endif()

# --- Install and package --------------------------------------------------
#
#     cmake --install build --prefix <prefix>
#
# installs libMFTL.a, the public headers and a CMake package, so a solver can
#
#     find_package(MFTL 2.0 REQUIRED)        # with -DCMAKE_PREFIX_PATH=<prefix>
#     target_link_libraries(solver PRIVATE MFTL::MFTL)
#
# MPI comes with it as a usage requirement. tests/consumer is the check.
include(CMakePackageConfigHelpers)

install(TARGETS MFTL EXPORT MFTLTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
install(EXPORT MFTLTargets
    NAMESPACE MFTL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MFTL
)
configure_package_config_file(cmake/MFTLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MFTLConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MFTL
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/MFTLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MFTLConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MFTLConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MFTL
)

# --- Examples ---
if(BUILD_EXAMPLES AND MFTL_IS_TOP_LEVEL)
    file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "examples/*.c")
    foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
        get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE MFTL)
    endforeach()

    # The cluster scaling drivers (scaling/strong_scaling.c, scaling/weak_scaling.c),
    # built here rather than by hand so they get exactly the library's toolchain,
    # build type, LTO setting and MFTL_COMM_PROFILE definition. See scaling/README.md.
    file(GLOB SCALING_SOURCES CONFIGURE_DEPENDS "scaling/*.c")
    foreach(SOURCE_FILE ${SCALING_SOURCES})
        get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE MFTL)
    endforeach()
endif()

# --- Testing ---
if(BUILD_TESTING AND MFTL_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Documentation ---------------------------------------------------------
#
# There is none here, on purpose. The Doxygen comments in include/ and src/**,
# the twelve src/<module>/README.md theory chapters and the measurement
# document docs/analysis.md are the sources; everything that RENDERS them -- the
# Doxyfile, the MkDocs site, the page-generation machinery and their Python
# environment -- lives in the separate MFTL2_Documentation repository, which
# reads this tree from wherever it is checked out.
#
#     cd ../MFTL2_Documentation
#     python3 view_manual.py                    # the web manual, one command
#     python3 build_mkdocs.py --serve --open    # ... with live reload
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
message(STATUS "Warnings = errors: ${MFTL_WARNINGS_AS_ERRORS}")
message(STATUS "-march=native    : ${MFTL_ARCH_NATIVE}")
message(STATUS "comm profiling   : ${MFTL_COMM_PROFILE}")
message(STATUS "Extra arch flags : ${MFTL_ARCH_FLAGS}")
if(BUILD_EXAMPLES AND MFTL_IS_TOP_LEVEL)
    message(STATUS "Build Examples   : ON")
else()
    message(STATUS "Build Examples   : OFF")
endif()
if(BUILD_TESTING AND MFTL_IS_TOP_LEVEL)
    message(STATUS "Build Tests      : ON")
else()
    message(STATUS "Build Tests      : OFF")
endif()
message(STATUS "Output Directory : ${CMAKE_BINARY_DIR}/lib")
message(STATUS "-------------------------------------------------")
