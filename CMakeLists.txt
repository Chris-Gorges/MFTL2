cmake_minimum_required(VERSION 3.13)

project(MFTL2 VERSION 2.0.0 LANGUAGES C)

# MFTL2 is either developed on its own (top level: tests, examples, sanitizers) or consumed
# by another project (add_subdirectory / find_package, e.g. multiflow).
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(MFTL2_TOP_LEVEL ON)
else()
    set(MFTL2_TOP_LEVEL OFF)
endif()

# --- Options ---
option(ENABLE_SANITIZERS "Build the library, tests and examples with ASan+UBSan in Debug mode (consumers of such an archive are linked with the sanitizer runtime)" ${MFTL2_TOP_LEVEL})
option(BUILD_EXAMPLES "Build example executables" ${MFTL2_TOP_LEVEL})
option(MFTL_BUILD_TESTS "Build the MFTL2 test suite" ${MFTL2_TOP_LEVEL})
option(MFTL_USE_LAPACK "Build with LAPACK/BLAS support (consumed by Stage 08 remeshing; falls back gracefully when off)" OFF)
# Portability: the defaults produce an archive that can be linked by any compatible
# toolchain on any node of the same ISA. -march=native is a local-development opt-in.
option(MFTL_NATIVE_ARCH "Compile the Release library with -march=native (NOT portable across CPUs; development/benchmarking only)" OFF)
option(MFTL_ENABLE_IPO "Link-time optimisation of the Release library (GCC: fat LTO objects, so non-LTO consumers still link)" ON)
option(MFTL_BUILD_MANUAL "Add the target `manual` that builds doc/MFTL2.pdf (needs pdflatex; never part of the library build)" OFF)

# --- Dependencies ---
find_package(MPI REQUIRED COMPONENTS C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Compiler Detection & Configuration (all target-scoped, nothing global) ---
set(MFTL2_GNU_LIKE OFF)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(MFTL2_GNU_LIKE ON)
endif()

# Sanitizers are decided at configure time (single-config builds): when the archive is
# instrumented, its INTERFACE carries the sanitizer link flag so every consumer gets the
# runtime it needs; a non-instrumented archive carries nothing.
set(MFTL2_SANITIZE OFF)
if(ENABLE_SANITIZERS AND MFTL2_GNU_LIKE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MFTL2_SANITIZE ON)
    set(MFTL2_SANITIZER_FLAGS -fsanitize=address,undefined)
endif()

# LTO: only when supported, and (GCC) with fat objects so the archive still contains
# regular object code for consumers that do not link with -flto.
set(MFTL2_IPO OFF)
if(MFTL_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
    if(IPO_SUPPORTED AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        set(MFTL2_IPO ON)
    elseif(NOT IPO_SUPPORTED)
        message(STATUS "Link Time Optimization not supported: ${IPO_ERROR}")
    else()
        message(STATUS "LTO disabled for ${CMAKE_C_COMPILER_ID}: no fat LTO objects (archive would be unusable without LTO)")
    endif()
endif()

# Compile options for MFTL2's own test/example executables (never exported).
function(mftl2_dev_target_options tgt)
    if(MFTL2_SANITIZE)
        target_compile_options(${tgt} PRIVATE ${MFTL2_SANITIZER_FLAGS} -fno-omit-frame-pointer -g)
        target_link_options(${tgt} PRIVATE ${MFTL2_SANITIZER_FLAGS})
    elseif(MFTL2_GNU_LIKE)
        target_compile_options(${tgt} PRIVATE $<$<CONFIG:Debug>:-g>)
    endif()
endfunction()

# --- Directories ---
set(MFTL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(MFTL_LIB_DIR ${PROJECT_BINARY_DIR}/lib)

# --- Library Target ------------------------------------------------------------
# Consumable target: another CMake project links `MFTL2` (or `MFTL2::MFTL2`) and
# gets the public headers, MPI, and (optionally) LAPACK transitively — without
# knowing anything about the src/ layout. The archive is libMFTL2.a, distinct
# from MFTL1's libMFTL.a so the two can coexist during the Stage-15 migration.
file(GLOB_RECURSE MFTL_SOURCES "src/*.c")

add_library(MFTL2 STATIC ${MFTL_SOURCES})

# Public include interface. `external/` holds khash.h, which mftl_core.h exposes
# in the public MFTL_Object struct and in public function signatures, so it is
# part of the public interface and must travel with the target. Internal source
# subdirectories (src/core, src/parallel, src/geometry, src/remeshing, …) are NOT exposed: consumers only
# ever include the flat public headers in include/.
target_include_directories(MFTL2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<INSTALL_INTERFACE:include>
)
# Private declarations shared across the source subdirectories (never installed or exported).
target_include_directories(MFTL2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/internal)
target_link_libraries(MFTL2 PUBLIC MPI::MPI_C)
# libm is part of the link interface (sqrt, pow, ... are used throughout)
if(UNIX AND NOT APPLE)
    target_link_libraries(MFTL2 PUBLIC m)
endif()

# Library compile flags: PRIVATE (they do not leak into consumers).
if(MFTL2_GNU_LIKE)
    target_compile_options(MFTL2 PRIVATE $<$<CONFIG:Release>:-O3>)
    if(MFTL_NATIVE_ARCH)
        target_compile_options(MFTL2 PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
endif()
if(MFTL2_IPO)
    set_property(TARGET MFTL2 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    target_compile_options(MFTL2 PRIVATE $<$<CONFIG:Release>:-ffat-lto-objects>)
endif()
if(MFTL2_SANITIZE)
    target_compile_options(MFTL2 PRIVATE ${MFTL2_SANITIZER_FLAGS} -fno-omit-frame-pointer -g)
    target_link_options(MFTL2 INTERFACE ${MFTL2_SANITIZER_FLAGS})
    message(STATUS "MFTL2 archive is sanitizer-instrumented; consumers are linked with ${MFTL2_SANITIZER_FLAGS}")
elseif(MFTL2_GNU_LIKE)
    target_compile_options(MFTL2 PRIVATE $<$<CONFIG:Debug>:-g>)
endif()

# Backwards-compatible aliases: MFTL2's own examples/tests (and any early
# consumer) may still refer to `MFTL`; the namespaced alias is the form external
# projects should use with find_package(MFTL2).
add_library(MFTL ALIAS MFTL2)
add_library(MFTL2::MFTL2 ALIAS MFTL2)

# --- Optional LAPACK/BLAS (Stage 08 paraboloid least-squares fit) ---
# Off by default: the library builds and functions without LAPACK. When
# requested and found, consumers of the MFTL2 target get the link interface
# and the MFTL_HAVE_LAPACK compile definition.
if(MFTL_USE_LAPACK)
    find_package(LAPACK)
    if(LAPACK_FOUND)
        target_link_libraries(MFTL2 PUBLIC LAPACK::LAPACK)
        target_compile_definitions(MFTL2 PUBLIC MFTL_HAVE_LAPACK)
    else()
        message(WARNING "MFTL_USE_LAPACK=ON but LAPACK/BLAS was not found; building without LAPACK support")
    endif()
endif()

set_target_properties(MFTL2 PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${MFTL_LIB_DIR}
    OUTPUT_NAME MFTL2
)

# --- Executable Target (Conditional) ---
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "examples/*.c")
    foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
        get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        
        # Link math library (m) and MFTL
        target_link_libraries(${TARGET_NAME} PRIVATE MFTL m)
        target_include_directories(${TARGET_NAME} PRIVATE ${MFTL_INCLUDE_DIR})
        mftl2_dev_target_options(${TARGET_NAME})
    endforeach()
endif()

# --- Manual (optional; LaTeX is never needed for the library) ---
if(MFTL_BUILD_MANUAL)
    find_program(MFTL2_PDFLATEX pdflatex)
    find_program(MFTL2_MAKE make)
    if(MFTL2_PDFLATEX AND MFTL2_MAKE)
        add_custom_target(manual
            COMMAND ${MFTL2_MAKE} -C ${CMAKE_CURRENT_SOURCE_DIR}/doc OUT=${CMAKE_CURRENT_BINARY_DIR}/doc PDFLATEX=${MFTL2_PDFLATEX}
            COMMENT "Building the MFTL2 manual (${CMAKE_CURRENT_BINARY_DIR}/doc/MFTL2.pdf)"
            VERBATIM)
    else()
        message(WARNING "MFTL_BUILD_MANUAL=ON but pdflatex or make was not found; no `manual` target")
    endif()
endif()

# --- Install / export (so external projects can `find_package(MFTL2)`) ---------
# add_subdirectory() consumers already work via the target above; these rules
# additionally support an installed / packaged MFTL2 that a separate build tree
# can discover. Internal src/ layout is never installed — only the public
# headers and khash.h.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS MFTL2 EXPORT MFTL2Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/external/khash.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT MFTL2Targets NAMESPACE MFTL2:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MFTL2)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MFTL2ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
# Minimal package config: re-find the PUBLIC dependencies (MPI always; LAPACK
# only when this build enabled it — otherwise the imported target's link
# interface would reference LAPACK::LAPACK without a find_dependency for it),
# then pull in the exported targets.
set(_MFTL2_CFG "include(CMakeFindDependencyMacro)\nfind_dependency(MPI COMPONENTS C)\n")
if(MFTL_USE_LAPACK AND LAPACK_FOUND)
    string(APPEND _MFTL2_CFG "find_dependency(LAPACK)\n")
endif()
string(APPEND _MFTL2_CFG "include(\"\${CMAKE_CURRENT_LIST_DIR}/MFTL2Targets.cmake\")\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/MFTL2Config.cmake" "${_MFTL2_CFG}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MFTL2Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MFTL2ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MFTL2)
# Also export from the build tree so a sibling build can point CMAKE_PREFIX_PATH
# here without installing.
export(EXPORT MFTL2Targets NAMESPACE MFTL2:: FILE "${CMAKE_CURRENT_BINARY_DIR}/MFTL2Targets.cmake")

# --- Status Output ---
message(STATUS "-------------------------------------------------")
message(STATUS "MFTL Build Configuration")
message(STATUS "-------------------------------------------------")
message(STATUS "Compiler ID      : ${CMAKE_C_COMPILER_ID}")
message(STATUS "Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers       : ${MFTL2_SANITIZE}")
message(STATUS "LTO (Release)    : ${MFTL2_IPO}")
message(STATUS "-march=native    : ${MFTL_NATIVE_ARCH}")
message(STATUS "Build Examples   : ${BUILD_EXAMPLES}")
if(MFTL_USE_LAPACK AND LAPACK_FOUND)
    message(STATUS "LAPACK           : ON")
else()
    message(STATUS "LAPACK           : OFF")
endif()
message(STATUS "Output Directory : ${MFTL_LIB_DIR}")
message(STATUS "-------------------------------------------------")

# --- Testing ---
if(MFTL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
