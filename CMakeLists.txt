cmake_minimum_required(VERSION 3.13)

project(MFTL VERSION 6.0 LANGUAGES C)

# --- Options ---
option(ENABLE_SANITIZERS "Enable Address and Undefined sanitizers in Debug mode" ON)
option(BUILD_EXAMPLES    "Build example executables" ON)
option(BUILD_TESTING     "Build the unit test suite" ON)

# Documentation is OFF by default and deliberately so: Doxygen with call graphs
# enabled takes longer than the library compiles, nothing in the normal build
# depends on it, and with the option OFF this file does not even call
# find_package(Doxygen), so configuration time is unchanged. The standalone
# route -- python3 build_docs.py -- needs no CMake at all.
option(MFTL_BUILD_DOCS        "Add the 'docs' target (Doxygen manual)"       OFF)
option(MFTL_DOCS_WARN_AS_ERROR "Fail the 'docs' target on a Doxygen warning" OFF)

# -march=native bakes in the ISA of the *build* machine. On a cluster the login
# node routinely differs from the compute nodes, which turns into SIGILL at run
# time, so it must be opted into explicitly. Set MFTL_ARCH_FLAGS instead to
# target a specific microarchitecture (e.g. -march=sapphirerapids).
option(MFTL_ARCH_NATIVE "Compile with -march=native (build machine ISA only!)" OFF)
set(MFTL_ARCH_FLAGS "" CACHE STRING "Extra architecture flags for Release builds")

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
# The same template (docs/doxygen/Doxyfile.in) drives this target and
# build_docs.py, and both substitute the same placeholders, so the two routes
# cannot drift apart.
#
# The target is NOT attached to `all`: `cmake --build build` never runs Doxygen,
# only `cmake --build build --target docs` does.
if(MFTL_BUILD_DOCS)
    find_package(Doxygen)

    if(NOT DOXYGEN_FOUND)
        message(WARNING
            "MFTL_BUILD_DOCS=ON but Doxygen was not found -- the 'docs' target "
            "will not exist. Install it with 'brew install doxygen graphviz' or "
            "'apt-get install doxygen graphviz'.")
    else()
        # Graphviz is optional. Without it the manual is complete except for the
        # call, caller, include and directory graphs.
        if(DOXYGEN_DOT_FOUND)
            set(MFTL_DOCS_HAVE_DOT "YES")
            get_filename_component(MFTL_DOCS_DOT_PATH "${DOXYGEN_DOT_EXECUTABLE}" DIRECTORY)
        else()
            set(MFTL_DOCS_HAVE_DOT "NO")
            set(MFTL_DOCS_DOT_PATH "")
            message(STATUS "Graphviz 'dot' not found -- documentation graphs disabled.")
        endif()

        set(MFTL_VERSION           "${PROJECT_VERSION}")
        set(MFTL_DOCS_SOURCE_DIR   "${CMAKE_CURRENT_SOURCE_DIR}")
        set(MFTL_DOCS_OUTPUT_DIR   "${CMAKE_BINARY_DIR}/docs")

        set(MFTL_DOXYFILE_IN  "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile.in")
        set(MFTL_DOXYFILE_OUT "${MFTL_DOCS_OUTPUT_DIR}/Doxyfile")

        # @ONLY: the template contains shell and Doxygen syntax that uses ${...},
        # which must survive untouched. Only @PLACEHOLDER@ is substituted.
        configure_file("${MFTL_DOXYFILE_IN}" "${MFTL_DOXYFILE_OUT}" @ONLY)

        set(MFTL_DOCS_STAMP "${MFTL_DOCS_OUTPUT_DIR}/html/index.html")

        # The manual is regenerated when a source, a page or the configuration
        # changes -- not on every build of the target.
        file(GLOB_RECURSE MFTL_DOCS_DEPENDS CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
             "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/src/*.md"
             "${CMAKE_CURRENT_SOURCE_DIR}/docs/*.md"
             "${CMAKE_CURRENT_SOURCE_DIR}/docs/pages/*.md"
             "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/*")

        add_custom_command(
            OUTPUT  "${MFTL_DOCS_STAMP}"
            COMMAND ${DOXYGEN_EXECUTABLE} "${MFTL_DOXYFILE_OUT}"
            DEPENDS "${MFTL_DOXYFILE_OUT}" ${MFTL_DOCS_DEPENDS}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Generating the MFTL manual with Doxygen"
            VERBATIM)

        add_custom_target(docs DEPENDS "${MFTL_DOCS_STAMP}")

        # Doxygen exits 0 even when it emits warnings, so "warnings are the
        # acceptance criterion" has to be enforced on the log rather than on the
        # exit code.
        if(MFTL_DOCS_WARN_AS_ERROR)
            add_custom_command(TARGET docs POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                        -DLOG=${MFTL_DOCS_OUTPUT_DIR}/doxygen-warnings.log
                        -P ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/check_warnings.cmake
                VERBATIM)
        endif()

        add_custom_command(TARGET docs POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo
                    "MFTL manual: ${MFTL_DOCS_STAMP}"
            VERBATIM)
    endif()
endif()

# --- Status Output ---
message(STATUS "-------------------------------------------------")
message(STATUS "MFTL Build Configuration")
message(STATUS "-------------------------------------------------")
message(STATUS "Compiler ID      : ${CMAKE_C_COMPILER_ID}")
message(STATUS "Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers       : ${ENABLE_SANITIZERS}")
message(STATUS "LTO (Release)    : ${IPO_SUPPORTED}")
message(STATUS "-march=native    : ${MFTL_ARCH_NATIVE}")
message(STATUS "Extra arch flags : ${MFTL_ARCH_FLAGS}")
message(STATUS "Build Examples   : ${BUILD_EXAMPLES}")
message(STATUS "Build Tests      : ${BUILD_TESTING}")
message(STATUS "Build Docs       : ${MFTL_BUILD_DOCS}   (target: docs)")
message(STATUS "Output Directory : ${CMAKE_BINARY_DIR}/lib")
message(STATUS "-------------------------------------------------")
