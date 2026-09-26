
# Link the main library
# We invoke the executable creation

# Serial Tests
add_executable(unit_tests_serial test_serial.c)
target_link_libraries(unit_tests_serial PRIVATE MFTL m)
target_include_directories(unit_tests_serial PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME SerialMathTopology COMMAND unit_tests_serial)

# MPI Tests
add_executable(unit_tests_mpi test_mpi.c)
target_link_libraries(unit_tests_mpi PRIVATE MFTL MPI::MPI_C m)
target_include_directories(unit_tests_mpi PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Register MPI Test with 2 Ranks
add_test(NAME MPIGhostSync COMMAND mpirun -np 2 $<TARGET_FILE:unit_tests_mpi>)

# --- Phase 1: Internals Test ---
add_executable(test_internals test_internals.c)
target_link_libraries(test_internals PRIVATE MFTL m)
target_include_directories(test_internals PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME UnitInternals COMMAND test_internals)

# --- Phase 2: Sphere Generation Test (Serial) ---
add_executable(test_sphere_gen test_sphere_gen.c)
target_link_libraries(test_sphere_gen PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_sphere_gen PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME UnitSphereGen COMMAND test_sphere_gen)

# --- Phase 3: MPI Stress Test ---
add_executable(test_mpi_stress test_mpi_stress.c)
target_link_libraries(test_mpi_stress PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_mpi_stress PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Run with 8 ranks for stress testing (or 4 if 8 is too heavy for local CI)
add_test(NAME MPIStressTest COMMAND mpirun -np 8 $<TARGET_FILE:test_mpi_stress>)

# --- Phase 4: Ghost Layer Integration Test ---
add_executable(test_ghost_layers test_ghost_layers.c)
target_link_libraries(test_ghost_layers PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_ghost_layers PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Requires exactly 4 ranks
add_test(NAME MPIGhostLayers COMMAND mpirun -np 4 $<TARGET_FILE:test_ghost_layers>)

# --- Phase 5: Remeshing Unit Tests ---
add_executable(test_remeshing test_remeshing.c)
target_link_libraries(test_remeshing PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_remeshing PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME UnitRemeshing COMMAND mpirun -np 1 $<TARGET_FILE:test_remeshing>)

# --- Phase 6: Intersection Test ---
add_executable(unit_tests_intersection test_intersection.c)
target_link_libraries(unit_tests_intersection PRIVATE MFTL m)
target_include_directories(unit_tests_intersection PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME UnitIntersection COMMAND unit_tests_intersection)

# --- LeakSanitizer suppressions for MPI-linked tests ---
# MPICH internals leak small bookkeeping allocations (see tests/lsan.supp);
# MFTL's own allocations must remain leak-clean.
set(MFTL_LSAN_ENV "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp:print_suppressions=0")
set_tests_properties(MPIGhostSync MPIStressTest MPIGhostLayers UnitRemeshing UnitSphereGen PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Phase 7: Object Integrals (Stage 02) ---
add_executable(test_object_integrals test_object_integrals.c)
target_link_libraries(test_object_integrals PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_object_integrals PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Rank-independence: the same binary is registered at 1/2/4/8 ranks
# (each run also asserts pre/post-redistribution equality internally).
foreach(NP 1 2 4 8)
    add_test(NAME ObjectIntegrals-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_object_integrals>)
endforeach()
set_tests_properties(ObjectIntegrals-np1 ObjectIntegrals-np2 ObjectIntegrals-np4 ObjectIntegrals-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 03: Rigid-body transforms ---
add_executable(test_transforms test_transforms.c)
target_link_libraries(test_transforms PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_transforms PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Transforms-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_transforms>)
endforeach()
set_tests_properties(Transforms-np1 Transforms-np2 Transforms-np4 Transforms-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 04: Object generators ---
add_executable(test_generators test_generators.c)
target_link_libraries(test_generators PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_generators PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Generators-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_generators>)
endforeach()
set_tests_properties(Generators-np1 Generators-np2 Generators-np4 Generators-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 05: Mesh I/O (GTS reader/writer + PVTU) ---
# Fixture location of the MFTL1 tree (GTS test meshes); override with
# -DMFTL1_ROOT=/path/to/mftl if checked out elsewhere.
set(MFTL1_ROOT "/home/berend/git/Code/mftl" CACHE PATH "Root of the MFTL1 source tree (GTS fixtures)")
add_executable(test_io test_io.c)
target_link_libraries(test_io PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_io PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(test_io PRIVATE MFTL_GTS_DIR="${MFTL1_ROOT}")
foreach(NP 1 2 4 8)
    add_test(NAME MeshIO-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_io>)
endforeach()
set_tests_properties(MeshIO-np1 MeshIO-np2 MeshIO-np4 MeshIO-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 06: Normals, curvatures, roughness ---
add_executable(test_curvature test_curvature.c)
target_link_libraries(test_curvature PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_curvature PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Curvature-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_curvature>)
endforeach()
set_tests_properties(Curvature-np1 Curvature-np2 Curvature-np4 Curvature-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 07: Point queries (point-in-object, point-triangle distance, AABB) ---
add_executable(test_point_query test_point_query.c)
target_link_libraries(test_point_query PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_point_query PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME PointQuery-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_point_query>)
endforeach()
set_tests_properties(PointQuery-np1 PointQuery-np2 PointQuery-np4 PointQuery-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 08: Serial remeshing A (np1 serial operators) ---
add_executable(test_remesh_quality test_remesh_quality.c)
target_link_libraries(test_remesh_quality PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_remesh_quality PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME RemeshQuality-np1 COMMAND mpirun -np 1 $<TARGET_FILE:test_remesh_quality>)
set_tests_properties(RemeshQuality-np1 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 09: Volume-conserving smoothing ---
add_executable(test_volume_smoothing test_volume_smoothing.c)
target_link_libraries(test_volume_smoothing PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_volume_smoothing PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME VolSmooth-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_volume_smoothing>)
endforeach()
set_tests_properties(VolSmooth-np1 VolSmooth-np2 VolSmooth-np4 VolSmooth-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 10: Parallel remeshing + load balancing (x-only decomposition) ---
add_executable(test_remesh_parallel test_remesh_parallel.c)
target_link_libraries(test_remesh_parallel PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_remesh_parallel PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME RemeshParallel-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_remesh_parallel>)
endforeach()
set_tests_properties(RemeshParallel-np1 RemeshParallel-np2 RemeshParallel-np4 RemeshParallel-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 11: Intersection + overlap volume ---
add_executable(test_overlap test_overlap.c)
target_link_libraries(test_overlap PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_overlap PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Overlap-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_overlap>)
endforeach()
set_tests_properties(Overlap-np1 Overlap-np2 Overlap-np4 Overlap-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 12: Opaque per-triangle data transport ---
add_executable(test_triangle_data test_triangle_data.c)
target_link_libraries(test_triangle_data PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_triangle_data PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME TriangleData-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_triangle_data>)
endforeach()
set_tests_properties(TriangleData-np1 TriangleData-np2 TriangleData-np4 TriangleData-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# --- Stage 13: Periodicity completion (end-to-end battery) ---
add_executable(test_periodic_endtoend test_periodic_endtoend.c)
target_link_libraries(test_periodic_endtoend PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_periodic_endtoend PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME PeriodicEndToEnd-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_periodic_endtoend>)
endforeach()
set_tests_properties(PeriodicEndToEnd-np1 PeriodicEndToEnd-np2 PeriodicEndToEnd-np4 PeriodicEndToEnd-np8 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")

# Sanitizer / debug options for every test executable of this directory (see
# mftl2_dev_target_options in the top-level CMakeLists.txt).
get_property(_mftl2_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t ${_mftl2_test_targets})
    get_target_property(_type ${_t} TYPE)
    if(_type STREQUAL "EXECUTABLE")
        mftl2_dev_target_options(${_t})
    endif()
endforeach()

# --- Stage 15: library items L1-L7 (rectilinear domains, twin gids, surface scalar, point
#     query objects, single-vertex curvature, soup builder, native mesh I/O, transfers) ---
add_executable(test_stage15 test_stage15.c)
target_link_libraries(test_stage15 PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage15 PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 3 4)
    add_test(NAME Stage15-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage15>)
    set_tests_properties(Stage15-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()

# --- Stage 16A: twin states (LOCAL/REMOTE/BOUNDARY), builders and declared class, copy/commit,
#     remote-twin discovery, ghost completion G1/G2, boundary loops, boundary-adjacent operators,
#     invariant fuzz (plans/16-generalized-surfaces.md §2, §10 T-A) ---
add_executable(test_stage16a test_stage16a.c)
target_link_libraries(test_stage16a PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage16a PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Stage16A-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16a>)
    set_tests_properties(Stage16A-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()

# --- Stage 16A: no raw twin_gid comparisons outside the state predicates (plans/16 §2.1) ---
add_test(NAME TwinGidReaderAudit COMMAND ${CMAKE_SOURCE_DIR}/tools/check_twin_gid_readers.sh ${CMAKE_SOURCE_DIR})

# --- Stage 16B: periodic image counters, winding/canonicalisation, phase-boundary sets
#     (plans/16 §3, tests T-B). The phase-set cases (every case of §3.6.10 and of the review
#     §19.3, with the executable model's statuses/values) are REGENERATED from
#     tools/phase_set_prototype (python3 + numpy) by a ctest fixture on every test run, so the
#     tests never run on stale data. The library build never needs Python: the generator is a
#     test-time dependency only, and without python3+numpy the model cases are skipped (the other
#     16B tests still run) unless MFTL_PHASE_SET_MODEL_TESTS=ON demands them. ---
set(MFTL_PHASE_SET_MODEL_TESTS "AUTO" CACHE STRING "Run the phase-set model cases (needs python3 + numpy at test time): AUTO, ON or OFF")
set_property(CACHE MFTL_PHASE_SET_MODEL_TESTS PROPERTY STRINGS AUTO ON OFF)
set(_mftl_ps_model OFF)
if(NOT MFTL_PHASE_SET_MODEL_TESTS STREQUAL "OFF")
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                        RESULT_VARIABLE _mftl_numpy_rc OUTPUT_QUIET ERROR_QUIET)
        if(_mftl_numpy_rc EQUAL 0)
            set(_mftl_ps_model ON)
        endif()
    endif()
    if(NOT _mftl_ps_model)
        if(MFTL_PHASE_SET_MODEL_TESTS STREQUAL "ON")
            message(FATAL_ERROR "MFTL_PHASE_SET_MODEL_TESTS=ON needs python3 with numpy")
        endif()
        message(STATUS "MFTL2: python3+numpy not found - the Stage-16B phase-set model cases are skipped")
    endif()
endif()
set(MFTL_PS_CASES ${CMAKE_BINARY_DIR}/phase_set_cases.txt)
set(MFTL_PS_PROTO ${CMAKE_SOURCE_DIR}/tools/phase_set_prototype)
set(_mftl_ps_arg "")
if(_mftl_ps_model)
    add_test(NAME Stage16B-export-model-cases
             COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE} export_cases.py ${MFTL_PS_CASES}
             WORKING_DIRECTORY ${MFTL_PS_PROTO})
    set_tests_properties(Stage16B-export-model-cases PROPERTIES FIXTURES_SETUP s16b_model_cases TIMEOUT 1200)
    set(_mftl_ps_arg ${MFTL_PS_CASES})
endif()
add_executable(test_stage16b test_stage16b.c)
target_link_libraries(test_stage16b PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage16b PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Stage16B-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16b> ${_mftl_ps_arg})
    set_tests_properties(Stage16B-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
    if(_mftl_ps_model)
        set_tests_properties(Stage16B-np${NP} PROPERTIES FIXTURES_REQUIRED s16b_model_cases)
    endif()
endforeach()
foreach(SL 1 2)
    add_test(NAME Stage16B-slab${SL}-np4 COMMAND mpirun -np 4 $<TARGET_FILE:test_stage16b> ${_mftl_ps_arg})
    set_tests_properties(Stage16B-slab${SL}-np4 PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV};S16B_SLABS=${SL};S16B_AXES=z")
    if(_mftl_ps_model)
        set_tests_properties(Stage16B-slab${SL}-np4 PROPERTIES FIXTURES_REQUIRED s16b_model_cases)
    endif()
endforeach()
# --- Stage 16B independent review (plans/16 §21): gauge invariance of transforms, winding
#     validity through every construction path, legacy v1 restart, analytic phase sets ---
add_executable(test_stage16b_gauge test_stage16b_gauge.c)
target_link_libraries(test_stage16b_gauge PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage16b_gauge PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Stage16B-gauge-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16b_gauge>)
    set_tests_properties(Stage16B-gauge-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()
# --- Stage 16C: generalized geometry kernels and boundary-aware remeshing (plans/16 §4, T-C) ---
add_executable(test_stage16c test_stage16c.c)
target_link_libraries(test_stage16c PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage16c PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Stage16C-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16c>)
    set_tests_properties(Stage16C-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}" TIMEOUT 1800)
endforeach()
# --- Stage 16E: geometry of the open-surface boundary models (plans/16 §6, §7; E2/E3 library part) ---
add_executable(test_stage16e test_stage16e.c)
target_link_libraries(test_stage16e PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage16e PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Stage16E-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16e>)
    set_tests_properties(Stage16E-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()

# --- Stage 16F: final validation of the wrapping topology (winding invariance under
#     redistribution, lattice re-gauge, remeshing, rank count and restart; the abort contract of
#     the unchecked single-body volume / inertia calls) ---
add_executable(test_stage16f test_stage16f.c)
target_link_libraries(test_stage16f PRIVATE MFTL MPI::MPI_C m)
target_include_directories(test_stage16f PRIVATE ${CMAKE_SOURCE_DIR}/include)
foreach(NP 1 2 4 8)
    add_test(NAME Stage16F-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16f>)
    set_tests_properties(Stage16F-np${NP} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()
add_test(NAME Stage16F-restart-write-np4 COMMAND mpirun -np 4 $<TARGET_FILE:test_stage16f> write ${CMAKE_BINARY_DIR}/stage16f_restart)
set_tests_properties(Stage16F-restart-write-np4 PROPERTIES FIXTURES_SETUP s16f_restart ENVIRONMENT "${MFTL_LSAN_ENV}")
foreach(NP 1 2 8)
    add_test(NAME Stage16F-restart-read-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16f> read ${CMAKE_BINARY_DIR}/stage16f_restart)
    set_tests_properties(Stage16F-restart-read-np${NP} PROPERTIES FIXTURES_REQUIRED s16f_restart ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()
# the unchecked calls abort with their documented message (the exit status of an MPI_Abort is not
# portable, so the message decides). The test runs as an MPI singleton (no launcher) with its output
# in a file: under load MPICH's launcher ends an aborting job before forwarding its output (observed
# 2 of 160 runs, even on one rank); without the launcher the message is always there (240 of 240).
foreach(CASE volume_tube:wraps inertia_ribbon:wraps volume_patch:open inertia_patch:open)
    string(REPLACE ":" ";" _c ${CASE})
    list(GET _c 0 WHAT)
    list(GET _c 1 KIND)
    if(KIND STREQUAL "wraps")
        set(_msg "the body wraps a periodic direction")
    else()
        set(_msg "the body is an open surface")
    endif()
    set(_out ${CMAKE_CURRENT_BINARY_DIR}/stage16f_unchecked_${WHAT}.out)
    add_test(NAME Stage16F-unchecked-${WHAT}
             COMMAND sh -c "'$<TARGET_FILE:test_stage16f>' unchecked ${WHAT} > '${_out}' 2>&1; grep -q '${_msg}' '${_out}' && ! grep -q 'NOT ABORTED' '${_out}'")
    set_tests_properties(Stage16F-unchecked-${WHAT} PROPERTIES ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()
add_test(NAME Stage16B-restart-write-np4 COMMAND mpirun -np 4 $<TARGET_FILE:test_stage16b> write ${CMAKE_BINARY_DIR}/stage16b_restart)
set_tests_properties(Stage16B-restart-write-np4 PROPERTIES FIXTURES_SETUP s16b_restart ENVIRONMENT "${MFTL_LSAN_ENV}")
foreach(NP 1 2 8)
    add_test(NAME Stage16B-restart-read-np${NP} COMMAND mpirun -np ${NP} $<TARGET_FILE:test_stage16b> read ${CMAKE_BINARY_DIR}/stage16b_restart)
    set_tests_properties(Stage16B-restart-read-np${NP} PROPERTIES FIXTURES_REQUIRED s16b_restart ENVIRONMENT "${MFTL_LSAN_ENV}")
endforeach()
