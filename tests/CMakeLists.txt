
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
