
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
