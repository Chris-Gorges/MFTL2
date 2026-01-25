
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
# Note: On some configs mpirun isn't in path or needs oversubscription handling.
# CTest keeps 'mpirun' simple. If fails, user checks env.
