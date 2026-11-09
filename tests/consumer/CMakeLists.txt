# A downstream project that uses MFTL the way a flow solver would.
#
# Two ways in, both checked in CI (.github/workflows/consumers.yml):
#
#   installed package:  cmake --install <mftl-build> --prefix <prefix>
#                       cmake -S tests/consumer -B <dir> -DCMAKE_PREFIX_PATH=<prefix>
#   source tree:        cmake -S tests/consumer -B <dir> -DMFTL_SOURCE_DIR=<checkout>
#
# and two languages: the same program as C11 and as C++17, because most
# solvers that would link MFTL are C++ and include its headers from C++.
cmake_minimum_required(VERSION 3.13)
project(MFTLConsumer LANGUAGES C CXX)

set(MFTL_SOURCE_DIR "" CACHE PATH "Use this MFTL checkout via add_subdirectory() instead of find_package()")

if(MFTL_SOURCE_DIR)
    add_subdirectory(${MFTL_SOURCE_DIR} mftl EXCLUDE_FROM_ALL)
else()
    find_package(MFTL 2.0 REQUIRED)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(consumer_c main.c)
target_link_libraries(consumer_c PRIVATE MFTL::MFTL)

# The C++ translation unit is the same source: one program, two front ends.
add_executable(consumer_cxx main.cpp)
target_link_libraries(consumer_cxx PRIVATE MFTL::MFTL)

enable_testing()
find_package(MPI REQUIRED COMPONENTS C)
foreach(_exe consumer_c consumer_cxx)
    add_test(NAME ${_exe}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:${_exe}> ${MPIEXEC_POSTFLAGS})
endforeach()
