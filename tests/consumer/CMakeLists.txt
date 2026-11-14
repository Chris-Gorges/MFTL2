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
#
# -DMFTL_CONSUMER_CXX_ONLY=ON makes it a C++-ONLY project (the C language is
# never enabled), the way a C++ solver declares itself with
# `project(Solver LANGUAGES CXX)`; only consumer_cxx is built then. A project
# that enables C as well would hide a package that needs it.
cmake_minimum_required(VERSION 3.13)
project(MFTLConsumer LANGUAGES CXX)

option(MFTL_CONSUMER_CXX_ONLY "Do not enable the C language: check that a C++-only project can use MFTL" OFF)
if(NOT MFTL_CONSUMER_CXX_ONLY)
    enable_language(C)
endif()

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

set(_consumers consumer_cxx)

# The C++ translation unit is the same source: one program, two front ends.
add_executable(consumer_cxx main.cpp)
target_link_libraries(consumer_cxx PRIVATE MFTL::MFTL)

if(NOT MFTL_CONSUMER_CXX_ONLY)
    add_executable(consumer_c main.c)
    target_link_libraries(consumer_c PRIVATE MFTL::MFTL)
    list(APPEND _consumers consumer_c)
endif()

enable_testing()
# Every language this project enabled: C and CXX, or CXX alone.
find_package(MPI REQUIRED)
foreach(_exe ${_consumers})
    add_test(NAME ${_exe}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:${_exe}> 2 ${MPIEXEC_POSTFLAGS})
endforeach()
