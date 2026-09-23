# Standalone project that consumes MFTL2 exactly as an external downstream would:
# it knows nothing about MFTL2's src/ layout — it only does find_package(MFTL2)
# and links the imported target. Point it at a configured/built (or installed)
# MFTL2 via CMAKE_PREFIX_PATH, e.g.
#
#   cmake -S external_link_test -B build-linktest \
#         -DCMAKE_PREFIX_PATH=$PWD/build-release
#   cmake --build build-linktest
#   ./build-linktest/mftl2_link_test
#
cmake_minimum_required(VERSION 3.13)
project(mftl2_link_test C)

find_package(MFTL2 REQUIRED)

add_executable(mftl2_link_test main.c)
# The ONLY thing the consumer names is the imported target; public headers
# (include/ + khash.h) and MPI arrive transitively through its usage interface.
target_link_libraries(mftl2_link_test PRIVATE MFTL2::MFTL2)

enable_testing()
add_test(NAME mftl2_link_test COMMAND mftl2_link_test)
