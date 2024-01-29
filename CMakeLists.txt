cmake_minimum_required(VERSION 2.6)

project(MFTL2)

#
# Set compiler
#
set(MPIHOME $ENV{MPI_HOME})
set(MFTL_SOURCE_DIR "/home/berend/git/Code/MFTL2")

IF(NOT MPIHOME)
  message(FATAL_ERROR "Please set MPI_HOME variable to the place where MPI resides before running CMake")
ENDIF(NOT MPIHOME)

set(CMAKE_C_COMPILER ${MPIHOME}/bin/mpicc)
set(CMAKE_CXX_COMPILER ${MPIHOME}/bin/mpiCC)

set(CMAKE_C_FLAGS "-O0 -g -fPIC")

# Includes
include_directories(${MFTL_SOURCE_DIR}/include)

# C files
FILE(GLOB_RECURSE MYFILES ABSOLUTE ${MFTL_SOURCE_DIR}/src/*.c)

add_library(MFTL STATIC ${MYFILES})

set(mylibs lapack blas gsl gslcblas)
set(mylibdirs $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
set(mylibdirs /usr/lib64/)
link_directories(${mylibdirs})

# add_executable(MFTL ${MYFILES})

# target_link_libraries(MFTL ${mylibs})

# find_package(GSL REQUIRED)
# target_link_libraries(MFTL)
