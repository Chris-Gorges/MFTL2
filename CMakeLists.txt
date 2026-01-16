# Set the minimum required version of CMake.
cmake_minimum_required(VERSION 3.10)

# Define the project name and specify that it's a C project.
project(MFTL C)

# --- Compiler and Build Settings ---

# This automatically finds mpicc and sets the necessary flags.
find_package(MPI REQUIRED)

# Set the C standard to C99.
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add compiler and linker flags for the Address Sanitizer, but *only*
# for the Debug build type, using a generator expression.
add_compile_options($<$<CONFIG:Debug>:-fsanitize=address>)
add_link_options($<$<CONFIG:Debug>:-fsanitize=address>)

# Set a variable for our library's include directory.
set(MFTL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
# Set a variable for our library's output directory.
set(MFTL_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

# Add the include directory to the search path for header files.
include_directories(${MFTL_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/external)

# --- Library Target ---

# Find all the .c files in the src/ directory and its subdirectories.
file(GLOB_RECURSE MFTL_SOURCES "src/*.c")

# Create a static library named "MFTL" from the source files.
# This will produce libMFTL.a
add_library(MFTL STATIC ${MFTL_SOURCES})

# Link the MFTL library itself against MPI. This provides the necessary
# include directories (for mpi.h) and compiler flags.
target_link_libraries(MFTL PUBLIC MPI::MPI_C)

# Set the output directory for the compiled library file (.a)
set_target_properties(MFTL PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${MFTL_LIB_DIR}
)

# --- Executable Target ---

# --- Executable Target ---

# Find all .c files in the examples/ directory
file(GLOB EXAMPLE_SOURCES "examples/*.c")

# Iterate over each example source file found
foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    # Extract the filename without extension to use as the target name
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)

    # Create an executable for each example
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    # Link the executable against our MFTL library and math library
    target_link_libraries(${TARGET_NAME} PRIVATE MFTL m)

    # Ensure the executable knows where to find the MFTL headers
    target_include_directories(${TARGET_NAME} PRIVATE ${MFTL_INCLUDE_DIR})
endforeach()



# A message to show where the library will be built.
message(STATUS "MFTL library will be built in: ${MFTL_LIB_DIR}")
