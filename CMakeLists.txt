cmake_minimum_required(VERSION 3.13)

project(MFTL C)

# --- Options ---
option(ENABLE_SANITIZERS "Enable Address and Undefined sanitizers in Debug mode" ON)
option(BUILD_EXAMPLES "Build example executables" ON)

# --- Dependencies ---
find_package(MPI REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Compiler Detection & Configuration ---

# 1. Enable Link Time Optimization (LTO/IPO) safely
# This works for GCC, Clang, and AppleClang automatically.
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED)
    # Enable LTO only for Release builds to speed up execution
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(WARNING "Link Time Optimization not supported: ${IPO_ERROR}")
endif()

# 2. Compiler Specific Flags
# We check if the compiler is GCC, Clang, or AppleClang before applying specific flags.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    
    # --- Release Optimizations ---
    # We add options individually to avoid CMake parsing errors.
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:Release>:-march=native>)
    # Note: -flto is handled by CMAKE_INTERPROCEDURAL_OPTIMIZATION above

    # --- Debug & Sanitizers ---
    if(ENABLE_SANITIZERS)
        # Apply Sanitizers (ASan + UBSan)
        # We separate them to ensure safety across different compiler versions
        add_compile_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
        add_compile_options($<$<CONFIG:Debug>:-fno-omit-frame-pointer>)
        add_compile_options($<$<CONFIG:Debug>:-g>)
        
        # Linker flags for sanitizers
        add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    else()
        # Debugging symbols only (for Valgrind)
        add_compile_options($<$<CONFIG:Debug>:-g>)
    endif()

elseif(MSVC)
    # (Optional) Placeholder if you ever compile on Windows with Visual Studio
    add_compile_options($<$<CONFIG:Release>:/O2>)
endif()

# --- Directories ---
set(MFTL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(MFTL_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

include_directories(${MFTL_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/external)

# --- Library Target ---
file(GLOB_RECURSE MFTL_SOURCES "src/*.c")

add_library(MFTL STATIC ${MFTL_SOURCES})
target_link_libraries(MFTL PUBLIC MPI::MPI_C)

set_target_properties(MFTL PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY ${MFTL_LIB_DIR}
)

# --- Executable Target (Conditional) ---
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "examples/*.c")
    foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
        get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        
        # Link math library (m) and MFTL
        target_link_libraries(${TARGET_NAME} PRIVATE MFTL m)
        target_include_directories(${TARGET_NAME} PRIVATE ${MFTL_INCLUDE_DIR})
    endforeach()
endif()

# --- Status Output ---
message(STATUS "-------------------------------------------------")
message(STATUS "MFTL Build Configuration")
message(STATUS "-------------------------------------------------")
message(STATUS "Compiler ID      : ${CMAKE_C_COMPILER_ID}")
message(STATUS "Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers       : ${ENABLE_SANITIZERS}")
message(STATUS "LTO (Release)    : ${IPO_SUPPORTED}")
message(STATUS "Build Examples   : ${BUILD_EXAMPLES}")
message(STATUS "Output Directory : ${MFTL_LIB_DIR}")
message(STATUS "-------------------------------------------------")

# --- Testing ---
enable_testing()
add_subdirectory(tests)
