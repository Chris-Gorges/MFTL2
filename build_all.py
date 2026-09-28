#!/usr/bin/env python3
"""
MFTL Build Script
=================

This script configures and builds the MFTL library and examples using CMake.
It handles platform differences (macOS/Linux) and build modes automatically.

Usage:
    python3 build_all.py [options]

Options:
    (No arguments) : Builds in Release mode (Optimized, No Sanitizers).
                     Use this for maximum performance.

    --debug        : Builds in Debug mode WITH Sanitizers (ASan + UBSan).
                     Use this to catch memory leaks, overflows, and undefined behavior.
                     Note: Not compatible with Valgrind.

    --valgrind     : Builds in Debug mode WITHOUT Sanitizers.
                     Use this if you intend to run the executable with Valgrind.

    --lib-only     : Builds only the static library (libMFTL.a).
                     Skips building the example executables in examples/.

Examples:
    python3 build_all.py             # Build everything for Speed
    python3 build_all.py --debug     # Build everything for Bug Hunting
    python3 build_all.py --lib-only  # Build only the library
"""

import os
import subprocess
import sys
import shutil
import argparse

def build_project(build_type="Release", use_sanitizers=True, build_examples=True, comm_profile=False):
    """
    Configures and builds the project with specific flags.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_root, "build")

    # Clean previous build to ensure flags don't get mixed
    if os.path.exists(build_dir):
        print(f"Cleaning old build directory: {build_dir}")
        shutil.rmtree(build_dir)
    
    os.makedirs(build_dir)
    print(f"Created build directory: {build_dir}")

    # Prepare CMake arguments
    cmake_args = [
        "cmake",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DENABLE_SANITIZERS={'ON' if use_sanitizers else 'OFF'}",
        f"-DBUILD_EXAMPLES={'ON' if build_examples else 'OFF'}",
        f"-DMFTL_COMM_PROFILE={'ON' if comm_profile else 'OFF'}",
        ".."
    ]

    # Run CMake
    print(f"Configuring [{build_type}] | Sanitizers: {use_sanitizers} | Examples: {build_examples} "
          f"| Comm profiling: {comm_profile}...")
    try:
        subprocess.check_call(cmake_args, cwd=build_dir)
    except subprocess.CalledProcessError:
        print("Error: CMake configuration failed.")
        sys.exit(1)

    # Run Make
    print("Building project...")
    try:
        # Use -j to build in parallel using all available CPU cores
        n_cores = os.cpu_count() or 1
        subprocess.check_call(["make", f"-j{n_cores}"], cwd=build_dir)
    except subprocess.CalledProcessError:
        print("Error: Build failed.")
        sys.exit(1)

    print("\nBuild successful!")
    print(f"Artifacts are located in: {build_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build MFTL project.",
        epilog="Example: python3 build_all.py --debug"
    )
    
    # Build Mode Group (Mutually Exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--release", action="store_true", default=True, help="Build in Release mode (Default)")
    mode_group.add_argument("--debug", action="store_true", help="Build in Debug mode with Sanitizers (ASan/UBSan)")
    mode_group.add_argument("--valgrind", action="store_true", help="Build in Debug mode WITHOUT Sanitizers (for Valgrind usage)")

    # Target Options
    parser.add_argument("--lib-only", action="store_true", help="Build ONLY the library, skip examples")
    parser.add_argument("--comm-profile", action="store_true",
                        help="Time the individual phases of MFTLRedistribute (MFTL_RedistributeStats t_* fields). "
                             "Off by default: the per-body timers cost 1.5-2 %% of the phase.")

    args = parser.parse_args()

    # Determine Build Settings
    build_examples = not args.lib_only
    
    if args.debug:
        build_project(build_type="Debug", use_sanitizers=True, build_examples=build_examples, comm_profile=args.comm_profile)
    elif args.valgrind:
        build_project(build_type="Debug", use_sanitizers=False, build_examples=build_examples, comm_profile=args.comm_profile)
    else:
        # Default to Release
        build_project(build_type="Release", use_sanitizers=False, build_examples=build_examples, comm_profile=args.comm_profile)
