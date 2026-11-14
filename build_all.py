#!/usr/bin/env python3
"""
MFTL Build Script
=================

Configures and builds the MFTL library, its examples and its tests with CMake, in
a fresh ``build/`` directory next to this script, so flags from an earlier build
never mix with the new ones. It can be run from any directory, and uses whatever
build tool CMake's default generator selects (``cmake --build``).

Usage:
    python3 build_all.py [options]

Options:
    (No arguments) : Builds in Release mode (Optimized, No Sanitizers).
                     Use this for maximum performance.

    --debug        : Builds in Debug mode WITH Sanitizers (ASan + UBSan).
                     Use this to catch memory leaks, overflows, and undefined behavior.
                     Note: Not compatible with Valgrind, and not installable
                     (cmake --install refuses a sanitized build).

    --valgrind     : Builds in Debug mode WITHOUT Sanitizers.
                     Use this if you intend to run the executable with Valgrind,
                     or to install a Debug build of the library.

    --lib-only     : Builds only the static library (libMFTL.a).
                     Skips the example executables in examples/ AND the test suite.

    --comm-profile : Times the individual phases of MFTLRedistribute.

Examples:
    python3 build_all.py             # Build everything for Speed
    python3 build_all.py --debug     # Build everything for Bug Hunting
    python3 build_all.py --lib-only  # Build only the library
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build"


def run(command):
    """Runs one command (an argument list, no shell) and exits with its status on failure."""
    print("Running:", " ".join(str(c) for c in command), flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: command failed with exit status {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def build_project(build_type="Release", use_sanitizers=True, lib_only=False, comm_profile=False):
    """Configures and builds the project in a fresh build/ directory."""
    if BUILD_DIR.exists():
        print(f"Cleaning old build directory: {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    def on_off(flag):
        return "ON" if flag else "OFF"

    print(f"Configuring [{build_type}] | Sanitizers: {use_sanitizers} | Examples and tests: {not lib_only} "
          f"| Comm profiling: {comm_profile}")
    run([
        "cmake", "-S", str(PROJECT_ROOT), "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DENABLE_SANITIZERS={on_off(use_sanitizers)}",
        f"-DBUILD_EXAMPLES={on_off(not lib_only)}",
        f"-DBUILD_TESTING={on_off(not lib_only)}",
        f"-DMFTL_COMM_PROFILE={on_off(comm_profile)}",
    ])

    print("Building project...")
    run(["cmake", "--build", str(BUILD_DIR), "--parallel", str(os.cpu_count() or 1)])

    print("\nBuild successful!")
    print(f"Artifacts are located in: {BUILD_DIR}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build MFTL project.", epilog="Example: python3 build_all.py --debug")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--release", action="store_true", help="Build in Release mode (Default)")
    mode_group.add_argument("--debug", action="store_true", help="Build in Debug mode with Sanitizers (ASan/UBSan)")
    mode_group.add_argument("--valgrind", action="store_true",
                            help="Build in Debug mode WITHOUT Sanitizers (for Valgrind, or an installable Debug build)")

    parser.add_argument("--lib-only", action="store_true", help="Build ONLY the library, skip examples and tests")
    parser.add_argument("--comm-profile", action="store_true",
                        help="Time the individual phases of MFTLRedistribute (MFTL_RedistributeStats t_* fields). "
                             "Off by default: the per-body timers cost 1.5-2 %% of the phase.")

    args = parser.parse_args(argv)

    if args.debug:
        build_type, sanitizers = "Debug", True
    elif args.valgrind:
        build_type, sanitizers = "Debug", False
    else:
        build_type, sanitizers = "Release", False

    build_project(build_type=build_type, use_sanitizers=sanitizers, lib_only=args.lib_only, comm_profile=args.comm_profile)


if __name__ == "__main__":
    main()
