#!/usr/bin/env python3
import os
import subprocess
import shutil
import sys

def build_project():
    """
    Configures and builds the MFTL library and examples using CMake.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_root, "build")

    # Create build directory if it doesn't exist
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print(f"Created build directory: {build_dir}")

    # Run CMake to configure
    print("Configuring with CMake...")
    try:
        subprocess.check_call(["cmake", ".."], cwd=build_dir)
    except subprocess.CalledProcessError:
        print("Error: CMake configuration failed.")
        sys.exit(1)

    # Run Make to build
    print("Building with Make...")
    try:
        subprocess.check_call(["make"], cwd=build_dir)
    except subprocess.CalledProcessError:
        print("Error: Build failed.")
        sys.exit(1)

    print("\nBuild successful!")
    print(f"Executables are located in: {build_dir}")

if __name__ == "__main__":
    build_project()
