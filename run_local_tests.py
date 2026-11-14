#!/usr/bin/env python3
"""
Builds MFTL in a fresh build/ directory and runs the whole CTest suite on every core.

Every argument is handed to build_all.py, so ``python3 run_local_tests.py --debug``
tests the sanitizer build. Works from any directory: paths are taken from this
script's location, and no command goes through a shell.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build"


def run(command, cwd=None):
    """Runs one command (an argument list, no shell) and exits with its status on failure."""
    print("Running:", " ".join(str(c) for c in command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: command failed with exit status {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    print("--- Building Project ---")
    run([sys.executable, str(PROJECT_ROOT / "build_all.py"), *sys.argv[1:]])

    # --output-on-failure gives details if stdout checks fail. -j: every MPI test
    # declares its rank count (PROCESSORS), so CTest schedules ranks rather than
    # tests and keeps the machine full without oversubscribing it.
    print("\n--- Running Tests (CTest) ---")
    run(["ctest", "--output-on-failure", "-j", str(os.cpu_count() or 1)], cwd=BUILD_DIR)


if __name__ == "__main__":
    main()
