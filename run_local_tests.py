import os
import subprocess
import sys

def run_command(command, cwd=None):
    """Runs a shell command and exits on failure."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    root_dir = os.getcwd()
    build_dir = os.path.join(root_dir, "build")

    # 1. Build using existing infrastructure; build options (--debug, --valgrind)
    #    are passed through, so `python3 run_local_tests.py --debug` tests the
    #    sanitizer build.
    print("--- Building Project ---")
    run_command(" ".join(["python3 build_all.py"] + sys.argv[1:]))

    # 2. Run Tests
    print("\n--- Running Tests (CTest) ---")
    # --output-on-failure gives details if stdout checks fail. -j: every MPI test
    # declares its rank count (PROCESSORS), so CTest schedules ranks rather than
    # tests and keeps the machine full without oversubscribing it.
    n_cores = os.cpu_count() or 1
    run_command(f"ctest --output-on-failure -j {n_cores}", cwd=build_dir)

if __name__ == "__main__":
    main()
