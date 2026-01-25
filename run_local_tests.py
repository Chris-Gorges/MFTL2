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

    # 1. Build using existing infrastructure (re-runs cmake if needed)
    print("--- Building Project ---")
    run_command("python3 build_all.py")

    # 2. Run Tests
    print("\n--- Running Tests (CTest) ---")
    # --output-on-failure gives details if stdout checks fail
    run_command("ctest --output-on-failure", cwd=build_dir)

if __name__ == "__main__":
    main()
