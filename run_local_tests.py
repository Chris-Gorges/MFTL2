import os
import shutil
import subprocess
import sys

def find_ctest():
    """Locate a working ctest binary (a broken Python shim may precede the real one in PATH)."""
    candidates = []
    path_ctest = shutil.which("ctest")
    if path_ctest:
        candidates.append(path_ctest)
    candidates.append("/usr/bin/ctest")
    candidates.append("/usr/local/bin/ctest")

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not os.path.exists(candidate):
            continue
        seen.add(candidate)
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True)
            if result.returncode == 0:
                return candidate
        except OSError:
            continue
    return None

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
    ctest = find_ctest()
    if ctest is None:
        print("Error: No working ctest binary found.")
        sys.exit(1)
    # --output-on-failure gives details if stdout checks fail
    run_command(f'"{ctest}" --output-on-failure', cwd=build_dir)

if __name__ == "__main__":
    main()
