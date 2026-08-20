#!/usr/bin/env python3
"""
MFTL Documentation Build Script
===============================

Generates the Doxygen manual WITHOUT going through CMake, so that documentation
can be regenerated without touching the build directory's CMake cache and
without any risk of slowing down the edit-compile-test cycle.

The CMake route (``-DMFTL_BUILD_DOCS=ON`` plus ``--target docs``) exists too and
produces identical output: both substitute the same placeholders into the single
template ``docs/doxygen/Doxyfile.in``.

Usage:
    python3 build_docs.py [options]

Options:
    --open           Open the generated index.html in the default browser.
    --strict         Fail (non-zero exit) if Doxygen emitted any warning.
    --no-graphs      Skip Graphviz. Much faster; loses the call/caller graphs.
    --latex          Additionally emit LaTeX sources for a PDF.
    --output-dir DIR Write the manual somewhere other than build/docs.
    --quiet          Suppress Doxygen's progress chatter.

Examples:
    python3 build_docs.py                 # build the HTML manual
    python3 build_docs.py --open          # ... and look at it
    python3 build_docs.py --strict        # treat documentation warnings as errors
    python3 build_docs.py --no-graphs     # quick pass while editing prose
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOXYFILE_IN = os.path.join(PROJECT_ROOT, "docs", "doxygen", "Doxyfile.in")

# Keep in step with the project() call in CMakeLists.txt.
MFTL_VERSION = "6.0"


def find_tool(name, human):
    """Locate an executable, or explain how to get it and exit."""
    path = shutil.which(name)
    if path:
        return path
    print(f"Error: {human} not found on PATH ('{name}').", file=sys.stderr)
    print("  macOS : brew install doxygen graphviz", file=sys.stderr)
    print("  Debian: sudo apt-get install doxygen graphviz", file=sys.stderr)
    sys.exit(1)


def configure_doxyfile(output_dir, have_dot, dot_path, want_latex):
    """Substitute the @PLACEHOLDER@ tokens of Doxyfile.in and write the result.

    This is deliberately the same substitution CMake's configure_file() performs,
    so the two build routes cannot drift apart.
    """
    if not os.path.isfile(DOXYFILE_IN):
        print(f"Error: template not found: {DOXYFILE_IN}", file=sys.stderr)
        sys.exit(1)

    with open(DOXYFILE_IN, encoding="utf-8") as fh:
        text = fh.read()

    substitutions = {
        "MFTL_VERSION": MFTL_VERSION,
        "MFTL_DOCS_SOURCE_DIR": PROJECT_ROOT,
        "MFTL_DOCS_OUTPUT_DIR": output_dir,
        "MFTL_DOCS_HAVE_DOT": "YES" if have_dot else "NO",
        "MFTL_DOCS_DOT_PATH": dot_path or "",
    }
    for key, value in substitutions.items():
        text = text.replace(f"@{key}@", value)

    leftovers = re.findall(r"@[A-Z_][A-Z0-9_]*@", text)
    if leftovers:
        print(f"Error: unsubstituted placeholders: {sorted(set(leftovers))}", file=sys.stderr)
        sys.exit(1)

    if want_latex:
        text = text.replace("GENERATE_LATEX         = NO", "GENERATE_LATEX         = YES")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Doxyfile")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path


def summarise_warnings(log_path):
    """Return (count, first_lines) for the Doxygen warning log."""
    if not os.path.isfile(log_path):
        return 0, []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    return len(lines), lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate the MFTL Doxygen manual.",
        epilog="Example: python3 build_docs.py --open",
    )
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="open the generated manual when it is ready")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if Doxygen emitted any warning")
    parser.add_argument("--no-graphs", dest="graphs", action="store_false",
                        help="skip Graphviz; much faster, loses the call graphs")
    parser.add_argument("--latex", action="store_true",
                        help="also emit LaTeX sources for a PDF")
    parser.add_argument("--output-dir", default=None,
                        help="output directory (default: build/docs)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress Doxygen's progress output")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "build", "docs")
    output_dir = os.path.abspath(output_dir)

    doxygen = find_tool("doxygen", "Doxygen")

    dot_path = ""
    have_dot = False
    if args.graphs:
        dot_exe = shutil.which("dot")
        if dot_exe:
            have_dot = True
            dot_path = os.path.dirname(dot_exe)
        else:
            print("Warning: Graphviz 'dot' not found -- call graphs will be omitted.")
            print("         brew install graphviz   /   sudo apt-get install graphviz")

    doxyfile = configure_doxyfile(output_dir, have_dot, dot_path, args.latex)

    print("-------------------------------------------------")
    print("MFTL Documentation Build")
    print("-------------------------------------------------")
    print(f"Doxygen          : {doxygen}")
    print(f"Graphviz (dot)   : {'yes -- ' + dot_path if have_dot else 'no (graphs disabled)'}")
    print(f"Output Directory : {output_dir}")
    print(f"LaTeX            : {'yes' if args.latex else 'no'}")
    print(f"Strict           : {'yes' if args.strict else 'no'}")
    print("-------------------------------------------------")

    stdout = subprocess.DEVNULL if args.quiet else None
    try:
        subprocess.check_call([doxygen, doxyfile], cwd=PROJECT_ROOT, stdout=stdout)
    except subprocess.CalledProcessError as exc:
        print(f"\nError: Doxygen failed with exit code {exc.returncode}.", file=sys.stderr)
        sys.exit(1)

    log_path = os.path.join(output_dir, "doxygen-warnings.log")
    n_warnings, warning_lines = summarise_warnings(log_path)

    index = os.path.join(output_dir, "html", "index.html")
    print("\nDocumentation built.")
    print(f"  Manual   : {index}")
    print(f"  Warnings : {n_warnings}" + (f"  ({log_path})" if n_warnings else ""))

    if n_warnings:
        for line in warning_lines[:25]:
            print(f"    {line}")
        if n_warnings > 25:
            print(f"    ... and {n_warnings - 25} more, see {log_path}")

    if args.open_browser and os.path.isfile(index):
        webbrowser.open(f"file://{index}")

    if args.strict and n_warnings:
        print(f"\nError: --strict and {n_warnings} Doxygen warning(s).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
