#!/usr/bin/env python3
"""
MFTL Manual Build Script (MkDocs / Material)
============================================

Builds the MFTL manual as a modern static site: MkDocs with the Material theme
for the prose, mkdoxy for the API reference, and the full Doxygen HTML manual
mounted inside the result at ``/doxygen/`` so there is one artefact and one URL.

Nothing about the site is a second copy of anything. The guide pages, the ten
module theory chapters and the three measurement documents are generated at
build time from ``docs/pages/*.md``, ``src/<module>/README.md`` and
``docs/*-analysis.md`` -- the same files Doxygen reads -- by
``docs/mkdocs/gen_pages.py``. The API reference is generated from Doxygen XML
using a Doxyfile derived from ``docs/doxygen/Doxyfile.in``, so both manuals
extract the same symbols with the same aliases.

The Python toolchain is kept in a private virtual environment (``.venv-docs``)
rather than in the system interpreter, for the reason ``build_all.py`` always
recreates ``build/``: a documentation dependency must never be able to change
what the library build does.

Usage:
    python3 build_mkdocs.py [options]

Options:
    --serve          Run the live-reload development server instead of building.
    --port N         Port for --serve (default: 8000).
    --open           Open the result in the default browser.
    --strict         Fail on any warning, from Doxygen and from MkDocs alike.
    --no-doxygen     Skip the Doxygen HTML pass; /doxygen/ keeps whatever is
                     already in build/docs/html (or a placeholder page).
    --no-graphs      Pass --no-graphs to the Doxygen pass. Much faster while
                     editing prose; loses the call and include graphs.
    --site-dir DIR   Write the site somewhere other than build/site.
    --reinstall      Re-install the documentation toolchain into .venv-docs.
    --quiet          Suppress progress chatter.

Examples:
    python3 build_mkdocs.py                 # build the whole site
    python3 build_mkdocs.py --serve --open  # write prose, watch it reload
    python3 build_mkdocs.py --strict        # what CI should run
    python3 build_mkdocs.py --serve --no-doxygen --no-graphs   # fastest loop
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv-docs")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements-docs.txt")
CONFIG = os.path.join(PROJECT_ROOT, "mkdocs.yml")
DEFAULT_SITE_DIR = os.path.join(PROJECT_ROOT, "build", "site")

#: Import names that prove the toolchain is present, and the distribution each
#: comes from -- reported when one is missing so the fix is one pip install.
REQUIRED_IMPORTS = {
    "mkdocs": "mkdocs",
    "material": "mkdocs-material",
    "mkdoxy": "mkdoxy",
    "mkdocs_gen_files": "mkdocs-gen-files",
}


def venv_python(venv_dir):
    """Return the interpreter inside a virtual environment.

    :param venv_dir: root of the virtual environment.
    :returns: absolute path to its ``python`` executable.
    """
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def missing_imports(python):
    """Report which parts of the documentation toolchain an interpreter lacks.

    :param python: path to the interpreter to probe.
    :returns: list of distribution names that are not importable.
    """
    script = "import importlib.util as u,sys;print(' '.join(n for n in sys.argv[1:] if u.find_spec(n) is None))"
    try:
        out = subprocess.run(
            [python, "-c", script, *REQUIRED_IMPORTS],
            capture_output=True, text=True, check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return list(REQUIRED_IMPORTS.values())
    return [REQUIRED_IMPORTS[name] for name in out]


def ensure_environment(reinstall, quiet):
    """Create ``.venv-docs`` if needed and install the toolchain into it.

    The interpreter running this script is used directly when it already has
    everything, so a system-wide or Conda installation is not duplicated.

    :param reinstall: install even when the environment already satisfies the
        requirements.
    :param quiet: suppress pip's progress output.
    :returns: path to the interpreter the documentation should be built with.
    """
    if not reinstall and not missing_imports(sys.executable):
        return sys.executable

    python = venv_python(VENV_DIR)
    if not os.path.isfile(python):
        print(f"Creating the documentation environment in {os.path.relpath(VENV_DIR, PROJECT_ROOT)} ...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        python = venv_python(VENV_DIR)

    needed = missing_imports(python)
    if reinstall or needed:
        if needed and not reinstall:
            print(f"Installing the documentation toolchain ({', '.join(needed)}) ...")
        pip = [python, "-m", "pip", "install", "--upgrade"]
        if quiet:
            pip.append("--quiet")
        subprocess.check_call(pip + ["pip"])
        subprocess.check_call(pip + ["-r", REQUIREMENTS])

    still_missing = missing_imports(python)
    if still_missing:
        print(f"Error: could not install {', '.join(still_missing)}.", file=sys.stderr)
        print(f"  Try: {python} -m pip install -r {REQUIREMENTS}", file=sys.stderr)
        sys.exit(1)
    return python


def build_doxygen(strict, graphs, quiet):
    """Run the Doxygen HTML pass that the site mounts at ``/doxygen/``.

    Delegated to ``build_docs.py`` rather than reimplemented, so the embedded
    manual is byte-for-byte the one a reader gets from that script.

    :param strict: treat Doxygen warnings as errors.
    :param graphs: generate the Graphviz call and include graphs.
    :param quiet: suppress Doxygen's progress output.
    :returns: nothing; exits non-zero if the pass fails.
    """
    if shutil.which("doxygen") is None:
        print("Warning: Doxygen not found on PATH -- skipping the embedded manual.")
        print("         macOS: brew install doxygen graphviz")
        return

    command = [sys.executable, os.path.join(PROJECT_ROOT, "build_docs.py")]
    if strict:
        command.append("--strict")
    if not graphs:
        command.append("--no-graphs")
    if quiet:
        command.append("--quiet")
    subprocess.check_call(command, cwd=PROJECT_ROOT)


def run_mkdocs(python, arguments, quiet):
    """Invoke MkDocs with the environment the site expects.

    :param python: interpreter to run MkDocs with.
    :param arguments: MkDocs sub-command and its options.
    :param quiet: suppress MkDocs' informational output.
    :returns: the process' exit status.
    """
    env = dict(os.environ)
    # An advertisement printed by a transitive dependency of mkdocs-gen-files.
    env["DISABLE_MKDOCS_2_WARNING"] = "true"

    command = [python, "-m", "mkdocs", *arguments, "--config-file", CONFIG]
    if quiet:
        command.append("--quiet")
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


def main():
    """Parse the command line and drive the two passes.

    :returns: nothing; exits with the status of the MkDocs run.
    """
    parser = argparse.ArgumentParser(
        description="Generate the MFTL manual as a MkDocs Material site.",
        epilog="Example: python3 build_mkdocs.py --serve --open",
    )
    parser.add_argument("--serve", action="store_true",
                        help="run the live-reload development server instead of building")
    parser.add_argument("--port", type=int, default=8000,
                        help="port for --serve (default: 8000)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="open the result in the default browser")
    parser.add_argument("--strict", action="store_true",
                        help="fail on any warning, from Doxygen and from MkDocs alike")
    parser.add_argument("--no-doxygen", dest="doxygen", action="store_false",
                        help="skip the Doxygen HTML pass mounted at /doxygen/")
    parser.add_argument("--no-graphs", dest="graphs", action="store_false",
                        help="skip Graphviz in the Doxygen pass; much faster")
    parser.add_argument("--site-dir", default=None,
                        help="output directory (default: build/site)")
    parser.add_argument("--reinstall", action="store_true",
                        help="re-install the documentation toolchain into .venv-docs")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress output")
    args = parser.parse_args()

    site_dir = os.path.abspath(args.site_dir or DEFAULT_SITE_DIR)
    python = ensure_environment(args.reinstall, args.quiet)

    if not args.quiet:
        print("-------------------------------------------------")
        print("MFTL Manual Build (MkDocs / Material)")
        print("-------------------------------------------------")
        print(f"Interpreter      : {python}")
        print(f"Doxygen pass     : {'yes' if args.doxygen else 'skipped'}"
              f"{'' if args.graphs else ' (no graphs)'}")
        print(f"Output Directory : {site_dir}")
        print(f"Mode             : {'serve' if args.serve else 'build'}")
        print(f"Strict           : {'yes' if args.strict else 'no'}")
        print("-------------------------------------------------", flush=True)

    if args.doxygen:
        build_doxygen(args.strict, args.graphs, args.quiet)

    if args.serve:
        arguments = ["serve", "--dev-addr", f"127.0.0.1:{args.port}"]
        if args.strict:
            arguments.append("--strict")
        if args.open_browser:
            # The server blocks, so the browser has to be pointed at it first.
            # One second is enough for MkDocs to bind the socket; if it is not,
            # a reload in the browser costs nothing.
            url = f"http://127.0.0.1:{args.port}/"
            print(f"Serving on {url} -- Ctrl-C to stop.")
            time.sleep(1.0)
            webbrowser.open(url)
        sys.exit(run_mkdocs(python, arguments, args.quiet))

    arguments = ["build", "--site-dir", site_dir]
    if args.strict:
        arguments.append("--strict")
    status = run_mkdocs(python, arguments, args.quiet)
    if status != 0:
        print("\nError: the MkDocs build failed.", file=sys.stderr)
        sys.exit(status)

    index = os.path.join(site_dir, "index.html")
    if not args.quiet:
        print("\nManual built.")
        print(f"  Site           : {index}")
        print(f"  Doxygen manual : {os.path.join(site_dir, 'doxygen', 'index.html')}")
        print("\n  Serve it locally with:  python3 build_mkdocs.py --serve --open")

    if args.open_browser and os.path.isfile(index):
        webbrowser.open(f"file://{index}")


if __name__ == "__main__":
    main()
