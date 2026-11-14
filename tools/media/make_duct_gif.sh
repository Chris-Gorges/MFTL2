#!/usr/bin/env bash
# Linux workstation animation: one mesh frame on disk at a time.
# Usage: tools/media/make_duct_gif.sh [BUILD_DIR] [WORK_DIR] [options]
# Run with --help for the 20,000-droplet defaults and a small smoke run.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$script_dir/duct_gif.py" "$@"
