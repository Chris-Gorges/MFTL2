#!/bin/sh
# Builds and runs the surface-fit benchmarks against a Release build of MFTL2.
#   ./run_benchmarks.sh [build-dir]     (default ../../build)
set -e
B=${1:-../../build}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
case "$(uname)" in Darwin) LAPACK="-framework Accelerate" ;; *) LAPACK="-llapack -lblas" ;; esac
mpicc -O3 -std=c11 -I"$ROOT/include" "$HERE/qrcp_vs_dgelsy.c" "$B/lib/libMFTL.a" $LAPACK -lm -o "$B/qrcp_vs_dgelsy"
"$B/qrcp_vs_dgelsy"
