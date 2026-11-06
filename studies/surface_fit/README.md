# Surface-fit benchmarks

`qrcp_vs_dgelsy.c` compares the library's least-squares solver
(`MFTLLeastSquaresQRCP`, `src/geometry/surface_fit.c`) with LAPACK's `dgelsy` on the
systems the paraboloid placement solves, and the placement's accuracy against the
exact sphere. It is not part of the build: the library does not link LAPACK.

    ./run_benchmarks.sh [build-dir]       # default ../../build; links Accelerate on macOS, -llapack elsewhere

`benchmark_output.txt` is the run cited in `src/geometry/README.md`,
`docs/analysis.md` §14.10 and `../leveque_remeshing/PAPER_NOTES.md` §9 (Apple M4,
Apple clang 21, Release build). In short:

- **Speed:** 0.38–0.52 µs per solve of a positions-only system (24 rows), 1.2–1.3 µs
  with normal slopes (72 rows), against 1.31–1.46 µs and 2.43–2.59 µs for `dgelsy`
  with a precomputed workspace: 1.96 to 3.43 times faster.
- **Agreement:** the solutions agree to 2.3e-15 relative, with the same rank on every
  problem.
- **Scale:** mftl's fit (8 samples, unscaled coordinates, `dgelsy` at
  `rcond = 1e-10`) loses its curvature columns once the edge length falls below about
  1e-5 in coordinate units (sqrt of rcond): on a sphere of radius 1e-4 every edge is
  rank-truncated and the placed height is off by 4.5 sagittas. The same 8 samples in
  scaled coordinates are exact to 3.2e-3 of the sagitta at every scale.
- **Rank deficiency:** 8 samples on the unit circle at height 1 (x^2 + y^2 = 1 couples
  the height and the curvature): grouped pivoting returns c0 = 1, the plane through
  the data; `dgelsy`'s minimum-norm solution returns c0 = 0.667 with curvature 0.333.
