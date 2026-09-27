# MFTL2 manual

`MFTL2.tex` is the user and developer manual of MFTL2: concepts (data model, parallel model,
twin states, periodicity, phase-boundary sets), the geometry, query, remeshing and I/O
functions, the multiflow integration, the invariants and the known limitations. The chapters
are in `chapters/`; the figures are TikZ drawings inside the chapters.

## Building

Requirements: `pdflatex` with the TikZ, `listings`, `booktabs`, `enumitem` and `lmodern`
packages (any TeX Live installation). The MFTL2 library build does **not** need LaTeX.

    make -C doc            # doc/MFTL2.pdf
    make -C doc clean

or, from a CMake build tree configured with `-DMFTL_BUILD_MANUAL=ON`:

    cmake --build <build> --target manual   # <build>/doc/MFTL2.pdf

## Listings are real programs

Every code listing of the "Worked examples" chapter is included with `\lstinputlisting` from
`examples/manual_*.c`. These programs are built with the library (`BUILD_EXAMPLES=ON`, the
default for a top-level build) and run by the test suite on 1, 2, 4 and 8 ranks
(`ctest -R Manual-`), so the manual cannot show a listing that does not compile or run.
