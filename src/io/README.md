`src/io/` — Mesh output {#mftl_mod_io}
=======================

**Public header:** `include/mftl_io.h` · **Layer:** 3 ·
**Module group:** @ref mftl_io

| File | Job |
|---|---|
| `vtk_legacy.c` | `MFTLWriteVtk` — a serial legacy ASCII dump of what is in memory |
| `vtu_writer.c` | `MFTLWriteVtu` — the parallel VTK XML writer: ASCII or raw binary, one file per rank or one shared MPI-IO file |

Nothing here is read back by the library.

---

## Two writers, two contracts {#io_contracts}

| | `MFTLWriteVtk` | `MFTLWriteVtu` |
|---|---|---|
| audience | debugging | visualisation and post-processing |
| entities | the whole pool, ghosts included | **owned** triangles (and the vertices they reference) |
| arrays | owner, `verts.normal`, `tris.area` — unvalidated | owner, body id, optional global ids — never geometry |
| periodic seam | not handled | straddling triangles unwrapped |
| parallel | one rank, one file | collective, two layouts |

`MFTLWriteVtu` never writes `tris.area` or `tris.normal`: nothing records whether
they are current and a redistribution leaves them unspecified, so such a file
could be stale while looking right. `MFTLWriteVtk` writes them precisely to show
the raw state.

---

## Only owned entities are written {#io_owned}

A ghost is a copy of a triangle another rank owns. Writing whole pools would put
every halo triangle into the merged dataset once per holder: invisible in the
picture (the copies coincide) but every integral a visualisation tool computes
over the surface — area, enclosed volume — would be wrong by the halo fraction
(10–30 % in a typical run). Ownership is an array prefix, so owned-only is a loop
bound. `include_ghosts` restores the whole pool for inspecting the halo
(`test_ghost_inclusion`, `test_cell_count_is_the_owned_count`).

The points of a piece are the vertices referenced by its written triangles (some
of them ghosts), found by one marking pass over `tris.verts` and a prefix scan. A
seam vertex is therefore written by every rank with a triangle on it.

---

## Periodic unwrapping {#io_unwrap}

Vertices are stored wrapped into the domain box, so a triangle straddling a
periodic face would be drawn across the whole domain. Such a triangle is given
**three private points** at its unwrapped corners

$$
  \mathbf{p}_0, \quad \mathbf{p}_0 + \mathbf{e}_1, \quad \mathbf{p}_0 + \mathbf{e}_2
$$

(minimum-image edges $\mathbf{e}_i$), which may lie outside the box; all other
triangles keep sharing their points. The mesh in memory is untouched.

A triangle wraps iff some minimum-image edge differs from its raw difference, i.e.
iff a periodic correction was applied: $d - L\,\mathrm{nearbyint}(d/L)$ returns
$d$ bit-exactly when the rounding is 0 and differs by about $L$ otherwise. The test
is exact, with no tolerance (`test_wrap_detection_paths_agree` checks both
connectivity paths). The private points follow all shared points of a piece, and
the connectivity addresses them as `n_marked + 3*w` — that order is load-bearing.

---

## Encoding and layout {#io_encoding}

**Raw binary is the default.** Formatting a double as `%.10e` costs about 100 ns
and 17 bytes, so an ASCII writer is bound by `printf`, not by the disk. VTK XML's
raw appended section turns the emit phase into a memcpy: 23–32× faster
(docs/analysis.md §3). Its byte order is the host's and is declared in the header.

| | `MFTL_IO_FILE_PER_RANK` | `MFTL_IO_SINGLE_FILE` |
|---|---|---|
| files per dump | $P+1$ (pieces + `.pvtu` index) | 1 |
| mechanism | buffered POSIX writes per rank | `MPI_File_write_at_all` |
| collectives | one reduction of the return code | one `MPI_Iexscan` + `MPI_Iallreduce` pair, the collective open/writes, the return code |
| encodings | ASCII or binary | binary only |

### Placing blocks in the shared file {#io_layout}

Each of the up to nine arrays (points, connectivity, offsets, types, vertex owner
and id, triangle owner, body id, triangle id) is one contiguous block whose size
is `unit` bytes times either the point or the cell count. One exclusive scan of the
pair $(n_\text{points}, n_\text{cells})$ therefore places every rank's slice of
every block:

$$
  \text{offset}_b = H + \sum_{b' < b} (8 + S_{b'}) + 8
                  + u_b \sum_{r < \text{rank}} n^{(r)}_{\text{kind}(b)} ,
$$

where $H$ is the length of the XML header — identical on every rank because it is
built from global counts only — and the 8 bytes are each block's UInt64 length
prefix. Rank 0 writes the header, the length prefixes and the footer; every rank
writes its slices with `MPI_File_write_at_all`, which lets the MPI library merge
many small non-contiguous slices into a few large stripe-aligned writes (two-phase
collective buffering). The file is truncated to its exact size. Connectivity and
offsets are Int32, which bounds a shared file at $2^{31}-1$ points and
$(2^{31}-1)/3$ cells.

### Which layout is faster depends on the machine

On the development laptop (one SSD) the shared file measured 1.0× at $P = 1$ and
0.45× at $P = 4$ relative to one file per rank: there is nothing to aggregate. On
a parallel file system with thousands of ranks, $P+1$ files per dump is what
overloads the metadata server first, which is why the shared layout exists.
**Measure on the target system before choosing.** `MPI_INFO_NULL` is passed, so
collective-buffering hints (`cb_nodes`, `striping_unit`, …) are the remaining
tuning knobs.
