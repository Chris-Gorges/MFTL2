`src/io/` — Mesh output {#mftl_mod_io}
=======================

**Public header:** `include/mftl_io.h` · **Layer:** 3 ·
**Module group:** @ref mftl_io

| File | Job |
|---|---|
| `vtk_legacy.c` | `MFTLWriteVtk` — a serial ASCII dump of **what is actually in memory** |
| `vtu_writer.c` | `MFTLWriteVtu` — the parallel VTK XML writer, four combinations of encoding and file layout |

Nothing here is read back by the library. This is the module every other module's
correctness is ultimately **judged** through, because it is how a human looks at
what the rest produced.

---

## The two writers have different contracts {#io_contracts}

| | `MFTLWriteVtk` | `MFTLWriteVtu` |
|---|---|---|
| audience | a developer debugging | a visualisation tool, a post-processor |
| entities | whatever the pool holds | **owned only**, by default |
| `tris.area` / `tris.normal` | **written** | **never written** |
| periodic seam | not handled | triangles unwrapped |
| parallel | no — one rank, one file | yes |

@warning `MFTLWriteVtu` must **never** write `tris.area` or `tris.normal`.
         Nothing on the object records whether they are current — `derived`
         tracks the topology and the frames, not the geometry — and a
         redistribution leaves them undefined by design, so a production file
         containing them would look right and be stale. `MFTLWriteVtk` writes
         them deliberately, because the raw dump's contract is to show what is in
         memory.

---

## Only owned entities are written {#io_owned}

MFTL 5 wrote every triangle a rank held, ghosts included. A ghost is a **copy** of
a triangle another rank owns, so the merged dataset contained every halo triangle
two or more times.

Invisible when you look at the surface — the copies are coincident — but **any
integral the visualisation tool computes over it is silently wrong by the halo
fraction**, which is 10–30 % in a typical run.

Ownership is an array prefix here, so writing owned-only costs nothing but a loop
bound. `include_ghosts` restores the old behaviour for when the halo itself is
what you want to look at.

Guarded by `test_ghost_inclusion` and `test_cell_count_is_the_owned_count`.

---

## Periodic unwrapping is free {#io_unwrap}

A triangle straddling a periodic face has corners stored at opposite ends of the
box; drawn naively it stretches across the whole domain.

MFTL 5 detected this by calling `MFTLGetMinimumImageVector` twice per triangle
inside a helper it then called **four times per triangle per file** — its own
`src/io/README.md` §3.3 records the redundancy.

Here `tris.anchor` and `tris.edge` already hold exactly what that helper computed
— corner 0 and the two seam-consistent edge vectors — so the unwrapped corners
are

$$
  \mathbf{p}_0, \quad \mathbf{p}_0 + \mathbf{e}_1, \quad \mathbf{p}_0 + \mathbf{e}_2
$$

with **no periodic arithmetic at all**, and the analysis is a single streaming
pass over two contiguous arrays.

### Shared points and private points

A triangle that does *not* wrap references three existing points, shared with its
neighbours. A triangle that **does** wrap gets **three private points of its own**
carrying the unwrapped coordinates, which may lie just outside the domain box.

`analyze_piece` produces both sets in one pass; `emit_points` writes the shared
ones first and the private ones after, and `emit_conn` addresses the private ones
as `n_marked + 3*w`. **That order is load-bearing** — the connectivity depends on
it.

### The detection test has no tolerance, on purpose {#io_wrap_test}

It asks whether a reconstructed corner $\mathbf{p}_0 + \mathbf{e}$ left the domain
box. That is a strict comparison, with no epsilon.

- A **false positive** costs three extra points carrying correct coordinates, and
  renders identically.
- A **false negative** draws a triangle across the whole domain.

Any slack added to suppress the first buys the second, so there is none.
`test_wrap_detection_paths_agree` pins it against the gathering criterion used
when the frames are stale.

---

## Encoding: raw binary by default {#io_encoding}

MFTL 5 emitted one `fprintf("%.10e %.10e %.10e\n")` per point and one per cell
record. Decimal formatting of a `double` costs on the order of a hundred
nanoseconds and produces ~25 bytes where the value occupies 8 — so an ASCII
writer is CPU-bound on `printf` long before it is bound by the disk.

VTK's XML format has carried a **raw appended-binary** section since 2003
precisely for this. Using it turns the emit phase into a `memcpy`.

$$
  \text{speed-up} = \mathbf{23\text{–}32\times}
$$

ASCII (`MFTL_VTK_ASCII`) is kept because a text file you can `less` is worth a
great deal when a mesh is wrong, and because it is the honest baseline the binary
path is measured against.

The file's byte order is the machine's, since the appended section is a plain
`memcpy`, so `host_byte_order()` **detects** it rather than assuming — the library
is expected to run on whatever the cluster has.

---

## Layout: one file per rank, or one shared file {#io_layout}

| | `MFTL_IO_FILE_PER_RANK` | `MFTL_IO_SINGLE_FILE` |
|---|---|---|
| files per dump | $P + 1$ (pieces + `.pvtu` index) | 1 |
| collectives | none | **one** `MPI_Exscan` |
| mechanism | POSIX `fwrite` per rank | `MPI_File_write_at_all` |
| contention | none | two-phase collective buffering merges slices into large stripe-aligned writes issued by aggregator ranks |

### One exclusive scan places every block

A block's size is `unit` bytes times **either** the piece's point count **or** its
cell count — there is no third possibility. So one exclusive scan of the *pair*
$\{\text{points}, \text{cells}\}$ places every block of the shared file:

$$
  \text{offset}_b = \text{header} + \sum_{b' < b} S_{b'}
                  + u_b \cdot \Big(\textstyle\sum_{r < \text{rank}} n^{(r)}_{\text{kind}(b)}\Big)
$$

That is why the collective cost of this writer is $O(1)$ messages in the number of
arrays, not $O(\text{arrays})$.

Every rank builds the **identical** header text and uses its length to place the
appended section — which is the whole reason the header is assembled into an
in-memory string buffer (`sbuf`) rather than being a sequence of `fprintf` calls.

### Which is faster is a property of the machine

This library will not pretend otherwise.

- On a laptop with one local SSD the **per-rank** layout usually wins: there is no
  aggregation to be had, and a shared file only adds contention. **Measured 1.0×
  at $P=1$ and 0.45× at $P=4$.**
- On a Lustre/GPFS system with $P$ in the thousands the **single file** wins by a
  margin that grows with $P$, and the per-rank layout eventually stops working at
  all — $P+1$ files per dump is what makes the metadata server, not the bandwidth,
  fall over first.

@note **MPI-IO did not make output faster; dropping ASCII did.** The 23–32× came
      entirely from raw binary replacing `printf`. `MFTL_IO_SINGLE_FILE` is kept
      for the property that was **not** measured, and it is the only one that
      matters at scale.

@note The shared-file backend passes `MPI_INFO_NULL`. Collective-buffering hints
      (`cb_nodes`, `striping_unit`, …) are untouched and are where the remaining
      tuning lives.

---

## The block table {#io_blocks}

Nine arrays, described uniformly so both backends share the emit code:

| Block | Type | Sized by | Written when |
|---|---|---|---|
| `BLK_POINTS` | `Float64` ×3 | points | always |
| `BLK_CONN` | `Int32` ×3 | cells | always |
| `BLK_OFFSETS` | `Int32` | cells | always |
| `BLK_TYPES` | `UInt8` | cells | always (5 = `VTK_TRIANGLE`) |
| `BLK_VOWNER` | `Int32` | points | `write_owner` |
| `BLK_VGID` | `Int32` | points | `write_global_id` |
| `BLK_TOWNER` | `Int32` | cells | `write_owner` |
| `BLK_BODYID` | `Int32` | cells | `write_body_id` |
| `BLK_TGID` | `Int32` | cells | `write_global_id` |

Adding an array is one `DECL(...)` row plus one `emit_*` call. Neither backend
knows what any block *means*.
