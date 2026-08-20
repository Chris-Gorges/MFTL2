Measurements {#mftl_page_analysis}
============

Three analysis documents, each answering a question the code comments can only
assert. Every number in them was produced by a program in `examples/` in a
Release build, and every one is reproducible by re-running it.

| Document | Question |
|---|---|
| @subpage mftl_page_perf | What does a timestep cost, where does the time go, and which design decisions did **not** pay off? |
| @subpage mftl_page_remesh_quality | Why did the LeVeque remeshing test end with a bad mesh? |
| @subpage mftl_page_parallel_remesh | Does the distributed remesher track the serial one, is it reproducible to the bit, and what does a phase cost? |

---

# Re-run them rather than trusting the tables {#analysis_rerun}

Every table carries the machine it was measured on, and the development machine
is a laptop with **four performance cores**. Three consequences run through all
three documents:

- **Above four ranks the absolute times are a lower bound.** An 8-rank run puts
  half the ranks on efficiency cores and the per-triangle costs roughly triple.
  The *ratios* survive that; the absolute numbers do not.
- **Cache-locality results do not transfer.** Morton renumbering buys no
  measurable time here despite cutting mean edge index distance 160×. That needs
  re-measuring on an x86 node with 1–2 MiB L2 per core.
- **I/O layout results transfer least of all.** One local SSD is the
  configuration least able to tell a shared collective file from $P$ per-rank
  files, and the one where the shared file cannot win.

```bash
python3 build_all.py            # Release; the benchmarks are meaningless otherwise
./build/geometry_update_benchmark
./build/vertex_normal_benchmark
./build/ensemble_update_benchmark
mpirun -n 4 ./build/parallel_advection_redistribute_list
mpirun -n 4 ./build/leveque_remeshing_parallel
```

---

# What each benchmark answers {#analysis_benchmarks}

| Program | Question |
|---|---|
| `geometry_update_benchmark.c` | half-edge walk vs. streaming frames; what Morton renumbering is worth |
| `layout_benchmark.c` | interleaved vs. padded vs. split xyz, on four access patterns |
| `vertex_normal_benchmark.c` | scatter vs. fan gather vs. CSR gather |
| `update_kernel_benchmark.c` | eleven ways to write the per-timestep kernel |
| `ensemble_update_benchmark.c` | the fused geometry+quality pass against the naive half-edge loop |
| `parallel_advection_redistribute_list.c` | what a whole timestep costs, and how much of it is communication |
| `orient3d_predicate_profile.c` | does the plain orientation determinant misclassify thin-film contact, and what does the adaptive one cost |
| `leveque_remeshing_parallel.c` | over 150 remeshed timesteps, does the distributed answer track the serial one, is it reproducible to the bit, and what does a phase cost |
| `remesh_phase_profile.c` | where the time goes inside one distributed remeshing phase |
| `remesh_regime_sweep.c` | where the spanning/dispersed crossover actually lies |

---

# The results that contradicted the design {#analysis_surprises}

These are the reason to read the documents rather than the code comments.

1. **The frame kernel is worth 1.25–1.57× non-periodic and 1.94–2.00× periodic**,
   not the 3× the previous version measured — because the general path got fast,
   not because the frames got slow. The figure the project carried before
   (1.03–1.24×) was **two thirds harness**: the benchmark timed the variants in
   blocks and averaged within each. Interleaving and taking minima moved it.
2. **Morton renumbering buys no measurable time on Apple silicon**, despite
   cutting mean edge index distance 160×.
3. **Splitting `verts.pos` into three component arrays is faster for gathers and
   slower for scatters**, by similar amounts. A wash — so the layout is decided on
   the API instead. The mechanism is address generation (a 24-byte stride needs a
   multiply; an 8-byte stride does not), not cache lines.
4. **The update kernel does not fail to vectorise because of the nine-way
   reduction.** The previous version's note said so and was wrong: splitting the
   reduction makes it slightly *worse*, and clang refuses the transformation even
   when a pragma forces it. 26 % of the pass is one `sqrt` plus one divide, which
   is irreducible.
5. **The quality statistics are not nearly free.** +5 % in the previous version,
   **+90 %** here (1.64 → 3.11 ns/triangle) — the statistics did not get slower,
   the streaming baseline got fast.
6. **MPI-IO did not make output faster; dropping ASCII did.** The 23–32× came
   entirely from raw binary replacing `printf`.
7. **The LeVeque test's poor final mesh was caused by feature detection on a
   smooth surface**, not by any single operator. Final minimum quality 0.76 with
   detection off against 0.0002 at 45°, monotone across five thresholds.
