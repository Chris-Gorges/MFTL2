# README animations

`make_readme_gifs.sh` regenerates the four existing animations. The larger Linux
workstation duct animation has a separate entry point, `make_duct_gif.sh`.

## Linux workstation: 20,000 droplets on 32 ranks

Install an MPI implementation, CMake, a C compiler, Python 3.9 or newer,
ParaView's `pvbatch` (5.11 or newer), and FFmpeg. Build on the workstation:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build-release --target droplets_in_duct_flow -j 32
```

Check the complete workflow on a small case first. Use a new or empty directory
on a local scratch disk; the scripts refuse to mix a run with old output.

```bash
tools/media/make_duct_gif.sh build-release /scratch/duct-smoke \
  --ranks 2 --drops 40 --refine 3 --steps 12 --time 0.05 \
  --width 480 --height 180 --zoom-body 1
```

Then run the workstation case, preferably inside `tmux` or `screen`:

```bash
tools/media/make_duct_gif.sh build-release /scratch/duct-20000
```

The equivalent entry through the original script is:

```bash
tools/media/make_readme_gifs.sh --duct-workstation build-release /scratch/duct-20000
```

Defaults: 32 MPI ranks, 20,000 droplets, refinement 4 (102.4 million initial
triangles), radii approximately 0.01026–0.02394, a 4 × 1 × 1 periodic duct,
swirl 0.15, 240 steps over one transit, and output every second step (121 PNGs).
The radius range scales with droplet count to keep approximately the original
dispersed volume fraction. The simulation checks that every requested droplet
fitted and runs `--verify` before the GIF is encoded.

The camera zooms onto body 42 with mesh edges; `--zoom-body ID` selects a different
droplet and `--no-zoom` keeps the wide view. At a periodic crossing that splits the
followed body across both ends of the duct, the renderer uses the wide view.
Tracking is smoothed using the current and previous frames, so no future meshes
need to be retained. A palette with 32 entries represents the 32 owning ranks.
Rendering uses 1600 × 600 pixels; the GIF is reduced to 1200 pixels wide at 15 fps.
Use `--help` for overrides or `--dry-run` to inspect commands without launching.

For a full-size timing pilot, use a separate directory and add
`--steps 24 --time 0.1`. This retains the production timestep. The simulation's
`ms/step` excludes mesh writing and rendering; measure elapsed wall time as well.

## One mesh frame on disk

The pipeline is deliberately paced by the renderer:

1. All ranks write and close their binary VTU pieces. Only then does rank 0
   atomically publish `duct_NNNN.ready`.
2. All MPI ranks sleep, leaving CPU time for a persistent, separate `pvbatch`.
3. ParaView reads that frame and saves a transparent PNG. The script checks that
   the PNG is complete, then deletes exactly the VTU pieces and PVTU index for
   that frame. It records the mesh size and triangle count in `frames.csv`.
4. A `.done` acknowledgement allows MPI to advance. No next mesh frame can be
   written before the previous one has been deleted.
5. After successful simulation verification, FFmpeg creates a global palette and
   encodes the retained PNG sequence into `duct.gif`.

The `.pvtu` index alone is **not** a readiness signal: rank 0 can write it while
other ranks are still writing their pieces. File-size stability is not used to
guess completion. The handshake is enabled only by the example's `--vtk-stream`
option; ordinary VTK output continues to work as before.

At 100–200 million triangles, binary mesh output is roughly **4–8 GB per frame**,
plus partition/seam overhead. Allow about **20 GB of scratch space**, including
the small PNG sequence, rather than keeping roughly 0.5–0.8 TB of mesh history.
The exact maximum is in `frames.csv`. This saves peak disk space; it does not
reduce the number of mesh bytes written over the whole run. A RAM-backed scratch
directory can avoid disk traffic, but its size counts against the 500 GB RAM.

The simulation remains in RAM while ParaView runs. Rendering also needs space
for the loaded mesh and graphics buffers; reducing scratch storage does not
remove that RAM requirement. For very large cases, software rendering can avoid
GPU-memory limits, at the cost of rendering time.

PNGs and logs remain for inspection and re-encoding. On a failure or interrupt,
the supervisor stops the processes it launched and preserves existing output;
it never deletes a mesh before its PNG is complete. The default timeout is one
hour per handoff (`--timeout SECONDS`); a stalled renderer is also stopped by the
supervisor. This is not a simulation restart mechanism: after a failed run, use
a fresh work directory to rerun it.

The result directory contains `duct.gif`, `frames/`, `frames.csv`, `job.json`,
`commands.json`, `simulation.log` and `render.log`. Check the final volume error
and measured triangle counts before updating the README claim. To replace the
README animation after reviewing the result:

```bash
cp /scratch/duct-20000/duct.gif .github/images/duct.gif
```

The workstation script does not automatically replace the published animation
or change its caption.

## Headless ParaView and process settings

`pvbatch` must be on `PATH`, or specify its executable with `PVBATCH` or
`--pvbatch`. On Linux the script passes `--force-offscreen-rendering`. A tiny
screenshot preflight runs **before** the large MPI job starts. A machine without
a display needs a ParaView build with EGL or OSMesa, or an X server such as Xvfb;
offscreen rendering alone does not add those capabilities to a build. See the
[ParaView rendering documentation](https://www.paraview.org/paraview-docs/v5.13.3/cxx/Offscreen.html).

Examples (choose the one supported by your installation):

```bash
# ParaView 6 with software headless rendering support:
PVBATCH_ARGS='--force-offscreen-rendering --opengl-window-backend=OSMesa' \
  tools/media/make_duct_gif.sh build-release /scratch/duct-20000

# An X-based build with Xvfb installed:
xvfb-run -a tools/media/make_duct_gif.sh build-release /scratch/duct-20000
```

`MPIEXEC` selects the MPI launcher (default `mpirun`), and `MPIEXEC_ARGS` adds
launcher options. `PVBATCH_ARGS` replaces the default ParaView options. Commands
are passed as argument arrays, without shell evaluation. The supervisor runs
ParaView separately from the simulation's MPI launcher, so ParaView may use its
own bundled MPI. Simulation ranks use one OpenMP thread each; the renderer's
thread limit is the selected rank count, while the simulation ranks sleep.

## Validation

The handoff tests need only Python; the optional integration checks exercise
the actual example on two MPI ranks. They check that a delayed consumer cannot
accumulate meshes, streaming preserves the numerical result, incomplete output
is retained, and a failed renderer stops the simulation.

```bash
python3 tools/media/test_duct_stream.py
python3 tools/media/test_duct_stream.py --binary build-release/droplets_in_duct_flow
```

The small run above additionally checks ParaView, camera tracking, transparency,
and FFmpeg together on the target workstation.
