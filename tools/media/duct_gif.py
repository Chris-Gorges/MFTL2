"""Supervise the MPI -> VTU -> PNG handoff and encode the completed duct GIF.

The standard-library helpers are also imported by render_frames.py in pvbatch.
The simulation and renderer are independent processes (and independent MPI
implementations); neither is launched inside the other's MPI communicator.
"""

import argparse
import colorsys
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zlib


def atomic_text(path, text=""):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(text)
    pending.replace(path)


def check_png(path):
    """Check chunk boundaries and CRCs before deleting an irreplaceable mesh."""
    with open(path, "rb") as fp:
        if fp.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Not a PNG: {path}")
        size = os.fstat(fp.fileno()).st_size
        seen_header = seen_data = False
        while True:
            header = fp.read(8)
            if len(header) != 8:
                raise RuntimeError(f"Incomplete PNG: {path}")
            length, kind = struct.unpack(">I4s", header)
            if length > size - fp.tell() - 4:
                raise RuntimeError(f"Truncated PNG chunk: {path}")
            data = fp.read(length)
            crc = struct.unpack(">I", fp.read(4))[0]
            if zlib.crc32(kind + data) != crc:
                raise RuntimeError(f"Corrupt PNG chunk: {path}")
            if not seen_header:
                if kind != b"IHDR" or length != 13 or 0 in struct.unpack(">II", data[:8]):
                    raise RuntimeError(f"Invalid PNG dimensions: {path}")
                seen_header = True
            seen_data = seen_data or kind == b"IDAT"
            if kind == b"IEND":
                if length or not seen_data or fp.read(1):
                    raise RuntimeError(f"Invalid PNG ending: {path}")
                return


def mesh_files(pvtu, ranks):
    """Validate the precise file set owned by this frame, never a broad glob."""
    pvtu = Path(pvtu)
    expected = [pvtu.with_name(f"{pvtu.stem}_{rank:05d}.vtu") for rank in range(ranks)]
    sources = [p.attrib["Source"] for p in ET.parse(pvtu).findall(".//Piece")]
    if sorted(sources) != sorted(p.name for p in expected):
        raise RuntimeError(f"Unexpected piece list in {pvtu}")
    for path in expected:
        with path.open("rb") as fp:
            fp.seek(max(0, path.stat().st_size - 64))
            if not fp.read().rstrip().endswith(b"</VTKFile>"):
                raise RuntimeError(f"Incomplete VTU piece: {path}")
    return expected + [pvtu]


def stream_frames(job):
    """Wait for the producer's collective completion marker for each frame."""
    stream = job["stream"]
    for step in stream["steps"]:
        pvtu = Path(stream["mesh_dir"]) / f"duct_{step:04d}.pvtu"
        ready = pvtu.with_suffix(".ready")
        deadline = time.monotonic() + stream["timeout"]
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {ready}")
            time.sleep(0.1)
        if ready.read_text().split() != [str(step), str(stream["ranks"])]:
            raise RuntimeError(f"Invalid completion marker: {ready}")
        mesh_files(pvtu, stream["ranks"])
        yield str(pvtu)


def finish_frame(job, pvtu, png, triangles):
    """Acknowledge only after a complete PNG exists and this mesh is removed."""
    check_png(png)
    paths = mesh_files(pvtu, job["stream"]["ranks"])
    mesh_bytes = sum(p.stat().st_size for p in paths)
    stats = Path(job["out"]).parent / "frames.csv"
    with stats.open("a") as fp:
        if fp.tell() == 0:
            fp.write("frame,triangles,mesh_bytes,png_bytes\n")
        fp.write(f"{Path(png).name},{triangles},{mesh_bytes},{Path(png).stat().st_size}\n")
    for path in paths:
        path.unlink()
    atomic_text(Path(pvtu).with_suffix(".done"), "rendered\n")
    print(f"Rendered {Path(png).name}: {triangles:,} triangles; removed {mesh_bytes / 1e9:.3f} GB", flush=True)


def stop_process(proc):
    """Stop this pipeline's process group, including MPI ranks, on any failure."""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def png_frames(work):
    return sorted(p for p in (work / "frames").glob("frame_*.png") if p.stem[6:].isdigit())


def supervise(simulation, renderer, work, timeout, render_threads, expected_frames):
    """Preflight rendering, then monitor both children and propagate failures."""
    sim = render = None
    env = os.environ.copy()
    render_env = dict(env, OMP_NUM_THREADS=str(render_threads), VTK_SMP_MAX_THREADS=str(render_threads))
    sim_env = dict(env, OMP_NUM_THREADS="1")
    with (work / "simulation.log").open("w") as sim_log, (work / "render.log").open("w") as render_log:
        try:
            render = subprocess.Popen(renderer, stdout=render_log, stderr=subprocess.STDOUT,
                                      env=render_env, start_new_session=True)
            deadline = time.monotonic() + min(timeout, 180)
            while not (work / "renderer.ready").exists():
                if render.poll() is not None:
                    raise RuntimeError(f"ParaView preflight failed ({render.returncode}); see {work / 'render.log'}")
                if time.monotonic() >= deadline:
                    raise RuntimeError("ParaView preflight timed out; check the headless OpenGL setup and render.log")
                time.sleep(0.1)
            print("ParaView preflight passed; starting MPI. Logs: " + str(work), flush=True)
            sim = subprocess.Popen(simulation, cwd=work / "mesh", stdout=sim_log, stderr=subprocess.STDOUT,
                                   env=sim_env, start_new_session=True)
            shown = 0
            progress = time.monotonic()
            while True:
                for label, proc in (("Simulation", sim), ("Renderer", render)):
                    if proc.poll() not in (None, 0):
                        raise RuntimeError(f"{label} failed with exit {proc.returncode}; see logs in {work}")
                frames = len(png_frames(work))
                if frames != shown:
                    print(f"Saved {frames} PNG frames; mesh storage is limited to one timestep.", flush=True)
                    shown = frames
                    progress = time.monotonic()
                if render.poll() == 0 and frames != expected_frames:
                    raise RuntimeError("Renderer exited before all expected frames were saved")
                if sim.poll() == 0 and render.poll() == 0:
                    break
                if time.monotonic() - progress > 2 * timeout:
                    raise RuntimeError("No completed frame within the simulation/render timeout; stopping both processes")
                time.sleep(0.2)
        finally:
            stop_process(sim)
            stop_process(render)


def encode_gif(work, ffmpeg, fps, width, colors, count):
    frames = png_frames(work)
    if len(frames) != count:
        raise RuntimeError(f"Expected {count} PNG frames, found {len(frames)}")
    for frame in frames:
        check_png(frame)
    pre = f"scale={width}:-1:flags=lanczos,format=rgba"
    source = [ffmpeg, "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(work / "frames/frame_%04d.png")]
    palette = work / "palette.png"
    subprocess.run(source + ["-vf", f"{pre},palettegen=reserve_transparent=1:max_colors={colors}:stats_mode=diff",
                             "-frames:v", "1", str(palette)], check=True)
    pending = work / "duct.pending.gif"
    subprocess.run(source + ["-i", str(palette), "-lavfi",
                             f"{pre}[x];[x][1:v]paletteuse=alpha_threshold=128:dither=bayer:bayer_scale=4:diff_mode=rectangle",
                             "-loop", "0", str(pending)], check=True)
    pending.replace(work / "duct.gif")


def arguments(argv=None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Render the 20,000-droplet workstation run with one mesh frame on disk.",
                                epilog="Small check: make_duct_gif.sh BUILD WORK --ranks 2 --drops 40 --refine 3 "
                                       "--steps 12 --time 0.05 --width 480 --height 180 --zoom-body 1")
    p.add_argument("build_dir", nargs="?", type=Path, default=root / "build")
    p.add_argument("work_dir", nargs="?", type=Path, help="new or empty directory; default: a new temporary directory")
    for name, default in (("ranks", 32), ("drops", 20000), ("refine", 4), ("steps", 240), ("vtk", 2),
                          ("width", 1600), ("height", 600), ("gif-width", 1200), ("fps", 15),
                          ("colors", 128), ("zoom-body", 42)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--rmin", type=float, help="default: 0.03 * min(1, (800/drops)^(1/3))")
    p.add_argument("--rmax", type=float, help="default: 0.07 * min(1, (800/drops)^(1/3))")
    p.add_argument("--time", type=float, default=1.0, help="duct transits; use steps/240 for a timing pilot")
    p.add_argument("--timeout", type=float, default=3600, help="seconds allowed for each frame/handoff")
    p.add_argument("--pvbatch", default=os.environ.get("PVBATCH", "pvbatch"))
    p.add_argument("--no-zoom", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the job and commands without creating files or launching anything")
    a = p.parse_args(argv)
    if any(getattr(a, k) <= 0 for k in ("ranks", "drops", "steps", "vtk", "width", "height", "gif_width", "fps")):
        p.error("counts and dimensions must be positive")
    if not 0 <= a.refine <= 10 or a.drops > 1000000:
        p.error("refine must be 0..10 and drops at most 1000000")
    if not 4 <= a.colors <= 256 or (not 0 <= a.zoom_body < a.drops and not a.no_zoom):
        p.error("colors must be 4..256 and zoom-body must name an existing droplet (or use --no-zoom)")
    if any(not math.isfinite(x) or x <= 0 for x in (a.time, a.timeout)):
        p.error("time and timeout must be positive and finite")
    scale = min(1.0, (800 / a.drops) ** (1 / 3))
    a.rmin = a.rmin if a.rmin is not None else 0.03 * scale
    a.rmax = a.rmax if a.rmax is not None else 0.07 * scale
    if not all(math.isfinite(x) for x in (a.rmin, a.rmax)) or not 0 < a.rmin <= a.rmax < 0.35:
        p.error("radii must satisfy 0 < rmin <= rmax < 0.35")
    return a


def main(argv=None):
    a = arguments(argv)
    media = Path(__file__).resolve().parent
    build = a.build_dir.resolve()
    if a.dry_run:
        work = a.work_dir.resolve() if a.work_dir else Path("/tmp/mftl-duct-NEW")
    else:
        work = a.work_dir.resolve() if a.work_dir else Path(tempfile.mkdtemp(prefix="mftl-duct-"))
    steps = list(range(0, a.steps + 1, a.vtk))
    palette = [list(colorsys.hsv_to_rgb((i * 0.61803398875) % 1, 0.48 + 0.14 * (i % 2), 0.88)) for i in range(a.ranks)]
    job = {"out": str(work / "frames"), "size": [a.width, a.height],
           "focal": [2.0, 0.5, 0.42], "position": [2.0, -3.1, 2.15], "up": [0, 0, 1], "angle": 30,
           "color": "TriangleOwner", "n_colors": a.ranks, "palette": palette,
           "edges": False, "ambient": 0.3, "diffuse": 0.75, "specular": 0.3,
           "box": [0, 4, 0, 1, 0, 1], "box_width": 3.0, "transparent": True,
           "stream": {"mesh_dir": str(work / "mesh"), "steps": steps, "ranks": a.ranks, "timeout": a.timeout}}
    if not a.no_zoom and len(steps) >= 5:
        job["zoom"] = {"in": int(0.25 * (len(steps) - 1)), "out": int(0.83 * (len(steps) - 1)),
                       "ramp": max(1, int(0.125 * (len(steps) - 1))), "body": a.zoom_body,
                       "distance": 0.4 * (a.rmax / 0.07), "edge_opacity": 0.6}
    simulation = [os.environ.get("MPIEXEC", "mpirun"), *shlex.split(os.environ.get("MPIEXEC_ARGS", "")),
                  "-n", str(a.ranks), str(build / "droplets_in_duct_flow"), "--drops", str(a.drops),
                  "--refine", str(a.refine), "--rmin", repr(a.rmin), "--rmax", repr(a.rmax),
                  "--steps", str(a.steps), "--time", str(a.time), "--vtk", str(a.vtk),
                  "--report", str(a.vtk), "--verify", "--vtk-stream", "--vtk-timeout", str(a.timeout)]
    pvargs = os.environ.get("PVBATCH_ARGS", "--force-offscreen-rendering" if sys.platform.startswith("linux") else "")
    renderer = [a.pvbatch, *shlex.split(pvargs), str(media / "render_frames.py"), str(work / "job.json")]
    print(f"{a.drops:,} droplets, {20 * 4 ** a.refine * a.drops:,} initial triangles, {len(steps)} frames", flush=True)
    print("Simulation: " + shlex.join(simulation), flush=True)
    print("Renderer:   " + shlex.join(renderer), flush=True)
    if a.dry_run:
        print(json.dumps(job, indent=2))
        return
    for executable in (simulation[0], renderer[0], "ffmpeg"):
        if not shutil.which(executable):
            raise RuntimeError(f"Executable not found: {executable}")
    if not os.access(build / "droplets_in_duct_flow", os.X_OK):
        raise RuntimeError(f"Build droplets_in_duct_flow first in {build}")
    cache = build / "CMakeCache.txt"
    if cache.exists() and "CMAKE_BUILD_TYPE:STRING=Debug" in cache.read_text():
        raise RuntimeError("Use a Release build for this animation, not Debug/sanitizers")
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise RuntimeError(f"Work directory must be empty: {work}")
    (work / "mesh").mkdir()
    (work / "frames").mkdir()
    (work / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    atomic_text(work / "commands.json", json.dumps({"simulation": simulation, "renderer": renderer}, indent=2) + "\n")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    supervise(simulation, renderer, work, a.timeout, a.ranks, len(steps))
    if "verify: ok" not in (work / "simulation.log").read_text():
        raise RuntimeError("Simulation did not report successful verification; keeping PNGs and logs")
    encode_gif(work, "ffmpeg", a.fps, min(a.gif_width, a.width), a.colors, len(steps))
    print(f"Finished: {work / 'duct.gif'}\nPNGs, frame counts, timings and volume checks remain in {work}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted; stopped children and preserved existing output.", file=sys.stderr)
        sys.exit(130)
