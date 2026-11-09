#!/usr/bin/env bash
# Regenerates the three animations in the README (.github/images/*.gif):
# runs the examples with VTU output, renders the frames with ParaView's pvbatch
# (render_frames.py) and encodes each GIF with a palette of its own (ffmpeg).
#
# Usage: tools/media/make_readme_gifs.sh [BUILD_DIR] [WORK_DIR]
#        BUILD_DIR  a Release build with the examples (default build)
#        WORK_DIR   where the runs and frames go (default a temporary directory)
#        PVBATCH    pvbatch to use (default: from PATH, else the ParaView app on macOS)
#
# Needs MPI with 8 slots, ParaView >= 5.11 and ffmpeg. Takes about two minutes.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
build="$(cd "${1:-$root/build}" && pwd)"
work="${2:-$(mktemp -d)}"
mkdir -p "$work"
work="$(cd "$work" && pwd)"
out="$root/.github/images"
render="$root/tools/media/render_frames.py"

PVBATCH="${PVBATCH:-$(command -v pvbatch || ls -d /Applications/ParaView-*.app/Contents/bin/pvbatch 2> /dev/null | tail -n 1)}"
[ -x "$PVBATCH" ] || { echo "pvbatch not found; set PVBATCH" >&2; exit 2; }

# gif NAME FPS WIDTH COLOURS [HOLD_SECONDS]: frames in $work/NAME_frames -> $out/NAME.gif
gif()
{
  local name=$1 fps=$2 width=$3 colours=$4 hold=${5:-0}
  local pre="scale=${width}:-1:flags=lanczos"
  [ "$hold" != 0 ] && pre="tpad=stop_mode=clone:stop_duration=${hold},$pre"
  ffmpeg -loglevel error -y -framerate "$fps" -i "$work/${name}_frames/frame_%04d.png" \
    -vf "$pre,palettegen=max_colors=${colours}:stats_mode=diff" "$work/${name}_palette.png"
  ffmpeg -loglevel error -y -framerate "$fps" -i "$work/${name}_frames/frame_%04d.png" -i "$work/${name}_palette.png" \
    -lavfi "$pre[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 "$out/${name}.gif"
  echo "wrote $out/${name}.gif ($(du -h "$out/${name}.gif" | cut -f1))"
}

# 1. LeVeque's deformation test, curvature sizing and Phong projection, 4 ranks:
#    a sphere stretched into a sheet and back, remeshed every step.
mkdir -p "$work/leveque" && cd "$work/leveque"
mpirun -n 4 "$build/leveque_remeshing_parallel" -refine 4 -sizing curvature -projection phong -vtu > run.log
cat > "$work/leveque.json" << EOF
{"pattern": "$work/leveque/leveque_par_*.pvtu", "out": "$work/leveque_frames", "stride": 2,
 "size": [640, 640], "focal": [0.5, 0.5, 0.5], "position": [2, -1, 1.5], "up": [0, 0, 1], "angle": 30,
 "autoframe": {"dir": [0.78, -0.44, 0.45], "k": 0.86, "sigma": 4, "periodic": true},
 "color": "TriangleOwner", "n_colors": 4, "edges": true, "edge_opacity": 0.45, "edge_width": 0.7, "ambient": 0.35, "diffuse": 0.7}
EOF
"$PVBATCH" "$render" "$work/leveque.json"
gif leveque 12 480 128

# 2. A kinematic Rayleigh-Taylor emulation whose cap rolls up and pinches off a
#    blob: a topology change that makes a new closed body. Coloured by body.
mkdir -p "$work/rt" && cd "$work/rt"
mpirun -n 4 "$build/rayleigh_taylor_open_surface" --roll 1.7 --growth 1.4 --topology --vtk 4 --report 50 --out rt > run.log
cat > "$work/rayleigh_taylor.json" << EOF
{"pattern": "$work/rt/rt_*.pvtu", "out": "$work/rayleigh_taylor_frames", "stride": 1,
 "size": [640, 640], "focal": [0.0, 0.0, -0.22], "position": [2.55, -3.45, 0.30], "up": [0, 0, 1], "angle": 30,
 "color": "BodyID", "n_colors": 4, "palette": [[0.42, 0.64, 0.90], [0.98, 0.62, 0.30], [0.52, 0.80, 0.54], [0.93, 0.50, 0.55]],
 "edges": true, "edge_opacity": 0.35, "edge_width": 0.6, "ambient": 0.35, "diffuse": 0.7}
EOF
"$PVBATCH" "$render" "$work/rayleigh_taylor.json"
gif rayleigh_taylor 14 480 128 1.5

# 3. 216 droplets in a periodic box, strained, drifted, remeshed every step and
#    redistributed over 8 ranks; coloured by owning rank.
mkdir -p "$work/drops" && cd "$work/drops"
mpirun -n 8 "$build/remesh_strong_scaling" --regime dispersed --per-axis 6 --strain 1.0 --drift 0.6 --steps 80 \
  --warmup 1 --trajectories 1 --vtk 1 --label drops > run.log
cat > "$work/droplets.json" << EOF
{"pattern": "$work/drops/drops_dispersed_*.pvtu", "out": "$work/droplets_frames", "stride": 1,
 "size": [640, 640], "focal": [3.0, 3.0, 3.0], "position": [16.0, -9.0, 12.0], "up": [0, 0, 1], "angle": 30,
 "color": "TriangleOwner", "n_colors": 8, "edges": false, "ambient": 0.3, "diffuse": 0.75, "box": [0, 6, 0, 6, 0, 6]}
EOF
"$PVBATCH" "$render" "$work/droplets.json"
gif droplets 12 440 96
