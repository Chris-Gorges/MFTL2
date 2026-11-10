#!/usr/bin/env bash
# Regenerates the four animations in the README (.github/images/*.gif) --
# LeVeque's deformation test face-on, the same test with uniform against
# curvature-adaptive sizing, droplets in a duct and a liquid jet breaking up:
# runs the examples with VTU output, renders the frames with ParaView's pvbatch
# (render_frames.py) on a TRANSPARENT background, and encodes each GIF with a
# palette of its own and one transparent colour (ffmpeg), so it sits on GitHub's
# light and dark themes alike.
#
# Usage: tools/media/make_readme_gifs.sh [BUILD_DIR] [WORK_DIR]
#        BUILD_DIR  a Release build with the examples (default build)
#        WORK_DIR   where the runs and frames go (default a temporary directory)
#        PVBATCH    pvbatch to use (default: from PATH, else the ParaView app on macOS)
#
# Needs MPI with 8 slots, ParaView >= 5.11 and ffmpeg. Takes about eight minutes and
# 9 GB of scratch space (the duct run writes 121 frames of up to 1.7 M triangles).
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

# gif NAME FPS WIDTH [HOLD_SECONDS] [COLOURS]: RGBA frames in $work/NAME_frames -> $out/NAME.gif.
# A GIF has 1-bit transparency, so pixels under half covered become transparent.
gif()
{
  local name=$1 fps=$2 width=$3 hold=${4:-0} colours=${5:-128}
  local pre="scale=${width}:-1:flags=lanczos,format=rgba"
  [ "$hold" != 0 ] && pre="tpad=stop_mode=clone:stop_duration=${hold},$pre"
  ffmpeg -loglevel error -y -framerate "$fps" -i "$work/${name}_frames/frame_%04d.png" \
    -vf "$pre,palettegen=reserve_transparent=1:max_colors=${colours}:stats_mode=diff" "$work/${name}_palette.png"
  ffmpeg -loglevel error -y -framerate "$fps" -i "$work/${name}_frames/frame_%04d.png" -i "$work/${name}_palette.png" \
    -lavfi "$pre[x];[x][1:v]paletteuse=alpha_threshold=128:dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 "$out/${name}.gif"
  echo "wrote $out/${name}.gif ($(du -h "$out/${name}.gif" | cut -f1))"
}

# 1. LeVeque's deformation test on 4 ranks, Phong projection, run twice:
#    curvature-adaptive and uniform sizing. The camera looks along the normal of
#    the stretched sheet at T/2 (its smallest principal axis) and follows each
#    frame's bounding box; the zoom wraps around, so the loop is seamless.
for sizing in curvature uniform; do
  mkdir -p "$work/leveque_$sizing" && cd "$work/leveque_$sizing"
  mpirun -n 4 "$build/leveque_remeshing_parallel" -refine 4 -sizing "$sizing" -projection phong -vtu > run.log
done
camera='"size": [640, 640], "focal": [0.5, 0.5, 0.5], "position": [2, -1, 1.5], "up": [0, 0, 1], "angle": 30'
follow='"autoframe": {"dir": [-0.257, 0.683, 0.683], "k": 0.86, "sigma": 4, "sigma_centre": 1, "periodic": true}'
cat > "$work/leveque.json" << EOF
{"pattern": "$work/leveque_curvature/leveque_par_*.pvtu", "out": "$work/leveque_frames", "stride": 2, $camera, $follow,
 "color": "TriangleOwner", "n_colors": 4, "edges": true, "edge_opacity": 0.45, "edge_width": 0.7,
 "ambient": 0.35, "diffuse": 0.7, "transparent": true}
EOF
"$PVBATCH" "$render" "$work/leveque.json"
gif leveque 12 480

for sizing in uniform curvature; do
  cat > "$work/sizing_$sizing.json" << EOF
{"pattern": "$work/leveque_$sizing/leveque_par_*.pvtu", "out": "$work/sizing_${sizing}_frames", "stride": 2,
 "size": [560, 560], "focal": [0.5, 0.5, 0.5], "position": [2, -1, 1.5], "up": [0, 0, 1], "angle": 30,
 "autoframe": {"dir": [-0.257, 0.683, 0.683], "k": 0.97, "sigma": 4, "sigma_centre": 1, "periodic": true},
 "area": {"range": [4e-6, 7e-5], "log": true, "preset": "Viridis (matplotlib)"},
 "edges": true, "edge_opacity": 0.35, "edge_width": 0.6, "ambient": 0.35, "diffuse": 0.7, "transparent": true}
EOF
  "$PVBATCH" "$render" "$work/sizing_$sizing.json"
done
mkdir -p "$work/leveque_sizing_frames"
for f in "$work"/sizing_uniform_frames/frame_*.png; do
  b=$(basename "$f")
  ffmpeg -loglevel error -y -i "$work/sizing_uniform_frames/$b" -i "$work/sizing_curvature_frames/$b" \
    -filter_complex "hstack,format=rgba" "$work/leveque_sizing_frames/$b"
done
gif leveque_sizing 12 640

# 2. A liquid jet entering through a nozzle, with a prescribed Rayleigh-Plateau
#    perturbation: necks pinch off and every drop becomes a body of its own.
#    Coloured by role (jet or drop), since a pinch can hand the jet a new id.
mkdir -p "$work/jet" && cd "$work/jet"
mpirun -n 4 "$build/liquid_jet_open_surface" 2 2 1 --sectors 64 --length 40 --eps 0.1 --weber 2 --topology --vtk 2 --report 100 --out jet > run.log
cat > "$work/liquid_jet.json" << EOF
{"pattern": "$work/jet/jet_*.pvtu", "out": "$work/liquid_jet_frames", "stride": 1,
 "size": [960, 320], "focal": [19.5, 0.0, 0.3], "position": [16.5, -30.0, 9.0], "up": [0, 0, 1], "angle": 30,
 "color": "Role", "role_from_wall": true, "n_colors": 5,
 "palette": [[0.42, 0.64, 0.90], [0.98, 0.62, 0.30], [0.52, 0.80, 0.54], [0.93, 0.50, 0.55], [0.70, 0.58, 0.88]],
 "edges": true, "edge_opacity": 0.3, "edge_width": 0.6, "ambient": 0.35, "diffuse": 0.7, "transparent": true}
EOF
"$PVBATCH" "$render" "$work/liquid_jet.json"
gif liquid_jet 15 880 1

# 3. 800 droplets of random size carried through a periodic duct by a swirling
#    flow, remeshed every step and redistributed over 8 ranks; coloured by rank.
#    Between frames 30 and 100 the camera closes in on droplet 42, a large one
#    on the near side (nothing between it and the camera), and follows it.
mkdir -p "$work/duct" && cd "$work/duct"
mpirun -n 8 "$build/droplets_in_duct_flow" --drops 800 --refine 3 --steps 240 --report 60 --vtk 2 > run.log
cat > "$work/duct.json" << EOF
{"pattern": "$work/duct/duct_*.pvtu", "out": "$work/duct_frames", "stride": 1,
 "size": [1080, 400], "focal": [2.0, 0.5, 0.42], "position": [2.0, -3.1, 2.15], "up": [0, 0, 1], "angle": 30,
 "color": "TriangleOwner", "n_colors": 8, "edges": false, "ambient": 0.3, "diffuse": 0.75, "specular": 0.3,
 "box": [0, 4, 0, 1, 0, 1], "box_width": 3.0, "transparent": true,
 "zoom": {"in": 30, "out": 100, "ramp": 15, "body": 42, "distance": 0.4, "edge_opacity": 0.6}}
EOF
"$PVBATCH" "$render" "$work/duct.json"
gif duct 15 800 0 64   # the close-up's mesh edges change every pixel of every frame: 64 colours keep it near 7 MB
