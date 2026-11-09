"""Renders a series of .pvtu files written by MFTLWriteVtu to PNG frames.

    pvbatch render_frames.py job.json

The job file (see make_readme_gifs.sh for the ones the README uses):

    pattern       glob of the .pvtu files, rendered in sorted order
    out           output directory for frame_NNNN.png
    stride        every n-th file (default 1); or "frames": [indices]
    size          [width, height] in pixels
    focal, position, up, angle
                  a fixed camera; or "autoframe": {"dir", "k", "sigma",
                  "periodic"} to follow each frame's bounding box from a fixed
                  direction, Gaussian-smoothed over sigma frames (wrapping
                  around for a periodic motion, so the loop is seamless)
    color         a cell array to colour by category ("TriangleOwner" for the
                  owning rank, "BodyID"), with "n_colors" and an optional
                  "palette"; null for a plain surface
    edges         draw the mesh edges, with "edge_opacity" and "edge_width"
    box           [xmin, xmax, ymin, ymax, zmin, zmax]: draw the domain outline
    ambient, diffuse, specular   lighting
"""
import glob, json, os, sys
from paraview.simple import *

job = json.load(open(sys.argv[1]))
files = sorted(glob.glob(job["pattern"]))[:: job.get("stride", 1)]
if "frames" in job:
    files = [files[i] for i in job["frames"]]
os.makedirs(job["out"], exist_ok=True)
w, h = job.get("size", [640, 640])

view = CreateView("RenderView")
view.ViewSize = [w, h]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.Background = job.get("background", [1.0, 1.0, 1.0])
view.CameraFocalPoint = job["focal"]
view.CameraPosition = job["position"]
view.CameraViewUp = job.get("up", [0, 0, 1])
view.CameraViewAngle = job.get("angle", 30)
view.EnableRenderOnInteraction = 0
Render(view)  # the first Render resets the camera; spend it on the empty view
try:
    view.UseFXAA = 1
except AttributeError:
    pass

if job.get("box"):
    b = job["box"]
    box = Box(XLength=b[1] - b[0], YLength=b[3] - b[2], ZLength=b[5] - b[4], Center=[(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    bd = Show(box, view)
    bd.Representation = "Outline"
    bd.AmbientColor = bd.DiffuseColor = [0.55, 0.58, 0.62]
    bd.LineWidth = 1.5

pal = job.get("palette", [[0.42, 0.64, 0.90], [0.98, 0.68, 0.38], [0.52, 0.80, 0.54], [0.93, 0.50, 0.55], [0.70, 0.58, 0.88],
                          [0.75, 0.60, 0.48], [0.95, 0.66, 0.84], [0.55, 0.82, 0.85]])
import math
frame = job.get("autoframe")
if frame:
    # Fixed direction, the camera follows each frame's bounding box: centre and
    # size are smoothed forwards AND backwards so the motion is steady.
    boxes = []
    for f in files:
        r = XMLPartitionedUnstructuredGridReader(FileName=[f])
        r.UpdatePipeline()
        boxes.append(r.GetDataInformation().GetBounds())
        Delete(r)
    cen = [[(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2] for b in boxes]
    rad = [0.5 * math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2) for b in boxes]

    sigma = frame.get("sigma", 3.0)
    periodic = frame.get("periodic", False)

    def smooth(xs):
        # Gaussian over neighbouring frames; a periodic motion wraps around so
        # the first and the last frame -- the same state -- are framed alike.
        n = len(xs)
        out = []
        for i in range(n):
            num = den = 0.0
            for j in range(-int(3 * sigma), int(3 * sigma) + 1):
                q = i + j
                if periodic:
                    q %= n - 1 if n > 1 else 1
                elif q < 0 or q >= n:
                    continue
                wgt = math.exp(-0.5 * (j / sigma) ** 2) if sigma > 0 else (1.0 if j == 0 else 0.0)
                num += wgt * xs[q]
                den += wgt
            out.append(num / den)
        return out

    rad = smooth(rad)
    cen = [list(c) for c in zip(*[smooth([c[k] for c in cen]) for k in range(3)])]
    dvec = frame["dir"]
    dn = math.sqrt(sum(x * x for x in dvec))
    dvec = [x / dn for x in dvec]
    dist_k = frame.get("k", 2.0) / math.tan(math.radians(job.get("angle", 30)) / 2)

for k, f in enumerate(files):
    src = XMLPartitionedUnstructuredGridReader(FileName=[f])
    if frame:
        dist = dist_k * max(rad[k], frame.get("min_radius", 0.0))
        view.CameraFocalPoint = cen[k]
        view.CameraPosition = [cen[k][i] + dist * dvec[i] for i in range(3)]
    d = Show(src, view)
    if frame:
        view.CameraFocalPoint = cen[k]
        view.CameraPosition = [cen[k][i] + dist * dvec[i] for i in range(3)]
    else:
        view.CameraFocalPoint = job["focal"]
        view.CameraPosition = job["position"]
    view.CameraViewUp = job.get("up", [0, 0, 1])
    view.CameraViewAngle = job.get("angle", 30)
    d.Specular = job.get("specular", 0.25)
    d.Ambient = job.get("ambient", 0.30)
    d.Diffuse = job.get("diffuse", 0.75)
    if job.get("color"):
        ColorBy(d, ("CELLS", job["color"]))
        lut = GetColorTransferFunction(job["color"])
        n = job.get("n_colors", 4)
        lut.InterpretValuesAsCategories = 1
        lut.Annotations = sum([[str(i), str(i)] for i in range(n)], [])
        lut.IndexedColors = sum([pal[i % len(pal)] for i in range(n)], [])
        d.SetScalarBarVisibility(view, False)
    else:
        ColorBy(d, None)
        d.DiffuseColor = d.AmbientColor = job.get("surface", [0.72, 0.80, 0.90])
    if job.get("edges"):
        d.Representation = "Surface With Edges"
        d.EdgeColor = job.get("edge_color", [0.15, 0.17, 0.20])
        try:
            d.EdgeOpacity = job.get("edge_opacity", 0.35)
        except AttributeError:
            pass
        d.LineWidth = job.get("edge_width", 1.0)
    Render(view)
    SaveScreenshot(os.path.join(job["out"], "frame_%04d.png" % k), view, ImageResolution=[w, h], TransparentBackground=0)
    Delete(src)
    del src
print("rendered", len(files), "frames")
