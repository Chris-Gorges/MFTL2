"""Renders a series of .pvtu files written by MFTLWriteVtu to PNG frames.

    pvbatch render_frames.py job.json

The job file (see make_readme_gifs.sh for the ones the README uses):

    pattern       glob of the .pvtu files, rendered in sorted order
    out           output directory for frame_NNNN.png
    stride        every n-th file (default 1); or "frames": [indices]
    size          [width, height] in pixels
    focal, position, up, angle
                  a fixed camera; or "autoframe": {"dir", "k", "sigma",
                  "periodic", "sigma_centre"} to follow each frame's bounding
                  box from a fixed direction, the zoom Gaussian-smoothed over
                  sigma frames and the centre over sigma_centre (wrapping
                  around for a periodic motion, so the loop is seamless)
    color         a cell array to colour by category ("TriangleOwner" for the
                  owning rank, "BodyID"), with "n_colors" and an optional
                  "palette"; null for a plain surface
    edges         draw the mesh edges, with "edge_opacity" and "edge_width"
    role_from_wall
                  colour by ROLE instead of by id: the body touching the plane
                  x = xmin (a jet at its nozzle) is role 0, every other body
                  1 + (id mod palette); ids change across a topology change,
                  roles do not. Sets the cell array "Role"
    area          {"range": [lo, hi], "log": true, "preset": name}: colour every
                  triangle by its area (ParaView's Cell Size) instead -- what
                  makes a size field, e.g. curvature-adaptive remeshing, visible
    label         a caption drawn in the upper left corner, in "label_color"
    transparent   save RGBA frames with a transparent background, for an
                  animation that sits on any page colour
    zoom          {"in": f0, "out": f1, "ramp": n, "body": id, "distance": d,
                   "edge_opacity": e}: a close-up between frames f0 and f1,
                  eased in and out over n frames, that follows one body (the
                  centre of its bounding box, by BodyID) from distance d along
                  the fixed camera's direction; mesh edges fade in to opacity e
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
    bd.AmbientColor = bd.DiffuseColor = job.get("box_color", [0.55, 0.58, 0.62])
    bd.LineWidth = job.get("box_width", 1.5)  # thicker for a transparent background, whose 1-bit GIF alpha cuts thin lines

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

    periodic = frame.get("periodic", False)

    def smooth(xs, sigma):
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

    # The zoom is smoothed more than the centre: a lagging centre pushes a
    # fast-moving shape out of the frame, a jittering zoom makes it pump.
    rad = smooth(rad, frame.get("sigma", 3.0))
    sc = frame.get("sigma_centre", frame.get("sigma", 3.0))
    cen = [list(c) for c in zip(*[smooth([c[k] for c in cen], sc) for k in range(3)])]
    dvec = frame["dir"]
    dn = math.sqrt(sum(x * x for x in dvec))
    dvec = [x / dn for x in dvec]
    dist_k = frame.get("k", 2.0) / math.tan(math.radians(job.get("angle", 30)) / 2)

ROLE_SCRIPT = """
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
ug = self.GetInputDataObject(0, 0)
out = self.GetOutputDataObject(0)
out.ShallowCopy(ug)
pts = vtk_to_numpy(ug.GetPoints().GetData())
conn = vtk_to_numpy(ug.GetCells().GetConnectivityArray()).reshape(-1, 3)
body = vtk_to_numpy(ug.GetCellData().GetArray("BodyID"))
cminx = pts[conn, 0].min(axis=1)
role = np.zeros(len(body), dtype=np.int32)
for b in np.unique(body):
    sel = body == b
    role[sel] = 0 if cminx[sel].min() < XMIN + 1e-6 else 1 + (int(b) % (NCOL - 1))
arr = numpy_to_vtk(role, deep=1)
arr.SetName("Role")
out.GetCellData().AddArray(arr)
"""

zoom = job.get("zoom")
if zoom:
    # Where the followed body is, frame by frame, lightly smoothed: its box
    # changes a little with every remesh.
    track = {}
    for k in range(zoom["in"], zoom["out"] + 1):
        r = XMLPartitionedUnstructuredGridReader(FileName=[files[k]])
        t = Threshold(Input=r, Scalars=["CELLS", "BodyID"], LowerThreshold=zoom["body"], UpperThreshold=zoom["body"], ThresholdMethod="Between")
        t.UpdatePipeline()
        b = t.GetDataInformation().GetBounds()
        track[k] = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        Delete(t)
        Delete(r)
    sz = zoom.get("sigma", 2.0)
    smooth_track = {}
    for k in track:
        num, den = [0.0, 0.0, 0.0], 0.0
        for q in track:
            wgt = math.exp(-0.5 * ((q - k) / sz) ** 2)
            num = [num[i] + wgt * track[q][i] for i in range(3)]
            den += wgt
        smooth_track[k] = [x / den for x in num]
    track = smooth_track

if job.get("label"):
    txt = Text(Text=job["label"])
    td = Show(txt, view)
    td.WindowLocation = "Upper Left Corner"
    td.FontSize = job.get("label_size", 26)
    td.Color = job.get("label_color", [0.15, 0.17, 0.20])

def zoom_weight(z, k):
    """0 for the wide view, 1 for the close-up, smoothstep ramps in between."""
    a = (k - z["in"]) / float(z["ramp"])
    b = (z["out"] - k) / float(z["ramp"])
    s = max(0.0, min(1.0, a, b))
    return s * s * (3.0 - 2.0 * s)

for k, f in enumerate(files):
    src = XMLPartitionedUnstructuredGridReader(FileName=[f])
    if job.get("role_from_wall"):
        merged = MergeBlocks(Input=src)
        src = ProgrammableFilter(Input=merged)
        src.OutputDataSetType = "vtkUnstructuredGrid"
        src.Script = ROLE_SCRIPT.replace("XMIN", repr(job.get("xmin", 0.0))).replace("NCOL", str(job.get("n_colors", 4)))
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
    zw = zoom_weight(zoom, k) if zoom else 0.0
    if zw > 0.0:
        wide_f, wide_p = list(view.CameraFocalPoint), list(view.CameraPosition)
        dirn = [wide_p[i] - wide_f[i] for i in range(3)]
        dl = math.sqrt(sum(x * x for x in dirn))
        near_f = track[k]
        near_p = [near_f[i] + zoom["distance"] * dirn[i] / dl for i in range(3)]
        view.CameraFocalPoint = [(1 - zw) * wide_f[i] + zw * near_f[i] for i in range(3)]
        view.CameraPosition = [(1 - zw) * wide_p[i] + zw * near_p[i] for i in range(3)]
    d.Specular = job.get("specular", 0.25)
    d.Ambient = job.get("ambient", 0.30)
    d.Diffuse = job.get("diffuse", 0.75)
    if job.get("area"):
        Hide(src, view)
        src = CellSize(Input=src)
        src.ComputeArea = 1
        src.ComputeVertexCount = src.ComputeLength = src.ComputeVolume = 0
        d = Show(src, view)
        d.Specular = job.get("specular", 0.25)
        d.Ambient = job.get("ambient", 0.30)
        d.Diffuse = job.get("diffuse", 0.75)
        ColorBy(d, ("CELLS", "Area"))
        a = job["area"]
        lut = GetColorTransferFunction("Area")
        lut.ApplyPreset(a.get("preset", "Viridis (matplotlib)"), True)
        lut.UseLogScale = 1 if a.get("log", True) else 0
        lut.RescaleTransferFunction(a["range"][0], a["range"][1])
        try:
            lut.AutomaticRescaleRangeMode = "Never"
        except AttributeError:
            pass
        d.SetScalarBarVisibility(view, False)
    elif job.get("color"):
        ColorBy(d, ("CELLS", job["color"]))
        lut = GetColorTransferFunction(job["color"])
        n = job.get("n_colors", 4)
        lut.InterpretValuesAsCategories = 1
        # Flattened by hand: a ProgrammableFilter replaces the builtin sum() in
        # this namespace with numpy_interface's.
        lut.Annotations = [str(i) for i in range(n) for _ in range(2)]
        lut.IndexedColors = [c for i in range(n) for c in pal[i % len(pal)]]
        d.SetScalarBarVisibility(view, False)
    else:
        ColorBy(d, None)
        d.DiffuseColor = d.AmbientColor = job.get("surface", [0.72, 0.80, 0.90])
    if zw > 0.0 and not job.get("edges"):
        d.Representation = "Surface With Edges"
        d.EdgeColor = job.get("edge_color", [0.15, 0.17, 0.20])
        d.EdgeOpacity = zw * zoom.get("edge_opacity", 0.4)
        d.LineWidth = job.get("edge_width", 0.6)
    if job.get("edges"):
        d.Representation = "Surface With Edges"
        d.EdgeColor = job.get("edge_color", [0.15, 0.17, 0.20])
        try:
            d.EdgeOpacity = job.get("edge_opacity", 0.35)
        except AttributeError:
            pass
        d.LineWidth = job.get("edge_width", 1.0)
    Render(view)
    SaveScreenshot(os.path.join(job["out"], "frame_%04d.png" % k), view, ImageResolution=[w, h], TransparentBackground=1 if job.get("transparent") else 0)
    Delete(src)
    del src
print("rendered", len(files), "frames")
