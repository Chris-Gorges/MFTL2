"""pvbatch script: renders mesh snapshots to PNG.

    pvbatch render_pv.py jobs.json

jobs.json is a list of {"in", "out", "focal", "position", "up", "parallel_scale",
"size", "edges", "edge_width", "array", "range", "cmap"}. One renderer, reused.
"""
import json
import sys

from paraview.simple import (ColorBy, GetActiveViewOrCreate, GetColorTransferFunction, Hide, LegacyVTKReader, Render, SaveScreenshot, Show,
                             Delete)

jobs = json.load(open(sys.argv[1]))
view = GetActiveViewOrCreate("RenderView")
view.OrientationAxesVisibility = 0
view.Background = [1.0, 1.0, 1.0]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.CameraParallelProjection = 1

for job in jobs:
    src = LegacyVTKReader(FileNames=[job["in"]])
    disp = Show(src, view)
    disp.Representation = "Surface With Edges" if job.get("edges", True) else "Surface"
    disp.EdgeColor = job.get("edge_color", [0.18, 0.20, 0.24])
    disp.LineWidth = job.get("edge_width", 1.0)
    disp.Specular = 0.15
    disp.Ambient = 0.15
    disp.Diffuse = 0.85
    arr = job.get("array")
    if arr:
        ColorBy(disp, ("POINTS", arr))
        lut = GetColorTransferFunction(arr)
        lut.ApplyPreset(job.get("cmap", "Cool to Warm"), True)
        lut.RescaleTransferFunction(job["range"][0], job["range"][1])
        try:
            lut.AutomaticRescaleRangeMode = "Never"
        except AttributeError:
            pass
        disp.SetScalarBarVisibility(view, False)
    else:
        disp.ColorArrayName = [None, ""]
        disp.AmbientColor = [0.86, 0.89, 0.94]
        disp.DiffuseColor = [0.86, 0.89, 0.94]
    view.CameraFocalPoint = job["focal"]
    view.CameraPosition = job["position"]
    view.CameraViewUp = job.get("up", [0.0, 0.0, 1.0])
    view.CameraParallelScale = job["parallel_scale"]
    w, h = job.get("size", [800, 800])
    view.ViewSize = [w, h]
    Render(view)
    SaveScreenshot(job["out"], view, ImageResolution=[w, h], TransparentBackground=0)
    Hide(src, view)
    Delete(src)
    del src
