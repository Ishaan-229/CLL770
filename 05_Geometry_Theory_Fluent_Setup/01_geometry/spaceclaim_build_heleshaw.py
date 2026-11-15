# -*- coding: utf-8 -*-
"""
spaceclaim_build_heleshaw.py  -  ANSYS SpaceClaim / Discovery script (IronPython, API V19+)
=========================================================================================
Builds the 'planar' Hele-Shaw fluid domain NATIVELY in SpaceClaim (no DXF, no polylines)
and creates the Named Selections used by ANSYS Meshing / Fluent:
    inlet, outlet, wall_bottom (glass / cell side, z = 0), wall_top (PDMS, z = h),
    wall_side (curved + straight side walls), fluid (body)

HOW TO RUN
  SpaceClaim -> File > New;  Design tab > Script Editor (or 'Prepare > Script')
  -> open this file -> set the API version box (bottom-right of the editor) to the
     newest V-number your install offers -> Run.
  Then File > Save As > .scdoc (and optionally Parasolid .x_t or STEP).

STATUS / HONESTY NOTE
  This script was written against the documented SpaceClaim scripting API but could
  NOT be executed in the environment where it was produced (no SpaceClaim licence).
  The STEP files produced by generate_geometry.py ARE verified. If a call below fails
  in your version, use ROUTE B at the bottom (import STEP + automatic named
  selections), which only needs the import call and face queries.

Units: SpaceClaim API works in metres internally; MM() converts.
"""
import math

# ---------------------------------------------------------------- parameters [mm]
L, W1, H, WMAX = 50.0, 1.0, 0.145, 10.0
L_IN, R_OUT, L_OUT = 2.0, 0.75, 1.0
N_PTS = 241
ROUTE = "A"                       # "A" = build natively, "B" = import verified STEP
STEP_PATH = r"C:\HeleShaw\HS_L50_W10_planar.step"   # used only by ROUTE B

X_T = L * (1.0 - W1 / WMAX)       # 45 mm : end of hyperbola (w = WMAX)
Y_T = WMAX / 2.0
X_TR = X_T + (Y_T - R_OUT)        # 45-deg wall reaches half-width R_OUT
X_END = X_TR + L_OUT


def half_w(x):
    return 0.5 * W1 * L / (L - x)


def P(x, y):
    return Point.Create(MM(x), MM(y), MM(0))


def hyperbola_pts(sign):
    xs = set()
    for i in range(N_PTS // 2):
        xs.add(round(X_T * i / (N_PTS // 2 - 1), 10))
        w = W1 * (WMAX / W1) ** (float(i) / (N_PTS // 2 - 1))
        xs.add(round(L * (1.0 - W1 / w), 10))
    xs = sorted(xs)
    return [P(x, sign * half_w(x)) for x in xs]


def seg(a, b):
    return CurveSegment.Create(a, b)


def nurbs_through(points):
    # NurbsCurve.CreateThroughPoints(isPeriodic, points, tolerance)
    crv = NurbsCurve.CreateThroughPoints(False, points, MM(1e-5))
    return CurveSegment.Create(crv)


def build_native():
    up = hyperbola_pts(+1.0)
    lo = hyperbola_pts(-1.0)
    lo.reverse()
    curves = [
        seg(P(-L_IN, -W1 / 2), P(-L_IN, W1 / 2)),      # inlet trace
        seg(P(-L_IN, W1 / 2), P(0, W1 / 2)),
        nurbs_through(up),
        seg(P(X_T, Y_T), P(X_TR, R_OUT)),
        seg(P(X_TR, R_OUT), P(X_END, R_OUT)),
        seg(P(X_END, R_OUT), P(X_END, -R_OUT)),        # outlet trace
        seg(P(X_END, -R_OUT), P(X_TR, -R_OUT)),
        seg(P(X_TR, -R_OUT), P(X_T, -Y_T)),
        nurbs_through(lo),
        seg(P(0, -W1 / 2), P(-L_IN, -W1 / 2)),
    ]
    profile = Profile(Plane.PlaneXY, curves)
    body = Body.ExtrudeProfile(profile, MM(H))
    return DesignBody.Create(GetRootPart(), "fluid", body)


def import_step():
    DocumentInsert.Execute(STEP_PATH, ImportOptions.Create(), None)
    return GetRootPart().Bodies[0]


def name(sel_faces, label):
    if not sel_faces:
        print("WARNING: no faces found for " + label)
        return
    ns = NamedSelection.Create(Selection.Create(sel_faces), Selection.Empty())
    NamedSelection.Rename(ns.CreatedNamedSelection.Name, label)


def classify_and_name(db):
    tol = MM(1e-4)
    inlet, outlet, bottom, top, side = [], [], [], [], []
    for f in db.Faces:
        ev = f.Shape.Geometry   # surface
        mid = f.EvalMid()
        p, n = mid.Point, mid.Normal
        if abs(abs(n.Z) - 1.0) < 1e-6 and abs(p.Z) < tol:
            bottom.append(f)
        elif abs(abs(n.Z) - 1.0) < 1e-6 and abs(p.Z - MM(H)) < tol:
            top.append(f)
        elif abs(abs(n.X) - 1.0) < 1e-6 and abs(p.X - MM(-L_IN)) < tol:
            inlet.append(f)
        elif abs(abs(n.X) - 1.0) < 1e-6 and abs(p.X - MM(X_END)) < tol:
            outlet.append(f)
        else:
            side.append(f)
    name(inlet, "inlet")
    name(outlet, "outlet")
    name(bottom, "wall_bottom")
    name(top, "wall_top")
    name(side, "wall_side")
    ns = NamedSelection.Create(Selection.Create(db), Selection.Empty())
    NamedSelection.Rename(ns.CreatedNamedSelection.Name, "fluid")
    print("faces: inlet %d outlet %d bottom %d top %d side %d"
          % (len(inlet), len(outlet), len(bottom), len(top), len(side)))


ClearAll()
db = build_native() if ROUTE == "A" else import_step()
classify_and_name(db)
# Make sure Meshing/Fluent treat the body as a fluid
try:
    db.SetFluid(True)        # available in recent versions; otherwise set "Fluid/Solid = Fluid" in Mechanical/Meshing
except Exception:
    print("Set the body to FLUID in Meshing (Details of Body > Fluid/Solid).")
print("Volume [mm^3] = %.6f  (expected 20.7447)" % (db.Shape.Volume * 1e9))
