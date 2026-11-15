#!/usr/bin/env python3
"""
generate_geometry.py
====================
Builds the 3D fluid domain of the widening (Hele-Shaw / Usami-type) microfluidic
channel described by Delon et al. (2020), MethodsX 7:100980, with the MODIFIED
dimensions of this project, and writes STEP, DXF and CSV deliverables.

Coordinate system (same as Delon et al. Fig. 2A)
    x : along the central axis, x = 0 where the full channel width equals w1
    y : across the width, y = 0 on the central axis (walls at y = +/- w(x)/2)
    z : across the channel height, z = 0 on the glass cover slip (cell side),
        z = h on the PDMS ceiling
All lengths are in MILLIMETRES (STEP files carry the mm unit explicitly).

Wall function (FULL width):     w(x) = w1 * L / (L - x),   0 <= x <= x_t
Termination of the hyperbola:   x_t = L * (1 - w1 / w_max)   (w(x_t) = w_max)
Outlet transition:              straight walls at 45 deg to the axis,
                                converging from y = +/- w_max/2 towards x_apex = x_t + w_max/2

Variants written (see VARIANTS at the bottom):
  planar  : straight lead-in of width w1 ending in a flat INLET face (x = -l_in);
            45-deg walls truncated at half-width r_out and a short straight outlet
            duct ending in a flat OUTLET face. Best for validation & mesh study.
  ports   : paper-like layout - circular inlet port (D = 3 mm) joined to the
            lead-in, rounded outlet tip formed by the outlet port circle
            (D = 1.5 mm) tangent to the 45-deg walls, plus short vertical port
            stubs; INLET/OUTLET are the top discs of the stubs.
  half    : y >= 0 half of 'planar' with a SYMMETRY face on y = 0.

Requirements:  pip install cadquery ezdxf numpy
Run:           python generate_geometry.py            (writes into ./output)
               python generate_geometry.py --wmax 15  (termination sensitivity)
"""
import argparse
import json
import math
import os

import numpy as np
import cadquery as cq
import ezdxf

# --------------------------------------------------------------------------
# Parameters (project values; everything marked ASSUMPTION is not published)
# --------------------------------------------------------------------------
P = dict(
    L=50.0,        # characteristic length in w(x) [mm]  (project value)
    w1=1.0,        # inlet channel width [mm]            (project value = paper)
    h=0.145,       # channel height [mm]                 (project value)
    w_max=10.0,    # width where hyperbola is terminated [mm] - matches the 1 cm
                   #   maximum width marked on Delon et al. Fig. 1C
    angle_deg=45.0,  # outlet wall angle to the axis (paper)
    l_in=2.0,      # ASSUMPTION straight lead-in length upstream of x = 0 [mm]
    D_in=3.0,      # inlet port (biopsy punch) diameter [mm]  (paper)
    D_out=1.5,     # outlet port (biopsy punch) diameter [mm] (paper)
    l_out=1.0,     # ASSUMPTION straight outlet duct length, planar variant [mm]
    H_port=1.0,    # ASSUMPTION modelled height of the vertical port stubs [mm]
                   #   (real punch goes through ~5 mm PDMS)
    n_spline=241,  # number of interpolation points on each hyperbolic wall
)


def w_of_x(x, L, w1):
    return w1 * L / (L - x)


def hyperbola_points(p):
    """Upper-wall points (x, +w/2) from x = 0 to x_t, clustered where the
    curvature is high (near x_t). Mixes uniform-x and uniform-log(w) spacing."""
    L, w1, w_max, n = p["L"], p["w1"], p["w_max"], p["n_spline"]
    x_t = L * (1.0 - w1 / w_max)
    xa = np.linspace(0.0, x_t, n // 2)
    wb = np.geomspace(w1, w_max, n // 2)
    xb = L * (1.0 - w1 / wb)
    x = np.unique(np.round(np.concatenate([xa, xb]), 10))
    x[0], x[-1] = 0.0, x_t
    return x, 0.5 * w_of_x(x, L, w1)


def derived(p):
    L, w1, w_max = p["L"], p["w1"], p["w_max"]
    a = math.radians(p["angle_deg"])
    x_t = L * (1 - w1 / w_max)
    y_t = w_max / 2
    x_apex = x_t + y_t / math.tan(a)          # where the 45-deg walls would meet
    r_out = p["D_out"] / 2
    r_in = p["D_in"] / 2
    # planar variant: 45-deg wall stops where half-width = r_out
    x_trunc = x_t + (y_t - r_out) / math.tan(a)
    # ports variant: outlet circle tangent to both 45-deg walls
    x_co = x_apex - r_out / math.sin(a)
    tang = (x_co + r_out * math.sin(a), r_out * math.cos(a))
    # inlet circle centre so that it meets the lead-in walls at x = -l_in
    x_ci = -p["l_in"] - math.sqrt(r_in ** 2 - (w1 / 2) ** 2)
    return dict(x_t=x_t, y_t=y_t, x_apex=x_apex, x_trunc=x_trunc, x_co=x_co,
                tangent=tang, x_ci=x_ci, r_in=r_in, r_out=r_out,
                slope_at_xt=w1 * L / (L - x_t) ** 2 / 2)


V = lambda x, y, z=0.0: cq.Vector(float(x), float(y), float(z))


def outline_edges(p, variant):
    """Return the ordered list of 2D edges (z = 0) of the closed planform, and a
    list of primitive descriptions for DXF export."""
    d = derived(p)
    xs, ys = hyperbola_points(p)
    w1, l_in = p["w1"], p["l_in"]
    up = [V(x, y) for x, y in zip(xs, ys)]
    lo = [V(x, -y) for x, y in zip(xs[::-1], ys[::-1])]
    E, prim = [], []

    def line(a, b):
        E.append(cq.Edge.makeLine(a, b)); prim.append(("line", (a.x, a.y), (b.x, b.y)))

    def spline(pts):
        E.append(cq.Edge.makeSpline(pts)); prim.append(("spline", [(q.x, q.y) for q in pts]))

    def arc3(a, m, b):
        E.append(cq.Edge.makeThreePointArc(a, m, b)); prim.append(("arc3", (a.x, a.y), (m.x, m.y), (b.x, b.y)))

    x_t, y_t = d["x_t"], d["y_t"]
    if variant in ("planar", "half"):
        x_tr, r = d["x_trunc"], d["r_out"]
        x_end = x_tr + p["l_out"]
        if variant == "planar":
            line(V(-l_in, -w1 / 2), V(-l_in, w1 / 2))          # INLET face trace
            line(V(-l_in, w1 / 2), V(0, w1 / 2))
            spline(up)
            line(V(x_t, y_t), V(x_tr, r))
            line(V(x_tr, r), V(x_end, r))
            line(V(x_end, r), V(x_end, -r))                    # OUTLET face trace
            line(V(x_end, -r), V(x_tr, -r))
            line(V(x_tr, -r), V(x_t, -y_t))
            spline(lo)
            line(V(0, -w1 / 2), V(-l_in, -w1 / 2))
        else:  # half model y >= 0
            line(V(-l_in, 0), V(-l_in, w1 / 2))                # INLET (half)
            line(V(-l_in, w1 / 2), V(0, w1 / 2))
            spline(up)
            line(V(x_t, y_t), V(x_tr, r))
            line(V(x_tr, r), V(x_end, r))
            line(V(x_end, r), V(x_end, 0))                     # OUTLET (half)
            line(V(x_end, 0), V(-l_in, 0))                     # SYMMETRY trace
    elif variant == "ports":
        r_in, x_ci = d["r_in"], d["x_ci"]
        tx, ty = d["tangent"]
        r_out, x_co = d["r_out"], d["x_co"]
        # inlet circle: from lower junction, around the back (x < x_ci), to upper junction
        arc3(V(-l_in, -w1 / 2), V(x_ci - r_in, 0), V(-l_in, w1 / 2))
        line(V(-l_in, w1 / 2), V(0, w1 / 2))
        spline(up)
        line(V(x_t, y_t), V(tx, ty))
        arc3(V(tx, ty), V(x_co + r_out, 0), V(tx, -ty))        # rounded outlet tip
        line(V(tx, -ty), V(x_t, -y_t))
        spline(lo)
        line(V(0, -w1 / 2), V(-l_in, -w1 / 2))
    else:
        raise ValueError(variant)
    return E, prim


def build_solid(p, variant):
    E, prim = outline_edges(p, variant)
    wire = cq.Wire.assembleEdges(E)
    assert wire.IsClosed(), "planform wire is not closed"
    face = cq.Face.makeFromWires(wire)
    solid = cq.Solid.extrudeLinear(face, V(0, 0, p["h"]))
    if variant == "ports":
        d = derived(p)
        cin = cq.Solid.makeCylinder(d["r_in"], p["H_port"], V(d["x_ci"], 0, p["h"]))
        cout = cq.Solid.makeCylinder(d["r_out"], p["H_port"], V(d["x_co"], 0, p["h"]))
        solid = solid.fuse(cin, cout).clean()
        solids = solid.Solids()
        assert len(solids) == 1, "port union produced %d solids" % len(solids)
        solid = solids[0]
    return solid, prim, face.Area()


def write_dxf(prim, path):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4          # millimetres
    doc.header["$MEASUREMENT"] = 1
    doc.layers.add("CHANNEL_BOUNDARY", color=7)
    msp = doc.modelspace()
    att = {"layer": "CHANNEL_BOUNDARY"}
    for item in prim:
        if item[0] == "line":
            msp.add_line(item[1], item[2], dxfattribs=att)
        elif item[0] == "spline":
            msp.add_spline(fit_points=[(x, y, 0) for x, y in item[1]], dxfattribs=att)
        elif item[0] == "arc3":
            a, m, b = (np.array(v) for v in item[1:])
            # circumcentre of the three points
            ax, ay = a; bx, by = m; cx, cy = b
            dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / dd
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / dd
            R = math.hypot(ax - ux, ay - uy)
            ang = lambda q: math.degrees(math.atan2(q[1] - uy, q[0] - ux)) % 360
            s, mid, e = ang(a), ang(m), ang(b)
            # DXF arcs run counter-clockwise from start to end; pick orientation through mid
            ccw = (mid - s) % 360 < (e - s) % 360
            msp.add_arc((ux, uy), R, s if ccw else e, e if ccw else s, dxfattribs=att)
    doc.saveas(path)


def write_csv(p, path, dx=0.1):
    d = derived(p)
    x = np.arange(0.0, d["x_t"] + 1e-9, dx)
    if abs(x[-1] - d["x_t"]) > 1e-9:
        x = np.append(x, d["x_t"])
    w = w_of_x(x, p["L"], p["w1"])
    with open(path, "w") as f:
        f.write("# Hele-Shaw wall coordinates, w(x) = w1*L/(L-x), L=%g mm, w1=%g mm, x_t=%g mm\n"
                % (p["L"], p["w1"], d["x_t"]))
        f.write("x_mm,y_upper_mm,y_lower_mm,w_full_mm,theory_factor_1_minus_x_over_L\n")
        for xi, wi in zip(x, w):
            f.write("%.4f,%.6f,%.6f,%.6f,%.6f\n" % (xi, wi / 2, -wi / 2, wi, 1 - xi / p["L"]))


def analytic_area(p, variant):
    d = derived(p)
    L, w1, w_max, l_in = p["L"], p["w1"], p["w_max"], p["l_in"]
    A_hyp = w1 * L * math.log(w_max / w1)              # integral of w dx, 0..x_t
    A = A_hyp + l_in * w1
    if variant in ("planar", "half"):
        r = d["r_out"]
        A += (w_max / 2 + r) * (d["x_trunc"] - d["x_t"]) + 2 * r * p["l_out"]
    return A / 2 if variant == "half" else A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmax", type=float, default=P["w_max"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    ap.add_argument("--variants", default="planar,ports,half")
    a = ap.parse_args()
    p = dict(P, w_max=a.wmax)
    os.makedirs(a.out, exist_ok=True)
    tag = "HS_L%g_W%g" % (p["L"], p["w_max"])
    d = derived(p)
    summary = dict(parameters=p, derived={k: v for k, v in d.items()}, variants={})
    for var in a.variants.split(","):
        solid, prim, A_face = build_solid(p, var)
        base = os.path.join(a.out, "%s_%s" % (tag, var))
        cq.exporters.export(cq.Workplane().add(solid), base + ".step")
        write_dxf(prim, base + "_planform.dxf")
        bb = solid.BoundingBox()
        summary["variants"][var] = dict(
            step=os.path.basename(base + ".step"), volume_mm3=solid.Volume(),
            planform_area_mm2=A_face, analytic_planform_area_mm2=(analytic_area(p, var) if var != "ports" else None),
            bbox_mm=[bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax], n_faces=len(solid.Faces()))
        print("%-7s V = %.4f mm^3  A = %.4f mm^2  faces = %d  bbox x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]"
              % (var, solid.Volume(), A_face, len(solid.Faces()), bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax))
    write_csv(p, os.path.join(a.out, "%s_wall_coordinates.csv" % tag))
    with open(os.path.join(a.out, "%s_geometry_summary.json" % tag), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary["derived"], indent=1, default=float))


if __name__ == "__main__":
    main()
