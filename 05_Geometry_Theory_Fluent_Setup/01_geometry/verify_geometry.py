#!/usr/bin/env python3
"""
verify_geometry.py
==================
Independent check of a STEP fluid domain produced by generate_geometry.py
(or exported from SpaceClaim). It re-imports the STEP file and checks:

  1. exactly one solid; BRepCheck validity
  2. watertightness: closed shell, no free (boundary) edges
  3. manifoldness: every edge shared by exactly two faces; no duplicate faces
  4. tight bounding box -> catches mm/m scaling errors
  5. volume vs analytical value  V = h * [ w1*L*ln(wmax/w1) + lead-in + outlet ]
  6. deviation of the curved side walls from the exact hyperbola w(x)/2
  7. identification of INLET / OUTLET / SYMMETRY planar faces and their areas
  8. optional: a PNG plan view for visual inspection

Usage:
  python verify_geometry.py output/HS_L50_W10_planar.step --L 50 --w1 1 --h 0.145 --wmax 10 --variant planar --png
Exit code 0 = all checks passed, 1 = at least one FAIL.
Requires: cadquery (OCP), numpy, matplotlib (only for --png)
"""
import argparse
import math
import sys

import numpy as np
import cadquery as cq
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopoDS import TopoDS

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(ok)
    print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, info))


def expected_volume(a):
    A = a.w1 * a.L * math.log(a.wmax / a.w1) + a.l_in * a.w1
    x_t = a.L * (1 - a.w1 / a.wmax)
    r_out, r_in = a.D_out / 2, a.D_in / 2
    if a.variant in ("planar", "half"):
        x_tr = x_t + (a.wmax / 2 - r_out)
        A += (a.wmax / 2 + r_out) * (x_tr - x_t) + 2 * r_out * a.l_out
        return (A / 2 if a.variant == "half" else A) * a.h
    return None  # 'ports' volume is reported, compared with the generator summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--L", type=float, default=50.0)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.145)
    ap.add_argument("--wmax", type=float, default=10.0)
    ap.add_argument("--l_in", type=float, default=2.0)
    ap.add_argument("--l_out", type=float, default=1.0)
    ap.add_argument("--D_in", type=float, default=3.0)
    ap.add_argument("--D_out", type=float, default=1.5)
    ap.add_argument("--variant", default="planar", choices=["planar", "ports", "half"])
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()

    shape = cq.importers.importStep(a.step).val()
    solids = shape.Solids()
    check("single solid body", len(solids) == 1, "n_solids = %d" % len(solids))
    s = solids[0]
    occ = s.wrapped
    check("BRepCheck_Analyzer valid", BRepCheck_Analyzer(occ).IsValid())

    shells = s.Shells()
    check("one closed shell (watertight)", len(shells) == 1 and shells[0].Closed(),
          "n_shells = %d" % len(shells))
    fb = ShapeAnalysis_FreeBounds(occ)
    n_free = 0
    for comp in (fb.GetClosedWires(), fb.GetOpenWires()):   # null compound = no free edges
        if not comp.IsNull():
            n_free += len(cq.Shape.cast(comp).Edges())
    check("no free/naked edges", n_free == 0, "free edges = %d" % n_free)

    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(occ, TopAbs_EDGE, TopAbs_FACE, emap)
    counts = [emap.FindFromIndex(i).Size() for i in range(1, emap.Extent() + 1)]
    bad = sum(c != 2 for c in counts)
    check("manifold: every edge has exactly 2 faces", bad == 0,
          "edges = %d, non-manifold = %d" % (len(counts), bad))

    faces = s.Faces()
    cents = np.array([f.Center().toTuple() + (f.Area(),) for f in faces])
    dup = 0
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            if np.allclose(cents[i], cents[j], atol=1e-9):
                dup += 1
    check("no duplicate faces", dup == 0, "faces = %d" % len(faces))

    box = Bnd_Box(); BRepBndLib.AddOptimal_s(occ, box, False, False)
    x0, y0, z0, x1, y1, z1 = box.Get()
    print("      tight bbox [mm]: x %.4f .. %.4f | y %.4f .. %.4f | z %.4f .. %.4f" % (x0, x1, y0, y1, z0, z1))
    zt = a.h if a.variant != "ports" else None
    if zt is not None:
        check("channel height = h (z extent)", abs((z1 - z0) - a.h) < 1e-6, "dz = %.6f mm" % (z1 - z0))
    check("width extent = wmax (mm units, no scaling)",
          abs((y1 - y0) - (a.wmax if a.variant != "half" else a.wmax / 2)) < 1e-3, "dy = %.5f mm" % (y1 - y0))
    check("length in expected range (40-70 mm)", 40 < (x1 - x0) < 70, "dx = %.4f mm" % (x1 - x0))

    V = s.Volume()
    Ve = expected_volume(a)
    if Ve:
        check("volume vs analytical", abs(V - Ve) / Ve < 1e-5,
              "V = %.6f mm^3, analytical = %.6f mm^3 (%.1e rel.)" % (V, Ve, abs(V - Ve) / Ve))
    else:
        print("      volume = %.6f mm^3 (ports variant; compare with geometry_summary.json)" % V)
    print("      volume = %.4f uL" % V)

    # curved wall deviation from exact hyperbola (sample bottom edges of the curved faces)
    x_t = a.L * (1 - a.w1 / a.wmax)
    maxdev = 0.0
    for f in faces:
        if f.geomType() != "PLANE":
            for e in f.Edges():
                if e.geomType() == "BSPLINE":
                    for t in np.linspace(0, 1, 400):
                        p = e.positionAt(t)
                        if 0 <= p.x <= x_t:
                            maxdev = max(maxdev, abs(abs(p.y) - 0.5 * a.w1 * a.L / (a.L - p.x)))
    check("hyperbolic walls follow w(x)/2 (< 0.1 um)", maxdev < 1e-4, "max |dev| = %.2e mm" % maxdev)

    # planar boundary faces
    print("      planar faces with normals along x or y (candidate boundaries):")
    for i, f in enumerate(faces):
        if f.geomType() == "PLANE":
            n = f.normalAt(); c = f.Center()
            lab = ""
            if abs(abs(n.x) - 1) < 1e-9:
                lab = "INLET" if c.x < 0 else "OUTLET"
            elif abs(abs(n.y) - 1) < 1e-9 and abs(c.y) < 1e-9:
                lab = "SYMMETRY"
            elif abs(abs(n.z) - 1) < 1e-9 and c.z > a.h + 1e-6:
                lab = "INLET (port top)" if c.x < 0 else "OUTLET (port top)"
            if lab:
                print("        face %2d  %-18s centre (%.4f, %.4f, %.4f)  area %.6f mm^2" % (i, lab, c.x, c.y, c.z, f.Area()))
    if a.variant == "planar":
        inl = [f for f in faces if f.geomType() == "PLANE" and abs(f.normalAt().x + 1) < 1e-9]
        check("inlet face area = w1*h", len(inl) == 1 and abs(inl[0].Area() - a.w1 * a.h) < 1e-9,
              "%.6f mm^2" % (inl[0].Area() if inl else -1))

    if a.png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 3.2))
        bottom = [f for f in faces if f.geomType() == "PLANE" and abs(f.normalAt().z + 1) < 1e-9 and abs(f.Center().z) < 1e-9]
        for f in bottom:
            for e in f.Edges():
                pts = np.array([e.positionAt(t).toTuple() for t in np.linspace(0, 1, 300)])
                ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1)
        xx = np.linspace(0, x_t, 400)
        ax.plot(xx, 0.5 * a.w1 * a.L / (a.L - xx), "r--", lw=0.8, label="exact w(x)/2")
        for xs in (5, 15, 25, 35, 45):
            ax.axvline(xs, color="tab:blue", lw=0.5, ls=":")
        ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title("%s - plan view at z = 0 (blue: x = 5, 15, 25, 35, 45 mm)" % a.step.split("/")[-1], fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); out = a.step.rsplit(".", 1)[0] + "_planview.png"
        fig.savefig(out, dpi=200); print("      wrote", out)

    print("\nSUMMARY: %d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
