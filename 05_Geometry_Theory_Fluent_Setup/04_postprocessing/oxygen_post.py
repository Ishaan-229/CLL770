#!/usr/bin/env python3
"""
oxygen_post.py
==============
Post-processing of the Fluent oxygen runs (UDS uds-0 = C [mol/m3]).

Input: one or more bottom-wall ASCII exports containing x/y/z-coordinate, wall-shear
and uds-0-scalar, each given as LABEL=path, e.g.
  python oxygen_post.py "0.22 IMP=exp_O_M2_Q022_IMP_wall_bottom.csv" "1 IMP=exp_O_M2_Q1_IMP_wall_bottom.csv" ...
Options: --csat 0.178  --ccrit 0.0178  --out oxygen_out

Produces
  * map of C_w/C_sat on the cell-covered wall (one panel per run)
  * C_w/C_sat along the central axis, and across the width at x = 5..45 mm
  * C_w vs wall shear (relationship between shear and oxygen availability)
  * table: area-mean C_w, min C_w, hypoxic area fraction for several thresholds
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fluent_io import read_ascii

STATIONS = np.array([5, 15, 25, 35, 45]) * 1e-3


def node_area_weights(x, y):
    """Approximate area weight of each wall node from a Delaunay triangulation of the
    planform (each triangle gives 1/3 of its area to each vertex). Triangles spanning
    outside the concave planform are removed by an edge-length filter."""
    tri = mtri.Triangulation(x, y)
    t = tri.triangles
    p = np.column_stack([x, y])
    e = np.max([np.linalg.norm(p[t[:, i]] - p[t[:, (i + 1) % 3]], axis=1) for i in range(3)], axis=0)
    keep = e < 5 * np.median(e)
    t = t[keep]
    A = 0.5 * np.abs((p[t[:, 1], 0] - p[t[:, 0], 0]) * (p[t[:, 2], 1] - p[t[:, 0], 1]) -
                     (p[t[:, 2], 0] - p[t[:, 0], 0]) * (p[t[:, 1], 1] - p[t[:, 0], 1]))
    wgt = np.bincount(t.ravel(), np.repeat(A / 3, 3), len(x))
    tri.set_mask(~keep)
    return wgt, tri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help='"LABEL=path_to_wall_bottom_export.csv"')
    ap.add_argument("--csat", type=float, default=0.178)
    ap.add_argument("--ccrit", type=float, default=0.0178)
    ap.add_argument("--xmin", type=float, default=0.0, help="start of cell-covered region [m]")
    ap.add_argument("--out", default="oxygen_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for r in a.runs:
        lab, path = r.split("=", 1)
        D = read_ascii(path)
        D = D[D["x-coordinate"] >= a.xmin]
        if "uds-0-scalar" not in D:
            raise SystemExit("%s has no uds-0-scalar column" % path)
        runs.append((lab, D))

    n = len(runs)
    fig, axs = plt.subplots(n, 1, figsize=(9, 1.9 * n + 0.4), squeeze=False)
    rows = []
    for k, (lab, D) in enumerate(runs):
        x, y = D["x-coordinate"].values, D["y-coordinate"].values
        c = D["uds-0-scalar"].values / a.csat
        wgt, tri = node_area_weights(x, y)
        ax = axs[k][0]
        cs = ax.tricontourf(tri, c, levels=np.linspace(0, 1, 21), cmap="RdYlBu", extend="both")
        ax.set_aspect("equal"); ax.set_ylabel("y [m]", fontsize=7); ax.set_title(lab, fontsize=8); ax.tick_params(labelsize=6)
        r = dict(run=lab, area_mm2=wgt.sum() * 1e6, mean_C_over_Csat=np.sum(c * wgt) / wgt.sum(), min_C_over_Csat=c.min())
        for thr in (0.5, 0.2, 0.1, 0.05):
            r["area_frac_below_%g" % thr] = wgt[c < thr].sum() / wgt.sum()
        r["area_frac_below_ccrit"] = wgt[c * a.csat < a.ccrit].sum() / wgt.sum()
        rows.append(r)
    fig.colorbar(cs, ax=axs.ravel().tolist(), label=r"$C_w/C_{sat}$ on cell wall")
    [fig.savefig(os.path.join(a.out, "o2_wall_maps.%s" % e), dpi=250) for e in ("png", "pdf")]
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "o2_summary.csv"), index=False)
    print(T.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for lab, D in runs:
        c = D["uds-0-scalar"] / a.csat
        ax_m = np.abs(D["y-coordinate"]) < 30e-6
        A = D[ax_m].sort_values("x-coordinate")
        ax[0].plot(A["x-coordinate"] * 1e3, A["uds-0-scalar"] / a.csat, label=lab)
        if "wall-shear" in D:
            ax[2].plot(D["wall-shear"].abs() * 10, c, ".", ms=1, alpha=0.4, label=lab)
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel(r"$C_w/C_{sat}$ on axis"); ax[0].legend(fontsize=6)
    ax[0].axhline(a.ccrit / a.csat, color="k", lw=0.6, ls=":")
    lab0, D0 = runs[0]
    for xs in STATIONS:
        m = np.abs(D0["x-coordinate"] - xs) < 0.15e-3
        S = D0[m].sort_values("y-coordinate")
        ax[1].plot(S["y-coordinate"] * 1e3, S["uds-0-scalar"] / a.csat, label="x = %g mm" % (xs * 1e3))
    ax[1].set_xlabel("y [mm]"); ax[1].set_ylabel(r"$C_w/C_{sat}$"); ax[1].set_title("across width: " + lab0, fontsize=8); ax[1].legend(fontsize=6)
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"$\tau_w$ [dyn cm$^{-2}$]"); ax[2].set_ylabel(r"$C_w/C_{sat}$")
    ax[2].set_title("shear vs oxygen at the monolayer", fontsize=8)
    fig.tight_layout(); [fig.savefig(os.path.join(a.out, "o2_profiles.%s" % e), dpi=250) for e in ("png", "pdf")]
    print("wrote", a.out)


if __name__ == "__main__":
    main()
