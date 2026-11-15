#!/usr/bin/env python3
"""
mesh_independence.py
====================
Grid Convergence Index (GCI) analysis following Celik et al. (2008), "Procedure for
estimation and reporting of uncertainty due to discretization in CFD applications",
J. Fluids Eng. 130, 078001 (Roache's GCI with Fs = 1.25 for three grids).

Input CSV (long format, one row per mesh x quantity) - see templates/mesh_independence_template.csv:
    mesh,n_cells,quantity,value
    M1,115000,tau_fit_S1_dyn,0.0389
    M2,690000,tau_fit_S1_dyn,0.0399
    M3,4100000,tau_fit_S1_dyn,0.0402
    ...
Volume of the domain (for the representative cell size h_rep = (V/N)^(1/3)):  --volume 20.7447e-9 (m^3)

Output: table with p (observed order), extrapolated value, e_a (fine-medium relative
difference), GCI_fine [%], asymptotic-range ratio, convergence type; and a plot.

Run:  python mesh_independence.py templates/mesh_independence_template.csv --volume 20.7447e-9
"""
import argparse
import math
import os

import numpy as np
import pandas as pd


def gci3(N, phi, V, Fs=1.25):
    """N, phi ordered fine -> coarse (N1 > N2 > N3)."""
    h = [(V / n) ** (1 / 3) for n in N]
    r21, r32 = h[1] / h[0], h[2] / h[1]
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    out = dict(h1=h[0], r21=r21, r32=r32, phi1=phi[0], phi2=phi[1], phi3=phi[2])
    if e21 == 0 or e32 == 0:
        out.update(p=np.nan, phi_ext=phi[0], e_a_pct=0.0, GCI_fine_pct=0.0, asym_ratio=np.nan, conv="identical")
        return out
    R = e21 / e32
    conv = "monotonic" if 0 < R < 1 else ("oscillatory" if R < 0 else "divergent")
    s = np.sign(e32 / e21)
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(200):                       # fixed-point iteration for unequal r
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    phi_ext = (r21 ** p * phi[0] - phi[1]) / (r21 ** p - 1)
    e_a = abs((phi[0] - phi[1]) / phi[0])
    gci21 = Fs * e_a / (r21 ** p - 1)
    gci32 = Fs * abs((phi[1] - phi[2]) / phi[1]) / (r32 ** p - 1)
    out.update(p=p, phi_ext=phi_ext, e_a_pct=100 * e_a, e_ext_pct=100 * abs((phi_ext - phi[0]) / phi_ext),
               GCI_fine_pct=100 * gci21, GCI_medium_pct=100 * gci32,
               asym_ratio=gci32 / (r21 ** p * gci21) if gci21 > 0 else np.nan, conv=conv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--volume", type=float, default=20.7447e-9, help="fluid volume [m^3]")
    ap.add_argument("--out", default="mesh_study_out")
    ap.add_argument("--target", type=float, default=2.0, help="acceptable GCI_fine [%%]")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = pd.read_csv(a.csv, comment="#")
    rows = []
    for q, g in D.groupby("quantity"):
        g = g.dropna(subset=["value"]).sort_values("n_cells", ascending=False)
        if len(g) < 3:
            continue
        g = g.iloc[:3]
        r = gci3(g["n_cells"].tolist(), g["value"].tolist(), a.volume)
        r["quantity"] = q
        r["mesh_independent"] = bool(r["conv"] in ("monotonic", "identical") and r.get("GCI_fine_pct", 0) <= a.target)
        rows.append(r)
    R = pd.DataFrame(rows)
    cols = ["quantity", "phi3", "phi2", "phi1", "phi_ext", "p", "conv", "e_a_pct", "GCI_fine_pct", "GCI_medium_pct", "asym_ratio", "mesh_independent"]
    R = R[[c for c in cols if c in R]]
    R.to_csv(os.path.join(a.out, "gci_table.csv"), index=False)
    print(R.round(5).to_string(index=False))
    print("\nCriteria: monotonic convergence, GCI_fine <= %.1f %%, asymptotic ratio ~ 1 (0.95-1.05)." % a.target)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    qs = R["quantity"].tolist()
    n = len(qs); ncol = min(4, n); nrow = int(math.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow), squeeze=False)
    for k, q in enumerate(qs):
        ax = axs[k // ncol][k % ncol]
        g = D[D.quantity == q].sort_values("n_cells")
        hrep = (a.volume / g["n_cells"]) ** (1 / 3) * 1e6
        ax.plot(hrep, g["value"], "o-")
        rr = R[R.quantity == q].iloc[0]
        if np.isfinite(rr["phi_ext"]):
            ax.axhline(rr["phi_ext"], ls="--", color="0.5", lw=0.8)
        ax.set_title("%s\nGCI$_{fine}$ = %.2f %%" % (q, rr.get("GCI_fine_pct", np.nan)), fontsize=7)
        ax.set_xlabel("h$_{rep}$ [µm]", fontsize=7); ax.tick_params(labelsize=6); ax.invert_xaxis()
    for k in range(n, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "gci_plots.png"), dpi=250)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
