#!/usr/bin/env python3
"""
validate_shear.py
=================
Systematic validation of the Fluent hydrodynamic solution against the analytical
Hele-Shaw relations of Delon et al. (2020), Eq. (2), for the PROJECT geometry.

Inputs (Fluent ASCII exports written by the journals, coordinates in metres):
  --zrakes  exp_<tag>_zrakes.csv      u(z) lines at x = 5,15,25,35,45 mm on the axis
  --yrakes  exp_<tag>_yrakes_z20.csv  across-width lines at z = 20 um (optional)
  --wall    exp_<tag>_wall_bottom.csv wall-shear on z = 0 (optional, recommended)
  --hs2d    ../02_analytical/output/hs2d_W10_Q1_centreline.csv (optional, 2D model)
  --Q 1.0   flow rate [uL/min] of THIS run;   --half if the half model was simulated

What it computes at every station
  * tau_fit   : mu * du/dz|wall from a least-squares fit u = a1 z + a2 z^2 + a3 z^3
                (no-slip built in) - bottom AND top wall
  * tau_flu   : Fluent's own wall-shear on the bottom wall near the axis
  * tau_1D    : 6 mu Q / (h^2 w1) (1 - x/L)
  * tau(20um) : mu * strain-rate at z = 20 um (CFD) vs tau_1D (1 - 2z/h)
  * tau_cell  : 3 tau_w (Gaver & Kute approximation used in the paper)
  * percentage deviations, and the uniform-shear half-width d(eps) across y
Outputs: CSV tables + PNG/PDF figures (Fig. 2, 3, 4C analogues) in --out.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_analytical"))
import hele_shaw_theory as th  # noqa: E402
from fluent_io import read_ascii, group_by  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
                     "savefig.dpi": 300, "figure.dpi": 110, "legend.frameon": False})
STATION_COLORS = ["#1f4e79", "#2e86c1", "#17a589", "#d68910", "#c0392b"]


def fit_wall_gradient(z, u, h, deg=3):
    """Least-squares fit of u(z) = sum_k a_k z^k, k = 1..deg (u(0) = 0 imposed).
    Returns du/dz at z = 0 and at z = h (from the mirrored fit)."""
    m = (z >= 0) & (z <= h) & np.isfinite(u)
    z, u = z[m], u[m]
    A = np.column_stack([z ** k for k in range(1, deg + 1)])
    a_bot = np.linalg.lstsq(A, u, rcond=None)[0]
    zt = h - z
    At = np.column_stack([zt ** k for k in range(1, deg + 1)])
    a_top = np.linalg.lstsq(At, u, rcond=None)[0]
    resid = np.sqrt(np.mean((A @ a_bot - u) ** 2)) / max(np.max(np.abs(u)), 1e-30)
    return a_bot[0], a_top[0], resid


def uniform_halfwidth(y, tau, eps):
    """Contiguous distance from the axis over which |tau - tau(0)|/tau(0) <= eps,
    evaluated separately on +y and -y; returns (d_plus, d_minus)."""
    o = np.argsort(y); y, tau = y[o], tau[o]
    t0 = np.interp(0.0, y, tau)
    res = []
    for side in (+1, -1):
        sel = (side * y) >= 0
        ys, ts = np.abs(y[sel]), tau[sel]
        o2 = np.argsort(ys); ys, ts = ys[o2], ts[o2]
        bad = np.where(np.abs(ts - t0) / t0 > eps)[0]
        res.append(ys[-1] if len(bad) == 0 else (ys[bad[0] - 1] if bad[0] > 0 else 0.0))
    return res[0], res[1], t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zrakes", required=True)
    ap.add_argument("--yrakes")
    ap.add_argument("--wall")
    ap.add_argument("--hs2d")
    ap.add_argument("--hs2d_Q", type=float, default=1.0, help="flow rate [uL/min] the hs2d file was computed for")
    ap.add_argument("--Q", type=float, required=True, help="uL/min")
    ap.add_argument("--L", type=float, default=50e-3)
    ap.add_argument("--w1", type=float, default=1e-3)
    ap.add_argument("--h", type=float, default=145e-6)
    ap.add_argument("--mu", type=float, default=th.MU)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="validation_out")
    ap.add_argument("--label", default="CFD (this project)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = dict(L=a.L, w1=a.w1, h=a.h, w_max=10e-3)
    Q = th.ulmin_to_m3s(a.Q)
    mu = a.mu
    stations = th.STATIONS_M

    # ------------------------------------------------------------------ z-profiles
    zr = group_by(read_ascii(a.zrakes), "x-coordinate", 6)
    rows, prof = [], {}
    for xs in stations:
        key = min(zr, key=lambda k: abs(k - xs))
        if abs(key - xs) > 1e-5:
            print("WARNING: no z-rake found near x = %.1f mm" % (xs * 1e3)); continue
        d = zr[key].sort_values("z-coordinate")
        z, u = d["z-coordinate"].values, d["x-velocity"].values
        g_bot, g_top, resid = fit_wall_gradient(z, u, a.h)
        sr = d["strain-rate-mag"].values if "strain-rate-mag" in d else np.abs(np.gradient(u, z))
        tau20_cfd = mu * np.interp(th.Z_CELL, z, sr)
        t1 = th.tau_wall(xs, Q, g, mu)
        r = dict(x_mm=xs * 1e3, w_mm=th.width(xs, g) * 1e3,
                 umax_cfd_mm_s=np.nanmax(u) * 1e3, umax_1D_mm_s=1.5 * th.mean_velocity(xs, Q, g) * 1e3,
                 tau_1D_Pa=t1, tau_fit_bottom_Pa=mu * g_bot, tau_fit_top_Pa=mu * g_top, fit_rel_rms=resid,
                 tau20_1D_Pa=th.tau_at_z(xs, th.Z_CELL, Q, g, mu), tau20_cfd_Pa=tau20_cfd)
        prof[xs] = (z, u, sr)
        rows.append(r)
    T = pd.DataFrame(rows)

    # ------------------------------------------------------------- wall export
    W = None
    if a.wall:
        W = read_ascii(a.wall)
        col = "wall-shear" if "wall-shear" in W else [c for c in W.columns if "shear" in c][0]
        vals = []
        for xs in stations:
            b = th.width(xs, g) / 2
            m = (np.abs(W["x-coordinate"] - xs) < 0.15e-3) & (np.abs(W["y-coordinate"]) < max(0.05 * b, 30e-6))
            vals.append(W.loc[m, col].abs().mean() if m.any() else np.nan)
        T["tau_fluent_wall_Pa"] = vals

    if a.hs2d and os.path.exists(a.hs2d):
        H2 = pd.read_csv(a.hs2d)
        scale = a.Q / a.hs2d_Q   # the 2D model is linear in Q
        T["tau_2D_Pa"] = np.interp(T["x_mm"], H2["x_mm"], H2["tau_2D_axis_dyn_cm2"]) / 10 * scale

    for c in [c for c in T.columns if c.startswith("tau_") and c != "tau_1D_Pa" and c.endswith("_Pa") and "20" not in c]:
        T["dev_%s_pct" % c[4:-3]] = 100 * (T[c] / T["tau_1D_Pa"] - 1)
    T["dev_tau20_pct"] = 100 * (T["tau20_cfd_Pa"] / T["tau20_1D_Pa"] - 1)
    for c in [c for c in T.columns if c.endswith("_Pa")]:
        T[c.replace("_Pa", "_dyn_cm2")] = T[c] * 10
    T["tau_cell_1D_dyn_cm2"] = 3 * T["tau_1D_dyn_cm2"]
    T["tau_cell_cfd_dyn_cm2"] = 3 * T["tau_fit_bottom_dyn_cm2"]
    T.to_csv(os.path.join(a.out, "%s_station_validation.csv" % a.tag), index=False)
    pd.set_option("display.width", 200)
    print(T[["x_mm", "tau_1D_dyn_cm2", "tau_fit_bottom_dyn_cm2"] + (["tau_fluent_wall_dyn_cm2"] if a.wall else [])
            + [c for c in T.columns if c.startswith("dev_")]].round(5).to_string(index=False))

    # ------------------------------------------------------------- figures
    # Fig A: velocity profiles u(z) (paper Fig. S1/S3 analogue)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for k, xs in enumerate(stations):
        if xs not in prof: continue
        z, u, sr = prof[xs]
        zz = np.linspace(0, a.h, 100)
        ax[0].plot(u * 1e3, z * 1e6, "o", ms=2.5, color=STATION_COLORS[k], label="x = %g mm" % (xs * 1e3))
        ax[0].plot(th.u_profile(xs, zz, Q, g) * 1e3, zz * 1e6, "-", lw=0.9, color=STATION_COLORS[k])
        ax[1].plot(sr, z * 1e6, "-", color=STATION_COLORS[k], label="x = %g mm" % (xs * 1e3))
        ax[1].plot(np.abs(6 * th.mean_velocity(xs, Q, g) / a.h * (1 - 2 * zz / a.h)), zz * 1e6, ":", lw=0.9, color=STATION_COLORS[k])
    ax[0].set_xlabel("u [mm s$^{-1}$]"); ax[0].set_ylabel("z [µm]")
    ax[0].set_title("Velocity across height (markers %s, lines 1D theory)" % a.label, fontsize=8)
    ax[1].set_xlabel("strain rate [s$^{-1}$]"); ax[1].set_ylabel("z [µm]"); ax[1].legend(fontsize=7)
    ax[1].set_title("Strain rate along z (Fig. 2B/D analogue)", fontsize=8)
    fig.tight_layout(); [fig.savefig(os.path.join(a.out, "%s_fig_profiles_z.%s" % (a.tag, e))) for e in ("png", "pdf")]

    # Fig B: tau_w and tau(20 um) vs x (Fig. 2C/E and Fig. 3A analogue)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    xx = np.linspace(0, 45e-3, 200)
    ax[0].plot(xx * 1e3, th.tau_wall(xx, Q, g, mu) * 10, "k-", label="Eq. 2 (theory)")
    ax[0].plot(xx * 1e3, th.tau_at_z(xx, th.Z_CELL, Q, g, mu) * 10, "k--", label="theory, z = 20 µm")
    ax[0].plot(T["x_mm"], T["tau_fit_bottom_dyn_cm2"], "o", color="#c0392b", label="%s, wall (profile fit)" % a.label)
    if a.wall:
        ax[0].plot(T["x_mm"], T["tau_fluent_wall_dyn_cm2"], "s", mfc="none", color="#c0392b", label="%s, Fluent wall-shear" % a.label)
        col = "wall-shear" if "wall-shear" in W else [c for c in W.columns if "shear" in c][0]
        bmask = np.abs(W["y-coordinate"]) < 30e-6
        Wa = W[bmask].sort_values("x-coordinate")
        ax[0].plot(Wa["x-coordinate"] * 1e3, Wa[col].abs() * 10, "-", lw=0.6, color="#e6b0aa", label="Fluent wall-shear on axis")
    ax[0].plot(T["x_mm"], T["tau20_cfd_dyn_cm2"], "^", color="#2e86c1", label="%s, z = 20 µm" % a.label)
    if "tau_2D_dyn_cm2" in T:
        ax[0].plot(T["x_mm"], T["tau_2D_dyn_cm2"], "x", color="0.3", label="2D Hele-Shaw model")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel(r"$\tau$ [dyn cm$^{-2}$]"); ax[0].legend(fontsize=6)
    ax[0].set_title("Q = %g µL/min" % a.Q, fontsize=8)
    ax[1].plot(xx * 1e3, 3 * th.tau_wall(xx, Q, g, mu) * 10, "k-", label=r"theory, $\tau_{cell}=3\tau_w$")
    ax[1].plot(T["x_mm"], T["tau_cell_cfd_dyn_cm2"], "o", color="#c0392b", label=a.label)
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel(r"$\tau_{cell}$ [dyn cm$^{-2}$]"); ax[1].legend(fontsize=7)
    ax[1].set_title("Cell FSS (Fig. 3B analogue)", fontsize=8)
    fig.tight_layout(); [fig.savefig(os.path.join(a.out, "%s_fig_tau_vs_x.%s" % (a.tag, e))) for e in ("png", "pdf")]

    # ------------------------------------------------- shear uniformity across y
    if a.yrakes:
        Y = group_by(read_ascii(a.yrakes), "x-coordinate", 6)
        drows = []
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
        for k, xs in enumerate(stations):
            key = min(Y, key=lambda kk: abs(kk - xs))
            d = Y[key].sort_values("y-coordinate")
            y = d["y-coordinate"].values
            tau = mu * (d["strain-rate-mag"].values if "strain-rate-mag" in d else np.abs(d["x-velocity"].values))
            b = th.width(xs, g) / 2 if xs < 45e-3 else 5e-3
            ax[0].plot(y * 1e3, tau * 10, color=STATION_COLORS[k], label="x = %g mm" % (xs * 1e3))
            ax[1].plot(y / b, tau / np.interp(0, y, tau), color=STATION_COLORS[k])
            for eps in (0.01, 0.05, 0.10):
                dp, dm, t0 = uniform_halfwidth(y, tau, eps)
                dd = min(dp, dm)
                inside = np.abs(y) <= dd
                drows.append(dict(x_mm=xs * 1e3, eps=eps, half_width_mm=b * 1e3, d_plus_mm=dp * 1e3, d_minus_mm=dm * 1e3,
                                  Dmax_mm=dd * 1e3, Dwall_mm=(b - dd) * 1e3, d_over_halfwidth=dd / b,
                                  mean_strain_rate_s=np.mean(tau[inside]) / mu, mean_tau_dyn_cm2=np.mean(tau[inside]) * 10,
                                  std_tau_dyn_cm2=np.std(tau[inside]) * 10))
        ax[1].axhspan(0.95, 1.05, color="0.92", zorder=0)
        ax[0].set_xlabel("y [mm]"); ax[0].set_ylabel(r"$\tau$(z = 20 µm) [dyn cm$^{-2}$]"); ax[0].legend(fontsize=7)
        ax[0].set_title("Shear across the width at z = 20 µm (Fig. 4C analogue)", fontsize=8)
        ax[1].set_xlabel("y / (w/2)"); ax[1].set_ylabel(r"$\tau(y)/\tau(0)$"); ax[1].set_title("Normalised; band = ±5 %", fontsize=8)
        fig.tight_layout(); [fig.savefig(os.path.join(a.out, "%s_fig_uniformity_y.%s" % (a.tag, e))) for e in ("png", "pdf")]
        D = pd.DataFrame(drows)
        D.to_csv(os.path.join(a.out, "%s_uniform_width_table.csv" % a.tag), index=False)
        print("\nUniform-shear region (Table 2 analogue):")
        print(D[D.eps == 0.05][["x_mm", "half_width_mm", "Dmax_mm", "Dwall_mm", "d_over_halfwidth", "mean_tau_dyn_cm2", "std_tau_dyn_cm2"]].round(4).to_string(index=False))

    # mesh-study row (umax, tau at stations) for mesh_independence.py
    mq = [("umax_S1_mm_s", T.loc[0, "umax_cfd_mm_s"])] + [("tau_fit_S%d_dyn" % (i + 1), v) for i, v in enumerate(T["tau_fit_bottom_dyn_cm2"])]
    if a.wall:
        mq += [("tau_fluent_S%d_dyn" % (i + 1), v) for i, v in enumerate(T["tau_fluent_wall_dyn_cm2"])]
    pd.DataFrame(mq, columns=["quantity", "value"]).to_csv(os.path.join(a.out, "%s_mesh_quantities.csv" % a.tag), index=False)
    print("\nwrote results to", a.out)


if __name__ == "__main__":
    main()
