#!/usr/bin/env python3
"""
hele_shaw_theory.py
===================
Analytical (1D, parallel-plate Poiseuille / Hele-Shaw) relations used by Delon et al.
(2020) Eq. (2), in consistent SI units, plus the numbers needed to set up Fluent.

    tau_w(x) = 6 mu Q / (h^2 w1) * (1 - x/L)                       [Pa]
    u(z)     = 6 U (z/h)(1 - z/h),  U = Q / (w(x) h)                 [m/s]
    tau(z)   = mu du/dz = tau_w (1 - 2 z / h)                        [Pa]
    dp/dx    = -12 mu Q / (h^3 w(x))                                 [Pa/m]
    tau_cell = 3 tau_w   (Gaver & Kute 1998 approximation used by Delon et al.)

Import as a module (used by the post-processing and design scripts) or run:
    python hele_shaw_theory.py            -> prints tables, writes CSV + figure
1 Pa = 10 dyn/cm^2.
"""
import math
import os

import numpy as np

# ---------------- fluid properties (Delon et al. 2020, 37 C culture medium) -----------
MU = 0.94e-3          # Pa s     (0.94 cP)
RHO = 981.8           # kg/m^3   (0.9818 g/cm^3)

# ---------------- geometries ---------------------------------------------------------
GEOM_PROJECT = dict(name="Project (L=50 mm, h=145 um)", L=50e-3, w1=1e-3, h=145e-6, w_max=10e-3)
GEOM_PAPER_EQ = dict(name="Paper Eq.2 (L=53 mm, h=145 um)", L=53e-3, w1=1e-3, h=145e-6, w_max=10e-3)
GEOM_PAPER_CFD = dict(name="Paper CFD height (L=53 mm, h=150 um)", L=53e-3, w1=1e-3, h=150e-6, w_max=10e-3)

FLOWS_UL_MIN = [0.22, 1.0, 120.0]
STATIONS_M = np.array([5, 15, 25, 35, 45]) * 1e-3     # S1..S5 (x = 0.5..4.5 cm)
Z_CELL = 20e-6                                       # 'cell height' used by Delon et al.


def ulmin_to_m3s(q):
    return q * 1e-9 / 60.0


def width(x, g):
    return g["w1"] * g["L"] / (g["L"] - x)


def mean_velocity(x, Q, g):
    return Q / (width(x, g) * g["h"])


def tau_wall(x, Q, g, mu=MU):
    return 6.0 * mu * Q / (g["h"] ** 2 * g["w1"]) * (1.0 - x / g["L"])


def tau_at_z(x, z, Q, g, mu=MU):
    return tau_wall(x, Q, g, mu) * (1.0 - 2.0 * z / g["h"])


def u_profile(x, z, Q, g):
    U = mean_velocity(x, Q, g)
    return 6.0 * U * (z / g["h"]) * (1.0 - z / g["h"])


def dpdx(x, Q, g, mu=MU):
    return -12.0 * mu * Q / (g["h"] ** 3 * width(x, g))


def pressure_drop_hyperbolic(Q, g, x_end=None, mu=MU):
    """Delta p over 0..x_end of the hyperbolic section (closed form)."""
    x_end = g["L"] * (1 - g["w1"] / g["w_max"]) if x_end is None else x_end
    # int dx / w = (1/(w1 L)) (L x - x^2/2)
    return 12 * mu * Q / (g["h"] ** 3) * (g["L"] * x_end - x_end ** 2 / 2) / (g["w1"] * g["L"])


def reynolds(Q, g, x=0.0, rho=RHO, mu=MU):
    """Re based on hydraulic diameter of the local rectangular section, and the
    reduced (Hele-Shaw) Reynolds number Re* = Re_h * (h / w) commonly used for
    thin-gap flows."""
    w = width(x, g)
    U = Q / (w * g["h"])
    Dh = 2 * w * g["h"] / (w + g["h"])
    return dict(U=U, Dh=Dh, Re_Dh=rho * U * Dh / mu, Re_h=rho * U * g["h"] / mu,
                Re_star=rho * U * g["h"] / mu * g["h"] / w)


def development_lengths(Q, g, x=0.0, rho=RHO, mu=MU):
    r = reynolds(Q, g, x, rho, mu)
    return dict(L_dev_m=(0.6 + 0.05 * r["Re_h"]) * g["h"],      # Durst-type estimate for parallel plates
                t_visc_s=g["h"] ** 2 * rho / mu)                  # momentum diffusion time over h


def aspect_flux_factor(h, w, nterms=50):
    """Exact flux factor of fully developed laminar flow in a w x h rectangle relative to
    infinite parallel plates:  Q_rect / Q_pp = 1 - (192 h / (pi^5 w)) * sum_{n odd} tanh(n pi w / 2h) / n^5
    (~ 1 - 0.630 h/w for w >> h)."""
    n = np.arange(1, 2 * nterms, 2)[:, None]
    w = np.atleast_1d(np.asarray(w, float))[None, :]
    s = (np.tanh(n * np.pi * w / (2 * h)) / n ** 5).sum(0)
    return np.squeeze(1 - 192 * h / (np.pi ** 5 * w) * s)


def tau_wall_axis_3d(x, Q, g, mu=MU):
    """Centre-line wall shear including the side-wall flux deficit of the local
    rectangular section (lubrication approximation, slowly varying w):
    tau_axis = tau_1D / f(h/w). Expected to sit ABOVE Eq. 2 by ~0.63 h/w."""
    return tau_wall(x, Q, g, mu) / aspect_flux_factor(g["h"], width(x, g))


def sidewall_layer(h, eps):
    """Distance from a side wall over which the bottom-wall shear is more than eps below
    its core value (leading term of the rectangular-duct series):
    tau/tau_core = 1 - (8/pi^2) exp(-pi*delta/h)  ->  delta_eps = (h/pi) ln(8/(pi^2 eps))."""
    return h / np.pi * np.log(8 / (np.pi ** 2 * eps))


def uniform_halfwidth_estimate(x, g, eps=0.05):
    """d(eps) ~ w(x)/2 - delta_eps : side-wall-limited estimate of the uniform-shear half-width
    (valid where the 2D Hele-Shaw field is uniform across y, i.e. S1-S4)."""
    return np.maximum(width(x, g) / 2 - sidewall_layer(g["h"], eps), 0.0)


def boundary_condition_table(g=GEOM_PROJECT, rho=RHO):
    rows = []
    for q in FLOWS_UL_MIN:
        Q = ulmin_to_m3s(q)
        rows.append(dict(Q_uL_min=q, Q_m3_s=Q, mdot_full_kg_s=rho * Q, mdot_half_kg_s=rho * Q / 2,
                         **reynolds(Q, g)))
    return rows


def station_table(Q, g, mu=MU):
    out = []
    for x in STATIONS_M:
        tw = tau_wall(x, Q, g, mu)
        out.append(dict(x_mm=x * 1e3, w_mm=width(x, g) * 1e3, U_mm_s=mean_velocity(x, Q, g) * 1e3,
                        umax_mm_s=1.5 * mean_velocity(x, Q, g) * 1e3,
                        tau_w_Pa=tw, tau_w_dyn=tw * 10, tau_z20_dyn=tau_at_z(x, Z_CELL, Q, g, mu) * 10,
                        tau_cell_dyn=3 * tw * 10, wall_halfslope=0.5 * g["w1"] * g["L"] / (g["L"] - x) ** 2,
                        tau_axis3d_dyn=float(tau_wall_axis_3d(x, Q, g, mu)) * 10,
                        aspect_corr_pct=100 * (1 / float(aspect_flux_factor(g["h"], width(x, g))) - 1),
                        d5_est_mm=float(uniform_halfwidth_estimate(x, g, 0.05)) * 1e3))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "output"); os.makedirs(out, exist_ok=True)
    g = GEOM_PROJECT
    print("=== Boundary conditions, project geometry (inlet section w1 x h = %.3g x %.3g m^2)" % (g["w1"], g["h"]))
    print("%9s %12s %14s %14s %11s %10s %10s" % ("Q uL/min", "Q m3/s", "mdot full kg/s", "mdot half kg/s", "U_in mm/s", "Re_Dh", "Re*"))
    for r in boundary_condition_table(g):
        print("%9.2f %12.4e %14.4e %14.4e %11.4f %10.4f %10.2e" % (r["Q_uL_min"], r["Q_m3_s"], r["mdot_full_kg_s"],
              r["mdot_half_kg_s"], r["U"] * 1e3, r["Re_Dh"], r["Re_star"]))
    d = development_lengths(ulmin_to_m3s(120), g)
    print("Entrance length (120 uL/min) ~ %.1f um; viscous time h^2/nu = %.3f s" % (d["L_dev_m"] * 1e6, d["t_visc_s"]))

    import csv
    with open(os.path.join(out, "theory_stations.csv"), "w", newline="") as f:
        wr = None
        for gg in (GEOM_PROJECT, GEOM_PAPER_EQ, GEOM_PAPER_CFD):
            for q in FLOWS_UL_MIN:
                for r in station_table(ulmin_to_m3s(q), gg):
                    row = dict(geometry=gg["name"], Q_uL_min=q, **r)
                    if wr is None:
                        wr = csv.DictWriter(f, fieldnames=list(row)); wr.writeheader()
                    wr.writerow(row)
    for q in FLOWS_UL_MIN:
        print("\n=== Theory at stations, %s, Q = %g uL/min" % (g["name"], q))
        print("%6s %7s %10s %12s %12s %12s %10s %12s %8s %9s" % ("x mm", "w mm", "umax mm/s", "tau_w dyn/cm2", "tau(20um)", "tau_cell",
              "dy/dx wall", "tau_axis3D", "corr %", "d5 mm"))
        for r in station_table(ulmin_to_m3s(q), g):
            print("%6.1f %7.3f %10.4f %12.5f %12.5f %12.5f %10.3f %12.5f %8.2f %9.3f" % (r["x_mm"], r["w_mm"], r["umax_mm_s"],
                  r["tau_w_dyn"], r["tau_z20_dyn"], r["tau_cell_dyn"], r["wall_halfslope"], r["tau_axis3d_dyn"],
                  r["aspect_corr_pct"], r["d5_est_mm"]))
        print("side-wall layer (h=%g um): delta_1%% = %.1f um, delta_5%% = %.1f um, delta_10%% = %.1f um" % (g["h"] * 1e6,
              sidewall_layer(g["h"], .01) * 1e6, sidewall_layer(g["h"], .05) * 1e6, sidewall_layer(g["h"], .10) * 1e6))
        print("Delta p over hyperbolic section = %.4g Pa" % pressure_drop_hyperbolic(ulmin_to_m3s(q), g))

    # Figure: theory only (clearly labelled)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.linspace(0, 45e-3, 300)
    for gg, ls in ((GEOM_PROJECT, "-"), (GEOM_PAPER_EQ, "--"), (GEOM_PAPER_CFD, ":")):
        for q, c in ((0.22, "#1f77b4"), (1.0, "#d62728")):
            ax[0].plot(x * 1e3, tau_wall(x, ulmin_to_m3s(q), gg) * 10, ls, color=c, lw=1.4,
                       label="%s, %g uL/min" % (gg["name"], q))
    for s in STATIONS_M:
        ax[0].axvline(s * 1e3, color="0.8", lw=0.6, zorder=0)
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel(r"$\tau_w$ [dyn cm$^{-2}$]  (THEORY, Eq. 2)")
    ax[0].legend(fontsize=6); ax[0].set_title("Wall shear stress, 1D Hele-Shaw theory", fontsize=9)
    z = np.linspace(0, g["h"], 100)
    for s in STATIONS_M:
        ax[1].plot(u_profile(s, z, ulmin_to_m3s(1.0), g) * 1e3, z * 1e6, label="x = %g mm" % (s * 1e3))
    ax[1].set_xlabel("u [mm/s]"); ax[1].set_ylabel("z [um]"); ax[1].legend(fontsize=7)
    ax[1].set_title("Poiseuille profiles, project geometry, 1 uL/min (THEORY)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "theory_shear_and_profiles.png"), dpi=200)
    print("\nwrote output/theory_stations.csv and output/theory_shear_and_profiles.png")


if __name__ == "__main__":
    main()
