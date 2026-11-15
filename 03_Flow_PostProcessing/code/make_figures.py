#!/usr/bin/env python3
"""
make_figures.py - every graph of the post-processing study.

Each figure is written twice:
  Graphs/<folder>/FigNN_<name>.png (300 dpi) and .pdf (vector)  - with figure number, title and caption
  output/report_figs/FigNN_<name>.png (220 dpi)                   - bare plot, the report supplies the caption
A JSON registry (output/figure_registry.json) records number, folder, file, title and caption.
Run after analysis.py:  python make_figures.py
"""
import json
import os
import pickle
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import theory as T

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": C.GRID, "grid.linewidth": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444", "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "savefig.facecolor": "white", "pdf.fonttype": 42,
})

RFIG = os.path.join(C.OUT, "report_figs")
REG = []
_counter = [0]
TH1 = dict(color=C.INK, ls="--", lw=1.3)          # Eq. 2 / parallel-plate theory
TH2 = dict(color=C.INK2, ls=":", lw=1.6)          # rectangular-duct theory
TH3 = dict(color="#9b59b6", ls="-.", lw=1.3)      # 2D Hele-Shaw model (distinct neutral-violet ink)
STN = list(C.STATIONS)


def save(fig, folder, name, title, caption, size=None):
    _counter[0] += 1
    n = _counter[0]
    fname = "Fig%02d_%s" % (n, name)
    os.makedirs(os.path.join(C.GRAPHS, folder), exist_ok=True); os.makedirs(RFIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(RFIG, fname + ".png"), dpi=220)
    # standalone version with number, title and caption
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(caption, width=int(w * 15.5))
    tlines = textwrap.wrap("Figure %d. %s" % (n, title), width=int(w * 9.6))
    cap_h = 0.15 * len(lines) + 0.12
    tit_h = 0.22 * len(tlines) + 0.18
    H = h + cap_h + tit_h
    fig.set_size_inches(w, H)
    fig.tight_layout(rect=[0, (cap_h + 0.05) / H, 1, 1 - tit_h / H])
    fig.suptitle("\n".join(tlines), fontsize=10.5, fontweight="bold", y=1 - 0.08 / H, va="top")
    fig.text(0.02, 0.02, "\n".join(lines), fontsize=7.8, color="#444444", va="bottom", ha="left")
    fig.savefig(os.path.join(C.GRAPHS, folder, fname + ".png"), dpi=300)
    fig.savefig(os.path.join(C.GRAPHS, folder, fname + ".pdf"))
    plt.close(fig)
    REG.append(dict(number=n, folder=folder, file=fname, title=title, caption=caption,
                    report_png=os.path.join(RFIG, fname + ".png")))
    return n


def qnorm(k):
    return {c["key"]: c for c in C.CASES}[k]


def tunit(Q_ulmin):
    """(scale, unit) for shear stress and pressure axes."""
    return (1.0, "Pa") if Q_ulmin > 10 else (1e3, "mPa")


def mark_stations(ax, label=True):
    for s, x in C.STATIONS.items():
        ax.axvline(x * 1e3, color="#bbbbbb", lw=0.7, ls=(0, (2, 3)), zorder=0)
        if label:
            ax.text(x * 1e3, 1.0, s, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#777777")


def main():
    with open(os.path.join(C.OUT, "results.pkl"), "rb") as fh:
        R = pickle.load(fh)
    cases = R["cases"]; tabs = R["tables"]; t2d = R["theory2d"]
    keys = [c["key"] for c in C.CASES]
    SH = os.path.join(C.ROOT, "02_CFD_Results_Raw", "fluent_screenshots")

    # ================================================================ PRESSURE
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.4))
    for ax, k in zip(axs, keys):
        ax.imshow(mpimg.imread(os.path.join(SH, "contour_pressure_%s.png" % k))); ax.axis("off")
        ax.set_title(qnorm(k)["label"] + "  (Fluent static pressure, colour range set by Fluent)", fontsize=9, loc="left")
    save(fig, "Pressure", "pressure_contours_all_Q", "Static pressure contours for the three flow rates",
         "Fluent contour images supplied with the results (plan view, raster). Colour-bar maxima 0.395, 1.80 and 215.6 Pa "
         "scale with Q to within 0.3 % (1.795-1.800 Pa per µL/min); the maxima are ~8 % above the centre-line value at x = 0 "
         "because the domain includes the 2 mm straight lead-in upstream of x = 0.")

    for k in keys:
        c = cases[k]; cl = c["cl"]; Q = c["meta"]["Q"]; s = Q / t2d["Qref"]
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        ax.plot(cl["x"] * 1e3, cl["p"], color=C.COLORS[k], label="CFD (centre-line)")
        ax.plot(t2d["xs"] * 1e3, t2d["p_axis"] * s, **TH3, label="T3: 2D Hele-Shaw model (with lead-in, outlet)")
        xh = np.linspace(0, C.XT, 200)
        p0 = np.interp(0, cl["x"], cl["p"])
        ax.plot(xh * 1e3, p0 - T.dp1(xh, Q), **TH1, label="T1: Eq. p(0) - p(x), anchored to CFD p(0)")
        mark_stations(ax); ax.axvline(C.XT * 1e3, color="#555555", lw=0.8)
        ax.text(C.XT * 1e3 + 0.4, 0.55, "end of hyperbolic\nsection (x_t = 45 mm)", transform=ax.get_xaxis_transform(), fontsize=7, color="#555555")
        ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Static pressure p [Pa]"); ax.set_xlim(-2.5, 51)
        ax.legend(loc="upper right")
        save(fig, "Pressure", "pressure_vs_x_%s" % k.replace(".", "p"), "Static pressure along the centre-line, %s" % c["meta"]["label"],
             "CFD pressure (gauge, outlet = 0 Pa) against the 2D Hele-Shaw model T3 (absolute, includes the lead-in and the outlet transition) and "
             "the analytical T1 profile anchored at the CFD value at x = 0. The CFD curve decreases monotonically; the T1 curve applies only to 0-45 mm.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for k in keys:
        c = cases[k]; cl = c["cl"]
        ax.plot(cl["x"] * 1e3, cl["p"] / c["meta"]["Q_ulmin"], color=C.COLORS[k], marker=C.MARKERS[k], markevery=60, ms=4.5, lw=1.4, label=c["meta"]["label"])
    ax.plot(t2d["xs"] * 1e3, t2d["p_axis"] / (t2d["Qref"] * 6e10), **TH3, label="T3: 2D Hele-Shaw model")
    mark_stations(ax); ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("p / Q  [Pa per µL/min]"); ax.set_xlim(-2.5, 51); ax.legend()
    save(fig, "Pressure", "pressure_vs_x_combined_normalised", "Centre-line pressure of all flow rates, normalised by Q",
         "Pressure divided by the volumetric flow rate. The three CFD curves collapse onto one line (largest difference 0.02 %), "
         "showing Stokes-flow linearity across a 545-fold range of Q. T3 lies ~5 % above the CFD along the whole channel.")

    P = tabs["pressure_summary"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    q = P.Q_uL_min.values
    qq = np.logspace(-1, 2.3, 50)
    ax.plot(qq, R["dp_fit"]["slope_Pa_per_uLmin"] * qq, color="#999999", lw=1, label="linear fit to CFD (R² = 1.000000)")
    ax.plot(q, P.dp_0_end_T3_Pa, **TH3, marker="D", ms=5, label="T3: 2D Hele-Shaw, x = 0 to outlet")
    ax.plot(q, P.dp_0_45_T1_Pa, **TH1, marker="x", ms=6, label="T1: Eq., x = 0 to 45 mm")
    for i, k in enumerate(keys):
        ax.plot(q[i], P.dp_0_end_CFD_Pa[i], marker="o", ls="none", color=C.COLORS[k], ms=8, mec="white", mew=1.5, zorder=5,
                label="CFD x = 0 to outlet, " + qnorm(k)["label"])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Volumetric flow rate Q [µL/min]"); ax.set_ylabel("Pressure drop Δp [Pa]")
    ax.legend(fontsize=7)
    save(fig, "Pressure", "pressure_drop_vs_Q", "Pressure drop across the channel versus flow rate",
         "Centre-line pressure drop from x = 0 to the outlet (CFD) on logarithmic axes. Δp = 1.657 Q (Q in µL/min) to within 0.02 %; "
         "hydraulic resistance 9.94e10 Pa s m^-3. The T1 points cover only the hyperbolic section and are therefore lower by construction.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    k = "Q1"; c = cases[k]; cl = c["cl"]; qn = c["meta"]["Q_ulmin"]
    ax.plot(cl["xg"] * 1e3, -cl["dpdx"] / qn, color=C.COLORS[k], label="CFD, Q = 1 µL/min (windowed regression ±0.4 mm)")
    for kk in ("Q0.22", "Q120"):
        ax.plot(cases[kk]["cl"]["xg"] * 1e3, -cases[kk]["cl"]["dpdx"] / cases[kk]["meta"]["Q_ulmin"], color=C.COLORS[kk], lw=1.0,
                marker=C.MARKERS[kk], markevery=40, ms=3.5, alpha=0.9, label="CFD, " + cases[kk]["meta"]["label"])
    ax.plot(cl["xg"] * 1e3, -cl["dpdx1"] / qn, **TH1, label="T1: -dp/dx = 12μQ(1-x/L)/(h³w₁)")
    ax.plot(cl["xg"] * 1e3, -cl["dpdx2"] / qn, **TH2, label="T2: T1 / f(h/w)  (rectangular section)")
    ax.plot(cl["xg"] * 1e3, -cl["dpdx2d"] / qn, **TH3, label="T3: 2D Hele-Shaw model")
    mark_stations(ax); ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("-dp/dx / Q  [Pa m⁻¹ per µL/min]"); ax.set_xlim(0, 50.5)
    ax.set_yscale("log"); ax.legend(fontsize=7, loc="upper right")
    save(fig, "Pressure", "pressure_gradient_vs_x", "Local axial pressure gradient along the channel",
         "Magnitude of dp/dx per unit flow rate from a ±0.4 mm least-squares window on the CFD centre-line pressure. The gradient falls "
         "linearly with x in the hyperbolic section, reaches a minimum at x ≈ 44 mm and rises again where the 45° walls converge to the 1.5 mm outlet.")

    # ================================================================ THEORETICAL COMPARISONS (pressure)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    c = cases["Q1"]; cl = c["cl"]; m = (cl["x"] >= 0) & (cl["x"] <= C.XT)
    ax.plot(cl["x"][m] * 1e3, cl["dp_cfd"][m], color=C.COLORS["Q1"], label="CFD, Q = 1 µL/min")
    ax.plot(cl["x"][m] * 1e3, cl["dp1"][m], **TH1, label="T1: (12μQ/h³w₁)(x - x²/2L)")
    ax.plot(cl["x"][m] * 1e3, cl["dp2"][m], **TH2, label="T2: with rectangular-section factor f(h/w)")
    ax.set_ylabel("p(0) - p(x)  [Pa]"); ax.legend(loc="lower right"); mark_stations(ax)
    for k in keys:
        cl_ = cases[k]["cl"]; mm = (cl_["x"] >= 1e-3) & (cl_["x"] <= C.XT)
        ax2.plot(cl_["x"][mm] * 1e3, 100 * (cl_["dp_cfd"][mm] / cl_["dp1"][mm] - 1), color=C.COLORS[k], lw=1.2, label="vs T1, " + qnorm(k)["label"])
    cl_ = cases["Q1"]["cl"]; mm = (cl_["x"] >= 1e-3) & (cl_["x"] <= C.XT)
    ax2.plot(cl_["x"][mm] * 1e3, 100 * (cl_["dp_cfd"][mm] / cl_["dp2"][mm] - 1), color=C.COLORS["Q1"], ls="--", lw=1.2, label="vs T2, Q = 1 µL/min")
    ax2.axhline(0, color="#999999", lw=0.8); ax2.set_ylabel("CFD deviation [%]"); ax2.set_xlabel("Axial position x [mm]")
    ax2.legend(fontsize=7, ncol=2); mark_stations(ax2, label=False); ax2.set_xlim(0, 46)
    save(fig, "Theoretical_Comparisons", "pressure_drop_CFD_vs_theory", "Cumulative pressure drop: CFD versus Hele-Shaw theory",
         "Top: pressure drop from x = 0 for Q = 1 µL/min. Bottom: signed deviation of the CFD from T1 for all flow rates (curves coincide) and "
         "from T2. Over 0-45 mm the CFD is 1.4 % above T1 and 4.9 % below T2.")

    # ================================================================ WALL SHEAR
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.4))
    for ax, k in zip(axs, keys):
        ax.imshow(mpimg.imread(os.path.join(SH, "contour_wall_shear_%s.png" % k))); ax.axis("off")
        ax.set_title(qnorm(k)["label"] + "  (Fluent wall shear stress, all walls)", fontsize=9, loc="left")
    save(fig, "Wall_Shear_Stress", "wall_shear_contours_all_Q", "Wall shear stress contours for the three flow rates",
         "Fluent contour images supplied with the results (raster). The colour-bar maxima (3.32e-3, 1.51e-2 and 1.825 Pa) occur at the inlet "
         "face of the lead-in, where the uniform mass-flow inlet profile creates a local shear peak; they scale with Q to within 0.7 %. "
         "The colour scale is dominated by that peak, so the gradient along the cell wall is quantified in the line plots that follow.")

    for k in keys:
        c = cases[k]; cl = c["cl"]; Q = c["meta"]["Q"]; sc, un = tunit(c["meta"]["Q_ulmin"])
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        ax.plot(cl["x"] * 1e3, cl["tw"] * sc, color=C.COLORS[k], label="CFD (centre-line, wall)")
        ax.plot(cl["x"] * 1e3, cl["tau1"] * sc, **TH1, label="T1: Eq. 2, 6μQ(1-x/L)/(h²w₁)")
        ax.plot(cl["x"] * 1e3, cl["tau2"] * sc, **TH2, label="T2: Eq. 2 / f(h/w)")
        ax.plot(t2d["xs"] * 1e3, t2d["tau_axis"] * Q / t2d["Qref"] * sc, **TH3, label="T3: 2D Hele-Shaw model")
        mark_stations(ax); ax.set_xlim(-2.5, 51); ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Wall shear stress τ_w [%s]" % un)
        ax.legend(loc="upper right", fontsize=7.5)
        save(fig, "Wall_Shear_Stress", "wall_shear_vs_x_%s" % k.replace(".", "p"), "Centre-line wall shear stress, %s" % c["meta"]["label"],
             "CFD wall shear along the y = 0 line on the channel wall compared with Eq. 2 (T1), the rectangular-section correction (T2) and the 2D "
             "Hele-Shaw model (T3, includes the outlet transition). The CFD decreases almost linearly to a minimum at x ≈ 44 mm and rises "
             "again in the converging outlet.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for k in keys:
        c = cases[k]; cl = c["cl"]
        ax.plot(cl["x"] * 1e3, cl["tw"], color=C.COLORS[k], marker=C.MARKERS[k], markevery=60, ms=4.5, label="CFD, " + c["meta"]["label"])
        ax.plot(cl["x"] * 1e3, cl["tau1"], **TH1, label="T1 (Eq. 2)" if k == "Q120" else None)
    ax.set_yscale("log"); mark_stations(ax); ax.set_xlim(-0.5, 51)
    ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Wall shear stress τ_w [Pa]"); ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    save(fig, "Wall_Shear_Stress", "wall_shear_vs_x_combined", "Centre-line wall shear stress for all flow rates",
         "Absolute wall shear on a logarithmic axis. The three cases are parallel (identical shape) and separated by the flow-rate ratios "
         "4.55 and 120; the dashed lines are Eq. 2 for each case.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for k in keys:
        c = cases[k]; cl = c["cl"]; t0 = 6 * C.MU * c["meta"]["Q"] / (C.H ** 2 * C.W1)
        ax.plot(cl["x"] / C.L, cl["tw"] / t0, color=C.COLORS[k], marker=C.MARKERS[k], markevery=55, ms=4.5, lw=1.4, label="CFD, " + c["meta"]["label"])
    xx = np.linspace(0, C.XT, 200)
    ax.plot(xx / C.L, 1 - xx / C.L, **TH1, label="T1: 1 - x/L")
    ax.plot(xx / C.L, (1 - xx / C.L) / T.flux_factor(C.H, T.width(xx)), **TH2, label="T2: (1 - x/L) / f(h/w)")
    ax.set_xlabel("Normalised axial position x / L"); ax.set_ylabel("τ_w / τ₁   (τ₁ = 6μQ/h²w₁)"); ax.set_xlim(-0.01, 1.01); ax.legend()
    save(fig, "Wall_Shear_Stress", "wall_shear_normalised", "Normalised wall shear stress versus normalised axial distance",
         "Wall shear divided by the Eq. 2 inlet value τ₁ = 6μQ/(h²w₁) (9.84e-4, 4.47e-3 and 0.537 Pa). All flow rates collapse onto one curve "
         "(differences < 0.05 %). The CFD crosses the T1 line near x/L = 0.5 and lies parallel to, but ~5 % below, T2.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for k in keys:
        cl = cases[k]["cl"]; mm = (cl["x"] >= 0) & (cl["x"] <= C.XT)
        ax.plot(cl["x"][mm] * 1e3, 100 * (cl["tw"][mm] / cl["tau1"][mm] - 1), color=C.COLORS[k], marker=C.MARKERS[k], markevery=50, ms=4,
                lw=1.3, label="CFD vs T1, " + qnorm(k)["label"])
    cl = cases["Q1"]["cl"]
    ax.plot(cl["x"][mm] * 1e3, 100 * (cl["tw"][mm] / cl["tau2"][mm] - 1), color=C.COLORS["Q1"], ls="--", lw=1.3, label="CFD vs T2, Q = 1 µL/min")
    ax.plot(cl["x"][mm] * 1e3, 100 * (cl["tw"][mm] / cl["tau2d"][mm] - 1), color=C.COLORS["Q1"], ls=":", lw=1.6, label="CFD vs T3, Q = 1 µL/min")
    ax.axhline(0, color="#999999", lw=0.8); mark_stations(ax); ax.set_xlim(0, 45.5); ax.set_ylim(-10, 50)
    ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Signed deviation  (CFD / theory - 1) [%]"); ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "Wall_Shear_Stress", "wall_shear_percentage_deviation", "Percentage deviation of the CFD wall shear stress from theory",
         "Signed deviation along the hyperbolic section. Against Eq. 2 the deviation runs from +3.5 % at the inlet through zero (x ≈ 25 mm) to "
         "-2 % at S4, then rises steeply to +46 % at S5 (outlet corner). Against T2 it is a near-constant -5.2 ± 0.4 %; against T3 it stays at -5 % "
         "including at S5.")

    # aspect-ratio figure
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for k in keys:
        cl = cases[k]["cl"]; mm = (cl["x"] >= 2e-3) & (cl["x"] <= 40e-3)
        hw = C.H / T.width(cl["x"][mm])
        ax.plot(hw, 100 * (cl["tw"][mm] / cl["tau1"][mm] - 1), ls="none", marker=C.MARKERS[k], ms=3, color=C.COLORS[k], alpha=0.7, label="CFD, " + qnorm(k)["label"])
    hh = np.linspace(0.02, 0.15, 100)
    ax.plot(hh, 100 * (1 / T.flux_factor(C.H, C.H / hh) - 1), **TH2, label="T2: 1/f(h/w) - 1  (exact side-wall effect)")
    af = R["cases"]["Q1"]["aspect_fit"]
    ax.plot(hh, 100 * ((1 + af["dev_T2_offset"] + af["dev_T2_slope_vs_hw"] * hh) / T.flux_factor(C.H, C.H / hh) - 1), color="#999999", lw=1,
            label="T2 shifted by the fitted uniform offset (%.1f %%)" % (100 * af["dev_T2_offset"]))
    ax.axhline(0, color="#999999", lw=0.8)
    for s in STN[:4]:
        hw = C.H / T.width(C.STATIONS[s]); ax.annotate(s, (hw, 100 * (1 / T.flux_factor(C.H, C.H / hw) - 1)), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center", color="#555555")
    ax.set_xlabel("Local aspect ratio h / w(x)"); ax.set_ylabel("τ_w,CFD / τ_Eq.2 - 1  [%]"); ax.legend(fontsize=7)
    save(fig, "Theoretical_Comparisons", "shear_deviation_vs_aspect_ratio", "Deviation from Eq. 2 as a function of the local aspect ratio",
         "Points: CFD centre-line shear between x = 2 and 40 mm. The CFD reproduces the slope of the exact side-wall correction (T2) "
         "but sits on a parallel line shifted by -4.9 %; the slope difference is only -0.026 per unit h/w.")

    # ================================================================ TRANSVERSE SHEAR
    for s in STN:
        fig, ax = plt.subplots(figsize=(6.4, 3.7))
        for k in keys:
            tp = cases[k]["tprof"][s]; qn = cases[k]["meta"]["Q_ulmin"]
            ax.plot(tp["yc"] * 1e3, tp["tauc"] / qn * 1e3, color=C.COLORS[k], marker=C.MARKERS[k], ms=4.2, lw=1.1, label="CFD, " + cases[k]["meta"]["label"])
        tp = cases["Q1"]["tprof"][s]; qn = cases["Q1"]["meta"]["Q_ulmin"]
        ax.plot(tp["y_side"] * 1e3, tp["tau_side"] / qn * 1e3, ls="none", marker="o", mfc="none", mec="#777777", ms=5,
                label="side-wall / corner nodes (excluded)")
        yf = np.concatenate([-tp["yf"][::-1], tp["yf"]]); rf = np.concatenate([tp["rT"][::-1], tp["rT"]])
        if s != "S5":
            ref = T.tau2(C.STATIONS[s], cases["Q1"]["meta"]["Q"]); lab = "T2: rectangular duct, exact"
            ax.plot(yf * 1e3, rf * ref / qn * 1e3, **TH2, label=lab)
        ref3 = np.interp(C.STATIONS[s], t2d["xs"], t2d["tau_axis"]) * cases["Q1"]["meta"]["Q"] / t2d["Qref"]
        r2 = np.concatenate([tp["r2d"][::-1], tp["r2d"]])
        ax.plot(yf * 1e3, r2 * ref3 / qn * 1e3, **TH3, label="T3: 2D Hele-Shaw (no side-wall drag)")
        ax.axvline(-tp["w"] / 2 * 1e3, color="#555555", lw=0.8); ax.axvline(tp["w"] / 2 * 1e3, color="#555555", lw=0.8)
        ax.set_xlabel("Transverse position y [mm]  (y = 0 on the axis)"); ax.set_ylabel("τ_w / Q  [mPa per µL/min]")
        ax.set_ylim(bottom=0); ax.legend(fontsize=7, loc="lower center")
        x_mm = C.STATIONS[s] * 1e3
        extra = (" The three flow rates overlap exactly (markers hidden behind each other); the central plateau is flat to within 0.8 % and the shear "
                 "drops to ~0.44 of the axis value at the corner nodes."
                 if s != "S5" else " S5 lies at the widest point, where the walls turn from diverging to converging (45°): there is no plateau; the shear is "
                 "highest on the axis and decreases continuously to ~0 at the corners, with the shape predicted by T3.")
        save(fig, "Transverse_Shear_S1_S5", "transverse_shear_%s" % s, "Transverse wall shear stress profile at %s (x = %.0f mm)" % (s, x_mm),
             "Top/bottom-wall nodes on the plane x = %.0f mm (each point is the mean of the two nearly coincident top and bottom wall nodes), divided by Q. "
             "Vertical lines: side walls at ±%.3f mm.%s" % (x_mm, tp["w"] / 2 * 1e3, extra))

    fig, axs = plt.subplots(5, 1, figsize=(6.6, 9.6))
    for ax, s, col in zip(axs, STN, C.STATION_COLORS):
        tp = cases["Q1"]["tprof"][s]
        ax.plot(tp["yc"] * 1e3, tp["tauc"] * 1e3, color=C.COLORS["Q1"], marker="s", ms=3.2, lw=1.0, label="CFD, Q = 1 µL/min")
        yf = np.concatenate([-tp["yf"][::-1], tp["yf"]])
        if s != "S5":
            ax.plot(yf * 1e3, np.concatenate([tp["rT"][::-1], tp["rT"]]) * T.tau2(C.STATIONS[s], cases["Q1"]["meta"]["Q"]) * 1e3, **TH2, label="T2")
        ref3 = np.interp(C.STATIONS[s], t2d["xs"], t2d["tau_axis"]) * cases["Q1"]["meta"]["Q"] / t2d["Qref"]
        ax.plot(yf * 1e3, np.concatenate([tp["r2d"][::-1], tp["r2d"]]) * ref3 * 1e3, **TH3, label="T3")
        ax.set_ylabel("τ_w [mPa]"); ax.set_ylim(bottom=0)
        ax.text(0.01, 0.06, "%s, x = %.0f mm, w = %.3f mm" % (s, C.STATIONS[s] * 1e3, tp["w"] * 1e3), transform=ax.transAxes, fontsize=8)
        if s == "S1":
            ax.legend(fontsize=7, loc="lower right", ncol=3)
    axs[-1].set_xlabel("Transverse position y [mm]")
    save(fig, "Transverse_Shear_S1_S5", "transverse_shear_all_sections", "Transverse wall shear profiles at the five sections (Q = 1 µL/min)",
         "Each panel has its own y range (the channel is 1.1 mm wide at S1 and 10 mm at S5). At S1-S4 the profile is a flat plateau with a "
         "side-wall layer of ~0.15 mm; at S5 the profile is non-uniform over the full width.")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for s, col in zip(STN, C.STATION_COLORS):
        tp = cases["Q1"]["tprof"][s]
        ax.plot(tp["yc"] * 1e3, tp["tauc"] * 1e3, color=col, marker="o", ms=3, lw=1.3, label="%s (x = %.0f mm)" % (s, C.STATIONS[s] * 1e3))
        ax.annotate(s, (0, tp["tau0"] * 1e3), textcoords="offset points", xytext=(0, 4), fontsize=7.5, ha="center", color=C.INK)
    ax.set_xlabel("Transverse position y [mm]"); ax.set_ylabel("Wall shear stress τ_w [mPa]"); ax.set_ylim(bottom=0); ax.legend(fontsize=7.5, loc="upper right")
    save(fig, "Transverse_Shear_S1_S5", "transverse_shear_evolution", "Evolution of the transverse wall shear profile from S1 to S5 (Q = 1 µL/min)",
         "As the channel widens the axis shear falls (4.15 → 0.66 mPa) and the ±5 % plateau widens from 0.78 mm (S1) to 3.04 mm (S4); the side-wall "
         "layer keeps a constant absolute thickness. At S5 the outlet corner reshapes the whole profile into a dome (±5 % band only 2.05 mm wide).")

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for s, col in zip(STN, C.STATION_COLORS):
        tp = cases["Q1"]["tprof"][s]
        ax.plot(tp["yc"] / (tp["w"] / 2), tp["r"], color=col, marker="o", ms=3, lw=1.3, label=s)
        if s != "S5":
            ax.plot(np.concatenate([-tp["yf"][::-1], tp["yf"]]) / (tp["w"] / 2), np.concatenate([tp["rT"][::-1], tp["rT"]]), color=col, ls=":", lw=1.2)
    tp = cases["Q1"]["tprof"]["S5"]
    ax.plot(np.concatenate([-tp["yf"][::-1], tp["yf"]]) / (tp["w"] / 2), np.concatenate([tp["rT"][::-1], tp["rT"]]), **TH3, label="T3 at S5")
    ax.plot([], [], **TH2, label="T2 (dotted, station colour)")
    ax.axhspan(0.95, 1.05, color="#eeeeee", zorder=0); ax.text(0.0, 1.06, "±5 % band", ha="center", fontsize=7, color="#666666")
    ax.set_xlabel("Normalised transverse position 2y / w"); ax.set_ylabel("τ(y) / τ(0)"); ax.set_ylim(0, 1.4); ax.legend(fontsize=7.5, ncol=2, loc="lower center")
    save(fig, "Transverse_Shear_S1_S5", "transverse_shear_normalised", "Normalised transverse wall shear profiles τ(y)/τ(0)",
         "Profiles normalised by the axis value and by the local half-width. At S1-S4 the layer of reduced shear has a fixed thickness of ~h, so "
         "its share of the width shrinks downstream. At S5 the shear falls monotonically from the axis to the corners.")

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    dd = np.linspace(0, 0.6, 300)
    for s, col in zip(STN[:4], C.STATION_COLORS[:4]):
        tp = cases["Q1"]["tprof"][s]
        dist = tp["w"] / 2 - np.abs(tp["yc"])
        ax.plot(dist * 1e3, tp["r"], ls="none", marker="o", ms=4, color=col, label="CFD %s" % s)
    ax.plot(dd, T.tau_bottom_ratio(T.width(5e-3) / 2 - dd * 1e-3, T.width(5e-3)), **TH2, label="T2: exact series (S1 width)")
    ax.plot(dd, 1 - 8 / np.pi ** 2 * np.exp(-np.pi * dd * 1e-3 / C.H), color="#bbbbbb", lw=1, label="leading term 1 - (8/π²)e^(-πd/h)")
    ax.axhline(0.95, color="#999999", lw=0.7, ls="--"); ax.text(0.45, 0.935, "5 % deficit", fontsize=7, color="#666666")
    ax.set_xlim(0, 0.6); ax.set_ylim(0.3, 1.03)
    ax.set_xlabel("Distance from the side wall d [mm]"); ax.set_ylabel("τ / τ(0)"); ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "Transverse_Shear_S1_S5", "sidewall_layer_collapse", "Side-wall layer: wall shear versus distance from the side wall (S1-S4)",
         "When plotted against the distance from the side wall the profiles from S1-S4 collapse, confirming that the layer thickness scales with h, "
         "not with w. The theoretical 5 % point is 0.129 mm from the wall; CFD wall nodes are 0.08-0.10 mm apart near the side walls.")

    U = tabs["uniformity"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    sub = U[U.case == "Q = 1 µL/min"]
    for e, mk in ((2, "v"), (5, "o"), (10, "^")):
        ss = sub[sub.eps_pct == e]
        ax.plot(ss.x_mm, 2 * ss.d_min_mm, marker=mk, color=C.COLORS["Q1"], lw=1.3, ms=6, label="CFD, ±%d %%" % e)
        ax.plot(ss.x_mm, 2 * ss.d_theory_mm, marker=mk, mfc="none", color=C.INK2, ls=":", lw=1.2, ms=6, label="theory, ±%d %%" % e)
    ax.plot(ss.x_mm, 2 * ss.half_width_mm, color="#bbbbbb", lw=1, marker="|", label="channel width w(x)")
    ax.set_yscale("log"); ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Width of uniform-shear zone 2d [mm]")
    ax.legend(fontsize=7, ncol=2, loc="upper left"); ax.set_xticks([5, 15, 25, 35, 45]); ax.set_xticklabels(["S1\n5", "S2\n15", "S3\n25", "S4\n35", "S5\n45"])
    save(fig, "Transverse_Shear_S1_S5", "uniform_zone_width_vs_x", "Width of the uniform-shear zone and its sensitivity to the tolerance",
         "2d = width of the central band in which |τ(y)/τ(0) - 1| ≤ ε, for ε = 2, 5 and 10 %, at Q = 1 µL/min (other flow rates differ by < 0.002 mm). "
         "Theory: T2 at S1-S4, T3 at S5. The uniform zone widens from S1 to S4 and collapses at S5.")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs_ = np.arange(5); wbar = 0.25
    for j, e in enumerate((2, 5, 10)):
        ss = sub[sub.eps_pct == e]
        ax.bar(xs_ + (j - 1) * wbar, 100 * ss.uniform_width_fraction, width=wbar - 0.03, color=C.STATION_COLORS[1 + j], label="CFD, ±%d %%" % e)
        ax.plot(xs_ + (j - 1) * wbar, 100 * 2 * ss.d_theory_mm / (2 * ss.half_width_mm), ls="none", marker="_", ms=12, mew=2, color=C.INK)
    ax.plot([], [], ls="none", marker="_", ms=12, mew=2, color=C.INK, label="theory")
    ax.set_xticks(xs_); ax.set_xticklabels(STN); ax.set_ylabel("Uniform fraction of the width 2d / w [%]"); ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, ncol=4, loc="upper left"); ax.grid(axis="x")
    save(fig, "Transverse_Shear_S1_S5", "uniform_fraction_bar", "Fraction of the channel width with uniform shear",
         "Percentage of the local width within the tolerance band (Q = 1 µL/min). At ±5 % the fraction grows from 70 % (S1) to 91 % (S4) "
         "and drops to 20 % at S5.")

    # ================================================================ TRANSVERSE VELOCITY
    for s in STN:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        tp = cases["Q1"]["tprof"][s]; qn1 = cases["Q1"]["meta"]["Q_ulmin"]
        ax.plot(tp["y_all"] * 1e3, tp["V_all"] / qn1 * 1e3, ls="none", marker=".", ms=3, color="#c8c8c8", label="CFD all nodes, Q = 1 (all depths z)")
        for k in keys:
            tq = cases[k]["tprof"][s]; qn = cases[k]["meta"]["Q_ulmin"]
            ax.plot(tq["col_y"] * 1e3, tq["col_u"] / qn * 1e3, color=C.COLORS[k], lw=1.2, marker=C.MARKERS[k], markevery=max(1, len(tq["col_y"]) // 25), ms=3.5,
                    label="upper envelope, " + cases[k]["meta"]["label"])
        yf = np.concatenate([-tp["yf"][::-1], tp["yf"]])
        lab = "T2: exact u(y, z = h/2), rectangular duct" if s != "S5" else "T3: 2D Hele-Shaw, mid-plane u = hτ/4μ"
        ax.plot(yf * 1e3, np.concatenate([tp["uT"][::-1], tp["uT"]]) / qn1 * 1e3, **(TH2 if s != "S5" else TH3), label=lab)
        ax.axvline(-tp["w"] / 2 * 1e3, color="#555555", lw=0.8); ax.axvline(tp["w"] / 2 * 1e3, color="#555555", lw=0.8)
        ax.set_ylim(bottom=0); ax.set_xlabel("Transverse position y [mm]"); ax.set_ylabel("|V| / Q  [mm/s per µL/min]"); ax.legend(fontsize=7, loc="lower center")
        save(fig, "Transverse_Velocity_S1_S5", "transverse_velocity_%s" % s, "Velocity magnitude across the width at %s (x = %.0f mm)" % (s, C.STATIONS[s] * 1e3),
             "Grey: every exported node of the plane x = %.0f mm (nodes at all heights z, so the scatter spans 0 to the mid-plane value). Lines: upper envelope "
             "(nodes that are the local maximum within ±0.015 mm), which approximates the velocity near mid-height. The exported quantity is the velocity magnitude, not the axial component; "
             "the export has no z coordinate, so u(z) cannot be reconstructed." % (C.STATIONS[s] * 1e3))

    fig, axs = plt.subplots(5, 1, figsize=(6.6, 9.6))
    for ax, s in zip(axs, STN):
        tp = cases["Q1"]["tprof"][s]
        ax.plot(tp["y_all"] * 1e3, tp["V_all"] * 1e3, ls="none", marker=".", ms=2.5, color="#c8c8c8")
        ax.plot(tp["col_y"] * 1e3, tp["col_u"] * 1e3, color=C.COLORS["Q1"], lw=1.2, label="CFD envelope")
        yf = np.concatenate([-tp["yf"][::-1], tp["yf"]])
        ax.plot(yf * 1e3, np.concatenate([tp["uT"][::-1], tp["uT"]]) * 1e3, **(TH2 if s != "S5" else TH3), label="T2" if s != "S5" else "T3")
        ax.set_ylabel("|V| [mm/s]"); ax.set_ylim(bottom=0)
        ax.text(0.01, 0.85, "%s, x = %.0f mm" % (s, C.STATIONS[s] * 1e3), transform=ax.transAxes, fontsize=8)
        ax.legend(fontsize=7, loc="lower center", ncol=2)
    axs[-1].set_xlabel("Transverse position y [mm]")
    save(fig, "Transverse_Velocity_S1_S5", "transverse_velocity_all_sections", "Velocity magnitude across the width at the five sections (Q = 1 µL/min)",
         "Peak velocity falls from 0.166 mm/s at S1 to 0.026 mm/s at S5 as the cross-section widens. Grey points are nodes at all depths; they fall on "
         "four discrete levels plus the walls, reflecting the mesh layers across the gap. Isolated dips in the envelope (S2, S4) are mesh columns "
         "that have no node near mid-height, not flow features.")

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for s, col in zip(STN, C.STATION_COLORS):
        tp = cases["Q1"]["tprof"][s]
        ax.plot(tp["col_y"] / (tp["w"] / 2), tp["col_u"] / tp["u0"], color=col, lw=1.4, label=s)
        ax.plot(np.concatenate([-tp["yf"][::-1], tp["yf"]]) / (tp["w"] / 2), np.concatenate([tp["uT"][::-1], tp["uT"]]) / tp["u_th0"], color=col, ls=":", lw=1.2)
    ax.plot([], [], color=C.INK2, ls=":", label="theory (station colour)")
    ax.set_xlabel("Normalised transverse position 2y / w"); ax.set_ylabel("u_env(y) / u_env(0)"); ax.set_ylim(0, 1.4); ax.legend(fontsize=7.5, ncol=3, loc="lower center")
    save(fig, "Transverse_Velocity_S1_S5", "transverse_velocity_normalised", "Normalised transverse velocity profiles",
         "Envelope velocity divided by its axis value. S1-S4 show a flat core and a side-wall layer of fixed thickness; at S5 the profile is "
         "non-uniform across the width because the flow converges towards the outlet.")

    V = tabs["velocity_summary"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    v1 = V[V.case == "Q = 1 µL/min"]
    ax.plot(v1.x_mm, v1.u_mid_axis_theory_m_s * 1e3, **TH2, marker="D", ms=5, label="theory: mid-plane axis velocity (T2; T3 at S5)")
    ax.plot(v1.x_mm, v1.u_parabola_from_CFD_gradient_m_s * 1e3, color="#999999", ls="--", marker="x", ms=6, lw=1, label="parabola from the CFD dp/dx: h²|dp/dx|/8μ")
    for k in keys:
        vk = V[V.case == qnorm(k)["label"]]; qn = qnorm(k)["Q_ulmin"]
        ax.plot(vk.x_mm, vk.u_max_sampled_CFD_m_s * 1e3 / qn, ls="none", marker=C.MARKERS[k], ms=6, color=C.COLORS[k], label="CFD sampled maximum / Q, " + qnorm(k)["label"])
    ax.set_yscale("log"); ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Peak velocity per unit Q [mm/s per µL/min]"); ax.legend(fontsize=7)
    ax.set_xticks([5, 15, 25, 35, 45])
    save(fig, "Transverse_Velocity_S1_S5", "peak_velocity_vs_x", "Peak velocity along the channel",
         "Highest sampled velocity magnitude at each section divided by Q (markers) against the theoretical mid-plane maximum. The CFD maxima are "
         "1.2-3.4 % below theory but 1.6-3.9 % above the parabola implied by the CFD's own pressure gradient.")

    # pathlines (velocity magnitude) screenshots
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.4))
    for ax, k in zip(axs, ("Q1", "Q120")):
        ax.imshow(mpimg.imread(os.path.join(SH, "pathlines_velocity_%s.png" % k))[60:560, :]); ax.axis("off")
        ax.set_title(qnorm(k)["label"] + "  (Fluent pathlines coloured by velocity magnitude)", fontsize=9, loc="left")
    save(fig, "Transverse_Velocity_S1_S5", "pathlines_Q1_Q120", "Pathlines coloured by velocity magnitude",
         "Fluent report images found in the results archive (raster). Maximum velocity 1.94e-4 m/s at 1 µL/min and 2.31e-2 m/s at 120 µL/min "
         "(ratio 119.1 vs 120: -0.8 %). Pathlines follow the diverging walls without separation and converge smoothly into the outlet.")

    # ================================================================ COMBINED COMPARISONS
    ST = tabs["station_comparison"]; s1 = ST[ST.case == "Q = 1 µL/min"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs_ = np.arange(5)
    ax.plot(xs_ - 0.15, 100 * (s1.dpdx_CFD_Pa_m / np.where(s1.station == "S5", np.nan, s1.dpdx_T2_Pa_m) - 1), ls="none", marker="o", ms=7, color=C.STATION_COLORS[2], label="-dp/dx vs T2")
    ax.plot(xs_, s1.signed_dev_T2_pct.where(s1.station != "S5"), ls="none", marker="s", ms=7, color=C.STATION_COLORS[3], label="τ_w (axis) vs T2")
    ax.plot(xs_[4], s1.signed_dev_T3_pct.values[4], ls="none", marker="s", ms=7, mfc="none", color=C.STATION_COLORS[3], label="τ_w (axis) vs T3 (S5)")
    ax.plot(xs_ + 0.15, 100 * (v1.ratio_max_sampled_to_theory - 1), ls="none", marker="^", ms=7, color=C.STATION_COLORS[4], label="peak |V| vs theory (≤ true peak)")
    ax.plot(xs_, s1.signed_dev_T1_pct, ls="none", marker="x", ms=7, color=C.INK, label="τ_w (axis) vs Eq. 2 (T1)")
    ax.axhline(0, color="#999999", lw=0.8); ax.axhspan(-5.16 - 0.36, -5.16 + 0.36, color="#eeeeee", zorder=0)
    ax.set_xticks(xs_); ax.set_xticklabels(STN); ax.set_ylabel("CFD deviation from theory [%]"); ax.set_ylim(-8, 8); ax.grid(axis="x")
    ax.annotate("τ_w vs Eq. 2 at S5: +46 %\n(off scale)", (4, 7.6), xytext=(3.75, 5.2), ha="center", fontsize=7, color=C.INK,
                arrowprops=dict(arrowstyle="->", color=C.INK, lw=0.8))
    ax.legend(fontsize=7, loc="upper right", ncol=1, bbox_to_anchor=(0.86, 1.0))
    save(fig, "Combined_Comparisons", "validation_ratios_by_station", "Agreement between CFD and theory at the five sections (Q = 1 µL/min)",
         "Deviation of the CFD pressure gradient, wall shear and peak velocity from the best applicable theory (T2 at S1-S4, T3 at S5); crosses: Eq. 2. "
         "Pressure gradient and shear are uniformly ~5 % low (grey band = mean ± s.d. over 2-40 mm); the peak velocity is within 3.5 %. The S5 point vs Eq. 2 (+46 %) is off-scale.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    cl = cases["Q1"]["cl"]
    ax.plot(cl["x"] * 1e3, cl["tw"] * 1e3, color=C.COLORS["Q1"], label="τ_w (CFD wall shear)")
    ax.plot(cl["xg"] * 1e3, 0.5 * C.H * np.abs(cl["dpdx"]) * 1e3, color=C.INK, ls="--", lw=1.2, label="(h/2)|dp/dx| from the CFD pressure")
    FB = tabs["force_balance"]; fb1 = FB[FB.case == "Q = 1 µL/min"]
    for _, r_ in fb1.iterrows():
        ax.annotate("%.3f" % r_.ratio_tau_over_half_h_dpdx, (r_.x_mm, r_.tau_CFD_Pa * 1e3), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    mark_stations(ax); ax.set_xlim(0, 50.5); ax.set_xlabel("Axial position x [mm]"); ax.set_ylabel("Stress [mPa]"); ax.legend()
    save(fig, "Combined_Comparisons", "pressure_gradient_vs_shear", "Momentum balance: wall shear versus pressure gradient (Q = 1 µL/min)",
         "For a wide gap, τ_w = (h/2)|dp/dx| on the axis. The CFD shear and the shear implied by the CFD pressure gradient agree to within 0.8 % at all "
         "stations (labels: ratio), i.e. the two exported fields are mutually consistent with h = 145 µm.")

    fig, axs = plt.subplots(5, 1, figsize=(6.4, 9.2))
    for ax, s in zip(axs, STN):
        tp = cases["Q1"]["tprof"][s]
        ax.plot(tp["yc"] / (tp["w"] / 2), tp["r"], color=C.COLORS["Q1"], marker="s", ms=3, lw=1.2, label="τ_w / τ_w(0)")
        ax.plot(tp["col_y"] / (tp["w"] / 2), tp["col_u"] / tp["u0"], color=C.STATION_COLORS[4], lw=1.2, label="u_env / u_env(0)")
        ax.set_ylim(0, 1.35); ax.set_ylabel("ratio"); ax.text(0.01, 0.08, s, transform=ax.transAxes, fontsize=8)
        if s == "S1":
            ax.legend(fontsize=7, loc="lower center", ncol=2)
    axs[-1].set_xlabel("Normalised transverse position 2y / w")
    save(fig, "Combined_Comparisons", "velocity_vs_shear_profiles", "Transverse velocity and wall shear profiles compared",
         "Normalised envelope velocity and wall shear at each section (Q = 1 µL/min). Both follow the same shape because τ_w = 4μu_mid/h holds "
         "locally across the core; near the side walls the shear drops slightly earlier than the velocity.")

    SC = tabs["flow_rate_scaling"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cols = ["dp_dev_from_proportional_pct"] + ["tau_%s_dev_pct" % s for s in STN] + ["u_%s_dev_pct" % s for s in STN] + ["contour_tau_max_dev_pct"]
    labs = ["Δp"] + ["τ %s" % s for s in STN] + ["u %s" % s for s in STN] + ["τ max\n(contour)"]
    xs_ = np.arange(len(cols))
    for j, k in enumerate(("Q0.22", "Q120")):
        row = SC[SC.case == qnorm(k)["label"]].iloc[0]
        ax.bar(xs_ + (j - 0.5) * 0.38, [row[c_] for c_ in cols], width=0.36, color=C.COLORS[k], label=qnorm(k)["label"] + " relative to Q = 1")
    ax.axhline(0, color="#777777", lw=0.8); ax.set_xticks(xs_); ax.set_xticklabels(labs, fontsize=7, rotation=0)
    ax.set_ylabel("Deviation from proportionality [%]"); ax.legend(fontsize=7.5); ax.grid(axis="x")
    save(fig, "Combined_Comparisons", "flow_rate_scaling", "Linearity of the response with flow rate",
         "Percentage departure of each quantity from exact proportionality with Q, relative to the 1 µL/min case. All interior quantities are "
         "proportional to within 0.08 %; the only measurable non-linearity is the +0.7 % inlet-face shear peak at 120 µL/min (Re = 3.6). "
         "The 0.22 µL/min bars are below 0.001 % except the contour maximum, whose -0.08 % reflects the 3-figure colour-bar label.")

    RG = tabs["regional_deviation"]; r1 = RG[RG.case == "Q = 1 µL/min"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs_ = np.arange(len(r1))
    ax.bar(xs_ - 0.2, r1.mean_dev_T1_pct, width=0.38, color=C.STATION_COLORS[1], label="mean deviation vs T1 (Eq. 2)")
    ax.bar(xs_ + 0.2, r1.mean_dev_T3_pct, width=0.38, color=C.STATION_COLORS[3], label="mean deviation vs T3 (2D Hele-Shaw)")
    ax.axhline(0, color="#777777", lw=0.8); ax.set_xticks(xs_); ax.set_xticklabels([t.replace(" ", "\n", 1) for t in r1.region], fontsize=7.5)
    ax.set_ylabel("Mean signed deviation of τ_w [%]"); ax.legend(fontsize=7.5); ax.grid(axis="x")
    for i, v in enumerate(r1.mean_dev_T1_pct):
        if np.isnan(v):
            ax.text(i - 0.2, 0.5, "n/a", ha="center", fontsize=7, color="#666666")
    save(fig, "Combined_Comparisons", "regional_deviation", "Inlet, central and outlet regions: deviation of the centre-line shear",
         "Mean signed deviation of the CFD wall shear in four regions (Q = 1 µL/min). Eq. 2 is accurate in the central region but fails in the "
         "outlet transition; the 2D model T3 removes the regional dependence and leaves a uniform -5 % offset.")

    with open(os.path.join(C.OUT, "figure_registry.json"), "w") as fh:
        json.dump(REG, fh, indent=1)
    print("figures:", len(REG))


if __name__ == "__main__":
    main()
