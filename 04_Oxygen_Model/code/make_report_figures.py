#!/usr/bin/env python3
"""
make_report_figures.py  --  new figures for the LaTeX term report.

(1) geometry schematic (planform + cross-section)
(2) published (Delon et al. 2020, Table 2) vs present uniform-shear widths
(3) oxygen transport: closed-form solutions and the 1.5-D reduced-order model
    (MODEL PREDICTIONS -- no oxygen CFD has been run)

All oxygen parameters are the baseline values of the project manual (Table 7.1);
q_max and K_m are illustrative / not cell-line specific.
Writes PDF figures to ../figures and a JSON of every number quoted in the report
to ../data/oxygen_numbers.json.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "regenerated", "figures")
DAT = os.path.join(HERE, "..", "regenerated")
os.makedirs(FIG, exist_ok=True); os.makedirs(DAT, exist_ok=True)
sys.path.insert(0, HERE)
import design_model as dm  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#e3e3e0",
    "grid.linewidth": 0.6, "xtick.color": "#444444", "ytick.color": "#444444", "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "savefig.facecolor": "white", "pdf.fonttype": 42,
})
COL = {0.22: "#2a78d6", 1.0: "#eb6834", 120.0: "#1baf7a"}
INK, INK2 = "#222222", "#666666"
QS = (0.22, 1.0, 120.0)

# ---------------------------------------------------------------- parameters
L, W1, H, WMAX = 50e-3, 1e-3, 145e-6, 10e-3
XT = L * (1 - W1 / WMAX)
D, CSAT, KM = 3.0e-9, 0.178, 6.3e-4
QMAX = 20e-18 * 2e9            # 4e-8 mol m^-2 s^-1 (illustrative)
KPDMS = 5.65e-6
A_S = 2 / 5.385; A_C = 2 / 8.235 - A_S
NUM = {}


def q_si(qul):
    return qul * 1e-9 / 60.0


def area(x):
    """Floor (cell) area between 0 and x of the hyperbolic section: A = w1 L ln(L/(L-x))."""
    return W1 * L * np.log(L / (L - x))


A_TOT = float(area(XT))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


# ================================================================= geometry
def fig_geometry():
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_axes([0.09, 0.42, 0.89, 0.55])
    x = np.linspace(0, XT * 1e3, 400)
    y = 0.5 * W1 * L / (L - x * 1e-3) * 1e3
    xo = np.array([45.0, 49.25]); yo = 5.0 - (xo - 45.0)
    up_x = np.r_[-2, 0, x, xo, 50.25]; up_y = np.r_[0.5, 0.5, y, yo, 0.75]
    ax.fill(np.r_[up_x, up_x[::-1]], np.r_[up_y, -up_y[::-1]], color="#dfe9f7", lw=0)
    ax.plot(up_x, up_y, color=INK, lw=1.2); ax.plot(up_x, -up_y, color=INK, lw=1.2)
    ax.plot([-2, -2], [-0.5, 0.5], color="#2a78d6", lw=3)
    ax.plot([50.25, 50.25], [-0.75, 0.75], color="#d6402a", lw=3)
    ax.axhline(0, color=INK2, lw=0.6, ls="-.")
    for i, xs in enumerate((5, 15, 25, 35, 45)):
        hw = 0.5 * W1 * L / (L - xs * 1e-3) * 1e3
        ax.plot([xs, xs], [-hw, hw], color="#104281", lw=1, ls=":")
        ax.text(xs, hw + 0.35, "S%d\nx = %d mm" % (i + 1, xs), ha="center", fontsize=7, color="#104281")
    ax.annotate("inlet (mass-flow inlet)\n$-2\\leq x\\leq0$ lead-in, 1 mm wide", xy=(-2, 0.3), xytext=(-1.5, 3.6),
                fontsize=7, color="#2a78d6", arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=0.8))
    ax.annotate("outlet (pressure outlet,\n0 Pa gauge), 1.5 mm wide", xy=(50.25, -0.6), xytext=(42.5, -7.2),
                fontsize=7, color="#d6402a", arrowprops=dict(arrowstyle="->", color="#d6402a", lw=0.8))
    ax.annotate("$w(x)=w_1L/(L-x)$", xy=(41.5, 2.65), xytext=(20, 5.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax.annotate("45° walls", xy=(47.2, 2.8), xytext=(48.6, 5.6), fontsize=7,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax.annotate("", xy=(44.3, 4.7), xytext=(44.3, -4.7), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK2))
    ax.text(43.9, -4.9, "$w_{max}$ = 10 mm", fontsize=7, color=INK2, ha="right")
    ax.annotate("", xy=(0, -2.6), xytext=(45, -2.6), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK2))
    ax.text(22.5, -3.3, "hyperbolic section, $x_t = L(1-w_1/w_{max})$ = 45 mm", ha="center", fontsize=7, color=INK2)
    ax.text(0.6, -1.5, "$w_1$ = 1 mm", fontsize=7, color=INK2)
    ax.text(11.0, -7.3, "side walls: wall_side;  floor z = 0: wall_bottom (glass, cells);\nceiling z = h: wall_top (PDMS)",
            fontsize=7, color=INK)
    ax.annotate("", xy=(6, -7.0), xytext=(0, -7.0), arrowprops=dict(arrowstyle="->", lw=1, color=INK))
    ax.annotate("", xy=(0, -5.4), xytext=(0, -7.0),
                arrowprops=dict(arrowstyle="->", lw=1, color=INK))
    ax.text(6.3, -7.2, "x (flow)", fontsize=7); ax.text(-0.9, -5.6, "y", fontsize=7)
    ax.set_xlim(-3, 54); ax.set_ylim(-8.0, 7.2); ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.grid(False)
    ax.set_title("(a) Planform of the fluid domain (true scale, plan view)", fontsize=9, loc="left")

    # ---- cross-section schematic
    b = fig.add_axes([0.07, 0.04, 0.9, 0.31]); b.set_axis_off()
    b.set_xlim(0, 10); b.set_ylim(-1.2, 4.2)
    b.add_patch(Rectangle((0.5, -0.9), 6.0, 0.9, color="#cfe3f0"))          # glass
    b.text(3.5, -0.5, "glass cover slip (impermeable)", ha="center", fontsize=7)
    b.add_patch(Rectangle((0.5, 1.6), 6.0, 2.4, color="#f3e6c8"))          # PDMS lid
    b.add_patch(Rectangle((0.5, 0.0), 0.5, 1.6, color="#f3e6c8")); b.add_patch(Rectangle((6.0, 0.0), 0.5, 1.6, color="#f3e6c8"))
    b.text(3.5, 3.4, "PDMS (5 mm, gas-permeable)", ha="center", fontsize=7)
    b.add_patch(Rectangle((1.0, 0.0), 5.0, 1.6, color="#eef4fb", ec=INK, lw=0.8))
    b.add_patch(Rectangle((1.0, 0.0), 5.0, 0.12, color="#b5527a"))
    b.text(2.6, 0.22, "cell monolayer (z = 0)", ha="left", fontsize=6.5, color="#b5527a")
    zz = np.linspace(0, 1.6, 50); uu = 1.6 * (zz / 1.6) * (1 - zz / 1.6) * 4
    b.plot(4.6 + 0.5 * uu, zz, color="#eb6834", lw=1.2)
    b.plot([4.6, 4.6], [0, 1.6], color="#eb6834", lw=0.6)
    b.text(5.45, 1.15, "u(z)", fontsize=7, color="#eb6834")
    b.annotate("", xy=(0.75, 1.6), xytext=(0.75, 0.0), arrowprops=dict(arrowstyle="<->", lw=0.8))
    b.text(0.05, 0.75, "h =\n145 µm", fontsize=6.5)
    b.annotate("", xy=(6.0, -1.1), xytext=(1.0, -1.1), arrowprops=dict(arrowstyle="<->", lw=0.8))
    b.text(3.5, -1.15, "w(x) = 1.11 mm (S1) ... 10 mm (S5)", ha="center", va="top", fontsize=6.5, backgroundcolor="white")
    for xx in (1.6, 2.2):
        b.add_patch(FancyArrowPatch((xx, 2.3), (xx, 1.7), arrowstyle="->", mutation_scale=9, color="#1baf7a"))
        b.add_patch(FancyArrowPatch((xx, 0.6), (xx, 0.15), arrowstyle="->", mutation_scale=9, color="#b5527a"))
    b.text(7.0, 2.6, "O$_2$ supply through PDMS\n(Robin: $N=k_{PDMS}(C_{sat}-C_{top})$)", fontsize=7, color="#1baf7a")
    b.text(7.0, 1.0, "O$_2$ uptake by cells\n(Michaelis–Menten flux)", fontsize=7, color="#b5527a")
    b.text(7.0, -0.4, "Cross-section (y–z), not to scale;\nflow along +x (into page)", fontsize=7, color=INK2)
    b.set_title("(b) Cross-section: boundaries relevant to flow and oxygen transport", fontsize=9, loc="left")
    save(fig, "geometry_schematic")


# ================================================================= published comparison
def fig_published_d():
    st = ["S1", "S2", "S3", "S4", "S5"]
    halfw = np.array([0.556, 0.714, 1.000, 1.667, 5.000])
    paper_dmax = np.array([0.34, 0.47, 0.82, 1.5, 3.84])
    paper_dwall = np.array([0.22, 0.24, 0.18, 0.16, 0.33])
    ours_d = np.array([0.390, 0.552, 0.834, 1.518, 1.024])
    ours_d2 = np.array([0.3475, 0.5115, 0.7895, 1.4715, 0.663])   # +/-2 % half widths (2d/2 from Table 8)
    ours_d10 = np.array([0.430, 0.5895, 0.873, 1.5355, 1.4495])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    xx = np.arange(5)
    ax[0].bar(xx - 0.27, paper_dmax, 0.26, color="#9aa5b1", label="Delon et al. (2020), $D_{max}$")
    ax[0].bar(xx, ours_d, 0.26, color="#eb6834", label="present CFD, $d$ at ±5 %")
    ax[0].errorbar(xx, ours_d, yerr=[ours_d - ours_d2, ours_d10 - ours_d], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax[0].plot(xx + 0.27, halfw, "k_", ms=12, mew=1.4, label="channel half-width $w/2$")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(st); ax[0].set_ylabel("half-width of uniform zone [mm]")
    ax[0].set_title("(a) Uniform-shear half-width from the axis", fontsize=9)
    ax[0].legend(fontsize=6.5, loc="upper left")
    ax[1].bar(xx[:4] - 0.15, paper_dwall[:4], 0.3, color="#9aa5b1", label="Delon et al., $D_{wall}$")
    ax[1].bar(xx[:4] + 0.15, (halfw - ours_d)[:4], 0.3, color="#eb6834", label="present CFD, $w/2-d$ (±5 %)")
    ax[1].axhline(0.129, color=INK, ls="--", lw=1, label="theory (rectangular duct), 0.129 mm")
    ax[1].set_xticks(xx[:4]); ax[1].set_xticklabels(st[:4]); ax[1].set_ylabel("side-wall layer [mm]")
    ax[1].set_title("(b) Non-uniform layer next to each side wall", fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper right"); ax[1].set_ylim(0, 0.33)
    save(fig, "published_vs_present_uniform_zone")


# ================================================================= oxygen: closed forms
def cb_imp_zero(x, qul, qmax=QMAX):
    return CSAT - qmax * area(x) / q_si(qul)


def k_eff(k):
    if np.isinf(k):
        return D / (A_S * H)
    Bi = k * H / D
    return k / (1 + Bi * A_S)


def pdms_closed(x, qul, k=KPDMS, qmax=QMAX):
    """Zero-order uptake + Robin (or saturated, k=inf) top wall: exponential relaxation in A(x)."""
    ke = k_eff(k)
    cbinf = CSAT + (H / D) * A_C * qmax - qmax / ke
    cb = cbinf + (CSAT - cbinf) * np.exp(-ke * area(x) / q_si(qul))
    N = ke * ((CSAT - cb) + (H / D) * A_C * qmax)
    cbot = cb - (H / D) * (A_S * qmax - A_C * N)
    return cb, cbot, cbinf, ke


def x_crit(qul, qmax=QMAX):
    s = q_si(qul) * CSAT / (qmax * W1 * L)
    return L * (1 - math.exp(-s))


def model_profile(qul, k):
    d = dm.Design(Q_ulmin=qul, k_pdms=k)
    return d, dm.oxygen_profile(d, n=900)


def fig_oxygen_axial():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
    xs = np.linspace(0, XT, 900)
    for qul in QS:
        d, P = model_profile(qul, 0.0)
        ax[0].plot(P.x * 1e3, P.Cbot / CSAT, color=COL[qul], label="%g µL/min, 1.5-D MM model" % qul)
        cz = cb_imp_zero(xs, qul) - (H / D) * A_S * QMAX
        m = cz > 0
        ax[0].plot(xs[m] * 1e3, cz[m] / CSAT, color=INK, ls="--", lw=0.9)
        if x_crit(qul) < XT:
            ax[0].axvline(x_crit(qul) * 1e3, color=COL[qul], ls=":", lw=0.9)
            ax[0].text(x_crit(qul) * 1e3 + 0.5, 0.55 if qul < 0.5 else 0.42, "$x_{crit}$ = %.1f mm" % (x_crit(qul) * 1e3),
                       fontsize=7, color=COL[qul])
        d2, P2 = model_profile(qul, KPDMS)
        ax[1].plot(P2.x * 1e3, P2.Cbot / CSAT, color=COL[qul], label="%g µL/min" % qul)
        _, cbot, _, _ = pdms_closed(xs, qul)
        ax[1].plot(xs * 1e3, cbot / CSAT, color=INK, ls="--", lw=0.9)
        _, cbs, _, _ = pdms_closed(xs, qul, k=np.inf)
        ax[1].plot(xs * 1e3, cbs / CSAT, color=COL[qul], ls=":", lw=1.0)
    ax[0].plot([], [], color=INK, ls="--", lw=0.9, label="closed form, zero-order uptake")
    ax[0].axhline(0.1, color="#b5527a", lw=0.8, ls="-."); ax[0].text(16, 0.12, "hypoxia threshold 0.1 $C_{sat}$", fontsize=6.5, color="#b5527a")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("$C_{bot}/C_{sat}$ at the cell layer")
    ax[0].set_title("(a) Impermeable walls (IMP)", fontsize=9); ax[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[0].set_ylim(-0.02, 1.05)
    ax[1].plot([], [], color=INK, ls="--", lw=0.9, label="closed form, PDMS lid")
    ax[1].plot([], [], color=INK2, ls=":", lw=1.0, label="closed form, saturated walls")
    ax[1].set_xlabel("x [mm]"); ax[1].set_title("(b) PDMS lid (Robin) and saturated walls", fontsize=9)
    ax[1].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); ax[1].set_ylabel("$C_{bot}/C_{sat}$")
    ax[1].set_ylim(0.93, 1.003)
    save(fig, "oxygen_axial_profiles")


def fig_oxygen_z():
    eta = np.linspace(0, 1, 200)
    th_imp = eta - eta ** 3 + eta ** 4 / 2
    th_bal = eta
    u = 6 * eta * (1 - eta)
    cup_imp = np.trapezoid(u * th_imp, eta) / np.trapezoid(u, eta)
    NUM["cup_imp_theta"] = cup_imp; NUM["Nu_one_sided"] = 2 / cup_imp
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(th_imp, eta, color="#eb6834", label="impermeable lid: $\\eta-\\eta^3+\\eta^4/2$")
    ax[0].plot(th_bal, eta, color="#1baf7a", label="PDMS, supply = uptake: $\\eta$")
    ax[0].axvline(cup_imp, color="#eb6834", ls=":", lw=0.9); ax[0].text(cup_imp + 0.01, 0.05, "cup-mixed\n13/35", fontsize=6.5, color="#eb6834")
    ax[0].set_xlabel("$[C(z)-C_{bot}]\\,/\\,(q h/D)$"); ax[0].set_ylabel("$\\eta = z/h$")
    ax[0].set_title("(a) Fully developed O$_2$ profile across the gap", fontsize=9); ax[0].legend(fontsize=6.5, loc="upper left")
    ax[0].set_ylim(0, 1); ax[0].set_xlim(0, 1.05)
    # dimensional for baseline
    dc = QMAX * H / D
    NUM["dC_gap_molm3"] = dc; NUM["dC_gap_pct"] = 100 * dc / CSAT
    ax[1].plot(100 * (1 - (dc * (th_imp[-1] - th_imp)) / CSAT), eta * H * 1e6, color="#eb6834", label="IMP (relative to the lid value)")
    ax[1].plot(100 * (1 - (dc * (1 - th_bal)) / CSAT), eta * H * 1e6, color="#1baf7a", label="PDMS (relative to the lid value)")
    ax[1].set_xlabel("$C(z)/C_{top}$ [%]  (baseline $q_{max}$)"); ax[1].set_ylabel("z [µm]")
    ax[1].set_title("(b) Transverse drop is ≤ 1.1 % of $C_{sat}$", fontsize=9); ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_xlim(98.8, 100.05)
    save(fig, "oxygen_transverse_profile")


def fig_oxygen_tau():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for qul in QS:
        for k, ls in ((0.0, "-"), (KPDMS, "--")):
            d, P = model_profile(qul, k)
            m = (P.x >= 5e-3) & (P.x <= 40e-3)
            ax.plot(P.tau[m] * 10, P.Cbot[m] / CSAT, ls, color=COL[qul],
                    label=("%g µL/min" % qul) if k == 0 else None)
    ax.plot([], [], "k-", lw=1, label="impermeable"); ax.plot([], [], "k--", lw=1, label="PDMS lid")
    ax.set_xscale("log"); ax.set_xlabel("wall shear stress $\\tau_w$ (T2) [dyn cm$^{-2}$]")
    ax.set_ylabel("$C_{bot}/C_{sat}$"); ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Shear and oxygen at the cell layer, 5 ≤ x ≤ 40 mm", fontsize=9)
    save(fig, "oxygen_vs_shear")


def fig_design_map():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    Qg = np.logspace(-1.3, 1.5, 300)
    for f, c in ((0.25, "#9ec5f4"), (0.5, "#6da7ec"), (1, "#256abf"), (2.5, "#104281"), (10, "#06204a")):
        xc = np.array([x_crit(qq, QMAX * f) for qq in Qg]) * 1e3
        xc[xc >= XT * 1e3] = np.nan
        ax[0].plot(Qg, xc, color=c, label="%.1f×10$^{-8}$" % (4 * f))
    for qul in (0.22, 1.0):
        ax[0].axvline(qul, color=COL[qul], ls=":", lw=0.9)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Q [µL/min]"); ax[0].set_ylabel("$x_{crit}$ [mm] (anoxia beyond)")
    ax[0].set_title("(a) Onset of anoxia, impermeable walls", fontsize=9); ax[0].legend(fontsize=6, title="$q_{max}$ in mol m$^{-2}$ s$^{-1}$", title_fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax[0].set_ylim(0, 46)
    qm = np.logspace(-9, -6.5, 200)
    qmin = qm * A_TOT / CSAT * 60e9
    ax[1].plot(qm, qmin, color=INK, label="$Q_{min}=q_{max}A_{cells}/C_{sat}$")
    ax[1].axvline(QMAX, color="#b5527a", ls="--", lw=0.9); ax[1].text(QMAX * 1.1, 0.15, "baseline\n4×10$^{-8}$", fontsize=6.5, color="#b5527a")
    for qul in QS:
        ax[1].axhline(qul, color=COL[qul], ls=":", lw=0.9); ax[1].text(1.2e-9, qul * 0.62, "%g µL/min" % qul, fontsize=6.5, color=COL[qul])
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("$q_{max}$ [mol m$^{-2}$ s$^{-1}$]")
    ax[1].set_ylabel("$Q_{min}$ [µL/min]"); ax[1].set_title("(b) Flow needed to feed the whole floor", fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "oxygen_design_map")


def fig_groups():
    Qg = np.logspace(-1, 2.3, 200)
    U0 = q_si(Qg) / (W1 * H)
    Pe_h = U0 * H / D
    Da_conv = QMAX * A_TOT / (q_si(Qg) * CSAT)
    t_res = A_TOT * H / q_si(Qg)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(Qg, Pe_h, color="#2a78d6", label="$Pe_h=\\bar U_0 h/D$ (inlet)")
    ax.plot(Qg, Da_conv, color="#eb6834", label="$Da_{conv}=q_{max}A/(QC_{sat})$")
    ax.plot(Qg, t_res / (H ** 2 / D), color="#1baf7a", label="$t_{res}/t_{D}$, $t_D=h^2/D$")
    ax.axhline(QMAX * H / (D * CSAT), color=INK2, ls="--", lw=1, label="$Da_w=q_{max}h/(DC_{sat})$")
    ax.axhline(KPDMS * CSAT / QMAX, color="#9b59b6", ls="-.", lw=1, label="$\\Phi_{PDMS}=k_{PDMS}C_{sat}/q_{max}$")
    ax.axhline(1, color=INK, lw=0.6)
    for qul in QS:
        ax.axvline(qul, color=COL[qul], ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Q [µL/min]"); ax.set_ylabel("dimensionless group")
    ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(1.02, 1.0)); ax.set_ylim(3e-3, 3e3)
    ax.set_title("Transport regime vs flow rate (baseline parameters)", fontsize=9)
    save(fig, "oxygen_dimensionless_groups")


def numbers():
    out = {}
    out["A_tot_mm2"] = A_TOT * 1e6
    out["q_max"] = QMAX
    out["Q_min_ulmin"] = QMAX * A_TOT / CSAT * 60e9
    out["Da_w"] = QMAX * H / (D * CSAT)
    out["Bi"] = KPDMS * H / D
    out["Phi"] = KPDMS * CSAT / QMAX
    out["k_eff_pdms"] = k_eff(KPDMS)
    out["k_eff_sat"] = k_eff(np.inf)
    for qul in QS:
        r = {}
        Q = q_si(qul)
        U0 = Q / (W1 * H)
        r["U0_mm_s"] = U0 * 1e3
        r["Pe_h"] = U0 * H / D
        r["Pe_L"] = U0 * L / D
        r["Da_conv"] = QMAX * A_TOT / (Q * CSAT)
        r["t_res_s"] = A_TOT * H / Q
        r["supply_conv_mol_s"] = Q * CSAT
        r["uptake_max_mol_s"] = QMAX * A_TOT
        r["supply_pdms_max_mol_s"] = KPDMS * CSAT * A_TOT
        r["x_crit_mm"] = x_crit(qul) * 1e3 if x_crit(qul) < XT else None
        r["drop_at_45_zero_order_pct"] = 100 * QMAX * A_TOT / (Q * CSAT)
        r["L_entry_conc_mm"] = 0.0115 * (U0 * 2 * H / D) * 2 * H * 1e3   # graetz entrance, parallel plates (Shah & London)
        cb, cbot, cbinf, ke = pdms_closed(np.array([XT]), qul)
        r["pdms_closed_Cbot45_over_Csat"] = float(cbot[0] / CSAT)
        r["pdms_relax_area_mm2"] = Q / ke * 1e6
        r["pdms_relax_x_mm"] = L * (1 - math.exp(-(Q / ke) / (W1 * L))) * 1e3
        _, cbs, _, kes = pdms_closed(np.array([XT]), qul, k=np.inf)
        r["sat_closed_Cbot45_over_Csat"] = float(cbs[0] / CSAT)
        r["sat_relax_x_mm"] = L * (1 - math.exp(-(Q / kes) / (W1 * L))) * 1e3
        for k, tag in ((0.0, "IMP"), (KPDMS, "PDMS"), (1.0, "SAT")):
            d = dm.Design(Q_ulmin=qul, k_pdms=k)
            ev = dm.evaluate(d)
            P = dm.oxygen_profile(d, n=900)
            r[tag] = dict(min_Cbot=float(ev["minCbot_over_Csat"]), mean_Cbot=float(ev["meanCbot_over_Csat"]),
                          hypoxic_frac_5_40=float(ev["hypoxic_area_frac"]),
                          Cbot_at_45=float(P.Cbot.iloc[-1] / CSAT),
                          uptake_total_mol_s=float(np.trapezoid(P.q * P.w, P.x)),
                          x_10pct_mm=float(P.x[P.Cbot < 0.1 * CSAT].min() * 1e3) if (P.Cbot < 0.1 * CSAT).any() else None,
                          Cbot_at_S=[float(np.interp(xx, P.x, P.Cbot) / CSAT) for xx in (5e-3, 15e-3, 25e-3, 35e-3, 45e-3)])
            # outlet mixed-cup and balance
            r[tag]["Cb_out_over_Csat"] = float(P.Cb.iloc[-1] / CSAT)
            r[tag]["balance_resid_pct"] = float(100 * ((Q * (CSAT - P.Cb.iloc[-1])) - (np.trapezoid((P.q - P.N) * P.w, P.x))) / max(Q * CSAT, 1e-30))
        out["Q%g" % qul] = r
    out.update(NUM)
    with open(os.path.join(DAT, "oxygen_numbers.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    fig_geometry(); fig_published_d(); fig_oxygen_axial(); fig_oxygen_z(); fig_oxygen_tau(); fig_design_map(); fig_groups()
    o = numbers()
    print(json.dumps(o, indent=1, default=float)[:6000])
