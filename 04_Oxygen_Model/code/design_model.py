#!/usr/bin/env python3
"""
design_model.py
===============
Reduced-order design model coupling (i) Hele-Shaw shear relations and (ii) an oxygen
mass balance, used to screen geometries / operating conditions BEFORE running Fluent.

SHEAR (see 02_analytical/hele_shaw_theory.py)
  tau_axis(x) = 6 mu Q (1 - x/L) / (h^2 w1 f(h/w))   (f = rectangular-duct flux factor)
  imaging window  x_a = 0.1 L  ...  x_b where the wall half-slope (1/2) dw/dx = s_max
                  (s_max = 0.25 keeps the 2D Hele-Shaw centreline within ~1 % of Eq. 2;
                   checked with hele_shaw_2d.py for L = 50 mm)
  side-wall non-uniform band  delta_5 = (h/pi) ln(8/(pi^2 * 0.05)) = 0.887 h  on each side

OXYGEN ("1.5-D" model: cup-mixing concentration C_b(x), fully developed transverse profile)
  Q dC_b/dx = w(x) [ N_top - q_bot ]
  q_bot = q_max C_bot / (K_m + C_bot)                 (Michaelis-Menten monolayer on z = 0)
  N_top = k_pdms (C_sat - C_top)                      (Robin supply through PDMS; k = 0: impermeable)
  wall concentrations from the fully developed parallel-plate solution (superposition of the
  one-sided Nu = 5.385 and symmetric Nu = 8.235 constant-flux cases, Shah & London 1978):
      C_b - C_bot = (h/D) (a_s q_bot - a_c N_top)
      C_top - C_b = (h/D) (a_s N_top - a_c q_bot)      a_s = 2/5.385, a_c = 2/8.235 - a_s
  Neglects: axial diffusion (Pe_L >> 1), entrance region (~ mm at 120 uL/min - conservative),
  across-width variations (slower fluid near side walls -> CFD will show extra depletion there).

USAGE
  python design_model.py                 # baseline predictions, sweeps, tornado, Pareto
Outputs in ./output : PNG figures + CSV tables (MODEL predictions, not CFD results).
"""
import itertools
import math
import os
import sys
from dataclasses import dataclass, replace, asdict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hele_shaw_theory as th  # noqa: E402

A_S = 2 / 5.385
A_C = 2 / 8.235 - A_S


@dataclass
class Design:
    # geometry & flow (SI)
    L: float = 50e-3
    w1: float = 1e-3
    h: float = 145e-6
    w_max: float = 10e-3
    Q_ulmin: float = 1.0
    mu: float = th.MU
    # oxygen (BASELINE - see manual Table 7.1 for sources/status)
    D: float = 3.0e-9          # m2/s
    C_sat: float = 0.178       # mol/m3
    ocr: float = 20e-18        # mol/cell/s  (ILLUSTRATIVE; replace with measured Caco-2/A549 value)
    rho_cell: float = 2e9      # cells/m2    (= 2e5 cells/cm2, seeding density of Delon et al.)
    K_m: float = 6.3e-4        # mol/m3      (0.5 mmHg)
    k_pdms: float = 5.65e-6    # m/s (0 = impermeable)
    C_crit: float = 0.0178     # mol/m3 (10 % of saturation) - reporting threshold
    s_max: float = 0.25

    @property
    def Q(self):
        return th.ulmin_to_m3s(self.Q_ulmin)

    @property
    def q_max(self):
        return self.ocr * self.rho_cell

    @property
    def g(self):
        return dict(L=self.L, w1=self.w1, h=self.h, w_max=self.w_max)

    @property
    def x_t(self):
        return self.L * (1 - self.w1 / self.w_max)


def width(d, x):
    return np.where(x <= d.x_t, d.w1 * d.L / (d.L - np.minimum(x, d.x_t)), d.w_max)


def imaging_window(d):
    x_a = 0.1 * d.L
    x_b = d.L - math.sqrt(d.w1 * d.L / (2 * d.s_max))
    return x_a, min(x_b, d.x_t)


# ------------------------------------------------------------------ oxygen
def local_fluxes(d, Cb):
    """Solve the local wall-flux problem at bulk concentration Cb -> (q_bot, N_top, C_bot, C_top)."""
    Cb = max(Cb, 0.0)
    hD = d.h / d.D

    def N_of_q(q):
        return d.k_pdms * (d.C_sat - Cb + hD * A_C * q) / (1 + d.k_pdms * hD * A_S)

    def Cbot(q):
        return Cb - hD * (A_S * q - A_C * N_of_q(q))

    def res(q):
        c = max(Cbot(q), 0.0)
        return q - d.q_max * c / (d.K_m + c)
    q = 0.0 if res(0.0) >= 0 else brentq(res, 0.0, d.q_max, xtol=1e-16)
    N = N_of_q(q)
    cb = max(Cbot(q), 0.0)
    ct = Cb + hD * (A_S * N - A_C * q)
    return q, N, cb, ct


def oxygen_profile(d, n=600):
    x_end = d.x_t
    def rhs(x, y):
        q, N, _, _ = local_fluxes(d, y[0])
        return [float(width(d, x)) * (N - q) / d.Q]
    xs = np.linspace(0, x_end, n)
    sol = solve_ivp(rhs, (0, x_end), [d.C_sat], t_eval=xs, max_step=x_end / 200, rtol=1e-8, atol=1e-12)
    Cb = np.clip(sol.y[0], 0, None)
    out = np.array([local_fluxes(d, c) for c in Cb])
    return pd.DataFrame(dict(x=sol.t, w=width(d, sol.t), Cb=Cb, q=out[:, 0], N=out[:, 1], Cbot=out[:, 2], Ctop=out[:, 3],
                             tau=th.tau_wall_axis_3d(sol.t, d.Q, d.g, d.mu)))


def dimensionless(d, x=0.0):
    w = float(width(d, np.array(x)))
    U = d.Q / (w * d.h)
    area = d.w1 * d.L * math.log(d.w_max / d.w1)
    return dict(
        Pe_h=U * d.h / d.D,                                   # transverse diffusion time / convection time over h
        Pe_L=U * d.L / d.D,                                   # axial convection vs axial diffusion
        Da_w=d.q_max * d.h / (d.D * d.C_sat),                 # wall uptake vs diffusion across h
        Da_conv=d.q_max * area / (d.Q * d.C_sat),             # total uptake vs convective supply (>1: flow alone cannot feed)
        Bi_pdms=d.k_pdms * d.h / d.D,                         # PDMS conductance vs liquid-layer conductance
        Phi_pdms=d.k_pdms * d.C_sat / d.q_max if d.q_max else np.inf,  # max PDMS supply / max uptake
        C_over_Km=d.C_sat / d.K_m,
        L_entry_mm=0.0115 * (U * 2 * d.h / d.D) * 2 * d.h * 1e3,  # concentration entrance length (parallel plates)
        t_res_s=area * d.h / d.Q, t_diff_h_s=d.h ** 2 / d.D)


def evaluate(d):
    """Scalar metrics used by sweeps and optimisation."""
    x_a, x_b = imaging_window(d)
    P = oxygen_profile(d)
    m = (P.x >= x_a) & (P.x <= x_b)
    tau_a = float(th.tau_wall_axis_3d(x_a, d.Q, d.g, d.mu)); tau_b = float(th.tau_wall_axis_3d(x_b, d.Q, d.g, d.mu))
    dlt = th.sidewall_layer(d.h, 0.05)
    nonuni = 2 * dlt / float(width(d, np.array(x_a)))
    hyp = np.trapezoid(((P.Cbot < d.C_crit) * P.w)[m], P.x[m]) / max(np.trapezoid(P.w[m], P.x[m]), 1e-30)
    return dict(x_a_mm=x_a * 1e3, x_b_mm=x_b * 1e3, tau_a_dyn=tau_a * 10, tau_b_dyn=tau_b * 10, range_ratio=tau_a / tau_b,
                nonuniform_frac_at_xa=nonuni, minCbot_over_Csat=P.Cbot[m].min() / d.C_sat,
                meanCbot_over_Csat=np.trapezoid((P.Cbot * P.w)[m], P.x[m]) / np.trapezoid(P.w[m], P.x[m]) / d.C_sat,
                hypoxic_area_frac=hyp, dp_Pa=th.pressure_drop_hyperbolic(d.Q, d.g, mu=d.mu),
                Re_star=th.reynolds(d.Q, d.g)["Re_star"], **dimensionless(d))


# ------------------------------------------------------------------ studies
def baseline_profiles(out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    rows = []
    for Qv, col in ((0.22, "#1f77b4"), (1.0, "#d62728"), (120.0, "#2ca02c")):
        for kp, ls, lab in ((0.0, "-", "impermeable"), (5.65e-6, "--", "PDMS 5 mm")):
            d = Design(Q_ulmin=Qv, k_pdms=kp)
            P = oxygen_profile(d)
            ax[0].plot(P.x * 1e3, P.Cbot / d.C_sat, ls, color=col, label="%g µL/min, %s" % (Qv, lab))
            ax[1].plot(P.tau * 10, P.Cbot / d.C_sat, ls, color=col)
            r = evaluate(d); r.update(Q_ulmin=Qv, wall=lab); rows.append(r)
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel(r"$C_{bot}/C_{sat}$ (cell surface)"); ax[0].legend(fontsize=6)
    ax[0].set_title("MODEL: O$_2$ at the monolayer", fontsize=9)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\tau_w$ [dyn cm$^{-2}$]"); ax[1].set_ylabel(r"$C_{bot}/C_{sat}$")
    ax[1].set_title("MODEL: shear vs O$_2$ availability", fontsize=9)
    # tornado for 1 uL/min impermeable and PDMS
    base = Design(Q_ulmin=1.0, k_pdms=0.0)
    pars = dict(D=(2.6e-9, 3.3e-9), C_sat=(0.167, 0.200), ocr=(5e-18, 100e-18), rho_cell=(1e9, 5e9),
                K_m=(6.3e-4, 6.3e-3), h=(120e-6, 170e-6))
    b0 = evaluate(base)["meanCbot_over_Csat"]
    lo, hi = [], []
    for p, (a, b) in pars.items():
        lo.append(evaluate(replace(base, **{p: a}))["meanCbot_over_Csat"] - b0)
        hi.append(evaluate(replace(base, **{p: b}))["meanCbot_over_Csat"] - b0)
    yy = np.arange(len(pars))
    ax[2].barh(yy, lo, color="#5dade2", label="low value"); ax[2].barh(yy, hi, color="#ec7063", label="high value")
    ax[2].set_yticks(yy); ax[2].set_yticklabels(list(pars), fontsize=7); ax[2].axvline(0, color="k", lw=0.6)
    ax[2].set_xlabel(r"$\Delta$ area-mean $C_{bot}/C_{sat}$ in imaging window (base %.2f)" % b0, fontsize=8); ax[2].legend(fontsize=6)
    ax[2].set_title("MODEL sensitivity, 1 µL/min, impermeable", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "design_oxygen_baseline.png"), dpi=220)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "design_baseline_metrics.csv"), index=False)
    return T


def sweeps(out):
    import matplotlib.pyplot as plt
    specs = [("L", np.linspace(30e-3, 80e-3, 11), 1e3, "L [mm]"),
             ("w1", np.linspace(0.5e-3, 2e-3, 11), 1e3, "w1 [mm]"),
             ("h", np.linspace(75e-6, 300e-6, 10), 1e6, "h [µm]"),
             ("Q_ulmin", np.geomspace(0.1, 200, 12), 1, "Q [µL/min]"),
             ("mu", np.linspace(0.7e-3, 1.5e-3, 9), 1e3, "µ [mPa s]"),
             ("D", np.linspace(2.4e-9, 3.4e-9, 6), 1e9, "D [1e-9 m²/s]"),
             ("ocr", np.geomspace(2e-18, 200e-18, 9), 1e18, "OCR [amol/cell/s]")]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6)); axs = axs.ravel()
    allrows = []
    for k, (p, vals, sc, lab) in enumerate(specs):
        for kp, ls in ((0.0, "-"), (5.65e-6, "--")):
            R = [dict(evaluate(replace(Design(k_pdms=kp), **{p: v})), value=v) for v in vals]
            R = pd.DataFrame(R); R["param"] = p; R["k_pdms"] = kp; allrows.append(R)
            ax = axs[k]; ax2 = ax.twinx()
            ax.plot(vals * sc, R.tau_a_dyn, "k" + ls, label=r"$\tau_w(x_a)$")
            ax2.plot(vals * sc, R.minCbot_over_Csat, ls, color="#c0392b", label="min C/Csat")
            ax.set_xlabel(lab, fontsize=8); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7, colors="#c0392b")
            ax2.set_ylim(-0.02, 1.02)
            if p in ("Q_ulmin", "ocr"):
                ax.set_xscale("log")
            if k == 0:
                ax.set_ylabel(r"$\tau_w(x_a)$ [dyn cm$^{-2}$]", fontsize=8)
    axs[-1].axis("off")
    axs[-1].text(0, 0.6, "black: wall shear at start of imaging window\nred: min O$_2$ at monolayer / C$_{sat}$\n"
                 "solid: impermeable walls\ndashed: 5 mm PDMS (Robin)\nAll curves: reduced-order MODEL", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "design_parameter_sweeps.png"), dpi=220)
    pd.concat(allrows).to_csv(os.path.join(out, "design_parameter_sweeps.csv"), index=False)


def pareto(out, tau_hi=0.04, ratio_min=5.0, min_img_len=20e-3, dp_max=1000.0, k_pdms=0.0, tag="IMP"):
    """Design problem (one wall scenario at a time):
       variables  L in [30, 80] mm, w1 in [0.5, 2] mm, h in [75, 300] um
       Q chosen so that tau_axis(x_a) = tau_hi   (shear target met exactly)
       constraints: tau(x_a)/tau(x_b) >= ratio_min ; x_b - x_a >= min_img_len ; h/w1 <= 0.2 ;
                    dp <= dp_max ; Re* < 1
       objectives (both minimised): f1 = non-uniform width fraction at x_a (2 delta_5 / w)
                                    f2 = oxygen deficit 1 - <C_bot>/C_sat (area mean over the window)"""
    import matplotlib.pyplot as plt
    rows = []
    for L, w1, h in itertools.product(np.linspace(30e-3, 80e-3, 11), np.linspace(0.5e-3, 2e-3, 7), np.linspace(75e-6, 300e-6, 10)):
        d = Design(L=L, w1=w1, h=h, w_max=10 * w1, k_pdms=k_pdms)
        x_a, x_b = imaging_window(d)
        f = float(th.aspect_flux_factor(h, float(width(d, np.array(x_a)))))
        Qm3 = tau_hi / 10 * h ** 2 * w1 * f / (6 * d.mu * (1 - x_a / L))
        d = replace(d, Q_ulmin=Qm3 * 60e9)
        feas = (h / w1 <= 0.2) and (x_b - x_a >= min_img_len) and ((L - x_a) / (L - x_b) >= ratio_min)
        if not feas:
            continue
        r = evaluate(d)
        if r["dp_Pa"] > dp_max or r["Re_star"] >= 1:
            continue
        r.update(L_mm=L * 1e3, w1_mm=w1 * 1e3, h_um=h * 1e6, Q_ulmin=d.Q_ulmin,
                 f1=r["nonuniform_frac_at_xa"], f2=1 - r["meanCbot_over_Csat"])
        rows.append(r)
    R = pd.DataFrame(rows)
    F = R[["f1", "f2"]].values
    nd = np.ones(len(R), bool)
    for i in range(len(R)):
        if np.any(np.all(F <= F[i], 1) & np.any(F < F[i], 1)):
            nd[i] = False
    R["pareto"] = nd
    R.sort_values(["pareto", "f1"], ascending=[False, True]).to_csv(os.path.join(out, "pareto_designs_%s.csv" % tag), index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    sc = ax.scatter(R.f1, R.f2, c=R.h_um, s=8, cmap="viridis", alpha=0.6)
    P = R[R.pareto].sort_values("f1")
    ax.plot(P.f1, P.f2, "r.-", lw=1, label="Pareto front")
    fig.colorbar(sc, label="h [µm]")
    ax.set_xlabel("f1: non-uniform width fraction at $x_a$")
    ax.set_ylabel(r"f2: O$_2$ deficit 1 - $\langle C_{bot}\rangle/C_{sat}$")
    ax.set_title("MODEL Pareto (%s walls), $\\tau_w(x_a)$ = %g dyn/cm²" % (tag, tau_hi), fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "design_pareto_%s.png" % tag), dpi=220)
    return R


def main():
    import matplotlib; matplotlib.use("Agg")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"); os.makedirs(out, exist_ok=True)
    print("Dimensionless groups at the inlet (baseline, impermeable):")
    for Qv in (0.22, 1.0, 120.0):
        dd = dimensionless(Design(Q_ulmin=Qv, k_pdms=0.0))
        print("  Q=%6.2f uL/min  " % Qv + "  ".join("%s=%.3g" % kv for kv in dd.items()))
    T = baseline_profiles(out)
    print(T[["Q_ulmin", "wall", "tau_a_dyn", "tau_b_dyn", "minCbot_over_Csat", "hypoxic_area_frac", "Da_conv", "Phi_pdms"]].round(4).to_string(index=False))
    sweeps(out)
    for kp, tag in ((0.0, "IMP"), (5.65e-6, "PDMS")):
        R = pareto(out, k_pdms=kp, tag=tag)
        print("\nPareto (%s): %d feasible, %d non-dominated designs. First 8:" % (tag, len(R), R.pareto.sum()))
        print(R[R.pareto].sort_values("f1")[["L_mm", "w1_mm", "h_um", "Q_ulmin", "f1", "f2", "minCbot_over_Csat", "range_ratio", "dp_Pa"]].head(8).round(4).to_string(index=False))
    with open(os.path.join(out, "design_baseline_parameters.txt"), "w") as f:
        for k, v in asdict(Design()).items():
            f.write("%s = %g\n" % (k, v))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
