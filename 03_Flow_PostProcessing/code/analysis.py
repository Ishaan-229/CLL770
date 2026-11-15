#!/usr/bin/env python3
"""
analysis.py - all calculations for the CFD post-processing of the widening Hele-Shaw chip.

Reads the raw Fluent XY exports (never modified), computes every derived quantity and every
theoretical comparison, and writes:
    output/results.pkl        full results (used by make_figures.py and build_workbook.py)
    output/results_summary.json  headline numbers
    output/tables/*.csv       all comparison tables
Run:  python analysis.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import theory as T
from fluent_xy import read_xy

EPS_LIST = (0.02, 0.05, 0.10)
CLUSTER_TOL = 4e-6      # m: points closer than this in y are treated as one mesh column
SIDE_TOL = 2e-6         # m: wall nodes within this of |y| = w/2 are side-wall/corner nodes


# ----------------------------------------------------------------------------- helpers
def jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def cluster(y):
    """Indices of y sorted, split into groups separated by gaps > CLUSTER_TOL."""
    o = np.argsort(y)
    br = np.where(np.diff(y[o]) > CLUSTER_TOL)[0]
    return [o[g] for g in np.split(np.arange(len(o)), br + 1)]


def dedup(x, v):
    """Average values at identical positions (used only for interpolation/derivatives)."""
    xr = np.round(x, 10)
    u, inv = np.unique(xr, return_inverse=True)
    return u, np.bincount(inv, v) / np.bincount(inv)


def local_slope(x, v, xc, half):
    """Least-squares slope of v(x) in the window [xc - half, xc + half]."""
    m = np.abs(x - xc) <= half
    if m.sum() < 4:
        return np.nan
    A = np.column_stack([x[m] - xc, np.ones(m.sum())])
    return np.linalg.lstsq(A, v[m], rcond=None)[0][0]


def uniform_halfwidth(yc, r, eps):
    """Walk outward from the axis on each side; return (d_left, d_right, limited_flags).
    yc: sorted cluster positions, r: tau/tau0. Linear interpolation of the eps crossing."""
    out = []
    for side in (-1, 1):
        m = (np.sign(yc) == side) | (yc == 0)
        ys = np.abs(yc[m]); rs = r[m]
        o = np.argsort(ys); ys, rs = ys[o], rs[o]
        dev = np.abs(rs - 1)
        bad = np.where(dev > eps)[0]
        if len(bad) == 0:
            out.append((ys[-1], True))
            continue
        k = bad[0]
        if k == 0:
            out.append((0.0, True))
            continue
        y0, y1, d0, d1 = ys[k - 1], ys[k], dev[k - 1], dev[k]
        out.append((y0 + (eps - d0) * (y1 - y0) / (d1 - d0), False))
    return out


def fine_halfwidth(y, r, eps):
    """Same criterion on a finely resolved theoretical profile (y >= 0 sorted)."""
    dev = np.abs(r - 1)
    bad = np.where(dev > eps)[0]
    if len(bad) == 0:
        return y[-1]
    k = bad[0]
    if k == 0:
        return 0.0
    return y[k - 1] + (eps - dev[k - 1]) * (y[k] - y[k - 1]) / (dev[k] - dev[k - 1])


# ----------------------------------------------------------------------------- loading
def load():
    data = {}
    inventory = []
    for c in C.CASES:
        d = os.path.join(C.RAW, c["folder"])
        p = read_xy(os.path.join(d, c["files"]["p"]))
        t = read_xy(os.path.join(d, c["files"]["tw"]))
        assert np.array_equal(p["x"], t["x"]), "pressure and shear exported on different points"
        case = dict(meta=c, cl_raw=dict(x=p["x"], p=p["y"], tw=t["y"]), trans={})
        for f, r in ((c["files"]["p"], p), (c["files"]["tw"], t)):
            inventory.append(dict(case=c["label"], folder=c["folder"], file=f, title=r["title"],
                                  x_label=r["labels"][0], y_label=r["labels"][1], key=r["key"], n_points=len(r["x"]),
                                  x_min_m=r["x"].min(), x_max_m=r["x"].max(), y_min=r["y"].min(), y_max=r["y"].max(),
                                  identified_as="axial position x [m] / %s [%s] on the y = 0 line along a wall"
                                  % (r["title"], "Pa")))
        for s in C.STATIONS:
            ts = read_xy(os.path.join(d, "trans-shear-stress-" + s.lower()))
            vs = read_xy(os.path.join(d, "trans-velocity-" + s.lower()))
            assert np.array_equal(ts["x"], vs["x"]), "shear and velocity exported on different points"
            case["trans"][s] = dict(y=ts["x"], tw=ts["y"], V=vs["y"])
            for f, r, unit in (("trans-shear-stress-" + s.lower(), ts, "Pa"), ("trans-velocity-" + s.lower(), vs, "m/s")):
                inventory.append(dict(case=c["label"], folder=c["folder"], file=f, title=r["title"],
                                      x_label=r["labels"][0], y_label=r["labels"][1], key=r["key"], n_points=len(r["x"]),
                                      x_min_m=r["x"].min(), x_max_m=r["x"].max(), y_min=r["y"].min(), y_max=r["y"].max(),
                                      identified_as="transverse position y [m] on the plane x = %.0f mm (all depths z) / %s [%s]"
                                      % (C.STATIONS[s] * 1e3, r["title"], unit)))
        data[c["key"]] = case
    return data, pd.DataFrame(inventory)


# ----------------------------------------------------------------------------- analysis
def analyse():
    data, inventory = load()
    os.makedirs(C.TABLES, exist_ok=True)
    R = dict(inventory=inventory, cases={}, theory2d={})

    # 2D quasi-3D Hele-Shaw reference at Q = 1 uL/min nominal case (Stokes: scale with Q)
    Qref = C.CASES[1]["Q"]
    s2d = T.solve_2d(Qref)
    R["theory2d"] = dict(xs=s2d["xs"], p_axis=s2d["p"][:, 0], tau_axis=s2d["tau"][:, 0], Qref=Qref,
                         Y=s2d["Y"], tau=s2d["tau"], p=s2d["p"])
    i0 = np.argmin(np.abs(s2d["xs"]))

    st_rows, cl_rows, p_rows, tr_rows, vel_rows, unif_rows, sens_rows, fb_rows = [], [], [], [], [], [], [], []
    for c in C.CASES:
        k = c["key"]; Q = c["Q"]; case = data[k]; s = Q / Qref
        x, p, tw = case["cl_raw"]["x"], case["cl_raw"]["p"], case["cl_raw"]["tw"]
        o = np.argsort(x, kind="stable"); xs_, ps_, ts_ = x[o], p[o], tw[o]
        xu, pu = dedup(x, p); _, tu = dedup(x, tw)
        n_dup = len(x) - len(xu)

        # monotonicity of pressure along x (sorted, de-duplicated)
        dpu = np.diff(pu)
        n_inc = int(np.sum(dpu > 0)); max_inc = float(dpu.max()) if n_inc else 0.0

        # local gradient by windowed regression (+-0.4 mm) on a 0.1 mm grid
        xg = np.arange(0.4e-3, C.X_END - 0.4e-3 + 1e-12, 0.1e-3)
        g_cfd = np.array([local_slope(xu, pu, xc, 0.4e-3) for xc in xg])

        # theory along the centre-line
        xa = xu.copy()
        hyp = (xa >= 0) & (xa <= C.XT)
        p0 = np.interp(0.0, xu, pu)
        p2d_axis = np.interp(xa, s2d["xs"], s2d["p"][:, 0]) * s
        tau2d_axis = np.interp(xa, s2d["xs"], s2d["tau"][:, 0]) * s

        case.update(cl=dict(x=xu, p=pu, tw=tu, x_sorted=xs_, p_sorted=ps_, tw_sorted=ts_,
                            xg=xg, dpdx=g_cfd,
                            dp_cfd=p0 - pu, dp1=np.where(hyp, T.dp1(xa, Q), np.nan), dp2=np.where(hyp, T.dp2(xa, Q), np.nan),
                            p2d=p2d_axis, tau1=np.where(hyp, T.tau1(xa, Q), np.nan), tau2=np.where(hyp, T.tau2(xa, Q), np.nan),
                            tau2d=tau2d_axis,
                            dpdx1=np.where((xg >= 0) & (xg <= C.XT), T.dpdx1(xg, Q), np.nan),
                            dpdx2=np.where((xg >= 0) & (xg <= C.XT), T.dpdx2(xg, Q), np.nan),
                            dpdx2d=np.gradient(np.interp(xg, s2d["xs"], s2d["p"][:, 0]) * s, xg)))

        # pressure summary
        p_end = float(pu[-1]); p45 = float(np.interp(C.XT, xu, pu))
        p_in_2d = float(s2d["p"][0, 0] * s); p0_2d = float(s2d["p"][i0, 0] * s)
        leg = C.CONTOUR_LEGEND[k]
        p_rows.append(dict(case=c["label"], Q_uL_min=c["Q_ulmin"], mdot_kg_s=c["mdot"],
                           p_x0_CFD_Pa=p0, p_x45_CFD_Pa=p45, p_end_CFD_Pa=p_end,
                           dp_0_45_CFD_Pa=p0 - p45, dp_0_45_T1_Pa=float(T.dp1(C.XT, Q)), dp_0_45_T2_Pa=float(T.dp2(C.XT, Q)[0]),
                           dp_0_45_T3_Pa=float((s2d["p"][i0, 0] - np.interp(C.XT, s2d["xs"], s2d["p"][:, 0])) * s),
                           dp_45_end_CFD_Pa=p45 - p_end, dp_45_end_T3_Pa=float(np.interp(C.XT, s2d["xs"], s2d["p"][:, 0]) * s),
                           dp_0_end_CFD_Pa=p0 - p_end, dp_0_end_T3_Pa=p0_2d,
                           p_inlet_T3_Pa=p_in_2d, p_max_contour_legend_Pa=leg["p_max"],
                           n_points=len(x), n_duplicate_x=n_dup, n_pressure_increases=n_inc, max_pressure_increase_Pa=max_inc,
                           Re_inlet=float(T.reynolds(0.0, Q))))

        # station table (centre-line)
        for sname, xs in C.STATIONS.items():
            tc = float(np.interp(xs, xu, tu)); t1 = float(T.tau1(xs, Q)); t2 = float(T.tau2(xs, Q))
            t3 = float(np.interp(xs, s2d["xs"], s2d["tau"][:, 0]) * s)
            gC = local_slope(xu, pu, xs, 0.4e-3)
            st_rows.append(dict(case=c["label"], Q_uL_min=c["Q_ulmin"], station=sname, x_mm=xs * 1e3, w_mm=float(T.width(xs)) * 1e3,
                                h_over_w=C.H / float(T.width(xs)), Re=float(T.reynolds(xs, Q)),
                                tau_CFD_Pa=tc, tau_T1_Pa=t1, tau_T2_Pa=t2, tau_T3_Pa=t3,
                                abs_err_T1_Pa=abs(tc - t1), pct_err_T1=100 * abs(tc - t1) / t1, signed_dev_T1_pct=100 * (tc / t1 - 1),
                                signed_dev_T2_pct=100 * (tc / t2 - 1), signed_dev_T3_pct=100 * (tc / t3 - 1),
                                dpdx_CFD_Pa_m=gC, dpdx_T1_Pa_m=float(T.dpdx1(xs, Q)), dpdx_T2_Pa_m=float(T.dpdx2(xs, Q)),
                                dpdx_dev_T1_pct=100 * (gC / float(T.dpdx1(xs, Q)) - 1), dpdx_dev_T2_pct=100 * (gC / float(T.dpdx2(xs, Q)) - 1),
                                p_CFD_Pa=float(np.interp(xs, xu, pu)), dp_from0_CFD_Pa=p0 - float(np.interp(xs, xu, pu)),
                                dp_from0_T1_Pa=float(T.dp1(xs, Q)), dp_from0_T2_Pa=float(T.dp2(xs, Q)[0])))
            fb_rows.append(dict(case=c["label"], station=sname, x_mm=xs * 1e3, tau_CFD_Pa=tc, dpdx_CFD_Pa_m=gC,
                                half_h_dpdx_Pa=0.5 * C.H * abs(gC), ratio_tau_over_half_h_dpdx=tc / (0.5 * C.H * abs(gC)),
                                h_eff_um=2 * tc / abs(gC) * 1e6))

        # centre-line comparison every 2.5 mm (0 .. 45) for the consolidated table
        for xmm in np.arange(0, 45.01, 2.5):
            xx = xmm * 1e-3
            tc = float(np.interp(xx, xu, tu)); t1 = float(T.tau1(xx, Q))
            cl_rows.append(dict(case=c["label"], Q_uL_min=c["Q_ulmin"], x_mm=xmm, tau_CFD_Pa=tc, tau_T1_Pa=t1,
                                tau_T2_Pa=float(T.tau2(xx, Q)), tau_T3_Pa=float(np.interp(xx, s2d["xs"], s2d["tau"][:, 0]) * s),
                                abs_err_T1_Pa=abs(tc - t1), pct_err_T1=100 * abs(tc - t1) / t1, signed_dev_T1_pct=100 * (tc / t1 - 1),
                                signed_dev_T2_pct=100 * (tc / float(T.tau2(xx, Q)) - 1)))

        # linearity of tau(x) in the central region 5..40 mm
        m = (xu >= 5e-3) & (xu <= 40e-3)
        A = np.column_stack([xu[m], np.ones(m.sum())]); coef, res, *_ = np.linalg.lstsq(A, tu[m], rcond=None)
        pred = A @ coef; r2 = 1 - np.sum((tu[m] - pred) ** 2) / np.sum((tu[m] - tu[m].mean()) ** 2)
        case["lin_fit"] = dict(slope=coef[0], intercept=coef[1], x_intercept_mm=-coef[1] / coef[0] * 1e3, r2=r2,
                               max_resid_pct=float(np.max(np.abs(tu[m] / pred - 1)) * 100))
        # aspect-ratio regressions over 2..40 mm
        m2 = (xu >= 2e-3) & (xu <= 40e-3)
        hw = C.H / T.width(xu[m2])
        dev1 = tu[m2] / T.tau1(xu[m2], Q) - 1                      # CFD vs Eq. 2
        dev2 = tu[m2] / T.tau2(xu[m2], Q) - 1                      # CFD vs rectangular-duct theory
        inv_f = 1 / T.flux_factor(C.H, T.width(xu[m2]))
        A2 = np.column_stack([hw, np.ones(m2.sum())])
        a_dev2 = np.linalg.lstsq(A2, dev2, rcond=None)[0]
        Af = np.column_stack([inv_f - 1, np.ones(m2.sum())])       # CFD/T1 - 1 = A (1/f - 1) + B
        a_f = np.linalg.lstsq(Af, dev1, rcond=None)[0]
        case["aspect_fit"] = dict(dev_T2_slope_vs_hw=a_dev2[0], dev_T2_offset=a_dev2[1],
                                  dev_T2_mean_pct=100 * dev2.mean(), dev_T2_std_pct=100 * dev2.std(),
                                  sidewall_fraction_A=a_f[0], uniform_offset_B=a_f[1])
        # T2 linear fit for comparison (T2 itself is not exactly linear in x)
        t2m = T.tau2(xu[m], Q); cf2 = np.linalg.lstsq(A, t2m, rcond=None)[0]
        case["lin_fit"].update(T2_x_intercept_mm=-cf2[1] / cf2[0] * 1e3,
                               T2_max_resid_pct=float(np.max(np.abs(t2m / (A @ cf2) - 1)) * 100))

        # ------------------------------------------------ transverse profiles
        case["tprof"] = {}
        for sname, xs in C.STATIONS.items():
            tr = case["trans"][sname]
            y, tws, V = tr["y"], tr["tw"], tr["V"]
            w = float(T.width(xs)); hw_ = w / 2
            wall = tws > 0; zero_v = V == 0
            side = wall & (np.abs(y) > hw_ - SIDE_TOL)
            tb = wall & ~side
            # pair the top/bottom wall nodes
            groups = cluster(y[tb])
            yt = y[tb]; tt = tws[tb]
            yc = np.array([yt[g].mean() for g in groups]); tc_ = np.array([tt[g].mean() for g in groups])
            ng = np.array([len(g) for g in groups])
            spread = np.array([(tt[g].max() - tt[g].min()) / tt[g].mean() if len(g) > 1 else np.nan for g in groups])
            tau0 = float(np.interp(0.0, yc, tc_))
            r = tc_ / tau0
            dvals = {e: uniform_halfwidth(yc, r, e) for e in EPS_LIST}
            # spacing of wall nodes near the side walls (resolution of d)
            dy_near = float(np.mean([np.diff(yc[yc > 0])[-1], np.diff(yc[yc < 0])[0]]))   # spacing of the two outermost columns
            gap_wall = float(np.mean([hw_ - yc.max(), yc.min() + hw_]))                     # outermost column to side wall

            # theory
            yf = np.linspace(0, hw_, 4001)
            if sname != "S5":
                rT = T.tau_bottom_ratio(yf, w)
                d_th = {e: hw_ - T.sidewall_delta(e, w) for e in EPS_LIST}
                th_name = "T2 (rectangular duct)"
            else:
                i = np.argmin(np.abs(s2d["xs"] - xs))
                rT = np.interp(yf, s2d["Y"][i], s2d["tau"][i] / s2d["tau"][i, 0])
                d_th = {e: fine_halfwidth(yf, rT, e) for e in EPS_LIST}
                th_name = "T3 (2D Hele-Shaw)"
            # the same theory sampled at the CFD wall-node positions and processed identically
            rT_nodes = np.interp(np.abs(yc), yf, rT)
            d_th_sampled = {e: uniform_halfwidth(yc, rT_nodes, e) for e in EPS_LIST}
            # 2D model profile at this station (in-plane effects only)
            i = np.argmin(np.abs(s2d["xs"] - xs))
            r2d = np.interp(yf, s2d["Y"][i], s2d["tau"][i] / s2d["tau"][i, 0])

            # velocity: upper envelope of |V| over all depths (max within +-ENV_HALF of each sample y)
            yi, Vi = y[V > 0], V[V > 0]
            oo = np.argsort(yi); yi, Vi = yi[oo], Vi[oo]
            env_half = 0.015e-3
            # keep only nodes that are the maximum within +-env_half of themselves (upper hull of the scatter);
            # this avoids shifting the steep near-wall part of the envelope towards the wall
            keep = np.array([Vi[j] >= Vi[np.abs(yi - yi[j]) <= env_half].max() for j in range(len(yi))])
            env_y, env_u = yi[keep], Vi[keep]
            umax_sample = float(V.max())
            core = np.abs(env_y) <= 0.1 * hw_
            u0 = float(np.interp(0.0, env_y, env_u))
            col_y, col_u, ncol_full = env_y, env_u, len(env_y)
            if sname != "S5":
                uT = T.u_rect(yf, C.H / 2, w, Q); uT43 = T.u_rect(yf, 0.43 * C.H, w, Q)
                u_th0 = float(uT[0])
            else:
                i = np.argmin(np.abs(s2d["xs"] - xs))
                uT = np.interp(yf, s2d["Y"][i], s2d["tau"][i]) * s * C.H / (4 * C.MU)   # mid-plane u = h tau / 4 mu
                uT43 = uT * 4 * 0.43 * 0.57
                u_th0 = float(uT[0])
            dvel = {e: uniform_halfwidth(col_y, col_u / u0, e) for e in EPS_LIST}

            case["tprof"][sname] = dict(w=w, yc=yc, tauc=tc_, ng=ng, spread=spread, tau0=tau0, r=r, d=dvals, dy_near=dy_near,
                                        y_side=y[side], tau_side=tws[side], yf=yf, rT=rT, r2d=r2d, d_th=d_th, th_name=th_name,
                                        col_y=col_y, col_u=col_u, env_y=env_y, env_u=env_u, u0=u0, umax_sample=umax_sample,
                                        uT=uT, uT43=uT43, u_th0=u_th0, dvel=dvel, d_th_sampled=d_th_sampled, yi=yi, Vi=Vi,
                                        y_all=y, V_all=V, n_wall=int(wall.sum()), n_zero_v=int(zero_v.sum()),
                                        n_wall_and_zero=int((wall & zero_v).sum()), n_side=int(side.sum()), n_full_cols=ncol_full)
            tau0_cl = float(np.interp(xs, xu, tu))
            tr_rows.append(dict(case=c["label"], station=sname, x_mm=xs * 1e3, w_mm=w * 1e3, n_points=len(y), n_wall_nodes=int(wall.sum()),
                                n_side_corner_nodes=int(side.sum()), n_top_bottom_nodes=int(tb.sum()), n_wall_columns=len(yc),
                                mean_top_bottom_spread_pct=float(np.nanmean(spread) * 100), max_top_bottom_spread_pct=float(np.nanmax(spread) * 100),
                                tau0_transverse_Pa=tau0, tau0_centreline_Pa=tau0_cl, diff_pct=100 * (tau0 / tau0_cl - 1),
                                tau_side_mean_Pa=float(tws[side].mean()) if side.any() else np.nan,
                                tau_side_over_tau0=float(tws[side].mean() / tau0) if side.any() else np.nan,
                                wall_node_spacing_near_side_mm=dy_near * 1e3, outermost_node_to_side_wall_mm=gap_wall * 1e3))
            for e in EPS_LIST:
                (dl, fl), (dr, fr) = dvals[e]
                dmin = min(dl, dr)
                unif_rows.append(dict(case=c["label"], Q_uL_min=c["Q_ulmin"], station=sname, x_mm=xs * 1e3, eps_pct=e * 100,
                                      half_width_mm=hw_ * 1e3, d_left_mm=dl * 1e3, d_right_mm=dr * 1e3, d_min_mm=dmin * 1e3,
                                      limited_by_sampling=bool(fl or fr), uniform_width_fraction=2 * dmin / w,
                                      d_theory_mm=d_th[e] * 1e3, theory=th_name, d_minus_theory_mm=(dmin - d_th[e]) * 1e3,
                                      d_theory_at_CFD_nodes_mm=min(d_th_sampled[e][0][0], d_th_sampled[e][1][0]) * 1e3,
                                      d_minus_theory_at_nodes_mm=(dmin - min(d_th_sampled[e][0][0], d_th_sampled[e][1][0])) * 1e3,
                                      sidewall_layer_CFD_mm=(hw_ - dmin) * 1e3, sidewall_layer_theory_mm=(hw_ - d_th[e]) * 1e3))
            vel_rows.append(dict(case=c["label"], Q_uL_min=c["Q_ulmin"], station=sname, x_mm=xs * 1e3, w_mm=w * 1e3,
                                 u_env_axis_CFD_m_s=u0, u_max_sampled_CFD_m_s=umax_sample,
                                 u_mid_axis_theory_m_s=u_th0, ratio_env_axis_to_theory=u0 / u_th0, ratio_max_sampled_to_theory=umax_sample / u_th0,
                                 u_parabola_from_CFD_gradient_m_s=(abs(local_slope(xu, pu, xs, 0.4e-3)) * C.H ** 2 / (8 * C.MU)),
                                 ratio_max_sampled_to_CFD_gradient_parabola=umax_sample / (abs(local_slope(xu, pu, xs, 0.4e-3)) * C.H ** 2 / (8 * C.MU)),
                                 mean_velocity_Q_over_wh_m_s=Q / (w * C.H),
                                 wall_angle_deg=float(np.degrees(np.arctan(0.5 * C.W1 * C.L / (C.L - xs) ** 2))) if sname != "S5" else 45.0))

        R["cases"][k] = case

    # ------------------------------------------------ flow-rate scaling
    P = pd.DataFrame(p_rows); ST = pd.DataFrame(st_rows); V = pd.DataFrame(vel_rows)
    scal = []
    base = C.CASES[1]
    for c in C.CASES:
        qr = c["Q"] / base["Q"]
        pr = P.loc[P.case == c["label"], "dp_0_end_CFD_Pa"].item() / P.loc[P.case == base["label"], "dp_0_end_CFD_Pa"].item()
        row = dict(case=c["label"], Q_ratio_to_1uL=qr, dp_ratio=pr, dp_dev_from_proportional_pct=100 * (pr / qr - 1),
                   dp_per_Q_Pa_per_uLmin=P.loc[P.case == c["label"], "dp_0_end_CFD_Pa"].item() / c["Q_ulmin"])
        for sname in C.STATIONS:
            tq = ST[(ST.case == c["label"]) & (ST.station == sname)].tau_CFD_Pa.item()
            tb_ = ST[(ST.case == base["label"]) & (ST.station == sname)].tau_CFD_Pa.item()
            uq = V[(V.case == c["label"]) & (V.station == sname)].u_env_axis_CFD_m_s.item()
            ub = V[(V.case == base["label"]) & (V.station == sname)].u_env_axis_CFD_m_s.item()
            row["tau_%s_dev_pct" % sname] = 100 * (tq / tb_ / qr - 1)
            row["u_%s_dev_pct" % sname] = 100 * (uq / ub / qr - 1)
        leg = C.CONTOUR_LEGEND[c["key"]]; legb = C.CONTOUR_LEGEND[base["key"]]
        row["contour_tau_max_dev_pct"] = 100 * (leg["tw_max"] / legb["tw_max"] / qr - 1)
        row["contour_p_max_dev_pct"] = 100 * (leg["p_max"] / legb["p_max"] / qr - 1)
        scal.append(row)
    SC = pd.DataFrame(scal)
    # linear regression dp vs Q through the three points
    qq = np.array([c["Q_ulmin"] for c in C.CASES]); dd = P.dp_0_end_CFD_Pa.values
    A = np.column_stack([qq, np.ones(3)]); cf = np.linalg.lstsq(A, dd, rcond=None)[0]
    r2 = 1 - np.sum((dd - A @ cf) ** 2) / np.sum((dd - dd.mean()) ** 2)
    R["dp_fit"] = dict(slope_Pa_per_uLmin=cf[0], intercept_Pa=cf[1], r2=r2,
                       resistance_Pa_s_m3=P.dp_0_end_CFD_Pa.values / np.array([c["Q"] for c in C.CASES]))

    # profile-shape collapse across flow rates
    coll = []
    for sname in C.STATIONS:
        ref = R["cases"]["Q1"]["tprof"][sname]
        for k in ("Q0.22", "Q120"):
            tp = R["cases"][k]["tprof"][sname]
            ri = np.interp(ref["yc"], tp["yc"], tp["r"])
            ui = np.interp(ref["col_y"], tp["col_y"], tp["col_u"] / tp["u0"])
            coll.append(dict(station=sname, compared=k + " vs Q1", max_abs_diff_tau_ratio=float(np.max(np.abs(ri - ref["r"]))),
                             max_abs_diff_u_ratio=float(np.max(np.abs(ui - ref["col_u"] / ref["u0"])))))
    COLL = pd.DataFrame(coll)

    # regional deviation summary (centre-line, from Eq. 2 / T2 / T3)
    reg = []
    for c in C.CASES:
        cl = R["cases"][c["key"]]["cl"]
        for name, a, b in (("inlet 0-5 mm", 0, 5e-3), ("central 5-40 mm", 5e-3, 40e-3), ("outlet transition 40-45 mm", 40e-3, 45e-3)):
            m = (cl["x"] >= a) & (cl["x"] <= b)
            d1 = cl["tw"][m] / cl["tau1"][m] - 1; d2 = cl["tw"][m] / cl["tau2"][m] - 1; d3 = cl["tw"][m] / cl["tau2d"][m] - 1
            reg.append(dict(case=c["label"], region=name, mean_dev_T1_pct=100 * np.nanmean(d1), max_abs_dev_T1_pct=100 * np.nanmax(np.abs(d1)),
                            mean_dev_T2_pct=100 * np.nanmean(d2), mean_dev_T3_pct=100 * np.nanmean(d3), max_abs_dev_T3_pct=100 * np.nanmax(np.abs(d3))))
        m = cl["x"] > C.XT
        reg.append(dict(case=c["label"], region="outlet section 45-50.25 mm", mean_dev_T1_pct=np.nan, max_abs_dev_T1_pct=np.nan,
                        mean_dev_T2_pct=np.nan, mean_dev_T3_pct=100 * np.mean(cl["tw"][m] / cl["tau2d"][m] - 1),
                        max_abs_dev_T3_pct=100 * np.max(np.abs(cl["tw"][m] / cl["tau2d"][m] - 1))))
    REG = pd.DataFrame(reg)

    # shape validation: ratios cancel any uniform magnitude offset
    shp = []
    for c in C.CASES:
        st = ST[ST.case == c["label"]].set_index("station")
        for sname in ("S1", "S2", "S4", "S5"):
            th = "tau_T3_Pa"
            cr = st.loc[sname, "tau_CFD_Pa"] / st.loc["S3", "tau_CFD_Pa"]
            tr = st.loc[sname, th] / st.loc["S3", th]
            shp.append(dict(case=c["label"], quantity="tau_w(%s) / tau_w(S3)" % sname, CFD=cr, theory=tr, theory_name="T3",
                            Eq2=st.loc[sname, "tau_T1_Pa"] / st.loc["S3", "tau_T1_Pa"], diff_vs_theory_pct=100 * (cr / tr - 1),
                            diff_vs_Eq2_pct=100 * (cr / (st.loc[sname, "tau_T1_Pa"] / st.loc["S3", "tau_T1_Pa"]) - 1)))
        cl = R["cases"][c["key"]]["cl"]; s_ = c["Q"] / Qref
        m = (cl["x"] >= 0) & (cl["x"] <= C.X_END)
        pc = cl["p"][m] / cl["p"][m][0]; pt = cl["p2d"][m] / cl["p2d"][m][0]
        shp.append(dict(case=c["label"], quantity="max |p(x)/p(0) CFD - T3| over 0-50.25 mm", CFD=np.nan, theory=np.nan, theory_name="T3",
                        Eq2=np.nan, diff_vs_theory_pct=100 * float(np.max(np.abs(pc - pt))), diff_vs_Eq2_pct=np.nan))
        mm = cl["x"][m] < 49.5e-3
        r3 = np.where(mm, cl["p"][m] / np.where(mm, cl["p2d"][m], 1.0), np.nan)
        shp.append(dict(case=c["label"], quantity="p_CFD / p_T3 along 0-49.5 mm: mean (theory col = std, %)", CFD=float(np.mean(r3[mm])), theory=100 * float(np.std(r3[mm])),
                        theory_name="T3", Eq2=np.nan, diff_vs_theory_pct=100 * (float(np.mean(r3[mm])) - 1), diff_vs_Eq2_pct=np.nan))
    SHP = pd.DataFrame(shp)

    tables = dict(file_inventory=inventory, pressure_summary=P, station_comparison=ST, centreline_comparison=pd.DataFrame(cl_rows),
                  transverse_summary=pd.DataFrame(tr_rows), uniformity=pd.DataFrame(unif_rows), velocity_summary=V,
                  flow_rate_scaling=SC, profile_collapse=COLL, force_balance=pd.DataFrame(fb_rows), regional_deviation=REG, shape_validation=SHP)
    for n, df in tables.items():
        df.to_csv(os.path.join(C.TABLES, n + ".csv"), index=False)
    R["tables"] = tables
    with open(os.path.join(C.OUT, "results.pkl"), "wb") as fh:
        pickle.dump(R, fh)

    summ = dict(dp_fit=R["dp_fit"], lin_fit={k: R["cases"][k]["lin_fit"] for k in R["cases"]},
                aspect_fit={k: R["cases"][k]["aspect_fit"] for k in R["cases"]})
    with open(os.path.join(C.OUT, "results_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2, default=jsonable)
    return R


if __name__ == "__main__":
    R = analyse()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for n in ("pressure_summary", "station_comparison", "transverse_summary", "uniformity", "velocity_summary",
              "flow_rate_scaling", "profile_collapse", "force_balance", "regional_deviation", "shape_validation"):
        print("\n=====", n); print(R["tables"][n].round(5).to_string(index=False))
    print(json.dumps(dict(dp_fit=R["dp_fit"], lin={k: R["cases"][k]["lin_fit"] for k in R["cases"]},
                          asp={k: R["cases"][k]["aspect_fit"] for k in R["cases"]}), indent=1, default=jsonable))
