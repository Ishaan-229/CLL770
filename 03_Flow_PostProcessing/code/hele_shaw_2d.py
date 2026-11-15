#!/usr/bin/env python3
"""
hele_shaw_2d.py
===============
Two-dimensional depth-averaged Hele-Shaw (Darcy-type) model of the planar channel,
solved with linear finite elements on a mapped structured mesh (numpy/scipy only).

Physics (lubrication limit h << w, Re*=Re_h*h/w << 1):
    depth-averaged velocity   U = -(h^2 / 12 mu) grad p
    continuity                div(U) = 0   ->   Laplace(p) = 0 in the planform
    side walls                dp/dn = 0 (no penetration; NO side-wall no-slip)
    inlet (x = -l_in)         uniform depth-integrated flux Q/w1
    outlet                    p = 0
    wall shear (bottom/top)   tau_w(x, y) = 6 mu |U| / h = (h/2) |grad p|

Why: the 1D formula assumes U is uniform across every cross-section x = const.
Near x_t the walls diverge at 45 deg and that assumption fails; this model shows
how much. It does NOT capture the side-wall boundary layer (thickness ~ h), which
only the 3D CFD resolves. So:   1D theory  <  2D Hele-Shaw  <  3D CFD  (fidelity).

Run:  python hele_shaw_2d.py --wmax 10      (writes CSV + PNG into ./output)
"""
import argparse
import math
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import hele_shaw_theory as th


def half_width_fun(L, w1, wmax, l_in, r_out, l_out):
    x_t = L * (1 - w1 / wmax)
    x_tr = x_t + (wmax / 2 - r_out)
    x_end = x_tr + l_out

    def b(x):
        x = np.asarray(x, float)
        out = np.where(x < 0, w1 / 2, 0.5 * w1 * L / (L - np.minimum(x, x_t)))
        out = np.where(x > x_t, wmax / 2 - (x - x_t), out)
        out = np.where(x > x_tr, r_out, out)
        return out
    return b, x_t, x_tr, x_end


def solve(L=50e-3, w1=1e-3, h=145e-6, wmax=10e-3, Q=th.ulmin_to_m3s(1.0), mu=th.MU,
          l_in=2e-3, r_out=0.75e-3, l_out=1e-3, dx=25e-6, n_eta=61):
    b, x_t, x_tr, x_end = half_width_fun(L, w1, wmax, l_in, r_out, l_out)
    # x nodes include the corners exactly
    corners = np.array([-l_in, 0.0, x_t, x_tr, x_end])
    xs = np.arange(-l_in, x_end, dx)
    xs = xs[np.min(np.abs(xs[:, None] - corners[None, :]), axis=1) > 0.3 * dx]   # avoid sliver columns
    xs = np.unique(np.concatenate([xs, corners]))
    eta = np.linspace(0, 1, n_eta)                      # half domain, y = eta * b(x)
    X, E = np.meshgrid(xs, eta, indexing="ij")
    Y = E * b(X)
    nx, ne = X.shape
    idx = lambda i, j: i * ne + j
    pts = np.column_stack([X.ravel(), Y.ravel()])
    tris = []
    for i in range(nx - 1):
        for j in range(ne - 1):
            a, bb, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [(a, bb, c), (a, c, d)]
    T = np.array(tris)
    # P1 stiffness
    P0, P1, P2 = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
    det = (P1[:, 0] - P0[:, 0]) * (P2[:, 1] - P0[:, 1]) - (P2[:, 0] - P0[:, 0]) * (P1[:, 1] - P0[:, 1])
    area = 0.5 * np.abs(det)
    bvec = np.stack([P1[:, 1] - P2[:, 1], P2[:, 1] - P0[:, 1], P0[:, 1] - P1[:, 1]], 1) / det[:, None]
    cvec = np.stack([P2[:, 0] - P1[:, 0], P0[:, 0] - P2[:, 0], P1[:, 0] - P0[:, 0]], 1) / det[:, None]
    k = h ** 3 / (12 * mu)
    Ke = k * area[:, None, None] * (bvec[:, :, None] * bvec[:, None, :] + cvec[:, :, None] * cvec[:, None, :])
    rows = np.repeat(T, 3, axis=1).ravel(); cols = np.tile(T, (1, 3)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(len(pts),) * 2)
    F = np.zeros(len(pts))
    # inlet Neumann: depth-integrated flux Q/w1 per unit length (half domain gets Q/2)
    q = Q / w1
    for j in range(ne - 1):
        a, c = idx(0, j), idx(0, j + 1)
        Lseg = abs(pts[c, 1] - pts[a, 1])
        F[a] += q * Lseg / 2; F[c] += q * Lseg / 2
    out_nodes = np.array([idx(nx - 1, j) for j in range(ne)])
    free = np.setdiff1d(np.arange(len(pts)), out_nodes)
    p = np.zeros(len(pts))
    p[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    # element gradients -> nodal (area-weighted average)
    gx = (bvec * p[T]).sum(1); gy = (cvec * p[T]).sum(1)
    Gx = np.bincount(T.ravel(), np.repeat(gx * area, 3), len(pts)) / np.bincount(T.ravel(), np.repeat(area, 3), len(pts))
    Gy = np.bincount(T.ravel(), np.repeat(gy * area, 3), len(pts)) / np.bincount(T.ravel(), np.repeat(area, 3), len(pts))
    tau = 0.5 * h * np.hypot(Gx, Gy)
    return dict(X=X, Y=Y, p=p.reshape(nx, ne), tau=tau.reshape(nx, ne), b=b, x_t=x_t, x_end=x_end,
                xs=xs, params=dict(L=L, w1=w1, h=h, wmax=wmax, Q=Q, mu=mu))


def uniform_half_width(sol, x_station, eps):
    """Largest |y| (from the axis) over which |tau(y) - tau(0)| <= eps * tau(0)."""
    i = np.argmin(np.abs(sol["xs"] - x_station))
    y, t = sol["Y"][i], sol["tau"][i]
    dev = np.abs(t - t[0]) / t[0]
    bad = np.where(dev > eps)[0]
    return y[-1] if len(bad) == 0 else y[bad[0] - 1] if bad[0] > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmax", type=float, default=10.0, help="mm")
    ap.add_argument("--L", type=float, default=50.0, help="mm")
    ap.add_argument("--h", type=float, default=0.145, help="mm")
    ap.add_argument("--Q", type=float, default=1.0, help="uL/min")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "output"); os.makedirs(out, exist_ok=True)
    g = dict(L=a.L * 1e-3, w1=1e-3, h=a.h * 1e-3, w_max=a.wmax * 1e-3)
    Q = th.ulmin_to_m3s(a.Q)
    sol = solve(L=g["L"], h=g["h"], wmax=g["w_max"], Q=Q)
    # self-check 1: lead-in gradient must equal the exact parallel-plate value
    i0 = np.argmin(np.abs(sol["xs"] + 1e-3))
    tw_lead = sol["tau"][i0].mean()
    exact = 6 * th.MU * Q / (g["h"] ** 2 * g["w1"])
    print("check: tau in straight lead-in = %.5e Pa, exact = %.5e Pa (rel err %.2e)" % (tw_lead, exact, abs(tw_lead / exact - 1)))
    dp1d = th.pressure_drop_hyperbolic(Q, g)
    ia = np.argmin(np.abs(sol["xs"])); ib = np.argmin(np.abs(sol["xs"] - sol["x_t"]))
    print("check: dp over hyperbola, 2D = %.4f Pa, 1D = %.4f Pa" % (sol["p"][ia, 0] - sol["p"][ib, 0], dp1d))

    rows = []
    print("\n%6s %12s %12s %8s %10s %10s %10s %10s" % ("x mm", "tau1D dyn", "tau2D(axis)", "dev %", "w/2 mm", "d(1%) mm", "d(5%) mm", "d(10%) mm"))
    for s in th.STATIONS_M:
        i = np.argmin(np.abs(sol["xs"] - s))
        t1 = th.tau_wall(s, Q, g); t2 = sol["tau"][i, 0]
        ds = [uniform_half_width(sol, s, e) for e in (0.01, 0.05, 0.10)]
        rows.append([s * 1e3, t1 * 10, t2 * 10, 100 * (t2 / t1 - 1), sol["b"](s) * 1e3] + [d * 1e3 for d in ds])
        print("%6.1f %12.5f %12.5f %8.2f %10.3f %10.3f %10.3f %10.3f" % tuple(rows[-1]))
    np.savetxt(os.path.join(out, "hs2d_W%g_Q%g_stations.csv" % (a.wmax, a.Q)), np.array(rows), delimiter=",",
               header="x_mm,tau_1D_dyn_cm2,tau_2D_axis_dyn_cm2,dev_percent,half_width_mm,d_1pct_mm,d_5pct_mm,d_10pct_mm", comments="")
    i_ax = sol["tau"][:, 0]
    np.savetxt(os.path.join(out, "hs2d_W%g_Q%g_centreline.csv" % (a.wmax, a.Q)),
               np.column_stack([sol["xs"] * 1e3, i_ax * 10, th.tau_wall(np.clip(sol["xs"], 0, None), Q, g) * 10]),
               delimiter=",", header="x_mm,tau_2D_axis_dyn_cm2,tau_1D_dyn_cm2", comments="")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 6.5))
    ax1 = fig.add_subplot(2, 1, 1)
    X, Y, T = sol["X"] * 1e3, sol["Y"] * 1e3, sol["tau"] * 10
    for sgn in (1, -1):
        cs = ax1.pcolormesh(X, sgn * Y, T, shading="gouraud", cmap="viridis", vmax=np.percentile(T, 99.5))
    for s in th.STATIONS_M:
        ax1.axvline(s * 1e3, color="w", lw=0.6, ls=":")
    fig.colorbar(cs, ax=ax1, label=r"$\tau_w$ [dyn cm$^{-2}$]")
    ax1.set_aspect("equal"); ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
    ax1.set_title("2D Hele-Shaw model (MODEL, not CFD): wall shear, L=%g mm, w_max=%g mm, Q=%g uL/min" % (a.L, a.wmax, a.Q), fontsize=9)
    ax2 = fig.add_subplot(2, 2, 3)
    xx = sol["xs"]
    ax2.plot(xx * 1e3, i_ax * 10, label="2D Hele-Shaw, axis")
    xm = xx[(xx >= 0) & (xx <= sol["x_t"])]
    ax2.plot(xm * 1e3, th.tau_wall(xm, Q, g) * 10, "--", label="1D Eq. 2")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel(r"$\tau_w$ [dyn cm$^{-2}$]"); ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(2, 2, 4)
    for s in th.STATIONS_M:
        i = np.argmin(np.abs(sol["xs"] - s))
        ax3.plot(sol["Y"][i] * 1e3, sol["tau"][i] / sol["tau"][i, 0], label="x=%g mm" % (s * 1e3))
    ax3.set_xlabel("distance from axis y [mm]"); ax3.set_ylabel(r"$\tau_w(y)/\tau_w(0)$"); ax3.legend(fontsize=7)
    ax3.axhspan(0.95, 1.05, color="0.9", zorder=0)
    fig.tight_layout(); fn = os.path.join(out, "hs2d_W%g_Q%g.png" % (a.wmax, a.Q)); fig.savefig(fn, dpi=180)
    print("wrote", fn)


if __name__ == "__main__":
    main()
