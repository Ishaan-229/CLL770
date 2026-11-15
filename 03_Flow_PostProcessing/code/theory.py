"""Theoretical references used for comparison with the CFD (SI units).

T1  Parallel-plate Hele-Shaw / lubrication theory (the project's Eq. 2):
        dp/dx = -12 mu Q (1 - x/L) / (h^3 w1),   tau_w = 6 mu Q (1 - x/L) / (h^2 w1)
        p(0) - p(x) = 12 mu Q / (h^3 w1) * (x - x^2 / 2L)            (0 <= x <= x_t)
T2  Same lubrication theory but with the exact fully developed flux of a w x h RECTANGULAR
    section (side-wall drag):  Q = (w h^3 / 12 mu)(-dp/dx) f(h/w),  f = 1 - 0.630 h/w + ...
        dp/dx_T2 = dp/dx_T1 / f,  centre-line tau_T2 = tau_T1 / f
    plus the exact transverse distribution of the bottom-wall shear and of u(y, z) in a rectangle.
T3  Two-dimensional depth-averaged Hele-Shaw model of the whole planform (lead-in, hyperbola,
    45-degree outlet transition, outlet duct), solved with P1 finite elements, with the
    permeability h^3 f(h/w(x)) / 12 mu so that T3 reduces to T2 in slowly varying regions.
    It captures in-plane 2D effects (outlet corner) but NOT the side-wall layer itself.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import config as C


def width(x):
    """Full width of the channel [m] for any x in the exported range (planar geometry)."""
    x = np.asarray(x, float)
    w = np.where(x < 0, C.W1, C.W1 * C.L / (C.L - np.minimum(x, C.XT)))
    x_tr = C.XT + (C.WMAX / 2 - C.R_OUT)
    w = np.where(x > C.XT, C.WMAX - 2 * (x - C.XT), w)
    w = np.where(x > x_tr, 2 * C.R_OUT, w)
    return w


def flux_factor(h, w, nterms=60):
    """Exact Q_rect / Q_parallel-plates for fully developed laminar flow in a w x h duct."""
    n = np.arange(1, 2 * nterms, 2)[:, None]
    w = np.atleast_1d(np.asarray(w, float))[None, :]
    s = (np.tanh(n * np.pi * w / (2 * h)) / n ** 5).sum(0)
    return np.squeeze(1 - 192 * h / (np.pi ** 5 * w) * s)


# ---------------- T1
def tau1(x, Q):
    """Eq. 2 for 0 <= x <= x_t (6 mu Q / h^2 w(x) with w = w1 L/(L-x)); the same local 1D
    formula with the actual width is returned elsewhere (lead-in), for reference only."""
    return 6 * C.MU * Q / (C.H ** 2 * width(x))


def dpdx1(x, Q):
    return -12 * C.MU * Q / (C.H ** 3 * width(x))


def dp1(x, Q):
    """p(0) - p(x) for 0 <= x <= x_t (user's equation)."""
    x = np.asarray(x, float)
    return 12 * C.MU * Q / (C.H ** 3 * C.W1) * (x - x ** 2 / (2 * C.L))


# ---------------- T2
def tau2(x, Q):
    return tau1(x, Q) / flux_factor(C.H, width(x))


def dpdx2(x, Q):
    return dpdx1(x, Q) / flux_factor(C.H, width(x))


def dp2(x, Q, n=4001):
    """p(0) - p(x) with the rectangular-section correction (numerical quadrature), 0 <= x <= x_t."""
    x = np.atleast_1d(np.asarray(x, float))
    xs = np.linspace(0, C.XT, n)
    g = -dpdx2(xs, Q)
    cum = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(xs))])
    return np.interp(x, xs, cum)


def tau_bottom_ratio(y, w, h=C.H, nterms=200):
    """Exact bottom-wall shear tau(y)/tau(y=0) in a fully developed w x h rectangle (y from axis)."""
    y = np.atleast_1d(np.asarray(y, float))
    n = np.arange(1, 2 * nterms, 2)[:, None]
    def tb(yy):
        arg = n * np.pi / h
        # cosh(a y)/cosh(a w/2) evaluated stably
        r = np.exp(arg * (np.abs(yy)[None, :] - w / 2)) * (1 + np.exp(-2 * arg * np.abs(yy)[None, :])) / (1 + np.exp(-arg * w))
        return 1 - (8 / (np.pi ** 2 * n ** 2) * r).sum(0)
    return tb(y) / tb(np.array([0.0]))


def u_rect(y, z, w, Q, h=C.H, nterms=200):
    """Exact fully developed velocity u(y, z) [m/s] in a w x h rectangle carrying flow Q."""
    y = np.atleast_1d(np.asarray(y, float)); z = float(z)
    n = np.arange(1, 2 * nterms, 2)[:, None]
    arg = n * np.pi / h
    r = np.exp(arg * (np.abs(y)[None, :] - w / 2)) * (1 + np.exp(-2 * arg * np.abs(y)[None, :])) / (1 + np.exp(-arg * w))
    G = 12 * C.MU * Q / (w * h ** 3 * flux_factor(h, w))
    s = (8 * h ** 2 / (np.pi ** 3 * n ** 3) * np.sin(n * np.pi * z / h) * r).sum(0)
    return G / (2 * C.MU) * (z * (h - z) - s)


def sidewall_delta(eps, w, h=C.H):
    """Distance from the side wall at which tau(y)/tau(0) = 1 - eps (exact series, root find)."""
    from scipy.optimize import brentq
    f = lambda d: tau_bottom_ratio(np.array([w / 2 - d]), w, h)[0] - (1 - eps)
    return brentq(f, 1e-9, w / 2 - 1e-9)


# ---------------- T3
def solve_2d(Q, dx=20e-6, n_eta=81, sidewall_corrected=True):
    """2D depth-averaged Hele-Shaw solution on the half planform (symmetry at y = 0)."""
    l_in, r_out, l_out = C.L_IN, C.R_OUT, C.L_OUT
    x_tr = C.XT + (C.WMAX / 2 - r_out)
    corners = np.array([-l_in, 0.0, C.XT, x_tr, C.X_END])
    xs = np.arange(-l_in, C.X_END, dx)
    xs = xs[np.min(np.abs(xs[:, None] - corners[None, :]), axis=1) > 0.3 * dx]
    xs = np.unique(np.concatenate([xs, corners]))
    eta = np.linspace(0, 1, n_eta)
    X, E = np.meshgrid(xs, eta, indexing="ij")
    Y = E * width(X) / 2
    nx, ne = X.shape
    idx = lambda i, j: i * ne + j
    pts = np.column_stack([X.ravel(), Y.ravel()])
    I, J = np.meshgrid(np.arange(nx - 1), np.arange(ne - 1), indexing="ij")
    I = I.ravel(); J = J.ravel()
    a, b_, c, d = idx(I, J), idx(I + 1, J), idx(I + 1, J + 1), idx(I, J + 1)
    T = np.vstack([np.column_stack([a, b_, c]), np.column_stack([a, c, d])])
    P0, P1, P2 = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
    det = (P1[:, 0] - P0[:, 0]) * (P2[:, 1] - P0[:, 1]) - (P2[:, 0] - P0[:, 0]) * (P1[:, 1] - P0[:, 1])
    area = 0.5 * np.abs(det)
    bv = np.stack([P1[:, 1] - P2[:, 1], P2[:, 1] - P0[:, 1], P0[:, 1] - P1[:, 1]], 1) / det[:, None]
    cv = np.stack([P2[:, 0] - P1[:, 0], P0[:, 0] - P2[:, 0], P1[:, 0] - P0[:, 0]], 1) / det[:, None]
    xc = (P0[:, 0] + P1[:, 0] + P2[:, 0]) / 3
    fk = flux_factor(C.H, width(xc)) if sidewall_corrected else np.ones(len(T))
    k = C.H ** 3 * fk / (12 * C.MU)
    Ke = (k * area)[:, None, None] * (bv[:, :, None] * bv[:, None, :] + cv[:, :, None] * cv[:, None, :])
    rows = np.repeat(T, 3, axis=1).ravel(); cols = np.tile(T, (1, 3)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(len(pts),) * 2)
    F = np.zeros(len(pts))
    q = Q / C.W1
    for j in range(ne - 1):
        a0, c0 = idx(0, j), idx(0, j + 1)
        Ls = abs(pts[c0, 1] - pts[a0, 1]); F[a0] += q * Ls / 2; F[c0] += q * Ls / 2
    out_nodes = idx(nx - 1, np.arange(ne))
    free = np.setdiff1d(np.arange(len(pts)), out_nodes)
    p = np.zeros(len(pts))
    p[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    gx = (bv * p[T]).sum(1); gy = (cv * p[T]).sum(1)
    wsum = np.bincount(T.ravel(), np.repeat(area, 3), len(pts))
    Gx = np.bincount(T.ravel(), np.repeat(gx * area, 3), len(pts)) / wsum
    Gy = np.bincount(T.ravel(), np.repeat(gy * area, 3), len(pts)) / wsum
    tau = 0.5 * C.H * np.hypot(Gx, Gy)          # tau at the centre of the gap depth-profile
    return dict(xs=xs, Y=Y, p=p.reshape(nx, ne), tau=tau.reshape(nx, ne), Q=Q)


def reynolds(x, Q):
    w = width(x)
    Dh = 2 * w * C.H / (w + C.H)
    U = Q / (w * C.H)
    return C.RHO * U * Dh / C.MU
