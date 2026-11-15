#!/usr/bin/env python3
"""
make_synthetic_exports.py  -  SOFTWARE TEST DATA ONLY
=====================================================
Writes files in Fluent's ASCII-export format, built from ANALYTICAL formulas
(parallel-plate profile x rectangular-duct side-wall decay, design-model oxygen),
so that validate_shear.py, mesh_independence.py and oxygen_post.py can be tested
before any CFD exists. These numbers are NOT simulation results and must never be
reported as such. Every file name starts with SYNTHETIC_.

Run from 04_postprocessing:   python selftest/make_synthetic_exports.py && bash selftest/run_selftest.sh
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "02_analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "05_design"))
import hele_shaw_theory as th  # noqa: E402

L, W1, H = 50e-3, 1e-3, 145e-6
G = dict(L=L, w1=W1, h=H, w_max=10e-3)
MU = th.MU
OUT = os.path.join(HERE, "synthetic_data"); os.makedirs(OUT, exist_ok=True)


def b_of_x(x):
    return np.where(x <= 45e-3, 0.5 * W1 * L / (L - np.minimum(x, 45e-3)), 5e-3)


def u_field(x, y, z, Q, nterms=15):
    """Fully developed rectangular-duct velocity scaled to the local section (lubrication)."""
    b = b_of_x(x); w = 2 * b
    f = th.aspect_flux_factor(H, w)
    G_ = 12 * MU * Q / (H ** 3 * w * f)
    u = G_ / (2 * MU) * z * (H - z)
    for n in range(1, 2 * nterms, 2):
        A_, C_ = n * np.pi * np.abs(y) / H, n * np.pi * b / H          # overflow-safe cosh(A)/cosh(C)
        ratio = np.exp(A_ - C_) * (1 + np.exp(-2 * A_)) / (1 + np.exp(-2 * C_))
        u -= 4 * G_ * H ** 2 / (MU * np.pi ** 3 * n ** 3) * np.sin(n * np.pi * z / H) * ratio
    return u


def write(fn, cols, data):
    with open(os.path.join(OUT, fn), "w") as f:
        f.write(", ".join(["nodenumber"] + cols) + "\n")
        for i, row in enumerate(data):
            f.write(", ".join(["%d" % (i + 1)] + ["%.8e" % v for v in row]) + "\n")


def run(Q_ul, tag, noise=0.0, dz_bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = th.ulmin_to_m3s(Q_ul)
    rows = []
    for xs in th.STATIONS_M:
        z = np.linspace(0, H, 146)
        u = u_field(np.full_like(z, xs), np.zeros_like(z), z, Q) * (1 + noise * rng.standard_normal(z.size))
        sr = np.abs(np.gradient(u, z))
        rows += [[xs, 0, zz, uu, 0, 0, abs(uu), s, 0] for zz, uu, s in zip(z, u, sr)]
    cols = ["x-coordinate", "y-coordinate", "z-coordinate", "x-velocity", "y-velocity", "z-velocity", "velocity-magnitude", "strain-rate-mag", "pressure"]
    write("SYNTHETIC_%s_zrakes.csv" % tag, cols, rows)
    rows = []
    for xs in th.STATIONS_M:
        b = float(b_of_x(xs)) * 0.999
        y = np.linspace(-b, b, 801)
        dz = 1e-7
        sr = (u_field(np.full_like(y, xs), y, np.full_like(y, 20e-6 + dz), Q) - u_field(np.full_like(y, xs), y, np.full_like(y, 20e-6 - dz), Q)) / (2 * dz)
        uu = u_field(np.full_like(y, xs), y, np.full_like(y, 20e-6), Q)
        rows += [[xs, yy, 20e-6, a, 0, 0, abs(a), abs(s), 0] for yy, a, s in zip(y, uu, sr)]
    write("SYNTHETIC_%s_yrakes_z20.csv" % tag, cols, rows)
    # bottom wall: wall shear from du/dz at z=0 (first-cell bias optional, mimics a coarse mesh)
    xs_ = np.linspace(-2e-3, 45e-3, 471)
    rows = []
    for x in xs_:
        b = float(b_of_x(max(x, 0)))
        for e in np.linspace(-1, 1, 41):
            y = e * b * 0.999
            dz = 1e-8
            tw = MU * u_field(np.array(max(x, 0)), np.array(y), np.array(dz), Q) / dz * (1 - dz_bias)
            rows.append([x, y, 0.0, tw, tw, 0, 0])
    write("SYNTHETIC_%s_wall_bottom.csv" % tag, ["x-coordinate", "y-coordinate", "z-coordinate", "wall-shear", "x-wall-shear", "y-wall-shear", "z-wall-shear"], rows)


def oxygen(tag, Q_ul, kp):
    from design_model import Design, oxygen_profile
    d = Design(Q_ulmin=Q_ul, k_pdms=kp)
    P = oxygen_profile(d)
    rows = []
    for x, cb, tau in zip(P.x[::5], P.Cbot[::5], P.tau[::5]):
        b = float(b_of_x(x))
        for e in np.linspace(-1, 1, 31):
            side = 1 - 0.3 * np.exp(-(1 - abs(e)) * b / H)      # crude side-wall depletion, test only
            rows.append([x, e * b * 0.999, 0.0, tau * side, cb * side])
    write("SYNTHETIC_%s_wall_bottom_o2.csv" % tag, ["x-coordinate", "y-coordinate", "z-coordinate", "wall-shear", "uds-0-scalar"], rows)


if __name__ == "__main__":
    run(1.0, "Q1")
    # pseudo mesh study: wall-shear bias ~ dz1/(2h) as a first-order wall-gradient would give
    for m, bias in (("M1", 1 / 16), ("M2", 1 / 24), ("M3", 1 / 36)):
        run(1.0, "Q1_%s" % m, dz_bias=bias)
    for q, kp, t in ((0.22, 0.0, "Q022_IMP"), (1.0, 0.0, "Q1_IMP"), (1.0, 5.65e-6, "Q1_PDMS")):
        oxygen(t, q, kp)
    print("SYNTHETIC test files written to", OUT)
