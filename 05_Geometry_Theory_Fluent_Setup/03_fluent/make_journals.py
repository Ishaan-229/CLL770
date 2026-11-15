#!/usr/bin/env python3
"""
make_journals.py
================
Writes Fluent TUI journal files (.jou) for every run of the simulation matrix,
plus simulation_matrix.csv. Strategy (robust across Fluent versions):

  1. Build ONE template case by hand in the GUI on the medium mesh
     (materials, zone types, schemes, UDS/UDF hooks...) -> template_hydro.cas.h5
     The GUI steps are in the manual, Phase 3.
  2. Each journal reads the template, swaps the mesh with /file/replace-mesh
     (zone names are preserved), sets the inlet mass flow, solves, reports and
     exports the surfaces needed by the Python post-processing scripts.

Lines marked  ; VERIFY  have prompt sequences that differ between Fluent releases.
Run each such command once interactively in your version and correct the
journal if the prompts differ (Fluent echoes the prompts in the console).

Run:  python make_journals.py      -> ./journals/*.jou and simulation_matrix.csv
Then in Fluent: File > Read > Journal...  or launch in batch:
      fluent 3ddp -t8 -g -i journals/H_M2_Q1.jou
"""
import csv
import os

RHO = 981.8
H = 145e-6
L, W1 = 50e-3, 1e-3
STATIONS = [5e-3, 15e-3, 25e-3, 35e-3, 45e-3]
Z_CELL = 20e-6
X_IN, X_OUT = -2e-3, 50.25e-3
MESHES = {"M1": "mesh_M1_coarse.msh.h5", "M2": "mesh_M2_medium.msh.h5", "M3": "mesh_M3_fine.msh.h5"}
FLOWS = {"Q022": 0.22, "Q1": 1.0, "Q120": 120.0}
HALF = False   # True if you mesh the half model (symmetry at y = 0): mass flow is halved


def mdot(q_ulmin):
    return RHO * q_ulmin * 1e-9 / 60.0 * (0.5 if HALF else 1.0)


def half_width(x):
    return 0.5 * W1 * L / (L - x) if x <= 45e-3 else 5e-3


def surfaces_block():
    s = ["; ---- post-processing surfaces (SI, metres) ----"]
    for x in STATIONS:
        tag = "%d" % round(x * 1e3)
        s.append("/surface/rake-surface zrake_x%s %.6g 0 0 %.6g 0 %.6g 146" % (tag, x, x, H))
        b = half_width(x) * 0.999
        s.append("/surface/rake-surface yrake_z20_x%s %.6g %.6g %.6g %.6g %.6g %.6g 801" % (tag, x, -b, Z_CELL, x, b, Z_CELL))
    s.append("/surface/rake-surface xrake_axis_z20 %.6g 0 %.6g %.6g 0 %.6g 2001" % (X_IN, Z_CELL, X_OUT, Z_CELL))
    s.append("/surface/rake-surface xrake_axis_zmid %.6g 0 %.6g %.6g 0 %.6g 2001" % (X_IN, H / 2, X_OUT, H / 2))
    s.append("/surface/plane-surface plane_z20 xy-plane %.6g          ; VERIFY" % Z_CELL)
    return s


def exports_block(tag, uds=False):
    zr = " ".join("zrake_x%d" % round(x * 1e3) for x in STATIONS)
    yr = " ".join("yrake_z20_x%d" % round(x * 1e3) for x in STATIONS)
    v = "x-velocity y-velocity z-velocity velocity-magnitude strain-rate-mag pressure"
    if uds:
        v += " uds-0-scalar"
    e = ["; ---- ASCII exports read by 04_postprocessing (comma delimited, node values) ----"]
    e.append('/file/export/ascii "exp_%s_zrakes.csv" %s () yes %s () no   ; VERIFY' % (tag, zr, v))
    e.append('/file/export/ascii "exp_%s_yrakes_z20.csv" %s () yes %s () no   ; VERIFY' % (tag, yr, v))
    e.append('/file/export/ascii "exp_%s_axis.csv" xrake_axis_z20 xrake_axis_zmid () yes %s () no   ; VERIFY' % (tag, v))
    wv = "wall-shear x-wall-shear y-wall-shear z-wall-shear" + (" uds-0-scalar" if uds else "")
    e.append('/file/export/ascii "exp_%s_wall_bottom.csv" wall_bottom () yes %s () no   ; VERIFY' % (tag, wv))
    return e


def hydro_journal(mesh_key, q_key):
    tag = "H_%s_%s" % (mesh_key, q_key)
    q = FLOWS[q_key]
    j = ["; %s : laminar steady hydrodynamics, mesh %s, Q = %g uL/min, mdot = %.6e kg/s" % (tag, mesh_key, q, mdot(q)),
         "/file/set-tui-version \"22.1\"   ; harmless if unsupported - delete if it errors",
         '/file/read-case "template_hydro.cas.h5"',
         '/file/replace-mesh "%s" yes   ; VERIFY (answer may be needed for "discard data?")' % MESHES[mesh_key],
         "/mesh/check",
         "/mesh/quality",
         "/define/boundary-conditions/set/mass-flow-inlet inlet () mass-flow no %.6e quit   ; VERIFY" % mdot(q),
         "/solve/set/p-v-coupling 21                ; 20 SIMPLE, 21 SIMPLEC, 24 Coupled",
         "/solve/set/discretization-scheme/pressure 12   ; 12 second order (14 = PRESTO!)",
         "/solve/set/discretization-scheme/mom 1         ; 1 second-order upwind",
         "/solve/monitors/residual/convergence-criteria 1e-8 1e-8 1e-8 1e-8",
         "/solve/initialize/hyb-initialization",
         "/solve/iterate 3000",
         '/report/fluxes/mass-flow no inlet outlet () yes "%s_massflow.flp"   ; VERIFY' % tag,
         '/report/surface-integrals/area-weighted-avg inlet outlet () pressure yes "%s_pressure.srp"   ; VERIFY' % tag,
         '/report/volume-integrals/maximum fluid () velocity-magnitude yes "%s_vmax.vrp"   ; VERIFY' % tag]
    j += surfaces_block() + exports_block(tag)
    j += ['/file/write-case-data "%s.cas.h5"' % tag, "; end"]
    return tag, j


O2_SCEN = {  # name: (bottom model, kpdms, description)
    "IMP": ("o2_flux_bottom_mm", 0.0, "impermeable PDMS (worst case, convective supply only)"),
    "PDMS": ("o2_flux_bottom_mm", 5.65e-6, "PDMS Robin supply, t_PDMS = 5 mm, P = 800 Barrer"),
    "SAT": ("o2_flux_bottom_mm", 1.0, "top/side walls held at C_sat (best case)"),
}


def o2_journal(q_key, scen, mesh_key="M2"):
    tag = "O_%s_%s_%s" % (mesh_key, q_key, scen)
    hook, kp, desc = O2_SCEN[scen]
    j = ["; %s : oxygen transport on frozen converged flow; %s" % (tag, desc),
         '/file/read-case-data "H_%s_%s.cas.h5"' % (mesh_key, q_key),
         "; The template must already contain: 1 UDS (flux = mass flow rate), libudf loaded,",
         "; UDS diffusivity = o2_diffusivity, wall_bottom uds flux = udf %s," % hook,
         "; wall_top & wall_side uds flux = udf o2_flux_pdms, inlet uds value = 0.178 (see manual Phase 8).",
         '(load "o2_params.scm")',
         "(rpsetvar 'o2/kpdms %g)" % kp,
         "/solve/set/equations/flow no     ; freeze velocity/pressure",
         "/solve/set/equations/uds-0 yes   ; VERIFY",
         "/solve/monitors/residual/convergence-criteria 1e-8 1e-8 1e-8 1e-8 1e-10   ; VERIFY (one value per equation)",
         "/solve/iterate 5000",
         "/define/user-defined/execute-on-demand \"o2_report::libudf\""]
    j += surfaces_block() + exports_block(tag, uds=True)
    j += ['/file/write-case-data "%s.cas.h5"' % tag, "; end"]
    return tag, j


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journals")
    os.makedirs(out, exist_ok=True)
    rows = []
    for m in MESHES:
        for qk in FLOWS:
            if m != "M2" and qk == "Q120":
                continue  # mesh study at 0.22 and 1 uL/min only
            tag, j = hydro_journal(m, qk)
            open(os.path.join(out, tag + ".jou"), "w").write("\n".join(j) + "\n")
            rows.append(dict(run=tag, phase="4-5 hydro", mesh=m, Q_uL_min=FLOWS[qk], mdot_kg_s="%.6e" % mdot(FLOWS[qk]),
                             physics="laminar steady, SIMPLEC, 2nd order", extract="umax, dp, tau_w S1-S5, u(z), mass imbalance, y-profiles"))
    for qk in FLOWS:
        for sc in O2_SCEN:
            tag, j = o2_journal(qk, sc)
            open(os.path.join(out, tag + ".jou"), "w").write("\n".join(j) + "\n")
            rows.append(dict(run=tag, phase="7-8 oxygen", mesh="M2", Q_uL_min=FLOWS[qk], mdot_kg_s="%.6e" % mdot(FLOWS[qk]),
                             physics="UDS O2, MM wall uptake, " + O2_SCEN[sc][2], extract="C_w map, C(axis), C(y) at S1-S5, uptake, hypoxic fraction"))
    with open(os.path.join(os.path.dirname(out), "simulation_matrix_journals.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote %d journals to %s" % (len(rows), out))


if __name__ == "__main__":
    main()
