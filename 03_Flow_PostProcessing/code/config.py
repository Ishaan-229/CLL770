"""Project constants, case definitions and folder layout (all SI unless the name says otherwise).

Values marked [SETUP] were read from the Fluent report files (report.xml) shipped with the results.
Values marked [PROJECT] are the project geometry; h is not contained in the exports and is taken
from the project specification (checked indirectly by the force balance tau = (h/2)|dp/dx|).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # package root
RAW = os.path.join(ROOT, "02_CFD_Results_Raw", "raw_fluent_exports")
OUT = os.path.join(ROOT, "03_Flow_PostProcessing", "regenerated", "output")
TABLES = os.path.join(OUT, "tables")
GRAPHS = os.path.join(ROOT, "03_Flow_PostProcessing", "regenerated", "Graphs")

# ---- fluid [SETUP: report.xml, material 'material']
RHO = 981.8          # kg/m^3
MU = 0.00094         # Pa s

# ---- geometry [PROJECT]
L = 50e-3            # m, hyperbola parameter
W1 = 1e-3            # m, inlet width
H = 145e-6           # m, channel height (NOT in the exports; project value)
WMAX = 10e-3         # m, width at the end of the hyperbolic section
XT = L * (1 - W1 / WMAX)   # 45 mm, end of the hyperbolic section
L_IN = 2e-3          # m, straight lead-in upstream of x = 0 (inferred, see report Sec. 2)
R_OUT = 0.75e-3      # m, outlet half-width (outlet duct 1.5 mm wide)
L_OUT = 1e-3         # m, outlet duct length
X_END = XT + (WMAX / 2 - R_OUT) + L_OUT   # 50.25 mm (matches last exported x)

STATIONS = {"S1": 5e-3, "S2": 15e-3, "S3": 25e-3, "S4": 35e-3, "S5": 45e-3}

# ---- cases: folder name -> mass flow [SETUP]; Q = mdot / rho
CASES = [
    dict(key="Q0.22", folder="q=3.6e-9", mdot=3.6e-9,
         files=dict(p="pressure-vs-x-centreline", tw="wall-shear-vs-x-centreline")),
    dict(key="Q1", folder="q=1.636e-8", mdot=1.636e-8,
         files=dict(p="pressure-vs-x", tw="wall-shear-stress-vs-x")),
    dict(key="Q120", folder="q = 120", mdot=1.964e-6,
         files=dict(p="pressure-vs-x", tw="wall-shear-stress-vs-x")),
]
for c in CASES:
    c["Q"] = c["mdot"] / RHO                      # m^3/s
    c["Q_ulmin"] = c["Q"] * 1e9 * 60              # uL/min
    c["nominal"] = {"Q0.22": "0.22", "Q1": "1", "Q120": "120"}[c["key"]]   # nominal flow rate, uL/min
    c["label"] = "Q = %s µL/min" % c["nominal"]

# contour colour-bar extrema read from the user's Fluent screenshots (whole domain)
CONTOUR_LEGEND = {
    "Q0.22": dict(p_max=0.395, tw_max=0.00332, tw_min=1.6e-7),
    "Q1": dict(p_max=1.8, tw_max=0.0151, tw_min=7.29e-7),
    "Q120": dict(p_max=215.6, tw_max=1.825, tw_min=8.744e-5),
}

# chart colours (validated categorical slots 1-3 of the reference palette) and neutral theory inks
COLORS = {"Q0.22": "#2a78d6", "Q1": "#eb6834", "Q120": "#1baf7a"}
MARKERS = {"Q0.22": "o", "Q1": "s", "Q120": "^"}
STATION_COLORS = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]   # ordinal blue ramp S1->S5
INK = "#222222"; INK2 = "#666666"; GRID = "#e3e3e0"
