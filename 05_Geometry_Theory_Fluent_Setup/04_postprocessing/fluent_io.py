"""
fluent_io.py - readers for ANSYS Fluent export files used by the post-processing scripts.

read_ascii(path)  : File > Export > Solution Data > ASCII (comma or space delimited,
                    node or cell values). Returns a pandas DataFrame with clean,
                    lower-case column names, e.g. 'x-coordinate', 'x-velocity',
                    'strain-rate-mag', 'wall-shear', 'uds-0-scalar'. Coordinates in m.
read_xy(path)     : XY-plot files written with "Write to File" (.xy). Returns
                    {curve_label: DataFrame(columns=['pos', 'val'])}.
read_report(path) : .flp / .srp / .vrp report files -> {zone_name: value}.
group_by(df, col, decimals) : split a multi-surface export into groups of constant x (etc.).
"""
import re

import numpy as np
import pandas as pd

ALIASES = {
    "x": "x-coordinate", "y": "y-coordinate", "z": "z-coordinate",
    "x-coord": "x-coordinate", "y-coord": "y-coordinate", "z-coord": "z-coordinate",
    "wall-shear-stress": "wall-shear", "strain-rate": "strain-rate-mag",
    "uds-0": "uds-0-scalar", "scalar-0": "uds-0-scalar",
}


def _clean(c):
    c = c.strip().strip('"').lower()
    return ALIASES.get(c, c)


def read_ascii(path):
    with open(path) as f:
        head = f.readline()
    sep = "," if "," in head else r"\s+"
    df = pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
    df.columns = [_clean(c) for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("unnamed")]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    need = {"x-coordinate", "y-coordinate", "z-coordinate"}
    if not need.issubset(df.columns):
        raise ValueError("%s: coordinates missing; columns are %s" % (path, list(df.columns)))
    return df


def read_xy(path):
    txt = open(path).read()
    curves = {}
    for m in re.finditer(r'\(\(xy/key/label\s+"([^"]*)"\)(.*?)\)', txt, re.S):
        vals = np.array([list(map(float, ln.split())) for ln in m.group(2).strip().splitlines() if ln.strip()])
        curves[m.group(1)] = pd.DataFrame(vals[:, :2], columns=["pos", "val"])
    return curves


def read_report(path):
    out = {}
    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    for ln in open(path):
        m = re.match(r"\s*([A-Za-z_][\w\-\.:]*)\s+(%s)\s*$" % num, ln)
        if m:
            out[m.group(1).lower()] = float(m.group(2))
    return out


def group_by(df, col="x-coordinate", decimals=6):
    key = df[col].round(decimals)
    return {k: g.copy() for k, g in df.groupby(key)}
