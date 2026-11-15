"""Parser for ANSYS Fluent XY-plot ASCII exports ((title ...)(labels ...)((xy/key/label ...) x y ...))."""
import re, numpy as np

def read_xy(path):
    title = labels = key = None
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("(title"):
                title = re.findall(r'"([^"]*)"', s)[0]
            elif s.startswith("(labels"):
                labels = re.findall(r'"([^"]*)"', s)
            elif s.startswith("((xy/key/label"):
                key = re.findall(r'"([^"]*)"', s)[0]
            elif s.startswith(")"):
                continue
            else:
                a, b = s.split()[:2]
                xs.append(float(a)); ys.append(float(b))
    return dict(title=title, labels=labels, key=key, x=np.array(xs), y=np.array(ys))
