"""Alloy-level splits shared by all phases (never split frames of one alloy)."""
import csv
import re
from pathlib import Path

import numpy as np

QC_CSV = Path(__file__).resolve().parents[2] / "data" / "processed" / "qc.csv"


def n_elements(qc_csv=QC_CSV):
    with open(qc_csv, newline="") as fh:
        return {r["alloy"]: len(set(re.findall(r"[A-Z][a-z]?", r["formula"])))
                for r in csv.DictReader(fh)}


def alloy_sets(qc_csv=QC_CSV, per_group=6, seed=0):
    """'binaries': all 2-element alloys; 'test': per_group random alloys with 3, 4 and 5
    elements plus every 6-element alloy (fixed seed); 'all'; 'rest' = all minus test."""
    nel = n_elements(qc_csv)
    rng = np.random.default_rng(seed)
    test = []
    for n in (3, 4, 5):
        ids = sorted(a for a, k in nel.items() if k == n)
        test += sorted(rng.choice(ids, min(per_group, len(ids)), replace=False).tolist())
    test += sorted(a for a, k in nel.items() if k == 6)
    key = int
    return {"binaries": sorted((a for a, k in nel.items() if k == 2), key=key),
            "test": sorted(test, key=key), "all": sorted(nel, key=key),
            "rest": sorted(set(nel) - set(test), key=key)}


def resolve(names, qc_csv=QC_CSV):
    """Expand set names (binaries, test, all, rest) and pass alloy ids through."""
    sets = alloy_sets(qc_csv)
    out = []
    for n in names:
        out += sets.get(n, [n])
    return out
