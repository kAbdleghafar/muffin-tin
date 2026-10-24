"""Alloy-level splits shared by all phases (never split frames of one alloy)."""
import csv
import re
from pathlib import Path

import numpy as np

QC_CSV = Path(__file__).resolve().parents[2] / "data" / "processed" / "qc.csv"

# Alloys left out of every set (user decision: mechanically unstable; CLAUDE.md data issue 11)
EXCLUDED = {
    "63": "Mo8Ta8Ti8Zr8: mechanically unstable (user)",
    "64": "Nb8Ta8Ti8Zr8: mechanically unstable (user)",
    "88": "Ta4Ti28: mechanically unstable (user); DFT C tensor has a negative eigenvalue (-7.1 GPa)",
}


def n_elements(qc_csv=QC_CSV):
    with open(qc_csv, newline="") as fh:
        return {r["alloy"]: len(set(re.findall(r"[A-Z][a-z]?", r["formula"])))
                for r in csv.DictReader(fh)}


def alloy_sets(qc_csv=QC_CSV, per_group=6, seed=0, excluded=EXCLUDED):
    """'binaries': all 2-element alloys; 'test': per_group random alloys with 3, 4 and 5
    elements plus every 6-element alloy (fixed seed); 'all'; 'rest' = all minus test.
    The test draw uses every alloy (so it never changes); excluded alloys are then removed
    from all sets."""
    nel = n_elements(qc_csv)
    rng = np.random.default_rng(seed)
    test = []
    for n in (3, 4, 5):
        ids = sorted(a for a, k in nel.items() if k == n)
        test += sorted(rng.choice(ids, min(per_group, len(ids)), replace=False).tolist())
    test += sorted(a for a, k in nel.items() if k == 6)
    keep = lambda ids: sorted((a for a in ids if a not in excluded), key=int)
    return {"binaries": keep(a for a, k in nel.items() if k == 2), "test": keep(test),
            "all": keep(nel), "rest": keep(set(nel) - set(test))}


def resolve(names, qc_csv=QC_CSV):
    """Expand set names (binaries, test, all, rest) and pass alloy ids through."""
    sets = alloy_sets(qc_csv)
    out = []
    for n in names:
        out += sets.get(n, [n])
    return out
