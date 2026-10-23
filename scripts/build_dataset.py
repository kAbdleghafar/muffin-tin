#!/usr/bin/env python3
"""
Build per-alloy training sets for the potential: data/processed/train/<id>.npz
(first + final ionic step of every work, E0, forces, stress, C_ij label, neighbour lists).

Usage:
    python scripts/build_dataset.py data/raw -j 4 [--ids 01 140] [--rc 6.0]
"""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtpot.potential.data import build_alloy
from tpw_to_dataset import alloy_id_from_path

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "train"


def one(path, rc, force):
    aid = alloy_id_from_path(path)
    out = OUT / f"{aid}.npz"
    if out.exists() and not force:
        return aid, "cached"
    d = build_alloy(path, rc=rc)
    np.savez_compressed(out, **d)
    return aid, f"{len(d['energy'])} frames, {len(d['nl_i'])} pairs"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--rc", type=float, default=6.0)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    paths = ([args.raw / f"elast_{i}.out" for i in args.ids] if args.ids
             else sorted(args.raw.glob("elast_*.out")))
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        for aid, msg in ex.map(partial(one, rc=args.rc, force=args.force), paths):
            print(f"{aid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
