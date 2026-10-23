#!/usr/bin/env python3
"""
Build per-alloy training sets for the potential: data/processed/train/<id>.npz
(first + final ionic step of every work, E0, forces, stress, C_ij label, neighbour lists);
with --relax, the vc-relax trajectories -> data/processed/train_relax/<id>.npz.

Usage:
    python scripts/build_dataset.py data/raw -j 4 [--ids 01 140] [--rc 6.0] [--relax]
"""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtpot.potential.data import build_alloy, build_relax
from tpw_to_dataset import alloy_id_from_path

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def one(path, rc, force, relax=False):
    aid = alloy_id_from_path(path)
    out = PROCESSED / ("train_relax" if relax else "train") / f"{aid}.npz"
    if out.exists() and not force:
        return aid, "cached"
    d = build_relax(path, rc=rc) if relax else build_alloy(path, rc=rc)
    if d is None:
        return aid, "no complete frames, skipped"
    np.savez_compressed(out, **d)
    return aid, f"{len(d['energy'])} frames, {len(d['nl_i'])} pairs"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--rc", type=float, default=6.0)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--relax", action="store_true", help="build the vc-relax sets instead")
    args = ap.parse_args(argv)
    kind = "relax" if args.relax else "elast"
    paths = ([args.raw / f"{kind}_{i}.out" for i in args.ids] if args.ids
             else sorted(args.raw.glob(f"{kind}_*.out")))
    (PROCESSED / ("train_relax" if args.relax else "train")).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        for aid, msg in ex.map(partial(one, rc=args.rc, force=args.force, relax=args.relax),
                               paths):
            print(f"{aid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
