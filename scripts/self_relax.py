#!/usr/bin/env python3
"""
Self-relaxation test: relax each alloy's DFT elastic reference structure at fixed cell with a
trained potential and report the energy gain and atomic displacements. DFT says the reference
is a minimum, so a good model barely moves it; a large gain exposes spurious minima that local
eigenvalues at the reference miss.

Usage:
    python scripts/self_relax.py models/adp_v1.pt --ids test -j 4
Writes models/<tag>_selfrelax.csv.
"""
import argparse, csv, sys, time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from mtpot.splits import resolve

RAW = ROOT / "data" / "raw"


def one(aid, model_path, fmax):
    import torch
    torch.set_num_threads(1)
    from mtpot.elastic.workflow import TorchCalculator, self_relaxation
    from mtpot.io.thermo_pw import tpw_reference
    from mtpot.potential.train import load
    model = load(model_path)[0]
    t0 = time.time()
    r = self_relaxation(tpw_reference(RAW / f"elast_{aid}.out")[0], TorchCalculator(model), fmax=fmax)
    r["seconds"] = time.time() - t0
    return aid, r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("--ids", nargs="+", default=["test"])
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for aid, r in ex.map(partial(one, model_path=args.model, fmax=args.fmax), resolve(args.ids)):
            rows.append(dict(alloy=aid, **r))
            print(f"{aid}: gain {r['gain_meV_atom']:8.1f} meV/atom, displacement max {r['disp_max_A']:.3f} "
                  f"mean {r['disp_mean_A']:.3f} A, d_min {r['dmin_ref_A']:.2f} -> {r['dmin_relaxed_A']:.2f} A",
                  flush=True)
    out = args.model.with_name(f"{args.model.stem}_selfrelax.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    g = np.array([r["gain_meV_atom"] for r in rows])
    print(f"{len(rows)} alloys: gain median {np.median(g):.1f}, worst {g.min():.1f} meV/atom -> {out}")


if __name__ == "__main__":
    main()
