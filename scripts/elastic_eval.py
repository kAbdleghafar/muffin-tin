#!/usr/bin/env python3
"""
Phase 3 evaluation: relaxed-ion C11/C12/C44 predicted by a trained potential vs the DFT
labels (data/processed/qc.csv, thermo_pw cubic projection).

Modes:
  dft_cell    strain the DFT reference cell (from elast_<id>.out), relax ions per strain
  relax_ions  relax ions in the DFT reference cell first, then as dft_cell
  own         deployment: start from the ideal SQS (first frame of relax_<id>.out), relax cell
              and ions with the potential, then strain

Usage:
    python scripts/elastic_eval.py models/eam_v1.pt --ids test --mode dft_cell -j 4
Writes models/<tag>_elastic_<mode>.csv.
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


def labels():
    with open(ROOT / "data" / "processed" / "qc.csv", newline="") as fh:
        return {r["alloy"]: np.array([float(r["C11_GPa"]), float(r["C12_GPa"]), float(r["C44_GPa"])])
                for r in csv.DictReader(fh)}


def one(aid, model_path, mode, fmax):
    import torch
    torch.set_num_threads(1)
    from mtpot.elastic.workflow import predict
    from mtpot.io.pwscf import parse_pw_relax
    from mtpot.io.thermo_pw import tpw_reference
    from mtpot.potential.train import load
    model, _, _ = load(model_path)
    if mode == "own":
        path = RAW / f"relax_{aid}.out"
        if not path.exists():
            return aid, None
        atoms = parse_pw_relax(path)[1]["initial"]
    else:
        atoms = tpw_reference(RAW / f"elast_{aid}.out")[0]
    t0 = time.time()
    try:
        r = predict(atoms, model, mode=mode, fmax=fmax)
    except Exception as e:                                   # report, do not stop the sweep
        return aid, dict(error=f"{type(e).__name__}: {e}")
    r["seconds"] = time.time() - t0
    return aid, r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("--ids", nargs="+", default=["test"])
    ap.add_argument("--mode", choices=["dft_cell", "relax_ions", "own"], default="dft_cell")
    ap.add_argument("--fmax", type=float, default=1e-3)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    ids, lab = resolve(args.ids), labels()
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for aid, r in ex.map(partial(one, model_path=args.model, mode=args.mode, fmax=args.fmax), ids):
            if r is None:
                print(f"{aid}: no structure for mode {args.mode}", flush=True)
                continue
            row = dict(alloy=aid)
            if "error" in r:
                row["error"] = r["error"]
            else:
                p, t = r["cubic"], lab[aid]
                row.update(C11=p[0], C12=p[1], C44=p[2], C11_dft=t[0], C12_dft=t[1], C44_dft=t[2],
                           dC11=p[0] - t[0], dC12=p[1] - t[1], dC44=p[2] - t[2],
                           Cp=(p[0] - p[1]) / 2, Cp_dft=(t[0] - t[1]) / 2,
                           unconverged_works=sum(not w["converged"] for w in r["works"]),
                           seconds=r["seconds"], **{f"pre_{k}": v for k, v in r["pre"].items()})
            rows.append(row)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = args.model.with_name(f"{args.model.stem}_elastic_{args.mode}.csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if "error" not in r]
    if ok:
        d = np.array([[r["dC11"], r["dC12"], r["dC44"], r["Cp"] - r["Cp_dft"]] for r in ok])
        print(f"\n{len(ok)} alloys ({len(rows) - len(ok)} failed), mode {args.mode}: MAE "
              f"C11 {np.abs(d[:, 0]).mean():.1f}  C12 {np.abs(d[:, 1]).mean():.1f}  "
              f"C44 {np.abs(d[:, 2]).mean():.1f}  C' {np.abs(d[:, 3]).mean():.1f} GPa -> {out}")


if __name__ == "__main__":
    main()
