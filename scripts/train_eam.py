#!/usr/bin/env python3
"""
Fit the EAM (Phase 2) to energies, forces, stresses and C_ij of the training alloys and
report errors on held-out alloys. Needs data/processed/train/<id>.npz (build_dataset.py).

Usage:
    python scripts/train_eam.py --train rest --val test --epochs 30 --tag eam_v1
Writes models/<tag>.pt, models/<tag>.log (one JSON line per epoch) and models/<tag>.json.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from mtpot.potential.data import reference_energies
from mtpot.potential.eam import EAM
from mtpot.potential.train import AlloyBatch, fit, fit_lbfgs, init_model, load, report, save
from mtpot.splits import resolve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", nargs="+", default=["rest"])
    ap.add_argument("--val", nargs="+", default=["test"])
    ap.add_argument("--optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    ap.add_argument("--epochs", type=int, default=30, help="Adam epochs / L-BFGS iterations")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-alloys", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--w", default="E=1,F=1,S=1,C=1,rE=1,rF=1,rS=0.25,H=10,R=1,kmin=0.5",
                    help="loss weights (rE/rF/rS: relax sets; H: stability, R: relaxed-ion C, "
                         "kmin: minimum force-constant eigenvalue in eV/A^2)")
    ap.add_argument("--hessian-train", type=int, default=24,
                    help="training alloys with Hessian terms (stratified by element count)")
    ap.add_argument("--hessian-val", action="store_true",
                    help="report stability and relaxed-ion C on all validation alloys")
    ap.add_argument("--relax", action="store_true",
                    help="add vc-relax trajectories of the training alloys (val: report only)")
    ap.add_argument("--exclude-relax", nargs="*", default=[],
                    help="alloys whose relax runs are left out (e.g. 87 88)")
    ap.add_argument("--max-train", type=int, help="random subset of training alloys")
    ap.add_argument("--rc", type=float, default=6.0)
    ap.add_argument("--n-pair", type=int, default=21)
    ap.add_argument("--n-dens", type=int, default=21)
    ap.add_argument("--n-embed", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--init", type=Path, help="warm start from a saved model (.pt)")
    ap.add_argument("--tag", default="eam")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    data_dir = ROOT / "data" / "processed" / "train"
    relax_dir = ROOT / "data" / "processed" / "train_relax"
    read = lambda ids: [dict(np.load(data_dir / f"{a}.npz")) for a in ids]
    read_relax = lambda ids: [dict(np.load(relax_dir / f"{a}.npz")) for a in ids
                              if (relax_dir / f"{a}.npz").exists()]
    train_ids, val_ids = resolve(args.train), resolve(args.val)
    if args.max_train and len(train_ids) > args.max_train:
        rng = np.random.default_rng(0)
        train_ids = sorted(rng.choice(train_ids, args.max_train, replace=False).tolist(), key=int)
    assert not set(train_ids) & set(val_ids), "train and validation alloys overlap"
    t0 = time.time()
    raw_train, raw_val = read(train_ids), read(val_ids)
    mu = reference_energies(raw_train)
    sets = {"train": [AlloyBatch(d, mu) for d in raw_train],
            "val": [AlloyBatch(d, mu) for d in raw_val]}
    if args.relax:
        sets["train_relax"] = [AlloyBatch(d, mu) for d in
                               read_relax([a for a in train_ids if a not in args.exclude_relax])]
        sets["val_relax"] = [AlloyBatch(d, mu) for d in read_relax(val_ids)]
    if args.hessian_train:
        from mtpot.splits import n_elements
        nel = n_elements()
        rng = np.random.default_rng(1)
        pool = {}
        for ab in sets["train"]:
            pool.setdefault(nel[ab.alloy], []).append(ab)
        quota = {k: max(1, round(args.hessian_train * len(v) / len(sets["train"]))) for k, v in pool.items()}
        for k, v in pool.items():
            for i in rng.choice(len(v), min(quota[k], len(v)), replace=False):
                v[i].hessian = True
        print("Hessian terms on", sorted((ab.alloy for ab in sets["train"] if ab.hessian), key=int), flush=True)
    if args.hessian_val:
        for ab in sets["val"]:
            ab.hessian = True
    train = sets["train"] + sets.get("train_relax", [])
    del raw_train, raw_val
    print(f"loaded {', '.join(f'{len(v)} {k}' for k, v in sets.items())} in "
          f"{time.time() - t0:.0f} s; mu (eV) = {np.round(mu, 3).tolist()}", flush=True)

    mcfg = dict(rc=args.rc, n_pair=args.n_pair, n_dens=args.n_dens, n_embed=args.n_embed)
    if args.init:
        model, mu0, cfg0 = load(args.init)
        mcfg = cfg0["model"]
        with torch.no_grad():                   # keep absolute energies: shift e0 by mu change
            model.e0 += torch.as_tensor(mu0 - mu)
        print(f"warm start from {args.init}", flush=True)
    else:
        model = init_model(EAM(**mcfg))
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.w.split(","))}
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    config = dict(model=mcfg, args={k: (str(v) if isinstance(v, Path) else v)
                                    for k, v in vars(args).items()},
                  train=train_ids, val=val_ids)

    def checkpoint(model, row):              # survive interruptions: resume with --init
        save(model, out / f"{args.tag}.pt", mu, dict(config, last=row))

    with open(out / f"{args.tag}.log", "w") as log:
        if args.optimizer == "lbfgs":
            hist = fit_lbfgs(model, train, sets, weights, iters=args.epochs, reg=args.reg,
                             log=log, eval_every=args.eval_every, checkpoint=checkpoint)
        else:
            hist = fit(model, train, sets, weights, epochs=args.epochs, lr=args.lr,
                       batch_alloys=args.batch_alloys, reg=args.reg, log=log,
                       eval_every=args.eval_every)
    save(model, out / f"{args.tag}.pt", mu, config)
    final = dict(report(model, sets), epochs=args.epochs,
                 seconds=sum(h["seconds"] for h in hist))
    (out / f"{args.tag}.json").write_text(json.dumps(final, indent=1, default=float))
    print(json.dumps(final, indent=1, default=float))


if __name__ == "__main__":
    main()
