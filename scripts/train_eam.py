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
from mtpot.potential.train import AlloyBatch, fit, fit_lbfgs, init_model, load, save, summarize
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
    ap.add_argument("--w", default="E=1,F=1,S=1,C=1", help="loss weights")
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
    read = lambda ids: [dict(np.load(data_dir / f"{a}.npz")) for a in ids]
    train_ids, val_ids = resolve(args.train), resolve(args.val)
    if args.max_train and len(train_ids) > args.max_train:
        rng = np.random.default_rng(0)
        train_ids = sorted(rng.choice(train_ids, args.max_train, replace=False).tolist(), key=int)
    assert not set(train_ids) & set(val_ids), "train and validation alloys overlap"
    t0 = time.time()
    raw_train, raw_val = read(train_ids), read(val_ids)
    mu = reference_energies(raw_train)
    train = [AlloyBatch(d, mu) for d in raw_train]
    val = [AlloyBatch(d, mu) for d in raw_val]
    del raw_train, raw_val
    print(f"loaded {len(train)} train / {len(val)} val alloys in {time.time() - t0:.0f} s; "
          f"mu (eV) = {np.round(mu, 3).tolist()}", flush=True)

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
    with open(out / f"{args.tag}.log", "w") as log:
        if args.optimizer == "lbfgs":
            hist = fit_lbfgs(model, train, val, weights, iters=args.epochs, reg=args.reg,
                             log=log, eval_every=args.eval_every)
        else:
            hist = fit(model, train, val, weights, epochs=args.epochs, lr=args.lr,
                       batch_alloys=args.batch_alloys, reg=args.reg, log=log,
                       eval_every=args.eval_every)
    config = dict(model=mcfg, args=vars(args), train=train_ids, val=val_ids)
    save(model, out / f"{args.tag}.pt", mu, config)
    final = {"train": summarize(model, train), "val": summarize(model, val),
             "epochs": args.epochs, "seconds": sum(h["seconds"] for h in hist)}
    (out / f"{args.tag}.json").write_text(json.dumps(final, indent=1, default=float))
    print(json.dumps(final, indent=1, default=float))


if __name__ == "__main__":
    main()
