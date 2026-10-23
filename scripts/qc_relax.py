#!/usr/bin/env python3
"""
QC of the pw.x relax runs (data/raw/relax_<id>.out): settings, convergence, how far each
trajectory travels (energy drop, forces, volume change) and whether its relaxed structure is
the reference cell of the alloy's elastic-constant run (elast_<id>.out).

Usage:
    python scripts/qc_relax.py data/raw -o data/processed/relax_qc.csv -j 4
"""
import argparse, csv, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import GPa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtpot.io.pwscf import parse_pw_relax
from mtpot.io.thermo_pw import tpw_reference
from tpw_to_dataset import alloy_id_from_path


def deviation(a, ref):
    """(max |cell difference|, max position difference modulo lattice) in Angstrom."""
    dc = np.abs(a.cell[:] - ref.cell[:]).max()
    d = (a.positions - ref.positions) @ np.linalg.inv(ref.cell[:])
    return float(dc), float(np.abs((d - np.round(d)) @ ref.cell[:]).max())


def qc_one(raw, aid):
    frames, info = parse_pw_relax(raw / f"relax_{aid}.out", alloy_id=aid)
    st = info["settings"]
    ok = [f for f in frames if f["F"] is not None and f["S"] is not None]
    row = dict(alloy=aid, ecut=f"{st['ecutwfc_Ry']}/{st['ecutrho_Ry']}",
               smearing=f"{st['smearing']} {st['degauss_Ry']}", nks=st["nks"],
               frames=len(frames), complete=len(ok), vc=info["vc"], converged=info["converged"],
               maxsteps=info["maxsteps"], final_scf=info["final_scf"], done=info["done"],
               errors=len(info["errors"]))
    if info["initial"] is not None:
        row.update(formula=info["initial"].get_chemical_formula(), natoms=len(info["initial"]))
    if ok:
        nat = len(ok[0]["symbols"])
        last = ok[-1]
        e = np.array([f["E"] for f in ok])
        row.update(
            E_drop_meV_atom=(e[0] - e[-1]) / nat * 1e3,
            F0_max_eV_A=float(np.linalg.norm(ok[0]["F"], axis=1).max()),
            F_end_max_eV_A=float(np.linalg.norm(last["F"], axis=1).max()),
            P0_GPa=float(-np.trace(ok[0]["S"]) / 3 / GPa),
            P_end_GPa=float(-np.trace(last["S"]) / 3 / GPa),
            dV_pct=100 * (abs(np.linalg.det(last["cell"])) / abs(np.linalg.det(ok[0]["cell"])) - 1),
            max_disp_A=float(np.abs(last["pos"] - ok[0]["pos"]).max()))
    ep = raw / f"elast_{aid}.out"
    relaxed = info["relaxed"] or (Atoms(ok[-1]["symbols"], positions=ok[-1]["pos"],
                                        cell=ok[-1]["cell"], pbc=True) if ok else None)
    if ep.exists() and relaxed is not None:
        ref, est = tpw_reference(ep)
        row["elast_ecut"] = f"{est['ecutwfc_Ry']}/{est['ecutrho_Ry']}"
        row["elast_smearing"] = f"{est['smearing']} {est['degauss_Ry']}"
        same = ref is not None and ref.get_chemical_symbols() == relaxed.get_chemical_symbols()
        row["same_species_order"] = same
        if same:
            row["ref_cell_dev_A"], row["ref_pos_dev_A"] = deviation(relaxed, ref)
            row["initial_vs_ref_pos_A"] = deviation(info["initial"], ref)[1]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("data/processed/relax_qc.csv"))
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    ids = sorted((alloy_id_from_path(p) for p in args.raw.glob("relax_*.out")), key=int)
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(qc_one, [args.raw] * len(ids), ids))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"{len(rows)} relax runs -> {args.out}")


if __name__ == "__main__":
    main()
