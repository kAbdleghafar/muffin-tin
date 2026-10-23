#!/usr/bin/env python3
"""
Phase 0 QC: one row per alloy (thermo_pw output + rho/ELF cubes) with flags for anything
inconsistent with CLAUDE.md. Alloys are processed one at a time; each row is cached as
<out dir>/qc_cache/<id>.json so an interrupted sweep resumes where it stopped. Flags are
recomputed from the cached numbers on every run, so thresholds can change without reparsing.

Usage:
    python scripts/qc.py data/raw -o data/processed/qc.csv [--ids 01 140] [--no-cubes] [--force]
"""
import argparse, csv, json, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtpot.dft import ECUTRHO_RY, ECUTWFC_RY, XC, ZVAL
from mtpot.elastic.protocol import N_WORKS, thermo_pw_strain
from mtpot.io.cube import integrate, read_cube, structure_deviation
from tpw_to_dataset import alloy_id_from_path, cubic_projection, parse_thermo_pw, refit_cij

TOL = dict(refit_GPa=0.02, residual_GPa=0.5, noncubic=0.08, electrons=0.05,
           structure_A=2e-4, strain=1e-6, strained_cell_A=1e-4)
ORDER = ["alloy", "formula", "natoms", "xc", "ecutwfc_Ry", "ecutrho_Ry", "nks", "smearing",
         "degauss_Ry", "nelec", "zval_sum", "works_total", "works_computed", "works_missing",
         "works_maxsteps", "works_unfinished", "frames", "frames_dropped", "strain_dev", "strained_cell_dev_A",
         "C11_GPa", "C12_GPa", "C44_GPa", "noncubic_frac", "refit_max_diff_GPa",
         "s0_xx_GPa", "s0_yy_GPa", "s0_zz_GPa", "s0_shear_max_GPa", "minus_TS_meV_atom",
         "E_span_meV_atom", "F_max_eV_A", "rho_electrons", "rho_min", "rho_struct_dev_A",
         "elf_min", "elf_max", "elf_struct_dev_A", "cube_grid", "elf_grid", "flags", "error"]


def ranges(ns):
    """[1, 2, 3, 7] -> '1-3,7'."""
    out, ns = [], sorted(ns)
    for n in ns:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in out)


def _num(x, cast=float):
    return cast(x) if x is not None else None


def qc_alloy(raw, aid, cubes=True):
    frames, info = parse_thermo_pw(str(Path(raw) / f"elast_{aid}.out"), alloy_id=aid)
    ref, st = info["reference"], info["settings"]
    ok = [f for f in frames if f["F"] is not None and f["S"] is not None]
    computed, ended = info["works_computed"], info["works_ended"]
    row = dict(alloy=aid, formula=ref.get_chemical_formula(), natoms=len(ref), xc=st["xc"],
               ecutwfc_Ry=_num(st["ecutwfc_Ry"]), ecutrho_Ry=_num(st["ecutrho_Ry"]),
               nks=_num(st["nks"], int), smearing=st["smearing"],
               degauss_Ry=_num(st["degauss_Ry"]), nelec=_num(st["nelec"]),
               zval_sum=float(sum(ZVAL.get(s, np.nan) for s in ref.get_chemical_symbols())),
               works_total=info["works_total"], works_computed=len(computed),
               works_missing=ranges(set(range(1, N_WORKS + 1)) - computed),
               works_maxsteps=ranges(info["works_maxsteps"]),
               works_unfinished=ranges(computed - ended),
               frames=len(ok), frames_dropped=len(frames) - len(ok))

    # strain protocol: each work's strain, and its first ionic step = reference x (I + eps)
    sdev, cdev = 0.0, 0.0
    for f in frames:
        eps = thermo_pw_strain(f["meta"]["work"])
        sdev = max(sdev, np.abs(f["meta"]["strain"] - eps).max())
        if f["meta"]["ionic_step"] == 1:
            cdev = max(cdev, np.abs(f["cell"] - ref.cell[:] @ (np.eye(3) + eps)).max())
    row.update(strain_dev=float(sdev), strained_cell_dev_A=float(cdev))

    C = info["C_tpw_kbar"]
    if C is not None:
        c11, c12, c44, dev, _ = cubic_projection(C / 10)
        row.update(C11_GPa=c11, C12_GPa=c12, C44_GPa=c44, noncubic_frac=dev)
    # refit only with all 4 points per strain: a partial (one-sided) fit picks up curvature
    # that thermo_pw's symmetric fit cancels, and would raise false mismatches
    if ended >= set(range(1, N_WORKS + 1)):
        Cfit, s0 = refit_cij(frames)                    # kbar, final frames (BFGS ended)
        if C is not None:
            row["refit_max_diff_GPa"] = float(np.abs(Cfit - C).max() / 10)
        s0 = s0 / 10
        row.update(s0_xx_GPa=s0[0], s0_yy_GPa=s0[1], s0_zz_GPa=s0[2],
                   s0_shear_max_GPa=np.abs(s0[3:]).max())
    if ok:
        e = np.array([f["E"] for f in ok]) / len(ref) * 1e3
        row.update(minus_TS_meV_atom=np.mean([f.get("mTS", 0.0) for f in ok]) / len(ref) * 1e3,
                   E_span_meV_atom=e.max() - e.min(),
                   F_max_eV_A=max(np.linalg.norm(f["F"], axis=1).max() for f in ok))

    if cubes:
        row["cubes_checked"] = True
        for kind in ("rho", "elf"):
            path = Path(raw) / f"{kind}_{aid}.cube"
            if not path.exists():
                continue
            atoms, data = read_cube(path)
            row[f"{kind}_struct_dev_A"] = structure_deviation(atoms, ref)
            row["cube_grid" if kind == "rho" else "elf_grid"] = "x".join(map(str, data.shape))
            if kind == "rho":
                row.update(rho_electrons=integrate(atoms, data), rho_min=data.min())
            else:
                row.update(elf_min=data.min(), elf_max=data.max())
            del data
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}


def flags(row):
    """Human-readable problems; the part before ' (' is the flag name."""
    if "error" in row:
        return [f"parse error ({row['error']})"]
    g, out = row.get, []
    if g("xc") != XC or g("ecutwfc_Ry") != ECUTWFC_RY or g("ecutrho_Ry") != ECUTRHO_RY:
        out.append(f"settings ({g('xc')}, {g('ecutwfc_Ry')}/{g('ecutrho_Ry')} Ry)")
    if g("nelec") != g("zval_sum"):
        out.append(f"nelec != sum Zval ({g('nelec')} vs {g('zval_sum')})")
    if g("works_total") != N_WORKS:
        out.append(f"works_total ({g('works_total')})")
    if g("works_missing"):
        out.append(f"restart works ({g('works_missing')})")
    if g("works_unfinished"):
        out.append(f"unfinished works ({g('works_unfinished')})")
    if g("works_maxsteps"):
        out.append(f"BFGS step limit ({g('works_maxsteps')})")
    if g("frames_dropped"):
        out.append(f"dropped frames ({g('frames_dropped')})")
    if g("strain_dev", 0) > TOL["strain"] or g("strained_cell_dev_A", 0) > TOL["strained_cell_A"]:
        out.append(f"strain protocol (strain {g('strain_dev'):.1e}, "
                   f"cell {g('strained_cell_dev_A'):.1e} A)")
    if g("C11_GPa") is None:
        out.append("no C_ij block")
    elif g("noncubic_frac") > TOL["noncubic"]:
        out.append(f"non-cubic ({g('noncubic_frac'):.3f})")
    if g("refit_max_diff_GPa", 0) > TOL["refit_GPa"]:
        out.append(f"refit mismatch ({g('refit_max_diff_GPa'):.3f} GPa)")
    s0 = [abs(g(k)) for k in ("s0_xx_GPa", "s0_yy_GPa", "s0_zz_GPa", "s0_shear_max_GPa")
          if g(k) is not None]
    if s0 and max(s0) > TOL["residual_GPa"]:
        out.append(f"residual stress ({max(s0):.2f} GPa)")
    if g("cubes_checked"):
        if g("rho_electrons") is None:
            out.append("rho cube missing")
        elif abs(g("rho_electrons") - g("nelec")) > TOL["electrons"]:
            out.append(f"electron count ({g('rho_electrons'):.3f} vs {g('nelec')})")
        if g("elf_min") is None:
            out.append("elf cube missing")
        elif g("elf_min") < -1e-6 or g("elf_max") > 1 + 1e-6:
            out.append(f"elf range ({g('elf_min'):.3g}..{g('elf_max'):.3g})")
        if g("elf_grid") and g("cube_grid") and g("elf_grid") != g("cube_grid"):
            out.append(f"cube grids differ ({g('cube_grid')} vs {g('elf_grid')})")
        devs = [g(k) for k in ("rho_struct_dev_A", "elf_struct_dev_A") if g(k) is not None]
        if devs and max(devs) > TOL["structure_A"]:
            out.append(f"cube structure ({max(devs):.1e} A)")
    return out


def _fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else v


def _cached_row(raw, aid, cache, cubes, force):
    cf = cache / f"{aid}.json"
    row = json.loads(cf.read_text()) if cf.exists() and not force else None
    if row is None or (cubes and not row.get("cubes_checked")):
        try:
            row = qc_alloy(raw, aid, cubes=cubes)
            cf.write_text(json.dumps(row))
        except Exception as e:                           # one broken file must not stop a sweep
            row = {"alloy": aid, "error": f"{type(e).__name__}: {e}"}
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path, help="directory with elast_<id>.out, rho_<id>.cube, elf_<id>.cube")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/processed/qc.csv"))
    ap.add_argument("--ids", nargs="+", help="alloy ids (default: every elast_*.out in RAW)")
    ap.add_argument("--no-cubes", action="store_true", help="skip the rho/ELF cubes (fast)")
    ap.add_argument("--force", action="store_true", help="ignore cached rows")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="worker processes")
    args = ap.parse_args(argv)

    ids = args.ids or sorted({alloy_id_from_path(p) for p in args.raw.glob("elast_*.out")},
                             key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
    cache = args.out.parent / "qc_cache"
    cache.mkdir(parents=True, exist_ok=True)
    get = partial(_cached_row, args.raw, cache=cache, cubes=not args.no_cubes, force=args.force)
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        results = ex.map(get, ids) if args.jobs > 1 else map(get, ids)
        for k, (aid, row) in enumerate(zip(ids, results), 1):
            row["flags"] = "; ".join(flags(row))
            rows.append(row)
            print(f"[{k}/{len(ids)}] {aid:>4} {row.get('formula', '?'):<22} "
                  f"{row['flags'] or 'ok'}", flush=True)

    keys = [k for k in ORDER if any(k in r for r in rows)]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: _fmt(v) for k, v in r.items()} for r in rows)

    names = Counter(f.split(" (")[0] for r in rows for f in flags(r))
    smear = Counter(f"{r.get('smearing')} {r.get('degauss_Ry')}" for r in rows if "error" not in r)
    print(f"\n{len(rows)} alloys, {sum(bool(r['flags']) for r in rows)} flagged -> {args.out}")
    print("smearing: " + ", ".join(f"{k}: {n}" for k, n in smear.most_common()))
    if names:
        print("flags:    " + ", ".join(f"{k}: {n}" for k, n in names.most_common()))
    return rows


if __name__ == "__main__":
    main()
