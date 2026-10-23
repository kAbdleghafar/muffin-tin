#!/usr/bin/env python3
"""
Phase 1: atom-centred density functions from the DFT valence density (CLAUDE.md).

  prepare  one pass over each alloy's rho/ELF cubes -> data/processed/density/<id>.npz
           (radial rho/ELF profiles per atom, a random grid sample with neighbour pairs,
           thin shells around every atom for the angular analysis)
  fit      fit per-species f_Z(r) on training alloys, evaluate on held-out alloys; each
           held-out alloy is also fitted on its own ("oracle") to separate transfer error
           from the limit of a spherical superposition
  angular  l = 1..4 angular power on the shells for DFT rho, the fitted superposition, the
           residual and ELF
  report   profile statistics per species, figures and tables -> reports/phase1/

Usage:
  python scripts/density_phase1.py prepare --ids binaries test -j 4
  python scripts/density_phase1.py fit --train binaries --test test
  python scripts/density_phase1.py angular --ids binaries test
  python scripts/density_phase1.py report --ids binaries test
"""
import argparse, csv, json, re, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtpot.density.basis import ExpBasis, RadialBasis
from mtpot.density.fit import SPECIES, DensityFit, metrics, sample_alloy
from mtpot.density.grid import grid_positions, neighbour_pairs
from mtpot.density.profiles import atom_pairs, radial_profiles, shell_power
from mtpot.io.cube import read_cube

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed" / "density"
REPORT = ROOT / "reports" / "phase1"
# categorical slots 1-6 (dataviz reference palette, validated; contrast relief = direct labels)
COLORS = dict(zip(SPECIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHELLS = (1.2, 1.4, 1.6)          # Angstrom; nearest-neighbour distance is ~2.8
SHELL_HALF = 0.04
PROFILE_RMAX, PROFILE_DR = 3.0, 0.02


def alloy_sets(qc_csv=ROOT / "data" / "processed" / "qc.csv", per_group=6, seed=0):
    """'binaries': all 2-element alloys; 'test': per_group random alloys with 3, 4 and 5
    elements plus every 6-element alloy (fixed seed)."""
    with open(qc_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    nel = {r["alloy"]: len(set(re.findall(r"[A-Z][a-z]?", r["formula"]))) for r in rows}
    rng = np.random.default_rng(seed)
    test = []
    for n in (3, 4, 5):
        ids = sorted(a for a, k in nel.items() if k == n)
        test += sorted(rng.choice(ids, min(per_group, len(ids)), replace=False).tolist())
    test += sorted(a for a, k in nel.items() if k == 6)
    key = lambda a: int(a)
    return {"binaries": sorted((a for a, k in nel.items() if k == 2), key=key),
            "test": sorted(test, key=key)}


def resolve(names):
    sets = alloy_sets()
    out = []
    for n in names:
        out += sets.get(n, [n])
    return out


# ------------------------------------------------------------------------------- prepare
def prepare_one(aid, n_points=20000, rc_max=6.5, force=False):
    path = OUT / f"{aid}.npz"
    if path.exists() and not force:
        return aid, "cached"
    atoms, rho = read_cube(RAW / f"rho_{aid}.cube")
    _, elf = read_cube(RAW / f"elf_{aid}.cube")
    pairs = atom_pairs(atoms, rho.shape, PROFILE_RMAX)
    r, prof_rho = radial_profiles(atoms, rho, PROFILE_RMAX, PROFILE_DR, pairs=pairs)
    _, prof_elf = radial_profiles(atoms, elf, PROFILE_RMAX, PROFILE_DR, pairs=pairs)
    out = dict(profile_r=r, prof_rho=prof_rho, prof_elf=prof_elf,
               symbols=np.array(atoms.get_chemical_symbols()))
    gi, ia, d, v = pairs
    for rs in SHELLS:
        m = np.abs(d - rs) < SHELL_HALF
        key = f"shell{rs:g}"
        out.update({f"{key}_idx": gi[m], f"{key}_atom": ia[m].astype(np.int16),
                    f"{key}_xyz": (v[m] / d[m][:, None]).astype(np.float32),
                    f"{key}_dr": (d[m] - rs).astype(np.float32),
                    f"{key}_rho": rho.ravel()[gi[m]], f"{key}_elf": elf.ravel()[gi[m]]})
    del pairs, gi, ia, d, v
    s = sample_alloy(atoms, rho, elf, n_points=n_points, rc_max=rc_max, seed=int(aid))
    out.update({f"sample_{k}": v for k, v in s.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return aid, "done"


def load(aid, prefix=""):
    """Cached arrays of one alloy (only keys starting with prefix: npz members load lazily)."""
    z = np.load(OUT / f"{aid}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files if k.startswith(prefix)}


def sample_of(data):
    return {k[len("sample_"):]: v for k, v in data.items() if k.startswith("sample_")}


# ------------------------------------------------------------------------------- fit
def parse_r_core(text):
    r = {s: 0.8 for s in SPECIES}
    for item in filter(None, (text or "").split(",")):
        k, v = item.split("=")
        if k == "all":
            r = {s: float(v) for s in SPECIES}
        else:
            r[k] = float(v)
    return r


def make_basis(kind, r0, rc, n):
    return ExpBasis(r0, rc, n) if kind == "exp" else RadialBasis(r0, rc, n)


def make_fit(samples, args, r_core):
    fit = DensityFit(make_basis(args.basis, args.r0, args.rc, args.n_int), r_core,
                     background=args.background)
    for s in samples:
        fit.add(s)
    return fit.solve(lam=args.lam, nonneg=not args.signed, monotone=args.monotone)


def save_fit(fit, path, extra):
    np.savez(path, coef=fit.coef, r0=fit.basis.r0, rc=fit.basis.rc, n_int=fit.basis.n,
             basis=type(fit.basis).__name__, background=fit.background,
             species=np.array(SPECIES), r_core=np.array([fit.r_core[s] for s in SPECIES]),
             **extra)


def load_fit(path):
    z = np.load(path)
    kind = "exp" if str(z["basis"]) == "ExpBasis" else "spline"
    fit = DensityFit(make_basis(kind, float(z["r0"]), float(z["rc"]), int(z["n_int"])),
                     dict(zip([str(s) for s in z["species"]], z["r_core"])),
                     background=bool(z["background"]))
    fit.coef = z["coef"]
    return fit


def cmd_fit(args):
    train, test = resolve(args.train), resolve(args.test or [])
    r_core = parse_r_core(args.r_core)
    samples = {a: sample_of(load(a, "sample_")) for a in train + test}
    fit = make_fit([samples[a] for a in train], args, r_core)
    rows = []
    for split, ids in (("train", train), ("test", test)):
        for a in ids:
            m = metrics(fit, samples[a])
            row = dict(alloy=a, split=split, formula=formula(samples[a]), **m)
            if split == "test" and args.oracle:
                row["oracle_rel_rmse"] = metrics(make_fit([samples[a]], args, r_core),
                                                 samples[a])["rel_rmse"]
            rows.append(row)
    tag = args.tag
    if fit.background:
        print("background e/atom: " + "  ".join(f"{k} {v:.3f}" for k, v in
                                                 fit.background_electrons().items()))
    save_fit(fit, OUT / f"fit_{tag}.npz", dict(train=np.array(train)))
    with open(OUT / f"fit_{tag}_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + (["oracle_rel_rmse"] if args.oracle and "oracle_rel_rmse" not in rows[0] else []))
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "test"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        keys = ["rel_rmse", "r2", "rel_rmse_0-1.2", "rel_rmse_1.2-1.6", "rel_rmse_1.6+",
                "charge_err_1.6+"] + (["oracle_rel_rmse"] if split == "test" and args.oracle else [])
        print(f"{split:5s} n={len(rs):2d} " + "  ".join(
            f"{k} {np.median([r[k] for r in rs if k in r]):.4f}" for k in keys) + "  (medians)")
    return fit, rows


def formula(sample):
    from ase import Atoms
    return Atoms(numbers=sample["numbers"]).get_chemical_formula()


# ------------------------------------------------------------------------------- angular
def cmd_angular(args):
    fit = load_fit(OUT / f"fit_{args.tag}.npz")
    rows = []
    for a in resolve(args.ids):
        data = load(a)
        cell, shape = data["sample_cell"], tuple(data["sample_shape"])
        from ase import Atoms
        atoms = Atoms(numbers=data["sample_numbers"], positions=data["sample_positions"],
                      cell=cell, pbc=True)
        sym = data["symbols"]
        for rs in SHELLS:
            k = f"shell{rs:g}"
            idx, ia, xyz, dr = data[f"{k}_idx"], data[f"{k}_atom"].astype(int), data[f"{k}_xyz"], data[f"{k}_dr"]
            # superposition at the shell points
            uniq, inv = np.unique(idx, return_inverse=True)
            pts = grid_positions(cell, shape, uniq)
            ip, ja, d = neighbour_pairs(pts, atoms, fit.basis.rc)
            spec = np.array([SPECIES.index(s) for s in atoms.get_chemical_symbols()])[ja]
            pred = np.zeros(len(uniq))
            for z, s in enumerate(SPECIES):
                m = spec == z
                if m.any():
                    pred += np.bincount(ip[m], weights=fit.f(s, d[m]), minlength=len(uniq))
            fields = {"rho": data[f"{k}_rho"], "fit": pred[inv], "elf": data[f"{k}_elf"]}
            fields["resid"] = fields["rho"] - fields["fit"]
            res = {name: shell_power(v, ia, xyz.astype(float), len(atoms), dr=dr.astype(float))
                   for name, v in fields.items()}
            rho_mean = res["rho"][0]
            for i in range(len(atoms)):
                row = dict(alloy=a, atom=i, symbol=sym[i], r=rs)
                for name, (mean, power, k4rel) in res.items():
                    row[f"{name}_mean"] = mean[i]
                    # residual amplitudes are expressed relative to the DFT mean on the shell
                    scale = (mean[i] / rho_mean[i]) if name == "resid" else 1.0
                    for l in range(power.shape[1]):
                        row[f"{name}_P{l + 1}"] = power[i, l] * abs(scale)
                    row[f"{name}_K4"] = k4rel[i] * scale
                rows.append(row)
        print(f"{a}: {formula(sample_of(data))}", flush=True)
    path = OUT / f"angular_{args.tag}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


def profile_stats(ids, radii=(0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.2)):
    """Per species and radius: mean rho/ELF, spread over atoms within an alloy and spread of
    the alloy means across alloys (transferability of atom-centred profiles)."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty innermost bins (r < voxel)
    curves = {s: [] for s in SPECIES}                  # (alloy, r, mean rho, mean elf, within std)
    for a in ids:
        z = np.load(OUT / f"{a}.npz")
        r, sym = z["profile_r"], z["symbols"]
        for s in set(sym.tolist()):
            m = sym == s
            curves[s].append((a, r, np.nanmean(z["prof_rho"][m], 0), np.nanmean(z["prof_elf"][m], 0),
                              np.nanstd(z["prof_rho"][m], 0)))
    rows = []
    for s, cs in curves.items():
        if not cs:
            continue
        r = cs[0][1]
        R, E, W = (np.array([c[k] for c in cs]) for k in (2, 3, 4))
        for x in radii:
            k = int(np.argmin(np.abs(r - x)))
            rows.append(dict(species=s, n_alloys=len(cs), r=round(float(r[k]), 2),
                             rho=R[:, k].mean(), rho_across_rel=R[:, k].std() / R[:, k].mean(),
                             rho_within_rel=np.mean(W[:, k]) / R[:, k].mean(),
                             elf=E[:, k].mean(), elf_across=E[:, k].std()))
    return curves, rows


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True, exist_ok=True)
    ids = resolve(args.ids)
    curves, rows = profile_stats(ids)
    with open(REPORT / "profiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    fit = load_fit(OUT / f"fit_{args.tag}.npz") if (OUT / f"fit_{args.tag}.npz").exists() else None
    # small multiples: one panel per species, every alloy's species-mean rho profile
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, s in zip(axes.ravel(), SPECIES):
        _style(ax)
        for _, r, rho, _, _ in curves[s]:
            ax.plot(r, rho, color=COLORS[s], lw=1.0, alpha=0.55)
        if fit is not None:
            ax.axvline(fit.r_core[s], color=MUTED, lw=1, ls=(0, (3, 3)))
            ax.text(fit.r_core[s] + 0.04, 3.0, "core mask", color=MUTED, fontsize=7, va="top")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 10)
        ax.set_title(f"{s}  ({len(curves[s])} alloys)", color=INK, fontsize=10, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("distance from nucleus r (Å)", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("spherical mean ρ (e/bohr³)", color=MUTED, fontsize=9)
    fig.suptitle("Atom-centred valence density: one line per alloy", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / "profiles_rho.png", dpi=150)
    plt.close(fig)

    if fit is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor="#fcfcfb")
        _style(ax)
        r = np.linspace(max(fit.r_core.values()), fit.basis.rc, 400)
        for s in SPECIES:
            y = fit.f(s, r)
            if not np.any(y):
                continue
            ax.plot(r, y, color=COLORS[s], lw=2, label=s)
            k = np.searchsorted(r, 2.0)
            ax.annotate(s, (r[k], y[k]), xytext=(4, 2), textcoords="offset points",
                        color=INK, fontsize=8)
        for x in (2.8, 3.25, 4.6, 5.4):
            ax.axvline(x, color=GRID, lw=1.2, zorder=0)
        ax.text(2.82, ax.get_ylim()[1] * 0.95, "bcc shells", color=MUTED, fontsize=7, va="top")
        ax.set_yscale("log")
        ax.set_xlabel("r (Å)", color=MUTED, fontsize=9)
        ax.set_ylabel("f_Z(r) (e/bohr³)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper right")
        ax.set_title(f"Fitted atomic density functions (trained on {args.tag})", color=INK,
                     fontsize=11, loc="left")
        fig.tight_layout()
        fig.savefig(REPORT / f"f_Z_{args.tag}.png", dpi=150)
        plt.close(fig)
    if args.compare:
        fits = {t: load_fit(OUT / f"fit_{t}.npz") for t in args.compare}
        colors = ["#2a78d6", "#eb6834", "#1baf7a"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor="#fcfcfb")
        r = np.linspace(1.0, 6.0, 500)
        for ax, s in zip(axes, ["Ti", "Mo"]):
            _style(ax)
            for (t, f), c in zip(fits.items(), colors):
                y = f.f(s, r) * 1e3
                ax.plot(r, y, color=c, lw=2, label=t)
                k = np.searchsorted(r, 1.6)
                ax.annotate(t, (r[k], y[k]), xytext=(6, 0), textcoords="offset points",
                            color=INK, fontsize=8)
            for x in (2.8, 3.25, 4.6, 5.4):
                ax.axvline(x, color=GRID, lw=1.2, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_ylim(-4, 20)
            ax.set_title(s, color=INK, fontsize=10, loc="left")
            ax.set_xlabel("r (Å); grey lines = bcc neighbour shells", color=MUTED, fontsize=9)
        axes[0].set_ylabel("f_Z(r) (10⁻³ e/bohr³)", color=MUTED, fontsize=9)
        axes[1].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle("Fits of equal quality disagree at neighbour distances", color=INK,
                     fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(REPORT / "f_Z_compare.png", dpi=150)
        plt.close(fig)
    print(f"-> {REPORT}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--train", nargs="+", required=True)
    f.add_argument("--test", nargs="*")
    f.add_argument("--r-core", default="", help="e.g. all=0.9,Ta=1.0 (Angstrom; default 0.8)")
    f.add_argument("--r0", type=float, default=0.6)
    f.add_argument("--rc", type=float, default=6.0)
    f.add_argument("--n-int", type=int, default=27, help="spline intervals / exp terms")
    f.add_argument("--basis", choices=["spline", "exp"], default="spline")
    f.add_argument("--background", action="store_true",
                   help="add a uniform background sum_Z N_Z/V * beta_Z per alloy")
    f.add_argument("--lam", type=float, default=1e-4)
    f.add_argument("--signed", action="store_true", help="allow negative coefficients")
    f.add_argument("--monotone", action="store_true", help="f_Z non-increasing and >= 0")
    f.add_argument("--oracle", action="store_true", help="also fit each test alloy alone")
    f.add_argument("--tag", default="binaries")
    g = sub.add_parser("angular")
    g.add_argument("--ids", nargs="+", required=True)
    g.add_argument("--tag", default="binaries")
    h = sub.add_parser("report")
    h.add_argument("--ids", nargs="+", required=True)
    h.add_argument("--tag", default="binaries")
    h.add_argument("--compare", nargs="*", help="fit tags to overlay for Ti and Mo")
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        ids = resolve(args.ids)
        with ProcessPoolExecutor(args.jobs) as ex:
            for aid, status in ex.map(prepare_one, ids, [20000] * len(ids), [6.5] * len(ids),
                                      [args.force] * len(ids)):
                print(f"{aid}: {status}", flush=True)
    elif args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "angular":
        cmd_angular(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
