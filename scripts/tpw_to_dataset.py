#!/usr/bin/env python3
"""
Turn thermo_pw elastic-constant outputs (what='scf_elastic_constants', ions relaxed)
into a training set for an interatomic potential.

For every ionic step of every strained geometry it stores:
  energy  (eV)      : the "!" total energy, i.e. the smearing free energy F,
                      which is the quantity forces and stress are derivatives of
  forces  (eV/A)
  stress  (eV/A^3)  : ASE sign convention (tension positive) = -(QE printed stress)
plus metadata: alloy id, work index, applied strain, ionic step, final-step flag,
smearing scheme and the -TS term (eV), so energies from different smearing schemes
can be put on the same footing (Gaussian: E0 = F - 0.5*(-TS); Marzari-Vanderbilt: E0 ~ F).

It also reads thermo_pw's C_ij, re-derives C_ij from the relaxed stresses as a
check, and reports the residual (zero-strain) stress and a cubic projection.

Usage:
    python tpw_to_dataset.py elast_*.out -o dataset.extxyz -s elastic_summary.csv
ASE's own espresso-out reader cannot read these files (no "Program PWSCF" marker).
"""
import argparse, csv, re, sys
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Ry, Bohr, GPa

KBAR = 0.1 * GPa  # 1 kbar in eV/A^3
RX_ALAT = re.compile(r"lattice parameter \(alat\)\s*=\s*([-\d.]+)")
RX_CELLDM1 = re.compile(r"(?:celldm\(1\)=|\balat=)\s*([-\d.]+)")
RX_AXIS = re.compile(r"a\((\d)\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
RX_TAU = re.compile(r"^\s*\d+\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
RX_FORCE = re.compile(r"atom\s+\d+\s+type\s+\d+\s+force =\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)")
RX_WORK = re.compile(r"Doing work\s+(\d+)\s*/\s*(\d+)")
RX_NUM = re.compile(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?")


def _settings(text):
    s = {}
    for key, rx in [("ecutwfc_Ry", r"kinetic-energy cutoff\s*=\s*([\d.]+)"),
                    ("ecutrho_Ry", r"charge density cutoff\s*=\s*([\d.]+)"),
                    ("nks", r"number of k points=\s*(\d+)"),
                    ("smearing", r"number of k points=\s*\d+\s+(.+?) smearing"),
                    ("degauss_Ry", r"smearing, width \(Ry\)=\s*([\d.]+)"),
                    ("xc", r"Exchange-correlation=\s*(\S+)"),
                    ("nelec", r"number of electrons\s*=\s*([\d.]+)")]:
        m = re.search(rx, text)
        s[key] = m.group(1).strip() if m else None
    return s


def alloy_id_from_path(path):
    """Numeric suffix of the file name: elast_01.out -> '01' (file stem if there is none)."""
    name = str(path).split("/")[-1]
    m = re.search(r"(\d+)\.\w+$", name)
    return m.group(1) if m else re.sub(r"\.\w+$", "", name)


def parse_thermo_pw(path, alloy_id=None):
    """Return (frames, info). info holds settings, reference structure, C_ij."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    alloy_id = alloy_id or alloy_id_from_path(path)
    info = {"alloy": alloy_id, "settings": _settings("\n".join(lines[:3000])),
            "works_total": None, "works_computed": set(), "works_converged": set(),
            "C_tpw_kbar": None}

    frames, alat, axes, symbols, cart = [], None, np.zeros((3, 3)), [], None
    strain, work, step, cur = None, None, 0, None
    ref = None
    i, n = 0, len(lines)

    def cell():
        return axes * alat * Bohr

    while i < n:
        L = lines[i]
        if (m := RX_CELLDM1.search(L)) and alat is None:
            alat = float(m.group(1))
        elif m := RX_ALAT.search(L):
            alat = float(m.group(1))
        elif m := RX_AXIS.search(L):
            axes[int(m.group(1)) - 1] = [float(x) for x in m.groups()[1:]]
        elif "positions (alat units)" in L:
            symbols, pos = [], []
            j = i + 1
            while j < n and (m := RX_TAU.match(lines[j])):
                symbols.append(m.group(1)); pos.append([float(x) for x in m.groups()[1:]])
                j += 1
            cart = np.array(pos) * alat * Bohr
            if ref is None:
                ref = Atoms(symbols, positions=cart, cell=cell(), pbc=True)
            i = j; continue
        elif "Applying the following strain" in L:
            strain = np.array([[float(x) for x in RX_NUM.findall(lines[i + k])] for k in (1, 2, 3)])
            step = 0
        elif m := RX_WORK.search(L):
            work = int(m.group(1)); info["works_total"] = int(m.group(2))
        elif L.startswith("ATOMIC_POSITIONS"):
            unit = L.split("(")[-1].strip(" )").lower() if "(" in L else "alat"
            pos = []
            for k in range(len(symbols)):
                parts = lines[i + 1 + k].split()
                pos.append([float(x) for x in parts[1:4]])
            pos = np.array(pos)
            if unit == "crystal":
                cart = pos @ cell()
            elif unit == "bohr":
                cart = pos * Bohr
            elif unit == "angstrom":
                cart = pos
            else:
                cart = pos * alat * Bohr
            i += 1 + len(symbols); continue
        elif L.startswith("!") and "total energy" in L and strain is not None:
            cur = {"E": float(RX_NUM.findall(L)[-1]) * Ry, "F": None, "S": None,
                   "cell": cell().copy(), "pos": cart.copy(), "symbols": list(symbols)}
            frames.append(cur); step += 1
            cur["meta"] = {"alloy": alloy_id, "work": work, "ionic_step": step,
                           "strain": strain.copy()}
            info["works_computed"].add(work)
        elif "smearing contrib. (-TS)" in L and cur is not None and "mTS" not in cur:
            cur["mTS"] = float(RX_NUM.findall(L)[-1]) * Ry
        elif "Forces acting on atoms" in L and cur is not None and cur["F"] is None:
            f, j = [], i + 1
            while len(f) < len(symbols) and j < n:
                if m := RX_FORCE.search(lines[j]):
                    f.append([float(x) for x in m.groups()])
                j += 1
            cur["F"] = np.array(f) * Ry / Bohr
            i = j; continue
        elif "total   stress" in L and cur is not None and cur["S"] is None:
            s = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in (1, 2, 3)])
            cur["S"] = -s * Ry / Bohr**3          # QE sign -> ASE sign
            i += 4; continue
        elif "bfgs converged" in L and work is not None:
            info["works_converged"].add(work)
        elif "Elastic constants C_ij (kbar)" in L:
            info["C_tpw_kbar"] = np.array([[float(x) for x in lines[i + 2 + k].split()[1:7]]
                                           for k in range(6)])
        i += 1

    # mark the last ionic step of each strained geometry, if its relaxation converged
    for a, b in zip(frames, frames[1:] + [None]):
        last = b is None or b["meta"]["work"] != a["meta"]["work"]
        a["meta"]["final"] = last and a["meta"]["work"] in info["works_converged"]
    info["reference"] = ref
    return frames, info


def to_atoms(fr, smearing=None):
    a = Atoms(fr["symbols"], positions=fr["pos"], cell=fr["cell"], pbc=True)
    calc = SinglePointCalculator(a, energy=fr["E"], forces=fr["F"],
                                 stress=fr["S"] if fr["S"] is not None else None)
    a.calc = calc
    m = fr["meta"]
    a.info.update(alloy=m["alloy"], work=m["work"], ionic_step=m["ionic_step"],
                  final=m["final"], strain_voigt=voigt_strain(m["strain"]),
                  minus_TS=fr.get("mTS", 0.0), smearing=smearing or "unknown")
    return a


def voigt_strain(e):
    """Engineering Voigt strain (e1..e6) from a 3x3 strain matrix."""
    return np.array([e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])


def voigt_stress(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def refit_cij(frames):
    """C_ij (kbar) and residual stress from relaxed stresses: linear fit per strain type."""
    groups = {}
    for fr in frames:
        if not fr["meta"]["final"] or fr["S"] is None:
            continue
        e = voigt_strain(fr["meta"]["strain"])
        j = int(np.argmax(np.abs(e)))
        if np.count_nonzero(np.abs(e) > 1e-8) != 1:
            continue
        groups.setdefault(j, []).append((e[j], voigt_stress(fr["S"]) / KBAR))  # tension +, kbar
    C, resid = np.full((6, 6), np.nan), []
    for j, pts in groups.items():
        x = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
        if len(x) < 2:
            continue
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        C[:, j] = coef[0]; resid.append(coef[1])
    return C, (np.mean(resid, axis=0) if resid else None)


def cubic_projection(C):
    Cs = 0.5 * (C + C.T)
    c11 = np.mean([Cs[0, 0], Cs[1, 1], Cs[2, 2]])
    c12 = np.mean([Cs[0, 1], Cs[0, 2], Cs[1, 2]])
    c44 = np.mean([Cs[3, 3], Cs[4, 4], Cs[5, 5]])
    Ccub = np.zeros((6, 6))
    Ccub[:3, :3] = c12; np.fill_diagonal(Ccub[:3, :3], c11)
    Ccub[3:, 3:] = np.eye(3) * c44
    dev = np.linalg.norm(Cs - Ccub) / np.linalg.norm(Cs)
    return c11, c12, c44, dev, Cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="dataset.extxyz")
    ap.add_argument("-s", "--summary", default="elastic_summary.csv")
    args = ap.parse_args()

    all_atoms, rows = [], []
    for path in args.files:
        frames, info = parse_thermo_pw(path)
        frames_ok = [f for f in frames if f["F"] is not None and f["S"] is not None]
        sm = (info["settings"]["smearing"] or "none").split()[0]
        all_atoms += [to_atoms(f, sm) for f in frames_ok]
        missing = sorted(set(range(1, (info["works_total"] or 0) + 1)) - info["works_computed"])
        st = info["settings"]
        print(f"\n== {info['alloy']}: {info['reference'].get_chemical_formula()}, "
              f"{len(info['reference'])} atoms")
        print(f"   {st['xc']}, ecut {st['ecutwfc_Ry']}/{st['ecutrho_Ry']} Ry, "
              f"{st['smearing']} {st['degauss_Ry']} Ry, {st['nks']} k-points")
        print(f"   frames: {len(frames_ok)}  strained geometries computed: "
              f"{len(info['works_computed'])}/{info['works_total']}")
        if missing:
            print(f"   WARNING: works {missing} not in this file (read from restart); "
                  f"their energies/forces/stresses are missing")
        unconverged = sorted(info["works_computed"] - info["works_converged"])
        if unconverged:
            print(f"   WARNING: works {unconverged} have no converged BFGS relaxation; "
                  f"none of their frames is marked final")
        if len(frames_ok) < len(frames):
            print(f"   WARNING: {len(frames) - len(frames_ok)} frames dropped "
                  f"(no forces or stress printed)")
        row = {"alloy": info["alloy"], "formula": info["reference"].get_chemical_formula(),
               "natoms": len(info["reference"]), "frames": len(frames_ok),
               "works_computed": len(info["works_computed"]),
               "smearing": f"{st['smearing']} {st['degauss_Ry']}"}
        if info["C_tpw_kbar"] is not None:
            c11, c12, c44, dev, _ = cubic_projection(info["C_tpw_kbar"])
            row.update(C11_GPa=round(c11 / 10, 1), C12_GPa=round(c12 / 10, 1),
                       C44_GPa=round(c44 / 10, 1), noncubic_frac=round(dev, 3))
            print(f"   thermo_pw cubic projection: C11 {c11/10:.1f}  C12 {c12/10:.1f}  "
                  f"C44 {c44/10:.1f} GPa  (non-cubic part {100*dev:.1f}%)")
        if not missing:
            Cfit, s0 = refit_cij(frames)
            if info["C_tpw_kbar"] is not None:
                d = np.nanmax(np.abs(Cfit - info["C_tpw_kbar"]))
                row["max_refit_diff_GPa"] = round(d / 10, 2)
                print(f"   refit from stresses vs thermo_pw: max |dC| = {d/10:.2f} GPa")
            if s0 is not None:
                row["residual_P_GPa"] = round(-np.mean(s0[:3]) / 10, 3)
                print(f"   residual stress at zero strain (GPa): "
                      f"{np.array2string(s0 / 10, precision=3)}")
        rows.append(row)

    write(args.out, all_atoms, format="extxyz")
    order = ["alloy", "formula", "natoms", "frames", "works_computed", "smearing", "C11_GPa",
             "C12_GPa", "C44_GPa", "noncubic_frac", "max_refit_diff_GPa", "residual_P_GPa"]
    keys = [k for k in order if any(k in r for r in rows)]
    with open(args.summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(all_atoms)} frames to {args.out} and {len(rows)} rows to {args.summary}")


if __name__ == "__main__":
    main()
