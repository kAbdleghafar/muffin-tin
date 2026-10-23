"""Parser for pw.x (v)c-relax outputs: one frame per SCF (BFGS steps, then the final SCF
that pw.x runs at the relaxed structure with recomputed G-vectors).

Units as in tpw_to_dataset: energies eV (the "!" free energy F, plus -TS separately), forces
eV/A, stress eV/A^3 with ASE sign (tension positive = -(QE printed)), cell and positions A."""
import re

import numpy as np
from ase import Atoms
from ase.units import Bohr, Ry

from tpw_to_dataset import RX_ALAT, RX_AXIS, RX_FORCE, RX_NUM, RX_TAU, _settings, alloy_id_from_path

RX_CELLP = re.compile(r"^CELL_PARAMETERS\s*\(\s*(angstrom|bohr|alat\s*=\s*([-\d.]+))", re.I)


def parse_pw_relax(path, alloy_id=None):
    """Return (frames, info). frames: dicts with E, mTS, F, S, cell, pos, symbols, step and
    kind ('bfgs' or 'final_scf'); info: settings, flags and initial / relaxed structures."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    text = "\n".join(lines[:4000])
    info = {"alloy": alloy_id or alloy_id_from_path(path), "settings": _settings(text),
            "vc": False, "converged": False, "maxsteps": False, "final_scf": False,
            "done": False, "errors": []}
    frames, alat, axes, cell, symbols, cart = [], None, np.zeros((3, 3)), None, [], None
    kind, cur, step = "bfgs", None, 0
    initial = relaxed = None
    i, n = 0, len(lines)
    while i < n:
        L = lines[i]
        if m := RX_ALAT.search(L):
            alat = float(m.group(1))
        elif m := RX_AXIS.search(L):
            axes[int(m.group(1)) - 1] = [float(x) for x in m.groups()[1:]]
            cell = axes * alat * Bohr
        elif "positions (alat units)" in L:
            symbols, pos = [], []
            j = i + 1
            while j < n and (m := RX_TAU.match(lines[j])):
                symbols.append(m.group(1)); pos.append([float(x) for x in m.groups()[1:]])
                j += 1
            cart = np.array(pos) * alat * Bohr
            if initial is None:
                initial = Atoms(symbols, positions=cart, cell=cell, pbc=True)
            i = j
            continue
        elif m := RX_CELLP.match(L):
            info["vc"] = True
            rows = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in (1, 2, 3)])
            unit = m.group(1).lower()
            cell = rows if unit == "angstrom" else rows * Bohr if unit == "bohr" \
                else rows * float(m.group(2)) * Bohr
            i += 4
            continue
        elif L.startswith("ATOMIC_POSITIONS"):
            unit = L.split("(")[-1].strip(" )").lower() if "(" in L else "alat"
            pos = np.array([[float(x) for x in lines[i + 1 + k].split()[1:4]]
                            for k in range(len(symbols))])
            cart = pos @ cell if unit == "crystal" else pos * Bohr if unit == "bohr" \
                else pos if unit == "angstrom" else pos * alat * Bohr
            i += 1 + len(symbols)
            continue
        elif L.startswith("!") and "total energy" in L:
            step += 1
            cur = {"E": float(RX_NUM.findall(L)[-1]) * Ry, "F": None, "S": None,
                   "cell": np.array(cell, float).copy(), "pos": cart.copy(),
                   "symbols": list(symbols), "step": step, "kind": kind}
            frames.append(cur)
        elif "smearing contrib. (-TS)" in L and cur is not None and "mTS" not in cur:
            cur["mTS"] = float(RX_NUM.findall(L)[-1]) * Ry
        elif "Forces acting on atoms" in L and cur is not None and cur["F"] is None:
            f, j = [], i + 1
            while len(f) < len(symbols) and j < n and "total   stress" not in lines[j] \
                    and not lines[j].startswith("!"):
                if m := RX_FORCE.search(lines[j]):
                    f.append([float(x) for x in m.groups()])
                j += 1
            cur["F"] = np.array(f) * Ry / Bohr if len(f) == len(symbols) else None
            i = j
            continue
        elif "total   stress" in L and cur is not None and cur["S"] is None:
            s = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in (1, 2, 3)])
            cur["S"] = -s * Ry / Bohr**3
            i += 4
            continue
        elif "bfgs converged" in L:
            info["converged"] = True
        elif "maximum number of steps has been reached" in L:
            info["maxsteps"] = True
        elif L.startswith("End final coordinates"):
            relaxed = Atoms(symbols, positions=cart, cell=cell, pbc=True)
        elif "Final scf calculation at the relaxed structure" in L:
            info["final_scf"] = True
            kind = "final_scf"
        elif "Error in routine" in L:
            info["errors"].append(L.strip())
        elif "JOB DONE" in L:
            info["done"] = True
        i += 1
    info["initial"], info["relaxed"] = initial, relaxed
    return frames, info
