"""Light readers for thermo_pw outputs (the full parser is scripts/tpw_to_dataset.py)."""
import numpy as np
from ase import Atoms
from ase.units import Bohr

from tpw_to_dataset import RX_AXIS, RX_CELLDM1, RX_TAU, _settings


def tpw_reference(path, max_lines=5000):
    """(unstrained reference structure, settings) from the head of a thermo_pw output."""
    with open(path, errors="replace") as fh:
        lines = [next(fh, "") for _ in range(max_lines)]
    alat, axes = None, np.zeros((3, 3))
    for k, L in enumerate(lines):
        if alat is None and (m := RX_CELLDM1.search(L)):
            alat = float(m.group(1))
        elif m := RX_AXIS.search(L):
            axes[int(m.group(1)) - 1] = [float(x) for x in m.groups()[1:]]
        elif "positions (alat units)" in L:
            sym, pos = [], []
            for L2 in lines[k + 1:]:
                if not (m := RX_TAU.match(L2)):
                    break
                sym.append(m.group(1)); pos.append([float(x) for x in m.groups()[1:]])
            return Atoms(sym, positions=np.array(pos) * alat * Bohr, cell=axes * alat * Bohr,
                         pbc=True), _settings("".join(lines))
    return None, _settings("".join(lines))
