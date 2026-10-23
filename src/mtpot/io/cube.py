"""Gaussian cube files from QE pp.x: bohr units, periodic grid without the repeated edge plane."""
import numpy as np
from ase.io.cube import read_cube_data
from ase.units import Bohr


def read_cube(path):
    """(atoms, data): atoms in Angstrom, data as written (rho in e/bohr^3, ELF dimensionless)."""
    data, atoms = read_cube_data(str(path))
    return atoms, data


def integrate(atoms, data):
    """Cell integral of a per-bohr^3 grid quantity, e.g. the electron count of rho."""
    return float(data.mean() * atoms.get_volume() / Bohr**3)


def structure_deviation(atoms, ref):
    """Largest cell or position difference (Angstrom, positions modulo lattice vectors)
    between two periodic structures; inf if the species sequence differs."""
    if list(atoms.numbers) != list(ref.numbers):
        return np.inf
    cell = ref.cell[:]
    d = (atoms.positions - ref.positions) @ np.linalg.inv(cell)
    dpos = np.abs((d - np.round(d)) @ cell).max()
    return float(max(np.abs(atoms.cell[:] - cell).max(), dpos))
