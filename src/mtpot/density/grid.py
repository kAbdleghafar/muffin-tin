"""Cube grids and periodic grid-point / atom neighbour pairs."""
import numpy as np
from scipy.spatial import cKDTree


def grid_positions(cell, shape, flat_idx):
    """Cartesian positions (Angstrom) of grid points given by flat C-order indices.
    QE pp.x cubes start at the origin; point (i, j, k) sits at (i/n1, j/n2, k/n3) fractional."""
    ijk = np.stack(np.unravel_index(flat_idx, shape), axis=-1)
    return (ijk / np.asarray(shape)) @ np.asarray(cell)


def image_atoms(atoms, rc):
    """Periodic images of all atoms that can lie within rc of a point in the home cell.
    Returns (positions, atom index of each image)."""
    cell = atoms.cell[:]
    frac = atoms.get_scaled_positions(wrap=True)
    vol = abs(np.linalg.det(cell))
    heights = [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
               for i in range(3)]
    n = [int(np.ceil(rc / h)) + 1 for h in heights]
    shifts = np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in n], indexing="ij")).reshape(3, -1).T
    pos = ((frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)) @ cell
    owner = np.tile(np.arange(len(atoms)), len(shifts))
    return pos, owner


def neighbour_pairs(points, atoms, rc, vectors=False):
    """All (point, atom) pairs closer than rc, periodic images included.
    Returns point index, atom index and distance (Angstrom), sorted by point, plus the
    displacement point - atom image if vectors=True."""
    pos, owner = image_atoms(atoms, rc)
    pairs = cKDTree(points).sparse_distance_matrix(cKDTree(pos), rc, output_type="ndarray")
    pairs = pairs[np.argsort(pairs["i"], kind="stable")]
    out = (pairs["i"].astype(np.int32), owner[pairs["j"]].astype(np.int32), pairs["v"])
    return out + (points[pairs["i"]] - pos[pairs["j"]],) if vectors else out
