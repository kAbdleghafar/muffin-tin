"""Atom-centred analysis of grid quantities: radial profiles and angular power on shells."""
import numpy as np
from scipy.special import sph_harm_y

from .grid import grid_positions, neighbour_pairs


def atom_pairs(atoms, shape, rmax):
    """(grid index, atom index, distance, displacement) for every grid point within rmax of an
    atom (periodic). Memory ~ n_atoms * (4/3 pi rmax^3) / voxel volume pairs."""
    idx = np.arange(int(np.prod(shape)))
    pts = grid_positions(atoms.cell[:], shape, idx)
    ip, ia, d, v = neighbour_pairs(pts, atoms, rmax, vectors=True)
    return idx[ip], ia, d, v


def radial_profiles(atoms, data, rmax=3.0, dr=0.02, pairs=None):
    """Spherical average of `data` around every atom: (r_centres, profiles[n_atoms, n_bins])."""
    gi, ia, d, _ = pairs if pairs is not None else atom_pairs(atoms, data.shape, rmax)
    nb = int(round(rmax / dr))
    b = np.minimum((d / dr).astype(int), nb - 1)
    key = ia * nb + b
    n = np.bincount(key, minlength=len(atoms) * nb)
    s = np.bincount(key, weights=data.ravel()[gi], minlength=len(atoms) * nb)
    with np.errstate(invalid="ignore"):
        prof = (s / n).reshape(len(atoms), nb)
    return (np.arange(nb) + 0.5) * dr, prof


def real_sph_harm(l, xyz):
    """Real orthonormal spherical harmonics Y_lm (m = -l..l) at unit vectors xyz -> (n, 2l+1)."""
    theta = np.arccos(np.clip(xyz[:, 2], -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    out = []
    for m in range(-l, l + 1):
        y = sph_harm_y(l, abs(m), theta, phi)
        if m < 0:
            out.append(np.sqrt(2) * (-1) ** m * y.imag)
        elif m == 0:
            out.append(y.real)
        else:
            out.append(np.sqrt(2) * (-1) ** m * y.real)
    return np.stack(out, axis=1)


def k4(xyz):
    """Cubic harmonic x^4 + y^4 + z^4 - 3/5 (zero mean on the sphere); + toward <100>, - toward <111>."""
    return (xyz**4).sum(axis=1) - 0.6


def _harmonics(xyz, lmax):
    return np.concatenate([real_sph_harm(l, xyz) for l in range(lmax + 1)], axis=1)


def _k4_coefficients(lmax):
    """Expansion of the K4 cubic harmonic in the real Y_lm basis (exact: K4 is l=4)."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(4000, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    return np.linalg.lstsq(_harmonics(xyz, lmax), k4(xyz), rcond=None)[0]


def shell_power(values, atom_idx, xyz, n_atoms, lmax=4, dr=None):
    """Angular content of `values` sampled on a thin shell around each atom, from a per-atom
    least-squares fit of real Y_lm up to lmax (robust to the uneven angular spread of grid
    points). Returns mean[n_atoms]; power[n_atoms, lmax] with power_l = sqrt(sum_m a_lm^2)/a_00
    (relative amplitude of the l component); k4rel[n_atoms] = K4 projection / mean
    (> 0: more density toward <100>, i.e. 2nd neighbours; < 0: toward <111>, 1st neighbours).
    dr: radial offset of each sample from the shell radius; fitted as an extra (radial-slope)
    term so the shell's finite thickness does not leak into the angular coefficients."""
    Y = _harmonics(xyz, lmax)
    nY = Y.shape[1]
    if dr is not None:
        Y = np.column_stack([Y, dr])
    kc = _k4_coefficients(lmax)
    mean = np.full(n_atoms, np.nan)
    power = np.full((n_atoms, lmax), np.nan)
    k4rel = np.full(n_atoms, np.nan)
    for a in range(n_atoms):
        m = atom_idx == a
        if m.sum() < 4 * Y.shape[1]:
            continue
        c = np.linalg.lstsq(Y[m], values[m], rcond=None)[0][:nY]
        mean[a] = c[0] / np.sqrt(4 * np.pi)
        for l in range(1, lmax + 1):
            power[a, l - 1] = np.linalg.norm(c[l * l:(l + 1) ** 2]) / abs(c[0])
        k4rel[a] = (c @ kc) / (kc @ kc) / mean[a]
    return mean, power, k4rel
