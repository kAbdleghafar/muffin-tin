"""Fit per-species radial density functions f_Z so that sum_i f_Zi(|r - R_i|) ~ DFT rho
outside per-species core masks (CLAUDE.md, Phase 1).

An alloy enters as a cached *sample*: random grid points, the DFT values there and every
(point, atom) pair within rc_max. The model is linear in the spline coefficients, so a fit
over many alloys only needs the accumulated normal equations."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.sparse import coo_matrix

from mtpot.dft import ZVAL

from .grid import grid_positions, neighbour_pairs

SPECIES = list(ZVAL)                                   # column blocks: Ti, Zr, Nb, Mo, Hf, Ta
REGIONS = ((0.0, 1.2), (1.2, 1.6), (1.6, np.inf))      # nearest-atom distance bins (Angstrom)


def species_index(atoms):
    return np.array([SPECIES.index(s) for s in atoms.get_chemical_symbols()], dtype=np.int8)


def sample_alloy(atoms, rho, elf=None, n_points=20000, rc_max=6.5, seed=0):
    """Uniform random grid points with their (point, species, distance) pairs within rc_max."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rho.size, n_points, replace=False))
    pts = grid_positions(atoms.cell[:], rho.shape, idx)
    ip, ia, d = neighbour_pairs(pts, atoms, rc_max)
    out = dict(idx=idx.astype(np.int64), rho=rho.ravel()[idx], ip=ip,
               spec=species_index(atoms)[ia], d=d.astype(np.float32),
               rc_max=rc_max, shape=np.array(rho.shape), cell=atoms.cell[:],
               numbers=atoms.numbers, positions=atoms.positions)
    if elf is not None:
        out["elf"] = elf.ravel()[idx]
    return out


def nearest(sample):
    """Nearest-atom distance of every sampled point."""
    dmin = np.full(len(sample["idx"]), np.inf)
    np.minimum.at(dmin, sample["ip"], sample["d"])
    return dmin


def outside_core(sample, r_core):
    """Points with no atom of species Z closer than r_core[Z] (dict symbol -> Angstrom)."""
    rcz = np.array([r_core[s] for s in SPECIES])
    bad = np.zeros(len(sample["idx"]), bool)
    bad[sample["ip"][sample["d"] < rcz[sample["spec"]]]] = True
    return ~bad


def design(sample, basis, keep=None):
    """Dense design matrix (n_points, n_species * basis.size): column block Z holds
    sum over atoms of species Z of B_j(|r - R_i|)."""
    m = sample["d"] < basis.rc
    d, ip, spec = sample["d"][m], sample["ip"][m], sample["spec"][m].astype(np.int64)
    B = basis.design(d).tocoo()
    n = len(sample["idx"])
    A = coo_matrix((B.data, (ip[B.row], spec[B.row] * basis.size + B.col)),
                   shape=(n, len(SPECIES) * basis.size)).toarray()
    return A if keep is None else A[keep]


class DensityFit:
    """Accumulate normal equations over alloys, then solve with a smoothness penalty."""

    def __init__(self, basis, r_core):
        self.basis, self.r_core = basis, r_core
        nc = len(SPECIES) * basis.size
        self.G, self.g, self.n = np.zeros((nc, nc)), np.zeros(nc), 0
        self.coef = None

    def add(self, sample, weight=1.0):
        keep = outside_core(sample, self.r_core)
        A = design(sample, self.basis, keep)
        self.G += weight * A.T @ A
        self.g += weight * A.T @ sample["rho"][keep]
        self.n += keep.sum()

    def solve(self, lam=1e-3, nonneg=True):
        """lam scales the second-difference penalty relative to the mean diagonal of G.
        Species absent from the training data get zero coefficients."""
        nb = self.basis.size
        P = np.kron(np.eye(len(SPECIES)), self.basis.penalty())
        present = np.repeat([self.G[k * nb:(k + 1) * nb, k * nb:(k + 1) * nb].trace() > 0
                             for k in range(len(SPECIES))], nb)
        G = self.G[np.ix_(present, present)]
        H = G + lam * np.mean(np.diag(G)) * P[np.ix_(present, present)]
        g = self.g[present]
        if nonneg:                                    # min c'Hc - 2g'c, c >= 0, as an NNLS
            L = cholesky(H, lower=True)
            c, _ = nnls(L.T, solve_triangular(L, g, lower=True), maxiter=50 * len(g))
        else:
            c = cho_solve(cho_factor(H), g)
        self.coef = np.zeros(len(self.g))
        self.coef[present] = c
        return self

    def f(self, symbol, r):
        nb = self.basis.size
        k = SPECIES.index(symbol)
        return self.basis(self.coef[k * nb:(k + 1) * nb], r)

    def predict(self, sample, keep=None):
        return design(sample, self.basis, keep) @ self.coef


def metrics(fit, sample):
    """Errors outside the core mask, overall and by nearest-atom distance region."""
    keep = outside_core(sample, fit.r_core)
    y, yhat = sample["rho"][keep], fit.predict(sample, keep)
    err, dmin = yhat - y, nearest(sample)[keep]
    out = dict(n=int(keep.sum()), mean=float(y.mean()), rmse=float(np.sqrt(np.mean(err**2))),
               r2=float(1 - np.sum(err**2) / np.sum((y - y.mean())**2)),
               max_abs=float(np.abs(err).max()))
    out["rel_rmse"] = out["rmse"] / out["mean"]
    for lo, hi in REGIONS:
        m = (dmin >= lo) & (dmin < hi)
        if m.any():
            key = f"{lo:g}-{hi:g}".replace("-inf", "+")
            out[f"rel_rmse_{key}"] = float(np.sqrt(np.mean(err[m]**2)) / y[m].mean())
            out[f"charge_err_{key}"] = float(err[m].sum() / y[m].sum())   # relative, signed
    return out
