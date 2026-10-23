"""Phase 1 density machinery on synthetic superpositions with known radial functions."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mtpot.density.basis import RadialBasis
from mtpot.density.fit import DensityFit, metrics, nearest, outside_core, sample_alloy
from mtpot.density.grid import grid_positions, image_atoms, neighbour_pairs
from mtpot.density.profiles import atom_pairs, radial_profiles, real_sph_harm, shell_power

RC = 5.0
TRUE = {"Ti": lambda r: 0.9 * np.exp(-r / 0.6) * (1 - r / RC) ** 3,
        "Mo": lambda r: 0.5 * np.exp(-r / 0.9) * (1 - r / RC) ** 3}


def disordered_bcc(seed, n_mo):
    a = bulk("Ti", "bcc", a=3.25, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    sym = np.array(["Ti"] * len(a))
    sym[rng.choice(len(a), n_mo, replace=False)] = "Mo"
    a.set_chemical_symbols(sym)
    a.positions += rng.normal(0, 0.05, a.positions.shape)
    a.set_cell(a.cell[:] + rng.normal(0, 0.02, (3, 3)), scale_atoms=True)    # triclinic
    return a


def superposed_grid(atoms, shape):
    pts = grid_positions(atoms.cell[:], shape, np.arange(np.prod(shape)))
    ip, ia, d = neighbour_pairs(pts, atoms, RC)
    sym = np.array(atoms.get_chemical_symbols())[ia]
    vals = np.zeros(len(d))
    for s, f in TRUE.items():
        vals[sym == s] = f(d[sym == s])
    return np.bincount(ip, weights=vals, minlength=len(pts)).reshape(shape)


def test_basis_vanishes_smoothly_at_cutoff():
    b = RadialBasis(0.8, 6.0, 26)
    c = np.random.default_rng(0).normal(size=b.size)
    h = 1e-4
    f = lambda r: b(c, np.array([r]))[0]
    assert abs(f(6.0 - h)) < 1e-9
    assert abs((f(6.0 - h) - f(6.0 - 2 * h)) / h) < 1e-6           # f' -> 0
    x = np.linspace(0.8, 6.0 - 3 * b.h, 50)
    assert np.allclose(b.design(x).toarray().sum(1), 1.0)            # partition of unity inside
    assert np.all(b(c, [6.0, 7.0]) == 0.0)


def test_neighbour_pairs_match_brute_force():
    a = disordered_bcc(1, 4)
    pts = grid_positions(a.cell[:], (10, 10, 10), np.arange(0, 1000, 37))
    ip, ia, d, v = neighbour_pairs(pts, a, 6.0, vectors=True)
    pos, own = image_atoms(a, 6.0)
    for p in range(len(pts)):
        dd = np.linalg.norm(pos - pts[p], axis=1)
        assert np.allclose(np.sort(dd[dd < 6.0]), np.sort(d[ip == p]))
    assert np.allclose(np.linalg.norm(v, axis=1), d)


def test_fit_recovers_known_radial_functions():
    shape = (30, 30, 30)
    samples = []
    for seed, n_mo in [(2, 4), (3, 8), (4, 12)]:
        a = disordered_bcc(seed, n_mo)
        rho = superposed_grid(a, shape)
        samples.append(sample_alloy(a, rho, n_points=6000, rc_max=RC, seed=seed))
    r_core = {s: 0.8 for s in ["Ti", "Zr", "Nb", "Mo", "Hf", "Ta"]}
    fit = DensityFit(RadialBasis(0.6, RC, 22), r_core)
    for s in samples[:2]:
        fit.add(s)
    fit.solve(lam=1e-8)
    r = np.linspace(0.8, 4.5, 40)
    for sym, f in TRUE.items():
        assert np.allclose(fit.f(sym, r), f(r), atol=2e-3 * f(0.8)), sym
    m = metrics(fit, samples[2])                                     # held-out composition
    assert m["rel_rmse"] < 1e-2 and m["r2"] > 0.999
    assert np.all(fit.f("Ta", r) == 0)                               # absent species
    fit.solve(lam=1e-8, monotone=True)                               # truth is monotone
    for sym, f in TRUE.items():
        y = fit.f(sym, r)
        assert np.all(np.diff(y) <= 1e-12) and np.allclose(y, f(r), atol=2e-3 * f(0.8)), sym


def test_core_mask_and_nearest():
    a = disordered_bcc(5, 4)
    s = sample_alloy(a, np.zeros((12, 12, 12)), n_points=500, rc_max=4.0)
    keep = outside_core(s, {k: 1.0 for k in ["Ti", "Zr", "Nb", "Mo", "Hf", "Ta"]})
    assert np.all(nearest(s)[keep] >= 1.0) and np.all(nearest(s)[~keep] < 1.0)


def test_radial_profile_of_single_atom():
    a = Atoms("Ti", positions=[[0.1, 0.2, 0.3]], cell=np.eye(3) * 9.0, pbc=True)
    rho = superposed_grid(a, (60, 60, 60))
    r, P = radial_profiles(a, rho, rmax=3.0, dr=0.1)
    ok = (r > 0.5) & (r < 3.0)
    assert np.allclose(P[0, ok], TRUE["Ti"](r[ok]), rtol=0.03)


def test_shell_power_sees_cubic_anisotropy():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(20000, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    vals = 1.0 + 0.3 * ((xyz**4).sum(1) - 0.6)                      # l=0 + cubic l=4
    mean, power, k4rel = shell_power(vals, np.zeros(len(xyz), int), xyz, 1)
    assert mean[0] == pytest.approx(1.0, abs=0.01)
    assert np.all(power[0, :3] < 0.02) and power[0, 3] > 0.05
    assert k4rel[0] == pytest.approx(0.3, rel=0.05)
    Y = real_sph_harm(2, xyz)                                        # orthonormal on the sphere
    assert np.allclose(4 * np.pi * Y.T @ Y / len(xyz), np.eye(5), atol=0.05)
