"""EAM model: autograd forces/stress vs finite differences, batching, setfl export vs ASE."""
import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.eam import EAM as AseEAM

from mtpot.potential.eam import EAM, SPECIES, collate, frame_graph
from mtpot.potential.export import write_setfl
from mtpot.potential.splines import Splines


def toy_model(seed=0):
    """A smooth, physically shaped random EAM (decaying f, repulsive-attractive phi,
    concave F) so the tests exercise realistic magnitudes."""
    g = torch.Generator().manual_seed(seed)
    m = EAM(rc=5.0, n_pair=16, n_dens=16, rho_max=3.0, n_embed=8)
    with torch.no_grad():
        r = m.f.x0 + m.f.h * (torch.arange(m.f.size, dtype=torch.float64) - 1)
        amp = 1 + 0.3 * torch.rand(len(SPECIES), 1, generator=g, dtype=torch.float64)
        m.f.coef.copy_(amp * torch.exp(-(r - 2.0) / 0.8)[None] * 0.3)
        rp = m.phi.x0 + m.phi.h * (torch.arange(m.phi.size, dtype=torch.float64) - 1)
        morse = lambda D, a, re: D * (torch.exp(-2 * a * (rp - re)) - 2 * torch.exp(-a * (rp - re)))
        m.phi.coef.copy_(torch.stack([morse(0.3 + 0.1 * k / 21, 1.4, 2.9) for k in range(21)]))
        x = m.F.x0 + m.F.h * (torch.arange(m.F.size, dtype=torch.float64) - 1)
        m.F.coef.copy_(-torch.sqrt(x.clamp(min=0))[None].repeat(len(SPECIES), 1)
                       * (1 + 0.2 * torch.rand(len(SPECIES), 1, generator=g, dtype=torch.float64)))
        m.e0.copy_(torch.randn(len(SPECIES), generator=g, dtype=torch.float64))
    return m


def alloy(seed=0, symbols=("Ti", "Mo", "Ta", "Zr")):
    a = bulk("Ti", "bcc", a=3.25, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    a.set_chemical_symbols(rng.choice(symbols, len(a)))
    a.positions += rng.normal(0, 0.06, a.positions.shape)
    a.set_cell(a.cell[:] @ (np.eye(3) + rng.normal(0, 0.01, (3, 3))), scale_atoms=True)
    return a


def test_splines_cutoff_and_extrapolation():
    s = Splines(1, 1.0, 4.0, 6, cutoff=True, init=torch.randn(1, 6, dtype=torch.float64))
    w = torch.zeros(5, dtype=torch.long)
    x = torch.tensor([4.0 - 1e-7, 4.0, 5.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    y = s(x, w)
    assert abs(y[0].item()) < 1e-12 and abs(y[1].item()) < 1e-12 and y[2] == 0
    d = torch.autograd.grad(y.sum(), x)[0]
    assert abs(d[0].item()) < 1e-9                                       # C1 at the cutoff
    h = 1e-6                                                             # continuous at x0
    lo, hi = s(torch.tensor([1.0 - h, 1.0 + h], dtype=torch.float64), w[:2])
    assert abs((lo - hi).item()) < 1e-5


def test_forces_and_stress_match_finite_differences():
    m, a = toy_model(1), alloy(1)
    out = m(collate([frame_graph(a, m.rc)]))
    F, S = out["forces"].detach().numpy(), out["stress"].detach().numpy()[0]
    E = lambda at: m.energies(collate([frame_graph(at, m.rc)])).item()
    h = 1e-5
    for k in [0, 5, 11]:
        for c in range(3):
            ap, am = a.copy(), a.copy()
            ap.positions[k, c] += h
            am.positions[k, c] -= h
            assert -(E(ap) - E(am)) / (2 * h) == pytest.approx(F[k, c], abs=1e-6)
    vol = a.get_volume()
    for v, (p, q) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
        eps = np.zeros((3, 3))
        eps[p, q] = eps[q, p] = h
        ap, am = a.copy(), a.copy()
        ap.set_cell(a.cell[:] @ (np.eye(3) + eps), scale_atoms=True)
        am.set_cell(a.cell[:] @ (np.eye(3) - eps), scale_atoms=True)
        de = (E(ap) - E(am)) / (2 * h) / vol / (1 if p == q else 2)
        assert de == pytest.approx(S[v], abs=1e-7)
    assert np.abs(F.sum(0)).max() < 1e-10                                # momentum


def test_batch_equals_single_frames():
    m = toy_model(2)
    frames = [alloy(3), alloy(4, ("Nb", "Hf")), alloy(5)[:-3]]
    both = m(collate([frame_graph(a, m.rc) for a in frames]))
    off = 0
    for k, a in enumerate(frames):
        one = m(collate([frame_graph(a, m.rc)]))
        assert torch.allclose(both["energy"][k], one["energy"][0])
        assert torch.allclose(both["stress"][k], one["stress"][0])
        assert torch.allclose(both["forces"][off:off + len(a)], one["forces"])
        off += len(a)


def test_setfl_export_matches_ase_eam(tmp_path):
    m, a = toy_model(3), alloy(6)
    path = write_setfl(m, tmp_path / "toy.eam.alloy", nr=10000, nrho=10000)
    a.calc = AseEAM(potential=str(path))
    out = m(collate([frame_graph(a, m.rc)]))
    assert a.get_potential_energy() == pytest.approx(out["energy"].item(), abs=1e-5)
    assert np.allclose(a.get_forces(), out["forces"].detach().numpy(), atol=1e-4)
    assert np.allclose(a.get_stress(), out["stress"].detach().numpy()[0], atol=1e-6)
