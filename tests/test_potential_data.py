"""Training-data pipeline on synthetic thermo_pw outputs with known answers."""
import numpy as np
import pytest
import torch
from ase.units import Ry

from mtpot.potential.data import build_alloy, reference_energies
from mtpot.potential.eam import EAM
from mtpot.potential.train import (EV_A3_GPA, AlloyBatch, cij_from_stress, evaluate, init_model,
                                   loss_of, summarize)
from synthetic_tpw import DEFAULT_S0_KBAR, SyntheticRun, default_C_kbar


@pytest.fixture
def built(tmp_path):
    run = SyntheticRun(default_C_kbar(), DEFAULT_S0_KBAR, n_steps=3, smearing="gaussian",
                       degauss=0.009, minus_TS_Ry=-7e-4, maxstep_works=[5])
    return run, build_alloy(run.write(tmp_path / "elast_07.out"), rc=5.0)


def test_build_alloy_selects_first_and_final_steps(built):
    run, d = built
    assert d["alloy"] == "07" and len(d["energy"]) == 48
    assert list(d["step"][:2]) == [1, 3] and d["final"].sum() == 24
    assert np.allclose(d["weight"], 0.5)
    assert not d["converged"][d["work"] == 5].any() and d["converged"][d["work"] == 6].all()
    # Gaussian smearing: E0 = F - (-TS)/2
    assert d["energy"][0] == pytest.approx((run.energy_Ry(1, 1) - 0.5 * run.minus_TS_Ry) * Ry)
    assert np.allclose(d["C_GPa"], run.C / 10)


def test_cij_from_final_frame_stresses(built):
    run, d = built
    mu = reference_energies([d])
    ab = AlloyBatch(d, mu)
    assert ab.cij is not None
    C = cij_from_stress(ab.s_target, ab.cij)
    assert torch.allclose(C, torch.as_tensor(run.C / 10), atol=0.03)     # GPa, print precision


def test_training_step_reduces_loss(built):
    _, d = built
    ab = AlloyBatch(d, reference_energies([d]))
    m = init_model(EAM(rc=5.0, n_pair=10, n_dens=10, n_embed=6))
    w = dict(E=1, F=1, S=1, C=1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        L = loss_of(evaluate(m, ab, create_graph=True), ab, w)
        L.backward()
        opt.step()
        losses.append(L.item())
    assert losses[-1] < losses[0]
    s = summarize(m, [ab])
    assert set(s) >= {"E_meV_atom", "F_eV_A", "S_GPa", "C11_MAE", "C44_MAE"}


def test_relax_sets_use_energy_differences_within_groups():
    from mtpot.potential.train import group_centred
    g = torch.tensor([0, 0, 0, 1])
    w = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    x = torch.tensor([1.0, 2.0, 3.0, 7.0], dtype=torch.float64)
    y = group_centred(x, g, w)
    assert torch.allclose(y, torch.tensor([-1.25, -0.25, 0.75, 0.0], dtype=torch.float64))
    assert torch.allclose(group_centred(x + 5.0, g, w), y)              # shift-invariant


def test_hessian_terms_in_training(built):
    _, d = built
    assert "ref_positions" in d and len(d["ref_i"]) > 0
    ab = AlloyBatch(d, reference_energies([d]))
    ab.hessian = True
    m = init_model(EAM(rc=5.0, n_pair=10, n_dens=10, n_embed=6))
    res = evaluate(m, ab, create_graph=True)
    assert res["eig"].shape == (3 * ab.nat,) and res["C_rel"].shape == (6, 6)
    w0 = dict(E=1, F=1, S=1, C=1)
    L0 = loss_of(res, ab, w0)
    L1 = loss_of(res, ab, dict(w0, H=1.0, kmin=1e3))                     # every mode below kmin
    assert L1 > L0
    L1.backward()                                                        # differentiable
    assert all(p.grad is not None for p in m.parameters())
    s = summarize(m, [ab])
    assert 0.0 <= s["stable_frac"] <= 1.0
