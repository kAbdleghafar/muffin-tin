"""Phase 3 workflow on toy EAM models (independent checks via ASE's EAM calculator)."""
import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.eam import EAM as AseEAM

from mtpot.elastic.workflow import TorchCalculator, cubic, elastic_constants, predict
from mtpot.potential.eam import collate, frame_graph
from mtpot.potential.export import write_setfl
from test_eam import alloy, toy_model


def test_calculator_matches_model_and_ase(tmp_path):
    m, a = toy_model(4), alloy(7)
    out = m(collate([frame_graph(a, m.rc)]))
    a.calc = TorchCalculator(m)
    assert a.get_potential_energy() == pytest.approx(out["energy"].item())
    assert np.allclose(a.get_forces(), out["forces"].detach().numpy())
    assert np.allclose(a.get_stress(), out["stress"][0].detach().numpy())
    b = a.copy()
    b.calc = AseEAM(potential=str(write_setfl(m, tmp_path / "t.eam.alloy", nr=10000, nrho=10000)))
    assert np.allclose(b.get_stress(), a.get_stress(), atol=1e-6)


def test_workflow_same_with_torch_and_ase_calculators(tmp_path):
    torch.set_num_threads(1)
    m, a = toy_model(5), alloy(8)
    C_t, s0_t, _ = elastic_constants(a, TorchCalculator(m), relax=False)
    ase_calc = AseEAM(potential=str(write_setfl(m, tmp_path / "t.eam.alloy", nr=10000, nrho=10000)))
    C_a, s0_a, _ = elastic_constants(a, ase_calc, relax=False)    # ASE's EAM is slow
    assert np.abs(C_t - C_a).max() < 0.2                                     # GPa
    assert np.abs(s0_t - s0_a).max() < 0.01


def test_perfect_bcc_is_cubic_and_needs_no_relaxation():
    torch.set_num_threads(1)
    m = toy_model(6)
    a = bulk("Mo", "bcc", a=3.15, cubic=True).repeat((2, 2, 2))
    calc = TorchCalculator(m)
    C, _, info = elastic_constants(a, calc, relax=True)
    C0, _, _ = elastic_constants(a, calc, relax=False)
    assert np.allclose(C, C0, atol=1e-6)
    assert np.allclose(np.diag(C)[:3], C[0, 0], atol=1e-6) and np.allclose(np.diag(C)[3:], C[3, 3], atol=1e-6)
    assert np.allclose([C[0, 1], C[0, 2], C[1, 2]], C[0, 1], atol=1e-6)
    assert np.abs(C[:3, 3:]).max() < 1e-6 and np.abs(C[3:, :3]).max() < 1e-6
    assert all(w["steps"] <= 1 for w in info)


def test_predict_modes_run():
    m, a = toy_model(7), alloy(9, ("Mo", "Nb"))
    r = predict(a, m, mode="relax_ions")
    assert r["cubic"].shape == (3,) and r["pre"]["converged"]
    assert np.allclose(r["cubic"], cubic(r["C"]))
