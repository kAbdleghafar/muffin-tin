"""pw.x relax parser on real files (skips unless data/raw/relax_<id>.out is present)."""
import numpy as np
import pytest
from ase.io import read

from conftest import raw_file
from mtpot.io.pwscf import parse_pw_relax
from qc_relax import deviation, tpw_reference

pytestmark = pytest.mark.data


@pytest.mark.parametrize("aid", ["74", "01"])
def test_matches_ase_reader(aid):
    """Forces, stress, cell and positions agree with ASE's independent pw.x reader (energies
    differ only by ASE's older Rydberg constant, ~1e-7 relative)."""
    path = raw_file("relax", aid, "out")
    frames, info = parse_pw_relax(path)
    ase_frames = read(path, index=":", format="espresso-out")
    assert len(frames) == len(ase_frames)
    for f, a in zip(frames, ase_frames):
        assert np.allclose(f["F"], a.get_forces(), atol=1e-6)
        assert np.allclose(f["S"], a.get_stress(voigt=False), atol=1e-8)
        assert np.allclose(f["cell"], a.cell[:], atol=1e-4)
        assert np.allclose(f["pos"], a.positions, atol=1e-4)
        assert f["E"] == pytest.approx(a.get_potential_energy(force_consistent=True), rel=1e-6)
    assert info["vc"] and info["converged"] and info["final_scf"] and info["done"]
    assert frames[-1]["kind"] == "final_scf" and frames[0]["kind"] == "bfgs"


def test_relaxed_structure_is_the_elastic_reference():
    path = raw_file("relax", "74", "out")
    _, info = parse_pw_relax(path)
    ref, _ = tpw_reference(raw_file("elast", "74", "out"))
    dc, dp = deviation(info["relaxed"], ref)
    assert dc < 1e-4 and dp < 1e-4
    _, dp0 = deviation(info["initial"], ref)                 # the run starts from ideal sites
    assert dp0 > 0.05


def test_crashed_run_keeps_complete_frames():
    frames, info = parse_pw_relax(raw_file("relax", "179", "out"))
    assert not info["done"] and info["errors"]
    assert sum(f["F"] is not None and f["S"] is not None for f in frames) == 10
