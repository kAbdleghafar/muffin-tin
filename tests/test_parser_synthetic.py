"""Parser logic, units and sign conventions on synthetic thermo_pw outputs with known answers.

Runs without the raw DFT data. See synthetic_tpw.py for what the synthetic files contain.
"""
import csv
import sys

import numpy as np
import pytest
from ase.io import read
from ase.units import Bohr, GPa, Ry

import tpw_to_dataset as tpw
from mtpot.elastic.protocol import N_WORKS, thermo_pw_strain
from synthetic_tpw import (DEFAULT_S0_KBAR, RY_BOHR3_KBAR, SyntheticRun, default_C_kbar,
                           engineering, voigt)

KBAR = 0.1 * GPa
STRESS_PRINT_TOL_KBAR = 2e-3       # stress printed to 1e-8 Ry/bohr^3 = 1.5e-3 kbar


def make_run(**kw):
    return SyntheticRun(default_C_kbar(), DEFAULT_S0_KBAR, **kw)


def parse(tmp_path, run, name="elast_07.out"):
    return tpw.parse_thermo_pw(str(run.write(tmp_path / name)))


@pytest.fixture(params=[("Marzari-Vanderbilt", 0.007, -2.0e-6), ("gaussian", 0.009, -7.2e-4)],
                ids=["mv", "gauss"])
def run(request):
    smearing, degauss, mTS = request.param
    return make_run(smearing=smearing, degauss=degauss, minus_TS_Ry=mTS)


# --------------------------------------------------------------------------- conventions
def test_unit_constants():
    assert tpw.KBAR == pytest.approx(0.1 * GPa)
    assert RY_BOHR3_KBAR == pytest.approx(147105.13, rel=1e-5)   # QE's RY_KBAR
    assert Ry / Bohr == pytest.approx(25.71104, rel=1e-6)          # Ry/bohr -> eV/A


def test_voigt_strain_is_engineering():
    e = np.array([[1, 6, 5], [6, 2, 4], [5, 4, 3]]) * 1e-3
    assert np.allclose(tpw.voigt_strain(e), [1e-3, 2e-3, 3e-3, 8e-3, 10e-3, 12e-3])
    assert np.allclose(tpw.voigt_stress(e), [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3])


def test_cubic_projection():
    C = np.zeros((6, 6))
    C[:3, :3] = 113.0
    np.fill_diagonal(C[:3, :3], 197.0)
    C[3:, 3:] = 30.0 * np.eye(3)
    c11, c12, c44, dev, _ = tpw.cubic_projection(C)
    assert (c11, c12, c44, dev) == pytest.approx((197.0, 113.0, 30.0, 0.0))
    # C12 comes from the symmetrised tensor; anisotropy shows up in the non-cubic fraction
    C2 = C.copy()
    C2[0, 1] += 4.0
    C2[0, 0] += 6.0
    c11, c12, c44, dev, Cs = tpw.cubic_projection(C2)
    assert (c11, c12, c44) == pytest.approx((199.0, 113.0 + 4.0 / 6, 30.0))
    assert np.allclose(Cs, Cs.T) and dev > 0


def test_protocol_matches_claude_md_table():
    values = [-0.0075, -0.0025, 0.0025, 0.0075]
    voigt_index = [0, 1, 2, 5, 4, 3]          # works 1-4 e1, ..., 13-16 e6, 17-20 e5, 21-24 e4
    for w in range(1, N_WORKS + 1):
        e = tpw.voigt_strain(thermo_pw_strain(w))
        k = voigt_index[(w - 1) // 4]
        expected = np.zeros(6)
        expected[k] = values[(w - 1) % 4] * (2 if k >= 3 else 1)
        assert np.allclose(e, expected), w
    with pytest.raises(ValueError):
        thermo_pw_strain(25)


# --------------------------------------------------------------------------- parsing
def test_settings(tmp_path, run):
    _, info = parse(tmp_path, run)
    st = info["settings"]
    assert st["smearing"] == run.smearing
    assert float(st["degauss_Ry"]) == pytest.approx(run.degauss)
    assert float(st["nelec"]) == run.nelec
    assert (float(st["ecutwfc_Ry"]), float(st["ecutrho_Ry"])) == (55.0, 550.0)
    assert st["xc"] == "PBESOL"


def test_frames_works_and_final_flags(tmp_path, run):
    frames, info = parse(tmp_path, run)
    assert len(frames) == N_WORKS * run.n_steps
    assert info["works_total"] == N_WORKS
    assert info["works_computed"] == set(range(1, N_WORKS + 1))
    for w in range(1, N_WORKS + 1):
        fw = [f for f in frames if f["meta"]["work"] == w]
        assert [f["meta"]["ionic_step"] for f in fw] == list(range(1, run.n_steps + 1))
        assert [f["meta"]["final"] for f in fw] == [run.is_final(s) for s in range(1, run.n_steps + 1)]


def test_energy_forces_stress_units_and_signs(tmp_path, run):
    frames, _ = parse(tmp_path, run)
    for f in frames:
        w, s = f["meta"]["work"], f["meta"]["ionic_step"]
        assert f["E"] == pytest.approx(run.energy_Ry(w, s) * Ry, abs=1e-7)
        assert f["mTS"] == pytest.approx(run.minus_TS_Ry * Ry, abs=1e-9)
        assert np.allclose(f["F"], run.forces_Ry_bohr(w, s) * Ry / Bohr, atol=1e-7)
        # ASE sign: tension positive = -(QE printed stress)
        assert np.allclose(f["S"] / KBAR, run.stress_kbar(w, s), atol=STRESS_PRINT_TOL_KBAR)


def test_reference_and_strained_geometry(tmp_path, run):
    frames, info = parse(tmp_path, run)
    ref = info["reference"]
    assert list(ref.symbols) == run.symbols
    assert np.allclose(ref.cell[:], run.ref_cell(), atol=1e-6)
    assert np.allclose(ref.get_scaled_positions(wrap=False), run.frac0, atol=1e-7)
    for f in frames:
        w, s = f["meta"]["work"], f["meta"]["ionic_step"]
        assert np.allclose(f["meta"]["strain"], thermo_pw_strain(w), atol=1e-7)
        assert np.allclose(f["cell"], run.cell(w), atol=1e-6)
        # step 1 from the pw header tau, later steps from BFGS ATOMIC_POSITIONS (crystal)
        assert np.allclose(f["pos"], run.positions(w, s), atol=1e-6), (w, s)


def test_refit_recovers_C_and_residual_stress(tmp_path, run):
    frames, info = parse(tmp_path, run)
    Cfit, s0 = tpw.refit_cij(frames)
    # least-squares error from print rounding: ~0.05 kbar; a row/column mix-up or using an
    # unrelaxed step (3 kbar offset) would be far larger
    assert np.abs(Cfit - run.C).max() < 0.3
    assert np.allclose(s0, DEFAULT_S0_KBAR, atol=0.01)
    assert np.allclose(info["C_tpw_kbar"], run.C, atol=1e-5)


def test_restart_works_are_detected(tmp_path):
    run = make_run(works=[23, 24], smearing="gaussian", degauss=0.009)
    frames, info = parse(tmp_path, run, "elast_150.out")
    assert info["works_computed"] == {23, 24}
    assert info["works_total"] == N_WORKS
    assert len(frames) == 2 * run.n_steps
    assert sorted({f["meta"]["work"] for f in frames}) == [23, 24]
    assert np.allclose(frames[0]["meta"]["strain"], thermo_pw_strain(23))


# --------------------------------------------------------------------------- CLI round trip
def run_cli(tmp_path, monkeypatch, *paths):
    out, summary = tmp_path / "dataset.extxyz", tmp_path / "summary.csv"
    monkeypatch.setattr(sys, "argv", ["tpw_to_dataset.py", *map(str, paths),
                                      "-o", str(out), "-s", str(summary)])
    tpw.main()
    with open(summary, newline="") as fh:
        rows = {r["alloy"]: r for r in csv.DictReader(fh)}
    return read(out, ":"), rows


def test_cli_extxyz_roundtrip(tmp_path, monkeypatch, run):
    path = run.write(tmp_path / "elast_07.out")
    atoms, rows = run_cli(tmp_path, monkeypatch, path)
    assert len(atoms) == N_WORKS * run.n_steps
    for a in atoms:
        w, s = a.info["work"], a.info["ionic_step"]
        assert a.info["final"] == run.is_final(s)
        assert a.info["smearing"] == run.smearing
        assert a.info["minus_TS"] == pytest.approx(run.minus_TS_Ry * Ry, abs=1e-9)
        assert np.allclose(a.info["strain_voigt"], engineering(thermo_pw_strain(w)))
        assert a.get_potential_energy() == pytest.approx(run.energy_Ry(w, s) * Ry, abs=1e-6)
        assert np.allclose(a.get_forces(), run.forces_Ry_bohr(w, s) * Ry / Bohr, atol=1e-6)
        assert np.allclose(a.get_stress() / KBAR, voigt(run.stress_kbar(w, s)),
                           atol=STRESS_PRINT_TOL_KBAR)          # Voigt xx yy zz yz xz xy
        assert np.allclose(a.cell[:], run.cell(w), atol=1e-6)

    row = rows["elast_07"]
    c11, c12, c44, dev, _ = tpw.cubic_projection(run.C)
    assert (float(row["C11_GPa"]), float(row["C12_GPa"]), float(row["C44_GPa"])) == \
        pytest.approx((round(c11 / 10, 1), round(c12 / 10, 1), round(c44 / 10, 1)))
    assert float(row["max_refit_diff_GPa"]) <= 0.03
    assert float(row["residual_P_GPa"]) == pytest.approx(-DEFAULT_S0_KBAR[:3].mean() / 10, abs=1e-3)
    assert int(row["works_computed"]) == N_WORKS and int(row["frames"]) == len(atoms)


def test_cli_reports_restart_works(tmp_path, monkeypatch, capsys):
    full = make_run().write(tmp_path / "elast_01.out")
    part = make_run(works=[23, 24], smearing="gaussian", degauss=0.009).write(tmp_path / "elast_150.out")
    atoms, rows = run_cli(tmp_path, monkeypatch, full, part)
    out = capsys.readouterr().out
    assert f"WARNING: works {list(range(1, 23))} not in this file" in out
    assert rows["elast_150"]["works_computed"] == "2"
    assert rows["elast_150"]["max_refit_diff_GPa"] == ""       # no refit on incomplete data
    assert rows["elast_01"]["max_refit_diff_GPa"] != ""
    assert len(atoms) == (N_WORKS + 2) * 2


# --------------------------------------------------------------------------- known gaps
@pytest.mark.xfail(strict=True, reason="CLAUDE.md convention: alloy id = numeric suffix; "
                                       "parser still stores 'elast_07'")
def test_alloy_id_is_numeric(tmp_path):
    frames, info = parse(tmp_path, make_run())
    assert info["alloy"] == "07" and frames[0]["meta"]["alloy"] == "07"


@pytest.mark.xfail(strict=True, reason="final = last frame of a work, even if BFGS never "
                                       "converged (interrupted run): refit would use it")
def test_unconverged_work_is_not_final(tmp_path):
    lines = make_run().lines()
    last_energy = max(i for i, L in enumerate(lines) if L.startswith("!"))
    path = tmp_path / "elast_150.out"
    path.write_text("\n".join(lines[:last_energy]) + "\n")   # killed after step 1 of work 24
    frames, info = tpw.parse_thermo_pw(str(path))
    last = [f for f in frames if f["meta"]["work"] == 24]
    assert len(last) == 1 and info["C_tpw_kbar"] is None
    assert not last[0]["meta"]["final"]
