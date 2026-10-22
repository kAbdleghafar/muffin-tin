"""Regression fixtures from the three examined samples (CLAUDE.md, "Regression fixtures").

Needs data/raw/{elast,rho,elf}_<id>.{out,cube} (or $MTPOT_RAW); each test skips if its file
is absent. Tolerances follow the precision the numbers are quoted with in CLAUDE.md.
"""
from functools import lru_cache

import numpy as np
import pytest
from ase.io.cube import read_cube_data
from ase.units import Bohr

import tpw_to_dataset as tpw
from conftest import raw_file
from mtpot.elastic.protocol import N_WORKS, thermo_pw_strain

pytestmark = pytest.mark.data

ZVAL = {"Ti": 12, "Zr": 12, "Nb": 13, "Mo": 14, "Hf": 12, "Ta": 27}
SAMPLES = {
    "01": dict(formula="Mo8Ti24", natoms=32, frames=278, works=set(range(1, 25)),
               smearing="marzari-vanderbilt", degauss=0.007, cubic=(197.0, 113.1, 30.1),
               minus_TS_meV=-0.1, nelec=400),
    "140": dict(formula="Hf4Mo9Nb9Ti9Zr9", natoms=40, frames=253, works=set(range(1, 25)),
                smearing="gaussian", degauss=0.009, cubic=(197.4, 116.0, 30.9),
                minus_TS_meV=-9.8, nelec=507),
    "150": dict(formula="Hf6Mo6Nb6Ta6Ti6Zr6", natoms=36, frames=30, works={23, 24},
                smearing="gaussian", degauss=0.009, cubic=(210.5, 118.8, 39.8),
                minus_TS_meV=-9.8, nelec=540),
}
COMPLETE = [k for k, v in SAMPLES.items() if len(v["works"]) == N_WORKS]
ids = pytest.mark.parametrize("aid", list(SAMPLES))
complete_ids = pytest.mark.parametrize("aid", COMPLETE)


@lru_cache(maxsize=None)
def _parse(path):
    return tpw.parse_thermo_pw(path)


def parsed(aid):
    return _parse(str(raw_file("elast", aid, "out")))


def usable(frames):
    return [f for f in frames if f["F"] is not None and f["S"] is not None]


@lru_cache(maxsize=None)
def _cube(path):
    data, atoms = read_cube_data(path)
    return atoms, data


def cube(kind, aid):
    return _cube(str(raw_file(kind, aid, "cube")))


# --------------------------------------------------------------------------- thermo_pw output
@ids
def test_structure_and_settings(aid):
    exp = SAMPLES[aid]
    _, info = parsed(aid)
    ref, st = info["reference"], info["settings"]
    assert ref.get_chemical_formula() == exp["formula"]
    assert len(ref) == exp["natoms"]
    assert sum(ZVAL[s] for s in ref.get_chemical_symbols()) == exp["nelec"]
    assert float(st["nelec"]) == exp["nelec"]
    assert (float(st["ecutwfc_Ry"]), float(st["ecutrho_Ry"])) == (55.0, 550.0)
    assert st["xc"].upper() == "PBESOL"
    assert st["smearing"].lower() == exp["smearing"]
    assert float(st["degauss_Ry"]) == pytest.approx(exp["degauss"])


@ids
def test_frames_and_works_present(aid):
    exp = SAMPLES[aid]
    frames, info = parsed(aid)
    assert len(usable(frames)) == exp["frames"]
    assert info["works_total"] == N_WORKS
    assert info["works_computed"] == exp["works"]       # 150: works 1-22 read from restart
    for w in exp["works"]:
        flags = [f["meta"]["final"] for f in frames if f["meta"]["work"] == w]
        assert flags.count(True) == 1 and flags[-1], w


@ids
def test_thermo_pw_cubic_projection(aid):
    _, info = parsed(aid)
    c11, c12, c44, dev, _ = tpw.cubic_projection(info["C_tpw_kbar"] / 10)    # GPa
    assert (c11, c12, c44) == pytest.approx(SAMPLES[aid]["cubic"], abs=0.051)
    assert 0.025 <= dev <= 0.055                       # non-cubic remainder 3-5% (SQS disorder)


@complete_ids
def test_refit_matches_thermo_pw(aid):
    frames, info = parsed(aid)
    Cfit, _ = tpw.refit_cij(frames)
    assert not np.isnan(Cfit).any()
    # quoted as "within 0.01 GPa" (2-decimal print); stress print rounding alone gives ~0.01
    assert np.abs(Cfit - info["C_tpw_kbar"]).max() / 10 <= 0.015


def test_residual_stress_01():
    _, s0 = tpw.refit_cij(parsed("01")[0])
    assert np.all((s0[:3] / 10 >= 0.22) & (s0[:3] / 10 <= 0.31))   # ~ +0.24-0.29 GPa


def test_residual_stress_140():
    _, s0 = tpw.refit_cij(parsed("140")[0])
    assert np.abs(s0 / 10).max() <= 0.12                           # within +-0.11 GPa


@ids
def test_strain_protocol(aid):
    """Every frame carries its work's strain; the first ionic step of each work is the
    reference cell strained as rows a_i (I + eps), fractional coordinates unchanged."""
    frames, info = parsed(aid)
    ref = info["reference"]
    for f in frames:
        w = f["meta"]["work"]
        eps = thermo_pw_strain(w)
        assert np.allclose(f["meta"]["strain"], eps, atol=1e-7), w
        if f["meta"]["ionic_step"] == 1:
            cell = ref.cell[:] @ (np.eye(3) + eps)
            assert np.allclose(f["cell"], cell, atol=1e-4), w
            dfrac = f["pos"] @ np.linalg.inv(cell) - ref.get_scaled_positions(wrap=False)
            assert np.abs(dfrac - np.round(dfrac)).max() < 1e-5, w


@ids
def test_smearing_entropy_per_atom(aid):
    exp = SAMPLES[aid]
    frames = usable(parsed(aid)[0])
    mts = np.mean([f["mTS"] / len(f["symbols"]) for f in frames]) * 1e3        # meV/atom
    if exp["smearing"] == "gaussian":
        assert mts == pytest.approx(exp["minus_TS_meV"], abs=0.5)
    else:
        assert -0.3 < mts <= 0.0


@ids
def test_narrow_configuration_coverage(aid):
    """Known data issue 3; also a units sanity check (a Ry/eV slip would break both bounds)."""
    frames = usable(parsed(aid)[0])
    e = np.array([f["E"] / len(f["symbols"]) for f in frames]) * 1e3          # meV/atom
    if aid in COMPLETE:
        assert 0.5 < e.max() - e.min() < 2.0           # quoted span 1-1.6 meV/atom
    assert max(np.linalg.norm(f["F"], axis=1).max() for f in frames) < 0.16


# --------------------------------------------------------------------------- cubes
def _same_structure(a, ref, tol=2e-4):
    assert list(a.numbers) == list(ref.numbers)
    assert np.allclose(a.cell[:], ref.cell[:], atol=tol)
    d = (a.positions - ref.positions) @ np.linalg.inv(ref.cell[:])
    assert np.abs((d - np.round(d)) @ ref.cell[:]).max() < tol


@ids
def test_rho_integrates_to_valence_count(aid):
    atoms, rho = cube("rho", aid)
    nelec = rho.mean() * atoms.get_volume() / Bohr**3    # e/bohr^3 on a periodic grid
    assert nelec == pytest.approx(SAMPLES[aid]["nelec"], abs=0.05)


@ids
def test_cubes_match_reference_structure(aid):
    ref = parsed(aid)[1]["reference"]
    atoms_rho, rho = cube("rho", aid)
    atoms_elf, elf = cube("elf", aid)
    _same_structure(atoms_rho, ref)
    _same_structure(atoms_elf, ref)
    assert elf.shape == rho.shape
    assert elf.min() >= -1e-6 and elf.max() <= 1 + 1e-6
