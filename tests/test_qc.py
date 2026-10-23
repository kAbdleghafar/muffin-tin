"""Phase 0 QC script on synthetic alloys (no raw data needed)."""
import csv

import numpy as np
import pytest
from ase.io import write
from ase.units import Bohr

import qc
import tpw_to_dataset as tpw
from mtpot.io.cube import integrate, read_cube, structure_deviation
from synthetic_tpw import DEFAULT_S0_KBAR, SyntheticRun, default_C_kbar

GRID = (12, 12, 24)


def add_alloy(raw, aid, rho_scale=1.0, cubes=True, **kw):
    run = SyntheticRun(default_C_kbar(), DEFAULT_S0_KBAR, **kw)
    path = run.write(raw / f"elast_{aid}.out")
    if cubes:
        ref = tpw.parse_thermo_pw(str(path))[1]["reference"]
        rho = np.full(GRID, rho_scale * run.nelec / (ref.get_volume() / Bohr**3))
        write(raw / f"rho_{aid}.cube", ref, format="cube", data=rho)
        write(raw / f"elf_{aid}.cube", ref, format="cube", data=np.full(GRID, 0.5))
    return run


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    add_alloy(d, "07")
    add_alloy(d, "150", cubes=False, works=[23, 24], smearing="gaussian", degauss=0.009)
    add_alloy(d, "08", rho_scale=1.01)
    return d


def test_cube_helpers(raw):
    atoms, rho = read_cube(raw / "rho_07.cube")
    ref = tpw.parse_thermo_pw(str(raw / "elast_07.out"))[1]["reference"]
    assert rho.shape == GRID
    assert integrate(atoms, rho) == pytest.approx(50.0, abs=1e-4)
    assert structure_deviation(atoms, ref) < 1e-5
    shifted = ref.copy()
    shifted.positions[1] += [0.01, 0, 0]
    assert structure_deviation(shifted, ref) == pytest.approx(0.01)
    lattice_shift = ref.copy()
    lattice_shift.positions[1] += ref.cell[2]                 # same structure modulo lattice
    assert structure_deviation(lattice_shift, ref) < 1e-9
    assert structure_deviation(ref[[1, 0, 2, 3]], ref) == np.inf


def test_ranges():
    assert qc.ranges([]) == ""
    assert qc.ranges([7, 1, 2, 3, 9, 10]) == "1-3,7,9-10"


def test_qc_sweep(raw, tmp_path, capsys):
    out = tmp_path / "processed" / "qc.csv"
    qc.main([str(raw), "-o", str(out)])
    with open(out, newline="") as fh:
        rows = {r["alloy"]: r for r in csv.DictReader(fh)}
    assert list(rows) == ["07", "08", "150"]                   # numeric order

    r = rows["07"]
    assert r["flags"] == ""
    assert (r["smearing"], float(r["degauss_Ry"])) == ("Marzari-Vanderbilt", 0.007)
    c11, c12, c44, dev, _ = tpw.cubic_projection(default_C_kbar() / 10)
    assert float(r["C11_GPa"]) == pytest.approx(c11, abs=1e-3)
    assert float(r["C44_GPa"]) == pytest.approx(c44, abs=1e-3)
    assert float(r["s0_zz_GPa"]) == pytest.approx(DEFAULT_S0_KBAR[2] / 10, abs=1e-3)
    assert float(r["rho_electrons"]) == pytest.approx(50.0, abs=1e-3)
    assert int(r["frames"]) == 48 and r["works_missing"] == ""

    assert rows["08"]["flags"] == "electron count (50.500 vs 50.0)"
    r = rows["150"]
    assert r["flags"] == "restart works (1-22); rho cube missing; elf cube missing"
    assert r["refit_max_diff_GPa"] == "" and r["s0_xx_GPa"] == ""   # no partial refits
    assert "smearing: Marzari-Vanderbilt 0.007: 2, gaussian 0.009: 1" in capsys.readouterr().out


def test_qc_resumes_from_cache(raw, tmp_path):
    out = tmp_path / "qc.csv"
    qc.main([str(raw), "-o", str(out), "--ids", "07"])
    (raw / "elast_07.out").unlink()                              # cached row is reused
    rows = qc.main([str(raw), "-o", str(out), "--ids", "07"])
    assert rows[0]["flags"] == "" and rows[0]["formula"] == "MoTi3"
    rows = qc.main([str(raw), "-o", str(out), "--ids", "07", "--force"])
    assert rows[0]["flags"].startswith("parse error (FileNotFoundError")


def test_qc_parallel_matches_serial(raw, tmp_path):
    serial = qc.main([str(raw), "-o", str(tmp_path / "a" / "qc.csv")])
    parallel = qc.main([str(raw), "-o", str(tmp_path / "b" / "qc.csv"), "-j", "2"])
    assert serial == parallel


def test_flags_unconverged_and_settings():
    row = {"xc": "PBE", "ecutwfc_Ry": 55.0, "ecutrho_Ry": 550.0, "nelec": 50.0, "zval_sum": 50.0,
           "works_total": 24, "works_unfinished": "24", "works_maxsteps": "5,21",
           "C11_GPa": 197.0, "noncubic_frac": 0.03}
    assert qc.flags(row) == ["settings (PBE, 55.0/550.0 Ry)", "unfinished works (24)",
                             "BFGS step limit (5,21)"]
