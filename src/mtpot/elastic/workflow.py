"""Phase 3: relaxed-ion elastic constants with a potential, reproducing thermo_pw's protocol:
24 strains (protocol.thermo_pw_strain) applied to the reference cell with fractional
coordinates fixed, ions relaxed at fixed cell, stress-strain slopes, cubic projection."""
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.units import GPa

from mtpot.potential.eam import collate, frame_graph

from .protocol import COMPONENTS, N_WORKS, STRAIN_VALUES, thermo_pw_strain

VOIGT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (0, 2): 4, (0, 1): 5}


class TorchCalculator(Calculator):
    """ASE calculator for a torch potential with the EAM interface (model(batch) -> energy,
    forces, stress). Stress is ASE-sign Voigt (eV/A^3)."""
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model, **kw):
        super().__init__(**kw)
        self.model = model

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        out = self.model(collate([frame_graph(self.atoms, self.model.rc)]))
        e = out["energy"][0].item()
        self.results = dict(energy=e, free_energy=e, forces=out["forces"].detach().numpy(),
                            stress=out["stress"][0].detach().numpy())


def voigt_stress_gpa(atoms):
    return atoms.get_stress(voigt=True) / GPa


def relax_ions(atoms, fmax=1e-3, steps=500):
    """Fixed-cell ionic relaxation (in place); returns (converged, n_steps)."""
    opt = BFGS(atoms, logfile=None)
    ok = opt.run(fmax=fmax, steps=steps)
    return bool(ok), opt.nsteps


def self_relaxation(atoms, calc, fmax=0.01, steps=1500):
    """Relax a DFT minimum (e.g. the elastic reference) at fixed cell with the model: how far
    does the model take it? A model whose minimum is near the DFT one moves atoms by << 0.1 A
    and gains ~nothing; a spurious deeper minimum shows up as a large gain (meV/atom) and
    displacement (A). Works on a copy."""
    from ase.neighborlist import neighbor_list
    a = atoms.copy()
    a.calc = calc
    e0 = a.get_potential_energy()
    p0 = a.positions.copy()
    ok, n = relax_ions(a, fmax=fmax, steps=steps)
    d = np.linalg.norm(a.positions - p0, axis=1)
    dmin = lambda x: float(neighbor_list("d", x, 4.0).min())
    return dict(converged=ok, steps=n, gain_meV_atom=1e3 * (a.get_potential_energy() - e0) / len(a),
                disp_max_A=float(d.max()), disp_mean_A=float(d.mean()),
                dmin_ref_A=dmin(atoms), dmin_relaxed_A=dmin(a))


def relax_cell(atoms, fmax=1e-3, steps=1000):
    """Cell + ionic relaxation to zero stress (in place)."""
    opt = BFGS(FrechetCellFilter(atoms), logfile=None)
    ok = opt.run(fmax=fmax, steps=steps)
    return bool(ok), opt.nsteps


def elastic_constants(reference, calc, relax=True, fmax=1e-3, steps=500):
    """6x6 C (GPa, C[i, j] = d sigma_i / d e_j, Voigt order) and the zero-strain intercept
    from the 24 thermo_pw strains of `reference` (cell and positions used as given)."""
    blocks = {}
    info = []
    for w in range(1, N_WORKS + 1):
        eps = thermo_pw_strain(w)
        a = reference.copy()
        a.calc = calc
        a.set_cell(reference.cell[:] @ (np.eye(3) + eps), scale_atoms=True)
        conv, n = relax_ions(a, fmax, steps) if relax else (True, 0)
        i, j = COMPONENTS[(w - 1) // len(STRAIN_VALUES)]
        e_eng = eps[i, j] * (1 if i == j else 2)
        blocks.setdefault(VOIGT[(i, j)], []).append((e_eng, voigt_stress_gpa(a)))
        info.append(dict(work=w, converged=conv, steps=n))
    C, s0 = np.zeros((6, 6)), []
    for v, pts in blocks.items():
        x = np.array([p[0] for p in pts])
        Y = np.array([p[1] for p in pts])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        C[:, v] = coef[0]
        s0.append(coef[1])
    return C, np.mean(s0, axis=0), info


def cubic(C):
    Cs = 0.5 * (C + C.T)
    c11 = np.mean(np.diag(Cs)[:3])
    c12 = np.mean([Cs[0, 1], Cs[0, 2], Cs[1, 2]])
    c44 = np.mean(np.diag(Cs)[3:])
    return np.array([c11, c12, c44])


def predict(atoms, model, mode="dft_cell", fmax=1e-3, threads=1):
    """Relaxed-ion cubic C (GPa) of one structure.
    mode 'dft_cell': strain the given (DFT reference) cell, as the labels were computed;
    'relax_ions': relax ions at the given cell first; 'own': relax cell and ions first
    (deployment: a new SQS from ideal sites)."""
    torch.set_num_threads(threads)
    calc = TorchCalculator(model)
    ref = atoms.copy()
    ref.calc = calc
    pre = {}
    if mode == "relax_ions":
        pre["converged"], pre["steps"] = relax_ions(ref, fmax)
    elif mode == "own":
        pre["converged"], pre["steps"] = relax_cell(ref, fmax)
        pre["volume_change_pct"] = 100 * (ref.get_volume() / atoms.get_volume() - 1)
    pre["max_disp_A"] = float(np.abs(ref.positions - atoms.positions).max()) if mode != "own" else None
    C, s0, info = elastic_constants(ref, calc, fmax=fmax)
    return dict(C=C, cubic=cubic(C), s0=s0, works=info, pre=pre)
