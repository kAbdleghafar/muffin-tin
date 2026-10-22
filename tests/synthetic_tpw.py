"""Synthetic thermo_pw scf_elastic_constants outputs with known answers, for parser tests.

The file layout reproduces the markers listed in CLAUDE.md ("Output-file format notes") using
pw.x print formats: thermo_pw reference header, then per work a strain block, `Doing work`,
a pw.x header and one block per ionic step (energy, -TS, forces, stress, BFGS positions).
Stresses follow a known linear law  sigma = s0 + C e  (kbar, tension positive, engineering
Voigt strain), so unit/sign conversions and the C_ij refit can be checked exactly.

These files test the parser's logic, not that thermo_pw prints exactly this; the regression
tests on the real samples (test_samples.py) cover that.
"""
import numpy as np
from ase.units import Ry, Bohr, GPa

from mtpot.elastic.protocol import N_WORKS, thermo_pw_strain

RY_BOHR3_KBAR = Ry / Bohr**3 / (0.1 * GPa)   # 1 Ry/bohr^3 in kbar


def voigt(t):
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])


def unvoigt(v):
    return np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]])


def engineering(e):
    return voigt(e) * np.array([1, 1, 1, 2, 2, 2])


class SyntheticRun:
    """bcc 1x1x2 supercell, MoTi3, with a known stress-strain law and a 2-step relaxation."""

    alat = 6.0                                     # bohr
    axes = np.diag([1.0, 1.0, 2.0])                # rows a(i), alat units
    symbols = ["Mo", "Ti", "Ti", "Ti"]
    tau0 = np.array([[0, 0, 0], [.5, .5, .5], [0, 0, 1], [.5, .5, 1.5]])  # alat units
    zval = {"Mo": 14, "Ti": 12}
    step_shift = 0.001                             # frac shift of atom 2 per ionic step
    unrelaxed_offset = 3.0                         # kbar added to sigma_xx per pending step

    def __init__(self, C_kbar, s0_kbar, works=range(1, N_WORKS + 1), n_steps=2,
                 smearing="Marzari-Vanderbilt", degauss=0.007, minus_TS_Ry=-2.0e-6,
                 E0_Ry=-400.0):
        self.C = np.asarray(C_kbar, float)
        self.s0 = np.asarray(s0_kbar, float)
        self.works = list(works)
        self.n_steps = n_steps
        self.smearing, self.degauss = smearing, degauss
        self.minus_TS_Ry, self.E0_Ry = minus_TS_Ry, E0_Ry

    @property
    def nat(self):
        return len(self.symbols)

    @property
    def nelec(self):
        return sum(self.zval[s] for s in self.symbols)

    @property
    def frac0(self):
        return self.tau0 @ np.linalg.inv(self.axes)

    def ref_cell(self):
        """Unstrained cell, Angstrom."""
        return self.axes * self.alat * Bohr

    def cell(self, work):
        """Strained cell, Angstrom: rows a_i (I + eps)."""
        return self.ref_cell() @ (np.eye(3) + thermo_pw_strain(work))

    def frac(self, step):
        f = self.frac0.copy()
        f[1, 0] += self.step_shift * (step - 1)
        return f

    def positions(self, work, step):
        return self.frac(step) @ self.cell(work)

    def is_final(self, step):
        return step == self.n_steps

    def energy_Ry(self, work, step):
        return self.E0_Ry - 1.0e-4 * work - 2.0e-5 * step

    def forces_Ry_bohr(self, work, step):
        f = np.zeros((self.nat, 3))
        f[1, 0] = 1.0e-3 * (self.n_steps - step)
        f[2, 0] = -f[1, 0]
        return f

    def stress_kbar(self, work, step):
        """ASE sign (tension positive), 3x3, kbar."""
        s = self.s0 + self.C @ engineering(thermo_pw_strain(work))
        s[0] += self.unrelaxed_offset * (self.n_steps - step)
        return unvoigt(s)

    # ---------------------------------------------------------------- writing
    def _axes_lines(self, axes):
        return ["     crystal axes: (cart. coord. in units of alat)"] + [
            "               a({}) = ({:11.6f}{:11.6f}{:11.6f} )  ".format(i + 1, *axes[i])
            for i in range(3)] + [""]

    def _tau_lines(self, tau):
        return ["     site n.     atom                  positions (alat units)"] + [
            "     {:4d}           {:<3s} tau({:4d}) = ({:12.7f}{:12.7f}{:12.7f}  )".format(
                k + 1, s, k + 1, *tau[k]) for k, s in enumerate(self.symbols)] + [""]

    def _pw_header(self, work):
        eps = thermo_pw_strain(work)
        axes = self.axes @ (np.eye(3) + eps)
        vol = abs(np.linalg.det(axes)) * self.alat**3
        return [
            "     Program PWSCF v.7.4.1 starts on 22Oct2026 at 10: 0: 0 ",
            "",
            "     bravais-lattice index     =            0",
            f"     lattice parameter (alat)  = {self.alat:12.4f}  a.u.",
            f"     unit-cell volume          = {vol:12.4f} (a.u.)^3",
            f"     number of atoms/cell      = {self.nat:12d}",
            f"     number of atomic types    = {len(set(self.symbols)):12d}",
            f"     number of electrons       = {self.nelec:12.2f}",
            "     kinetic-energy cutoff     =      55.0000  Ry",
            "     charge density cutoff     =     550.0000  Ry",
            "     scf convergence threshold =      1.0E-08",
            "     Exchange-correlation= PBESOL",
            "                           (   1   4  10   8   0   0   0)",
            "",
            f"     celldm(1)={self.alat:11.6f}  celldm(2)=   0.000000  celldm(3)=   0.000000",
            "     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000",
            "",
            *self._axes_lines(axes),
            *self._tau_lines(self.tau0 @ (np.eye(3) + eps)),
            f"     number of k points=    10  {self.smearing} smearing, width (Ry)={self.degauss:8.4f}",
            "",
        ]

    def _force_lines(self, f):
        types = sorted(set(self.symbols))
        return ["     atom {:4d} type {:2d}   force = {:14.8f}{:14.8f}{:14.8f}".format(
            k + 1, 1 + types.index(s), *f[k]) for k, s in enumerate(self.symbols)]

    def _step(self, work, step):
        f = self.forces_Ry_bohr(work, step)
        s_ase = self.stress_kbar(work, step)
        s_qe = -s_ase                                   # QE prints compression positive
        P = -np.trace(s_ase) / 3
        out = [
            f"!    total energy              = {self.energy_Ry(work, step):17.8f} Ry",
            "     estimated scf accuracy    <       0.00000001 Ry",
            f"     smearing contrib. (-TS)   = {self.minus_TS_Ry:17.8f} Ry",
            "",
            "     Forces acting on atoms (cartesian axes, Ry/au):",
            "",
            *self._force_lines(f),
            # verbosity='high' decomposition: same line format, must not be read as forces
            "     The non-local contrib.  to forces", *self._force_lines(f + 0.5),
            "     The ionic contribution  to forces", *self._force_lines(f - 0.25),
            f"     Total force = {np.linalg.norm(f):12.6f}     Total SCF correction =     0.000000",
            "",
            "     Computing stress (Cartesian axis) and pressure",
            "",
            f"          total   stress  (Ry/bohr**3)                   (kbar)     P={P:12.2f}",
            *["{:13.8f}{:13.8f}{:13.8f}    {:12.2f}{:12.2f}{:12.2f}".format(
                *(s_qe[r] / RY_BOHR3_KBAR), *s_qe[r]) for r in range(3)],
            "",
        ]
        if self.is_final(step):
            out += ["     bfgs converged in   2 scf cycles and   1 bfgs steps",
                    "     End of BFGS Geometry Optimization", "",
                    f"     Final energy             = {self.energy_Ry(work, step):17.8f} Ry",
                    "", "Begin final coordinates", ""]
            frac = self.frac(step)
        else:
            out += ["     BFGS Geometry Optimization", ""]
            frac = self.frac(step + 1)
        out += ["ATOMIC_POSITIONS (crystal)"] + [
            "{:<3s}  {:20.10f}{:20.10f}{:20.10f}".format(s, *frac[k])
            for k, s in enumerate(self.symbols)] + [""]
        if self.is_final(step):
            out += ["End final coordinates", ""]
        return out

    def lines(self):
        out = ["     Program THERMO_PW v.7.4.1 starts on 22Oct2026 at 10: 0: 0 ", "",
               "     Computing the elastic constants (synthetic test file)", "",
               f"     ibrav=    0    alat= {self.alat:14.8f} a.u.", "",
               *self._axes_lines(self.axes), *self._tau_lines(self.tau0)]
        for work in range(1, N_WORKS + 1):
            eps = thermo_pw_strain(work)
            out += ["     Applying the following strain"]
            out += ["     ({:13.5f},{:13.5f},{:13.5f}  )".format(*eps[r]) for r in range(3)]
            out += ["", f"     Doing work {work:6d} / {N_WORKS:6d}", ""]
            if work not in self.works:
                out += ["     (read from restart file)", ""]
                continue
            out += self._pw_header(work)
            for step in range(1, self.n_steps + 1):
                out += self._step(work, step)
        out += ["     Elastic constants C_ij (kbar) ",
                " i j=" + "".join(f"{j:16d}" for j in range(1, 7))]
        out += ["{:6d}".format(i + 1) + "".join(f"{c:16.5f}" for c in self.C[i])
                for i in range(6)]
        return out + ["", "     JOB DONE."]

    def write(self, path):
        path.write_text("\n".join(self.lines()) + "\n")
        return path


def default_C_kbar(seed=0):
    """Near-cubic (C11 197, C12 113, C44 30 GPa) with a non-symmetric perturbation,
    so a row/column (stress/strain) mix-up in the refit cannot go unnoticed."""
    C = np.zeros((6, 6))
    C[:3, :3] = 1131.0
    np.fill_diagonal(C[:3, :3], 1970.0)
    C[3:, 3:] = np.eye(3) * 301.0
    return C + np.random.default_rng(seed).normal(0.0, 20.0, (6, 6))


DEFAULT_S0_KBAR = np.array([2.5, 2.7, 2.9, 0.1, -0.2, 0.05])
