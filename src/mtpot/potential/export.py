"""Export an EAM model to the LAMMPS eam/alloy (setfl) format."""
import numpy as np
import torch
from ase.data import atomic_masses, atomic_numbers

from .eam import PAIR_INDEX, SPECIES


def write_setfl(model, path, species=SPECIES, nr=5000, nrho=5000, rho_max=None,
                comment="mtpot EAM"):
    """Tabulate F_Z(rho) (+ e0_Z, so absolute energies match), f_Z(r) and r*phi_ZZ'(r) on
    uniform grids. LAMMPS/ASE interpolate between points, so use dense grids."""
    rc = model.rc
    rho_max = rho_max or model.F.x1
    dr, drho = rc / (nr - 1), rho_max / (nrho - 1)
    r = torch.arange(nr, dtype=torch.float64) * dr
    rho = torch.arange(nrho, dtype=torch.float64) * drho
    idx = [SPECIES.index(s) for s in species]
    fmt = lambda v: "\n".join(" ".join(f"{x:.16e}" for x in v[k:k + 5]) for k in range(0, len(v), 5))
    with torch.no_grad(), open(path, "w") as fh:
        fh.write(f"{comment}\n\n\n")
        fh.write(f"{len(species)} {' '.join(species)}\n")
        fh.write(f"{nrho} {drho:.16e} {nr} {dr:.16e} {rc:.16e}\n")
        for s, k in zip(species, idx):
            z = atomic_numbers[s]
            fh.write(f"{z} {atomic_masses[z]:.4f} 0.0 bcc\n")
            kk = torch.full((nrho,), k, dtype=torch.long)
            fh.write(fmt((model.F(rho, kk) + model.e0[k]).numpy()) + "\n")
            fh.write(fmt(model.f(r, torch.full((nr,), k, dtype=torch.long)).numpy()) + "\n")
        for a, ka in enumerate(idx):
            for b in range(a + 1):
                p = PAIR_INDEX[ka, idx[b]].item()
                fh.write(fmt((r * model.phi(r, torch.full((nr,), p, dtype=torch.long))).numpy()) + "\n")
    return path
