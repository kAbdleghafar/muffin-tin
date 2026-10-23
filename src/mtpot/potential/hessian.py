"""Second-derivative training terms at a DFT reference structure (a minimum in DFT):

- stability: the model's force-constant matrix H_uu must be positive definite there
  (translations aside); otherwise its own relaxation leaves bcc (EAM's classic failure);
- relaxed-ion elastic constants by implicit differentiation instead of relaxation:
      C_rel = (1/V) [E_ee - E_eu (E_uu)^-1 E_ue]
  with u the atomic positions and e the Voigt engineering strain (r -> r (I + eps)).
  This is the same quantity elastic.workflow computes by relaxing ions, linearised at the
  reference (exact when the model's forces vanish there)."""
import torch

from .eam import collate

EV_A3_GPA = 160.21766208


def strain_tensor(e):
    """3x3 symmetric strain from Voigt engineering strain (e4 = 2 eps_yz, ...)."""
    return torch.stack([torch.stack([e[0], e[5] / 2, e[4] / 2]),
                        torch.stack([e[5] / 2, e[1], e[3] / 2]),
                        torch.stack([e[4] / 2, e[3] / 2, e[2]])])


def full_hessian(model, graph, create_graph=False):
    """Hessian of the energy w.r.t. (positions, Voigt strain): shape (3N + 6, 3N + 6)."""
    batch = collate([graph])
    n3 = batch["positions"].numel()

    def energy(z):
        pos = z[:n3].reshape(-1, 3)
        return model.energies(dict(batch, positions=pos), strain_tensor(z[n3:])[None]).sum()

    z0 = torch.cat([batch["positions"].reshape(-1), batch["positions"].new_zeros(6)])
    return torch.autograd.functional.hessian(energy, z0, create_graph=create_graph), \
        float(batch["volume"][0])


def translation_projector(n_atoms, dtype=torch.float64):
    return torch.kron(torch.ones(n_atoms, n_atoms, dtype=dtype) / n_atoms, torch.eye(3, dtype=dtype))


def hessian_terms(H, volume, n_atoms, kappa=50.0):
    """(eigenvalues of H_uu with translations shifted to kappa [eV/A^2],
        clamped-ion C [GPa], relaxed-ion C [GPa])."""
    n3 = 3 * n_atoms
    Huu, Hue, Hee = H[:n3, :n3], H[:n3, n3:], H[n3:, n3:]
    Hreg = Huu + kappa * translation_projector(n_atoms, H.dtype)
    eig = torch.linalg.eigvalsh(0.5 * (Hreg + Hreg.T))
    C_clamped = Hee / volume * EV_A3_GPA
    C_relaxed = (Hee - Hue.T @ torch.linalg.solve(Hreg, Hue)) / volume * EV_A3_GPA
    return eig, C_clamped, C_relaxed


def stability_penalty(eig, k_min=1.0):
    """Mean squared shortfall of force-constant eigenvalues below k_min (eV/A^2)."""
    return (torch.relu(k_min - eig) ** 2).mean()
