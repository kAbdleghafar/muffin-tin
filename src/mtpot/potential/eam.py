"""Multi-species EAM in torch:  E = sum_i [F_Zi(rho_i) + e_Zi] + 1/2 sum_{i!=j} phi_ZiZj(r_ij),
rho_i = sum_{j!=i} f_Zj(r_ij).  Forces and stress (ASE convention, tension positive,
stress = (1/V) dE/d(strain)) come from autograd through positions and a strain tensor."""
import numpy as np
import torch
from ase.neighborlist import neighbor_list

from mtpot.dft import ZVAL

from .splines import Splines

SPECIES = list(ZVAL)                     # Ti, Zr, Nb, Mo, Hf, Ta
NS = len(SPECIES)
PAIRS = [(a, b) for a in range(NS) for b in range(a, NS)]
PAIR_INDEX = torch.full((NS, NS), -1, dtype=torch.long)
for _k, (_a, _b) in enumerate(PAIRS):
    PAIR_INDEX[_a, _b] = PAIR_INDEX[_b, _a] = _k


class EAM(torch.nn.Module):
    def __init__(self, rc=6.0, r0_pair=1.8, r0_dens=1.8, n_pair=21, n_dens=21,
                 rho_max=3.0, n_embed=8):
        super().__init__()
        self.rc = float(rc)
        self.phi = Splines(len(PAIRS), r0_pair, rc, n_pair, cutoff=True)
        self.f = Splines(NS, r0_dens, rc, n_dens, cutoff=True)
        self.F = Splines(NS, 0.0, rho_max, n_embed, cutoff=False)
        self.e0 = torch.nn.Parameter(torch.zeros(NS, dtype=torch.float64))

    # ------------------------------------------------------------------ core
    def energies(self, batch, strain=None):
        """Per-frame energies (eV). batch: dict of tensors from `collate` (positions may
        require grad); strain: (n_frames, 3, 3) symmetric, applied as r -> r (I + strain)."""
        pos, Z = batch["positions"], batch["species"]
        i, j = batch["i"], batch["j"]
        d = pos[j] - pos[i] + batch["shift_cart"]
        if strain is not None:
            d = d + torch.einsum("pa,pab->pb", d, strain[batch["pair_frame"]])
        r = torch.linalg.norm(d, dim=1)
        rho = torch.zeros(len(Z), dtype=pos.dtype).index_add(0, i, self.f(r, Z[j]))
        e_atom = self.F(rho, Z) + self.e0[Z]
        e_pair = 0.5 * self.phi(r, PAIR_INDEX[Z[i], Z[j]])
        n = batch["n_frames"]
        E = torch.zeros(n, dtype=pos.dtype).index_add(0, batch["atom_frame"], e_atom)
        return E.index_add(0, batch["pair_frame"], e_pair)

    def forward(self, batch, forces=True, stress=True, create_graph=False):
        """Energies, forces and stresses (Voigt xx yy zz yz xz xy, eV/A^3) for a batch."""
        pos = batch["positions"].detach().requires_grad_(forces)
        strain = torch.zeros(batch["n_frames"], 3, 3, dtype=pos.dtype, requires_grad=stress)
        E = self.energies(dict(batch, positions=pos), strain if stress else None)
        out = {"energy": E}
        inputs = [t for t, need in ((pos, forces), (strain, stress)) if need]
        if inputs:
            grads = torch.autograd.grad(E.sum(), inputs, create_graph=create_graph)
            k = 0
            if forces:
                out["forces"] = -grads[k]
                k += 1
            if stress:
                g = grads[k]
                g = 0.5 * (g + g.transpose(1, 2)) / batch["volume"][:, None, None]
                out["stress"] = torch.stack([g[:, 0, 0], g[:, 1, 1], g[:, 2, 2],
                                             g[:, 1, 2], g[:, 0, 2], g[:, 0, 1]], dim=1)
        return out

    def regularization(self):
        return (self.phi.curvature_penalty() + self.f.curvature_penalty()
                + self.F.curvature_penalty())


# ---------------------------------------------------------------------- batching
def frame_graph(atoms, rc):
    """Neighbour list (full, both directions) of one structure as numpy arrays."""
    i, j, S = neighbor_list("ijS", atoms, rc)
    return dict(positions=atoms.positions.astype(np.float64), cell=atoms.cell[:].astype(np.float64),
                species=np.array([SPECIES.index(s) for s in atoms.get_chemical_symbols()]),
                i=i.astype(np.int64), j=j.astype(np.int64), S=S.astype(np.float64))


def collate(graphs):
    """Concatenate frame graphs into one batch dict of torch tensors."""
    pos, Z, ii, jj, sh, af, pf, vol, cells = [], [], [], [], [], [], [], [], []
    off = 0
    for f, g in enumerate(graphs):
        n = len(g["species"])
        pos.append(g["positions"]); Z.append(g["species"])
        ii.append(g["i"] + off); jj.append(g["j"] + off)
        sh.append(g["S"] @ g["cell"])
        af.append(np.full(n, f)); pf.append(np.full(len(g["i"]), f))
        vol.append(abs(np.linalg.det(g["cell"]))); cells.append(g["cell"])
        off += n
    t = lambda a, dt=torch.float64: torch.as_tensor(np.concatenate(a), dtype=dt)
    return dict(positions=t(pos), species=t(Z, torch.long), i=t(ii, torch.long),
                j=t(jj, torch.long), shift_cart=t(sh), atom_frame=t(af, torch.long),
                pair_frame=t(pf, torch.long), volume=torch.tensor(vol, dtype=torch.float64),
                cells=torch.tensor(np.array(cells)), n_frames=len(graphs),
                n_atoms=torch.tensor([len(g["species"]) for g in graphs]))
