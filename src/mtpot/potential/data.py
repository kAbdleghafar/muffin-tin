"""Training data: per-alloy frame sets from thermo_pw outputs, with neighbour lists.

Ionic steps within a work are near-duplicates (CLAUDE.md, data issue 4), so each work keeps
its first step (largest forces) and its final step (relaxed stress, used for C_ij), each
weighted 1/2 (1 if the work has a single step). With all_steps=True the intermediate steps are
kept too, with weight 0: they only enter through force differences between consecutive steps
(dF ~ -H du, curvature along the BFGS path). Energies are E0: F - (-TS)/2 for Gaussian
smearing, F for Marzari-Vanderbilt."""
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from tpw_to_dataset import parse_thermo_pw, voigt_stress, voigt_strain

from .eam import SPECIES

KBAR_EV_A3 = 0.1 / 160.21766208          # 1 kbar in eV/A^3


def e0_energy(frame, smearing):
    mts = frame.get("mTS", 0.0)
    return frame["E"] - 0.5 * mts if smearing.lower().startswith("gauss") else frame["E"]


def build_alloy(path, rc=6.0, all_steps=False):
    """Arrays for one alloy: stacked frames (same atom count), labels, neighbour lists."""
    frames, info = parse_thermo_pw(str(path))
    smearing = (info["settings"]["smearing"] or "").split()[0]
    ok = [f for f in frames if f["F"] is not None and f["S"] is not None]
    by_work = {}
    for f in ok:
        by_work.setdefault(f["meta"]["work"], []).append(f)
    keep, weight = [], []
    for w, fs in sorted(by_work.items()):
        ends = 1.0 / min(len(fs), 2)
        sel = fs if all_steps else [fs[0]] + ([fs[-1]] if len(fs) > 1 else [])
        keep += sel
        weight += [ends if k in (0, len(sel) - 1) else 0.0 for k in range(len(sel))]
    ref = info["reference"]
    species = np.array([SPECIES.index(s) for s in ref.get_chemical_symbols()], dtype=np.int8)
    nl_i, nl_j, nl_S, ptr = [], [], [], [0]
    for f in keep:
        at = Atoms(f["symbols"], positions=f["pos"], cell=f["cell"], pbc=True)
        i, j, S = neighbor_list("ijS", at, rc)
        nl_i.append(i.astype(np.int16)); nl_j.append(j.astype(np.int16)); nl_S.append(S.astype(np.int8))
        ptr.append(ptr[-1] + len(i))
    final = np.array([f["meta"]["final"] for f in keep])
    C = info["C_tpw_kbar"]
    ri, rj, rS = neighbor_list("ijS", ref, rc)                 # unstrained DFT reference cell
    return dict(
        alloy=info["alloy"], smearing=smearing, species=species,
        positions=np.array([f["pos"] for f in keep]), cells=np.array([f["cell"] for f in keep]),
        energy=np.array([e0_energy(f, smearing) for f in keep]),
        forces=np.array([f["F"] for f in keep]),
        stress=np.array([voigt_stress(f["S"]) for f in keep]),            # ASE sign, eV/A^3
        strain=np.array([voigt_strain(f["meta"]["strain"]) for f in keep]),
        work=np.array([f["meta"]["work"] for f in keep], dtype=np.int16),
        step=np.array([f["meta"]["ionic_step"] for f in keep], dtype=np.int16),
        final=final, converged=np.array([f["meta"]["converged"] for f in keep]),
        weight=np.array(weight), works_ended=np.array(sorted(info["works_ended"])),
        C_GPa=(C / 10.0) if C is not None else np.full((6, 6), np.nan),
        nl_i=np.concatenate(nl_i), nl_j=np.concatenate(nl_j), nl_S=np.concatenate(nl_S),
        nl_ptr=np.array(ptr, dtype=np.int64), rc=rc,
        ref_positions=ref.positions, ref_cell=ref.cell[:],
        ref_i=ri.astype(np.int16), ref_j=rj.astype(np.int16), ref_S=rS.astype(np.int8))


FRAME_KEYS = ("positions", "cells", "energy", "forces", "stress", "strain", "work", "step",
              "final", "converged", "weight", "energy_group")


def subset_frames(d, idx):
    """Copy of a built alloy restricted to frames idx (neighbour lists sliced to match)."""
    idx = np.asarray(idx)
    out = {k: (v[idx] if k in FRAME_KEYS else v) for k, v in d.items()}
    ptr = d["nl_ptr"]
    parts = [np.arange(ptr[k], ptr[k + 1]) for k in idx]
    sel = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    for k in ("nl_i", "nl_j", "nl_S"):
        out[k] = d[k][sel]
    out["nl_ptr"] = np.concatenate([[0], np.cumsum([len(q) for q in parts])]).astype(np.int64)
    return out


def limit_steps(d, k):
    """Keep the first k ionic steps of each work plus its last (k = None: all)."""
    if k is None:
        return d
    keep = []
    for w in np.unique(d["work"]):
        f = np.where(d["work"] == w)[0]
        keep += sorted(set(f[:k].tolist()) | {int(f[-1])})
    return subset_frames(d, keep)


def reference_graph(d):
    """Frame graph of the unstrained DFT reference structure (for Hessian terms)."""
    return dict(positions=d["ref_positions"], cell=d["ref_cell"], species=d["species"].astype(np.int64),
                i=d["ref_i"].astype(np.int64), j=d["ref_j"].astype(np.int64), S=d["ref_S"].astype(np.float64))


def frame_graphs(d):
    """Frame graphs (as eam.frame_graph produces) for every frame of a built alloy."""
    out = []
    for k in range(len(d["energy"])):
        a, b = d["nl_ptr"][k], d["nl_ptr"][k + 1]
        out.append(dict(positions=d["positions"][k], cell=d["cells"][k],
                        species=d["species"].astype(np.int64),
                        i=d["nl_i"][a:b].astype(np.int64), j=d["nl_j"][a:b].astype(np.int64),
                        S=d["nl_S"][a:b].astype(np.float64)))
    return out


def composition(d):
    return np.bincount(d["species"], minlength=len(SPECIES))


def reference_energies(alloys):
    """Per-species chemical potentials mu_Z (eV/atom) from a least-squares fit of every
    frame's E0 to its composition; subtracting them leaves formation-like energies."""
    A = np.array([composition(d) for d in alloys for _ in d["energy"]], dtype=float)
    y = np.concatenate([d["energy"] for d in alloys])
    used = A.sum(0) > 0
    mu = np.zeros(len(SPECIES))
    mu[used] = np.linalg.lstsq(A[:, used], y, rcond=None)[0]
    return mu


def build_relax(path, rc=6.0, max_frames=12):
    """Arrays for one pw.x vc-relax run: up to max_frames BFGS frames, log-spaced along the
    trajectory (always the ideal start and the last BFGS step), plus the final SCF.
    Cut-offs and smearing differ between runs, so only energy differences within a run are
    meaningful: `energy_group` 0 = BFGS frames (one basis set); the final SCF (new G-vectors)
    is its own group and contributes forces and stress only. Returns None without frames."""
    from mtpot.io.pwscf import parse_pw_relax
    frames, info = parse_pw_relax(path)
    smearing = (info["settings"]["smearing"] or "").split()[0]
    ok = [f for f in frames if f["F"] is not None and f["S"] is not None]
    bfgs = [f for f in ok if f["kind"] == "bfgs"]
    if not bfgs:
        return None
    idx = np.unique(np.round(np.geomspace(1, len(bfgs), max_frames - 1)).astype(int) - 1)
    keep = [bfgs[i] for i in idx] + [f for f in ok if f["kind"] == "final_scf"][:1]
    group = np.array([0 if f["kind"] == "bfgs" else 1 for f in keep], dtype=np.int8)
    species = np.array([SPECIES.index(s) for s in keep[0]["symbols"]], dtype=np.int8)
    nl_i, nl_j, nl_S, ptr = [], [], [], [0]
    for f in keep:
        at = Atoms(f["symbols"], positions=f["pos"], cell=f["cell"], pbc=True)
        i, j, S = neighbor_list("ijS", at, rc)
        nl_i.append(i.astype(np.int16)); nl_j.append(j.astype(np.int16)); nl_S.append(S.astype(np.int8))
        ptr.append(ptr[-1] + len(i))
    return dict(
        alloy=info["alloy"], smearing=smearing, species=species, source="relax",
        ecut=f"{info['settings']['ecutwfc_Ry']}/{info['settings']['ecutrho_Ry']}",
        positions=np.array([f["pos"] for f in keep]), cells=np.array([f["cell"] for f in keep]),
        energy=np.array([e0_energy(f, smearing) for f in keep]),
        forces=np.array([f["F"] for f in keep]),
        stress=np.array([voigt_stress(f["S"]) for f in keep]),
        step=np.array([f["step"] for f in keep], dtype=np.int16), energy_group=group,
        weight=np.full(len(keep), 1.0 / len(keep)),
        nl_i=np.concatenate(nl_i), nl_j=np.concatenate(nl_j), nl_S=np.concatenate(nl_S),
        nl_ptr=np.array(ptr, dtype=np.int64), rc=rc)
