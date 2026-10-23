"""Fitting the EAM to energies, forces, stresses and C_ij (from stresses at DFT frames)."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import composition, frame_graphs, reference_energies
from .eam import EAM, SPECIES, collate

EV_A3_GPA = 160.21766208
VOIGT_OF_BLOCK = [0, 1, 2, 5, 4, 3]       # works 1-4 e1, 5-8 e2, 9-12 e3, 13-16 e6, 17-20 e5, 21-24 e4


class AlloyBatch:
    """One alloy's frames, collated once, with labels as torch tensors. Elastic sets carry
    composition-referenced energies and C_ij labels; relax sets (`energy_group` present) are
    fitted on energy differences within each group only (their cut-offs differ)."""

    def __init__(self, d, mu):
        self.alloy = str(d["alloy"])
        self.relax = "energy_group" in d
        self.groups = torch.as_tensor(d["energy_group"].astype(np.int64)) if self.relax else None
        self.batch = collate(frame_graphs(d))
        nat = len(d["species"])
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
        self.nat = nat
        self.e_target = t((d["energy"] - composition(d) @ mu) / nat)          # eV/atom
        self.f_target = t(d["forces"].reshape(-1, 3))
        self.s_target = t(d["stress"] * EV_A3_GPA)                            # GPa
        self.w = t(d["weight"])
        self.w_atom = self.w.repeat_interleave(nat)
        # C_ij from stresses at the 24 final frames, if every relaxation ended
        self.cij = None
        if self.relax:
            return
        final = np.where(d["final"])[0]
        works = d["work"][final]
        if len(final) == 24 and sorted(works) == list(range(1, 25)):
            order = final[np.argsort(works)]
            e = np.array([d["strain"][order[w], VOIGT_OF_BLOCK[w // 4]] for w in range(24)])
            self.cij = (torch.as_tensor(order), t(e.reshape(6, 4)), t(d["C_GPa"]))


def cij_from_stress(stress_gpa, cij):
    """6x6 C (GPa) from Voigt stresses at the 24 final frames: slope of a straight-line fit
    per strain component (C[i, j] = d sigma_i / d e_j)."""
    order, e, _ = cij
    s = stress_gpa[order].reshape(6, 4, 6)                    # [work block, point, sigma_i]
    ec = e - e.mean(1, keepdim=True)
    slope = (ec[:, :, None] * (s - s.mean(1, keepdim=True))).sum(1) / (ec**2).sum(1)[:, None]
    block_of_voigt = [VOIGT_OF_BLOCK.index(v) for v in range(6)]
    return slope[block_of_voigt].T                            # C[i, j], j in Voigt order


def cubic(C):
    Cs = 0.5 * (C + C.T)
    return torch.stack([torch.diagonal(Cs)[:3].mean(), (Cs[0, 1] + Cs[0, 2] + Cs[1, 2]) / 3,
                        torch.diagonal(Cs)[3:].mean()])


def init_model(model):
    """Physically shaped start: f ~ exp decay (rho ~ 1 in bcc), F ~ -sqrt(rho),
    phi ~ weak Morse; e0 = 0 (energies are referenced to mu)."""
    with torch.no_grad():
        r = model.f.x0 + model.f.h * (torch.arange(model.f.size, dtype=torch.float64) - 1)
        model.f.set_coef(0.07 * torch.exp(-(r - 2.8) / 0.8)[None].repeat(len(SPECIES), 1))
        rp = model.phi.x0 + model.phi.h * (torch.arange(model.phi.size, dtype=torch.float64) - 1)
        morse = 0.2 * (torch.exp(-2 * 1.5 * (rp - 2.9)) - 2 * torch.exp(-1.5 * (rp - 2.9)))
        model.phi.set_coef(morse[None].repeat(model.phi.coef.shape[0], 1))
        x = model.F.x0 + model.F.h * (torch.arange(model.F.size, dtype=torch.float64) - 1)
        model.F.set_coef(-2.0 * torch.sqrt(x.clamp(min=0))[None].repeat(len(SPECIES), 1))
    return model


def group_centred(x, groups, w):
    """x minus its w-weighted mean within each group."""
    n = int(groups.max()) + 1
    m = torch.zeros(n, dtype=x.dtype).index_add(0, groups, w * x) / \
        torch.zeros(n, dtype=x.dtype).index_add(0, groups, w)
    return x - m[groups]


def evaluate(model, ab, create_graph=False, cij=True):
    out = model(ab.batch, create_graph=create_graph)
    e = out["energy"] / ab.nat
    dE = e - ab.e_target
    if ab.relax:
        dE = group_centred(dE, ab.groups, ab.w)
    res = dict(dE=dE, dF=out["forces"] - ab.f_target,
               dS=out["stress"] * EV_A3_GPA - ab.s_target)
    if cij and ab.cij is not None:
        C = cij_from_stress(out["stress"] * EV_A3_GPA, ab.cij)
        res["C"] = C
        res["dC"] = C - ab.cij[2]
    return res


def loss_of(res, ab, w):
    """Weighted squared errors in natural scales: 1 meV/atom, 0.05 eV/A, 0.1 GPa, 5 GPa.
    Relax sets use the weights rE, rF, rS (default: same as E, F, S)."""
    p = "r" if ab.relax else ""
    wE, wF, wS = (w.get(p + k, w[k]) for k in ("E", "F", "S"))
    L = wE * (ab.w * (res["dE"] * 1e3) ** 2).sum() / ab.w.sum()
    L = L + wF * (ab.w_atom[:, None] * (res["dF"] / 0.05) ** 2).sum() / (3 * ab.w_atom.sum())
    L = L + wS * (ab.w[:, None] * (res["dS"] / 0.1) ** 2).sum() / (6 * ab.w.sum())
    if "dC" in res and w.get("C", 0) > 0:
        dC = 0.5 * (res["dC"] + res["dC"].T)
        L = L + w["C"] * ((dC / 5.0) ** 2).mean()
    return L


def report(model, sets):
    """summarize() for several named sets, flattened as '<name>_<metric>'."""
    return {f"{name}_{k}": v for name, alloys in sets.items() if alloys
            for k, v in summarize(model, alloys).items()}


def summarize(model, alloys):
    """RMSEs over alloys (frame-weighted; relax energies within-group centred) and
    C11/C12/C44 errors from stress slopes."""
    se_e = se_f = se_s = 0.0
    n_e = n_f = n_s = 0.0
    cub = []
    with torch.enable_grad():
        for ab in alloys:
            res = {k: v.detach() for k, v in evaluate(model, ab).items()}
            se_e += float((ab.w * res["dE"] ** 2).sum()); n_e += float(ab.w.sum())
            se_f += float((ab.w_atom[:, None] * res["dF"] ** 2).sum()); n_f += 3 * float(ab.w_atom.sum())
            se_s += float((ab.w[:, None] * res["dS"] ** 2).sum()); n_s += 6 * float(ab.w.sum())
            if "C" in res:
                cub.append((cubic(res["C"]).numpy(), cubic(ab.cij[2]).numpy()))
    out = dict(E_meV_atom=1e3 * np.sqrt(se_e / n_e), F_eV_A=np.sqrt(se_f / n_f),
               S_GPa=np.sqrt(se_s / n_s))
    if cub:
        p, t = np.array([c[0] for c in cub]), np.array([c[1] for c in cub])
        for k, name in enumerate(["C11", "C12", "C44"]):
            out[f"{name}_MAE"] = float(np.abs(p[:, k] - t[:, k]).mean())
        out["n_cij"] = len(cub)
    return out


def fit(model, train, val, weights, epochs=50, lr=1e-2, batch_alloys=4, reg=1e-4,
        log=None, seed=0, eval_every=5):
    """Adam over batches of alloys; returns the history of metrics."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(train))
        tot = 0.0
        for k in range(0, len(order), batch_alloys):
            opt.zero_grad()
            L = sum(loss_of(evaluate(model, train[i], create_graph=True), train[i], weights)
                    for i in order[k:k + batch_alloys]) / len(order[k:k + batch_alloys])
            L = L + reg * model.regularization()
            L.backward()
            opt.step()
            tot += L.item()
        sched.step()
        row = dict(epoch=ep, loss=tot / int(np.ceil(len(order) / batch_alloys)),
                   seconds=time.time() - t0)
        if ep % eval_every == 0 or ep == epochs:
            row.update(report(model, val if isinstance(val, dict) else {"train": train, "val": val}))
        hist.append(row)
        if log:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
                  file=log, flush=True)
    return hist


def fit_lbfgs(model, train, val, weights, iters=100, reg=1e-4, log=None, eval_every=10,
              history_size=30, checkpoint=None):
    """Full-batch L-BFGS (strong Wolfe line search). Gradients are accumulated alloy by alloy,
    so memory stays at one alloy's graph. Logs (and calls checkpoint(model, row)) every
    eval_every iterations."""
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=eval_every,
                            history_size=history_size, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)
    n_eval = [0]

    def closure():
        opt.zero_grad()
        total = 0.0
        for ab in train:
            L = loss_of(evaluate(model, ab, create_graph=True), ab, weights) / len(train)
            L.backward()
            total += L.item()
        R = reg * model.regularization()
        R.backward()
        n_eval[0] += 1
        return torch.tensor(total + R.item())

    hist = []
    for it in range(eval_every, iters + 1, eval_every):
        t0 = time.time()
        loss = float(opt.step(closure))
        row = dict(iteration=it, evaluations=n_eval[0], loss=loss, seconds=time.time() - t0)
        row.update(report(model, val if isinstance(val, dict) else {"train": train, "val": val}))
        hist.append(row)
        if log:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
                  file=log, flush=True)
        if checkpoint:
            checkpoint(model, row)
    return hist


def save(model, path, mu, config):
    torch.save({"state": model.state_dict(), "mu": mu, "config": config}, path)


def load(path):
    ck = torch.load(path, weights_only=False)
    m = EAM(**ck["config"]["model"])
    m.load_state_dict(ck["state"])
    return m, ck["mu"], ck["config"]
