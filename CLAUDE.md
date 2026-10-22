# CLAUDE.md — Muffin-tin-like potential for refractory SQS alloys

## Goal

Build a physics-informed, muffin-tin / EAM-like interatomic potential for bcc refractory
multi-principal-element alloys, fitted to existing DFT data, that predicts **elastic
constants of new SQS structures** at near-DFT accuracy and a tiny fraction of the cost.

Deployment target: given a new SQS (species, coordinates, lattice vectors) →
relax with the potential → apply strains → relaxed-ion C_ij (cubic-projected C11, C12, C44).

The DFT electron density (rho) and ELF are used to shape the model's physics
(atom-centred density functions, angular terms). **Energies, forces and stresses are the
fitting targets** — rho/ELF alone cannot determine the energy.

## Data (252 alloys, 3 files each)

| File | Contents |
|---|---|
| `elast_<id>.out` | thermo_pw v7.4.1 `scf_elastic_constants` run, ions relaxed at every strain |
| `rho_<id>.cube` | QE pp.x valence density (pseudo + augmentation), e/bohr³, unstrained reference cell |
| `elf_<id>.cube` | QE pp.x ELF on the same grid and structure |

Samples examined so far: `01`, `140`, `150`. Raw files are 4–30 MB each (~13 GB of cubes
in total): never load all cubes at once, process one alloy at a time and cache reduced
results (`.npz`).

### DFT settings (verified in samples)
- Quantum ESPRESSO + thermo_pw, PBEsol, PAW with core corrections
- ecutwfc 55 Ry, ecutrho 550 Ry, scf threshold 1e-8, BFGS ionic relaxation, fixed cell per strain
- Valence (Zval): Ti 12, Zr 12, Nb 13, Mo 14, Hf 12, **Ta 27 (4f in valence)**
- Smearing is **not consistent**: `01` Marzari–Vanderbilt 0.007 Ry; `140`, `150` Gaussian 0.009 Ry
- Supercells are bcc-based, aligned with cubic axes, slightly triclinic after relaxation:
  2×2×4 (32 atoms), 2×2×5 (40), 3×3×2 (36); a ≈ 3.18–3.30 Å.
  thermo_pw finds point group C_1, so it computes all 21 C_ij.

### thermo_pw strain protocol (24 "works")
| Works | Strain component | Values |
|---|---|---|
| 1–4 | ε_xx (Voigt 1) | −0.0075, −0.0025, +0.0025, +0.0075 |
| 5–8 | ε_yy (Voigt 2) | same |
| 9–12 | ε_zz (Voigt 3) | same |
| 13–16 | ε_xy = ε_yx (Voigt 6) | same (engineering e6 = 2ε) |
| 17–20 | ε_xz = ε_zx (Voigt 5) | same |
| 21–24 | ε_yz = ε_zy (Voigt 4) | same |

Strain is applied to the reference cell with fractional coordinates held fixed, then ions
are relaxed at fixed cell. The potential's elastic workflow **must reproduce this protocol**
so predictions are comparable to the DFT labels.

### Output-file format notes (thermo_pw, needed for parsing)
- **ASE's `espresso-out` reader fails on these files** (no `Program PWSCF` marker). Use the
  custom parser `scripts/tpw_to_dataset.py`.
- Reference header uses `alat=  X a.u.`; each work's pw header uses `lattice parameter (alat)`.
- Per work: `Applying the following strain` + 3 matrix lines (`(  -0.00750,  0.00000,  0.00000  )`),
  `Doing work n / 24`,
  `crystal axes`, `site n. atom positions (alat units)`.
- Per ionic step: `!    total energy` (= free energy F, Ry), `smearing contrib. (-TS)`,
  `Forces acting on atoms` (take the first nat lines only; Ry/bohr),
  `total   stress  (Ry/bohr**3)` (first 3 columns), then `ATOMIC_POSITIONS (crystal)`.
- Final block: `Elastic constants C_ij (kbar)`; row i = stress component, column j = strain.
- Works read from restart files appear only as headers, with no energies/forces/stresses.

## Conventions (do not change)

- Internal units: eV, Å, eV/Å³. Report elastic quantities in GPa (1 kbar = 0.1 GPa).
- Stress sign: ASE convention, tension positive = **−(QE printed stress)**.
- Voigt order xx, yy, zz, yz, xz, xy with engineering shear strains (e4 = 2ε_yz, etc.).
- C[i, j] = dσ_i / de_j. Cubic projection: C11 = mean(C11, C22, C33), C12 = mean of the
  three off-diagonal normal terms of the symmetrised tensor, C44 = mean(C44, C55, C66).
- Training energy: E0 = F − ½(−TS) for Gaussian smearing; E0 ≈ F for Marzari–Vanderbilt.
  Forces and stresses are used as printed (derivatives of F).
- Alloy ID = numeric suffix of the filename (`elast_01.out` → `01`). The current parser
  stores `alloy="elast_01"`; normalise to the numeric ID.

## Existing code

`scripts/tpw_to_dataset.py` (working, validated):
- `parse_thermo_pw(path)` → `(frames, info)`; frames carry energy, forces, stress, cell,
  positions, strain, work index, ionic step, `final` flag and −TS.
- Writes `dataset.extxyz` (ASE SinglePointCalculator; info keys: `alloy`, `work`,
  `ionic_step`, `final`, `strain_voigt`, `minus_TS`, `smearing`) and `elastic_summary.csv`.
- `refit_cij()` re-derives C_ij from relaxed stresses as a check; `cubic_projection()`.

```bash
python scripts/tpw_to_dataset.py data/raw/elast_*.out \
    -o data/processed/dataset.extxyz -s data/processed/elastic_summary.csv
```

`src/mtpot/elastic/protocol.py`: `thermo_pw_strain(work)` = the 24-work strain table below.

Tests (`python -m pytest`; `-m "not data"` for the data-free subset):
- `tests/test_parser_synthetic.py`: parser units, signs, final flags, C_ij refit and CLI round trip
  on synthetic thermo_pw outputs with known answers (`tests/synthetic_tpw.py`). Two strict
  `xfail` tests mark known gaps: numeric alloy ID, and `final` set on a work whose BFGS never
  converged (an interrupted run would feed an unrelaxed stress into the refit).
- `tests/test_samples.py`: the regression fixtures below; skip unless the files are in
  `data/raw/` or `$MTPOT_RAW`. Sample `01` passes all of them (incl. strained cell =
  reference rows × (I + ε) with fixed fractional coordinates); `140`, `150` not yet uploaded.

## Regression fixtures (from the three samples)

| ID | Formula | Atoms | Frames | Works in file | Smearing | C11 / C12 / C44 (GPa) |
|---|---|---|---|---|---|---|
| 01 | Mo8Ti24 | 32 | 278 | 24/24 | MV 0.007 | 197.0 / 113.1 / 30.1 |
| 140 | Hf4Mo9Nb9Ti9Zr9 | 40 | 253 | 24/24 | Gauss 0.009 | 197.4 / 116.0 / 30.9 |
| 150 | Hf6Mo6Nb6Ta6Ti6Zr6 | 36 | 30 | 2/24 (works 23–24) | Gauss 0.009 | 210.5 / 118.8 / 39.8 |

- Refit C_ij from stresses matches thermo_pw within 0.01 GPa (01, 140).
- rho integrates to 400.0 / 507.0 / 540.0 electrons = valence count.
- Cube structures match the thermo_pw unstrained reference exactly (cell and positions).
- Non-cubic remainder of the tensor is 3–5% (SQS disorder; sets an error floor).
- −TS: ≈ −0.1 meV/atom (MV) vs ≈ −9.8 meV/atom (Gaussian).
- Residual stress at zero strain: `01` ≈ +0.24–0.29 GPa normal components; `140` within ±0.11 GPa.

Turn these into pytest tests before refactoring the parser.

## Known data issues

1. **`150` is incomplete in its output file**: works 1–22 came from restart files.
   The earlier interrupted run's output is lost (the user does not know where it is), so
   for `150` only works 23–24 give frames; thermo_pw's C_ij is still usable as a label.
   Recovering works 1–22 would mean recomputing them. Detect and report restarted works
   for every alloy; never silently drop them.
2. **Mixed smearing schemes** across alloys: always correct to E0 and store the scheme
   as metadata; check for systematic C_ij differences between schemes.
3. **Narrow configuration coverage**: within an alloy, energies span only 1–1.6 meV/atom and
   |F| < 0.16 eV/Å. A potential trained only on this may fail to relax a fresh SQS from
   ideal positions. Plan to add the vc-relax trajectories (ask the user for them) and
   ±3–5% volume scans.
4. **Ionic steps are near-duplicates** (~10 per work): weight per work, not per frame,
   and never treat them as independent samples in validation.
5. **Near-core rho/ELF are pseudopotential artefacts** (r ≲ 0.8 Å): Ta peaks at
   19 e/bohr³ with ELF ≈ 1, while Zr/Nb/Mo show ELF ≈ 0.1 at the nucleus. Mask the core
   region when fitting. About 16–17% of valence electrons lie > 1.2 Å from any nucleus;
   that is where the bonding signal is. Ti's near-core profile is identical in `01` and `150`
   (evidence that atom-centred functions transfer between alloys).

## Roadmap

**Phase 0 – ingestion & QC.** Parse all 252 alloys; one QC table: settings, smearing,
works present, frames, residual stress, refit-vs-thermo_pw difference, rho electron count.
Flag anything inconsistent.

**Phase 1 – density functions (the muffin-tin part).** Fit per-species radial functions
f_Z(r) so that Σ_i f_Zi(|r − R_i|) reproduces DFT rho outside a per-species core mask.
Use splines with a smooth cutoff (~5.5–6 Å; bcc neighbour shells at ~2.8, 3.25, 4.6, 5.4 Å).
Analyse the residual (DFT − superposition) and ELF for non-spherical, bonding content
(e.g. l = 2, 4 cubic-harmonic moments around each atom) to decide whether angular terms are needed.

**Phase 2 – energy model.** EAM form E = Σ_i F_Zi(ρ̄_i) + ½ Σ_ij φ_ZiZj(r_ij),
ρ̄_i = Σ_j f_Zj(r_ij), with f_Z initialised/regularised from Phase 1; MEAM-style angular
terms if Phase 1 shows they matter (pure EAM cannot give negative Cauchy pressure).
Implement in PyTorch with autodiff for forces and stress (virial). Loss on E0 (per atom),
forces and stresses, stress weighted strongly (elastic constants are curvatures).
Export pure-EAM versions to LAMMPS `eam/alloy` setfl format.

**Phase 3 – elastic workflow.** ASE calculator wrapping the model; relax new SQS;
apply the 24 thermo_pw strains; relax ions at fixed cell; linear fit of stress vs strain;
cubic projection. Validate the workflow on DFT-labelled alloys first.

**Phase 4 – validation.** Group k-fold split by chemistry (held-out compositions / element
sets), never random frames. Report energy (meV/atom), force (eV/Å), stress (GPa) errors and
C11, C12, C44, C′, Cauchy-pressure-sign errors vs DFT. Baselines: a MACE or ACE model
fitted to the same data, and an ablation without rho supervision.

## Suggested layout

```
data/raw/          # elast_*.out, rho_*.cube, elf_*.cube  (read-only, not in git)
data/processed/    # dataset.extxyz, elastic_summary.csv, qc.csv, radial profiles (.npz)
scripts/           # tpw_to_dataset.py and CLI entry points
src/mtpot/io/      # thermo_pw + cube readers
src/mtpot/density/ # radial fits, core masks, ELF analysis
src/mtpot/potential/  # EAM/MEAM model, training
src/mtpot/elastic/ # strain protocol, C_ij fitting
tests/             # pytest, fixtures from the table above
```

Environment: Python ≥ 3.10, numpy, scipy, ase, torch, pytest
(`pip install ase --break-system-packages` if the system Python is managed).

## Working rules

- Never modify files in `data/raw/`.
- Do not use `ase.io.read(..., format="espresso-out")` on thermo_pw outputs.
- Do not fit density inside the core mask, and do not mix F from different smearing schemes.
- Keep the unit and sign conventions above; add a test whenever a conversion is touched.
- Ask the user before launching anything long (full 252-alloy cube processing, training runs).

## Open questions for the user

- How many of the 252 alloys used each smearing scheme?
- ~~Where is the full output of the interrupted `150` run?~~ Unknown/lost (see Known data
  issue 1). Phase 0 will list any other restarted runs; decide then whether to recompute.
- Are the vc-relax outputs (ideal → relaxed SQS) available as extra training data?
- Target accuracy for C_ij (provisional: MAE ≤ 5–10 GPa vs DFT)?
- Is LAMMPS export required, or is an ASE/PyTorch calculator enough?
