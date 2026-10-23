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

## Data (255 alloys, 3 files each)

| File | Contents |
|---|---|
| `elast_<id>.out` | thermo_pw v7.4.1 `scf_elastic_constants` run, ions relaxed at every strain |
| `rho_<id>.cube` | QE pp.x valence density (pseudo + augmentation), e/bohr³, unstrained reference cell |
| `elf_<id>.cube` | QE pp.x ELF on the same grid and structure |

Full data set: three RAR archives on the user's Google Drive (shared "anyone with the link"):
`elast.rar` `19ncdMpoeH1ACApIul9PHoFghg0jFWXKq` (1.1 GB → 6.0 GB),
`rho.rar` `1iozSRV_pG14wqAj0kjVXkZhGpsZUUVc8` (1.4 GB → 5.9 GB),
`elf.rar` `1c8yQFm7i_OejH3GtDaqPxJiSpvjyr_jH` (1.3 GB → 5.9 GB). Download with
`curl -L -o X.rar "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"`
(network policy must allow `drive.google.com`, `drive.usercontent.google.com`), extract with
`unrar e` (`apt-get install unrar`) into a scratch dir, move into `data/raw/`, delete the
archive (disk is tight: ~17 GB unpacked). Archives hold flat `<kind>_<id>.<ext>` files only.
255 alloys: IDs 1–259 except 107, 108, 109, 240 (IDs 1–9 are zero-padded: `01`…`09`).
Samples examined so far: `01`, `140`, `150`. Raw files are 4–30 MB each (~12 GB of cubes
in total): never load all cubes at once, process one alloy at a time and cache reduced
results (`.npz`).

### DFT settings (verified in samples)
- Quantum ESPRESSO + thermo_pw, PBEsol, PAW with core corrections
- ecutwfc 55 Ry, ecutrho 550 Ry, scf threshold 1e-8, BFGS ionic relaxation, fixed cell per strain
  (`nstep` = 20; BFGS criteria energy < 5e-6 Ry, force < 5e-4 Ry/bohr)
- Valence (Zval): Ti 12, Zr 12, Nb 13, Mo 14, Hf 12, **Ta 27 (4f in valence)**
- Smearing is **not consistent** (Phase 0, all 255): Gaussian 0.009 Ry in 235 alloys;
  Marzari–Vanderbilt 0.007 Ry in 13 (`01`–`08`, `92`, `93`, `104`–`106`); Gaussian 0.007 Ry
  in 7 (`110`–`116`). The MV group is Ti-rich binaries/ternaries, so scheme and chemistry are
  confounded: no clean test of a systematic C_ij shift between schemes from these data.
- Supercells are bcc-based, aligned with cubic axes, slightly triclinic after relaxation:
  2×2×4 (32 atoms, 241 alloys), 2×2×5 (40, 9), 3×3×2 (36, 5); a ≈ 3.18–3.30 Å.
  36 element sets; 2–6 elements per alloy (2: 23, 3: 101, 4: 101, 5: 25, 6: 5).
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
- Alloy ID = numeric suffix of the filename (`elast_01.out` → `01`), as the parser stores it.

## Existing code

`scripts/tpw_to_dataset.py` (working, validated):
- `parse_thermo_pw(path)` → `(frames, info)`; frames carry energy, forces, stress, cell,
  positions, strain, work index, ionic step, `final` flag and −TS. `final` = last ionic step
  of a work whose BFGS ended — converged, or stopped at the `nstep` limit (thermo_pw uses that
  stress for C_ij too); per-frame `converged`. `info` sets: `works_computed`, `works_ended`,
  `works_converged`, `works_maxsteps`. A run cut off mid-relaxation has no final frame.
- Writes `dataset.extxyz` (ASE SinglePointCalculator; info keys: `alloy`, `work`,
  `ionic_step`, `final`, `relax_converged`, `strain_voigt`, `minus_TS`, `smearing`) and
  `elastic_summary.csv`.
- `refit_cij()` re-derives C_ij from relaxed stresses as a check; `cubic_projection()`.

```bash
python scripts/tpw_to_dataset.py data/raw/elast_*.out \
    -o data/processed/dataset.extxyz -s data/processed/elastic_summary.csv
```

`src/mtpot/elastic/protocol.py`: `thermo_pw_strain(work)` = the 24-work strain table below.
`src/mtpot/io/cube.py`: `read_cube`, `integrate` (electron count), `structure_deviation`.
`src/mtpot/dft.py`: Zval table and expected settings.

`scripts/qc.py` (Phase 0): one row per alloy → `data/processed/qc.csv`, with a `flags` column
(settings, nelec vs ΣZval, restart works, BFGS step limit, unfinished works, dropped frames,
strain protocol,
non-cubic > 8%, refit mismatch > 0.02 GPa, residual stress > 0.5 GPa, rho electron count,
ELF range, cube grid/structure). Rows are cached in `data/processed/qc_cache/<id>.json`, so a
sweep resumes; `--no-cubes` skips the cubes; `-j 4` runs 4 worker processes. The refit and
residual stress are only computed when all 24 relaxations ended (a one-sided fit picks up
curvature). Sample `01`: no flags.

```bash
python scripts/qc.py data/raw -o data/processed/qc.csv -j 4
```

`src/mtpot/density/` (Phase 1): `grid` (periodic grid-point/atom pairs), `basis`
(`RadialBasis` cubic B-splines with C2 cutoff; `ExpBasis`), `fit` (`sample_alloy`,
`DensityFit` with core mask, monotone / non-negative / background options, `metrics`),
`profiles` (radial profiles, shell angular power with real Y_lm to l = 4).
`scripts/density_phase1.py`: `prepare` (one pass per alloy over rho/ELF cubes →
`data/processed/density/<id>.npz`, ~15 s and 20 MB each), `fit`, `angular` (slow:
~30 s/alloy), `report` (→ `reports/phase1/`). Train/test sets: `binaries`, `test`.

```bash
python scripts/density_phase1.py prepare --ids binaries test -j 4
python scripts/density_phase1.py fit --train binaries --test test --r-core all=1.0 \
    --rc 6.0 --n-int 27 --lam 1e-6 --monotone --oracle --tag binaries
python scripts/density_phase1.py angular --ids binaries test --tag binaries
python scripts/density_phase1.py report --ids binaries test --tag binaries
```

Tests (`python -m pytest`; `-m "not data"` for the data-free subset):
- `tests/test_parser_synthetic.py`: parser units, signs, final flags, C_ij refit and CLI round trip
  on synthetic thermo_pw outputs with known answers (`tests/synthetic_tpw.py`), including a
  run killed mid-relaxation (its last work must not be `final`) and BFGS stopped at `nstep`.
- `tests/test_qc.py`: QC sweep, flags and cache on synthetic alloys and cubes.
- `tests/test_density.py`: neighbour pairs vs brute force, basis cutoff, recovery of known
  f_Z from synthetic triclinic superpositions (incl. monotone), harmonic analysis.
- `tests/test_samples.py`: the regression fixtures below; skip unless the files are in
  `data/raw/` or `$MTPOT_RAW`. Sample `01` passes all of them (incl. strained cell =
  reference rows × (I + ε) with fixed fractional coordinates), and so do `140` and `150`.

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

1. **Restarted runs: 14 alloys miss 185 works in total** (`05` 1–10, `11` 1–16, `46` 1–19,
   `60` 1–12, `76` 1–7, `96` 1–19, `114` 1–13, `123` 1–16, `133` 1–5, `141` 1–17, `150` 1–22,
   `159` 1–12, `181` 1–8, `194` 1–9). `150` example: works 1–22 came from restart files.
   The earlier interrupted run's output is lost (the user does not know where it is), so
   for `150` only works 23–24 give frames; thermo_pw's C_ij is still usable as a label.
   Recovering works 1–22 would mean recomputing them. Detect and report restarted works
   for every alloy; never silently drop them.
2. **Mixed smearing schemes** across alloys: always correct to E0 and store the scheme
   as metadata; check for systematic C_ij differences between schemes.
3. **Narrow configuration coverage**: within an alloy, energies span 0.6–4.8 meV/atom
   (median 1.8) and max |F| is 0.04–0.30 eV/Å (median 0.08). A potential trained only on this may fail to relax a fresh SQS from
   ideal positions. Plan to add the vc-relax trajectories (ask the user for them) and
   ±3–5% volume scans.
4. **Ionic steps are near-duplicates** (~10 per work): weight per work, not per frame,
   and never treat them as independent samples in validation.
5. **Near-core rho/ELF are pseudopotential artefacts** (r ≲ 0.8 Å): Ta peaks at
   19 e/bohr³ with ELF ≈ 1, while Zr/Nb/Mo show ELF ≈ 0.1 at the nucleus. Mask the core
   region when fitting. About 16–17% of valence electrons lie > 1.2 Å from any nucleus;
   that is where the bonding signal is. Ti's near-core profile is identical in `01` and `150`
   (evidence that atom-centred functions transfer between alloys).

6. **BFGS step limit**: 159 works in 56 alloys stopped at `nstep` = 20 before meeting the
   criteria (e.g. `28` work 21: gradient 8.5e-4 vs 5e-4 Ry/bohr). thermo_pw used their last
   stress for C_ij, so their last frame is `final` with `converged=False`. Frames are valid
   DFT data; the labels contain slightly under-relaxed stresses.
7. **Residual stress**: the zero-strain intercept exceeds 0.5 GPa in 18 alloys (max 0.91, `118`),
   i.e. the reference cell is not exactly at zero stress; their C_ij labels refer to that cell.
   Phase 3 must decide whether to strain the DFT cell or the potential's own relaxed cell.
8. **Large non-cubic part (> 8%) in 17 alloys, all Ti-rich (Ti ≥ 31%)**, up to 20% (`88`
   Ta4Ti28): physical (soft bcc Ti), raises the C_ij error floor for those alloys.

## Phase 0 results (complete; `data/processed/qc.csv`)

- 255 alloys, no parse errors; 52,797 frames (51,233 in the 241 complete alloys).
- Settings identical everywhere (PBEsol, 55/550 Ry); nelec = ΣZval for all.
- Refit C_ij vs thermo_pw ≤ 0.011 GPa for all 241 complete alloys; strain protocol holds for
  every work (strained cell vs reference × (I + ε) ≤ 7e-5 Å).
- Range: C11 134–463, C12 86–178, C44 19–102 GPa (medians 191 / 117 / 38).
- **Cauchy pressure C12 − C44 > 0 for all 255** and C′ ≥ 18 GPa: pure EAM is not ruled out by
  the Cauchy-pressure sign; angular terms must be justified by Phase 1 or by fit quality.
- Cubes (all 255, rho and ELF): rho integrates to ΣZval within ±0.006 e and is positive
  everywhere (min 0.018 e/bohr³); ELF within 0.003–1.0; every cube matches its thermo_pw
  reference cell and positions within 6.3e-5 Å; rho and ELF share the grid in every alloy
  (16 distinct grids, 96×96×192 in 193 alloys). No cube flags.
- Flags (91 alloys): BFGS step limit 56, residual stress 18, non-cubic 17, restart works 14 —
  all described under "Known data issues"; nothing else inconsistent.
- Full sweep: ~75 s for thermo_pw outputs with `-j 4`; ~14 min with cubes from a cold disk.

## Phase 1 results (23 binaries; details and figures in `reports/phase1/README.md`)

- Core mask 1.0 Å for all species: inside it every element's profile is identical across
  alloys and atoms (< 1.3 %); ELF minimum (semicore/valence boundary) at 1.0–1.2 Å.
- Monotone spline f_Z (r_c 6 Å) trained on binaries reproduce rho outside the mask to 3.2 %
  (train) / 3.4 % (23 held-out 3–6-element alloys) relative RMS; 3.1 % when each held-out
  alloy is fitted on its own. Transfer works; Zr (2 binaries only) is the weakest.
- **Ill-posed where EAM needs it**: equally good fits (monotone, unconstrained,
  exponential) disagree at neighbour distances; the implied ρ̄ᵢ at atom sites differ in
  mean and even sign (correlation 0.02–0.36). rho does not determine f_Z beyond ~2.3 Å.
- Residual is 1–3 % of the local density and mostly l = 4 (cubic), with a group-dependent
  sign: Mo/Nb/Ta excess toward ⟨111⟩, Ti/Hf (and mostly Zr) toward ⟨100⟩ — directional
  d-bonding linked to C′/C44.
- For Phase 2: learn f_Z from energies/forces/stresses (rho only as a soft prior for
  r < 2.3 Å or as per-atom shell-density targets); EAM baseline plus one angular channel
  as an ablation.

## Roadmap

**Phase 0 – ingestion & QC (done, see "Phase 0 results").** Parse all 255 alloys; one QC table: settings, smearing,
works present, frames, residual stress, refit-vs-thermo_pw difference, rho electron count.
Flag anything inconsistent.

**Phase 1 – density functions (the muffin-tin part) (done on binaries, see "Phase 1 results").** Fit per-species radial functions
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
- Ask the user before launching anything long (full 255-alloy cube processing, training runs).

## Open questions for the user

- ~~How many alloys used each smearing scheme?~~ Answered in "DFT settings" (Phase 0).
- ~~Where is the full output of the interrupted `150` run?~~ Unknown/lost (see Known data
  issue 1). Phase 0 will list any other restarted runs; decide then whether to recompute.
- Are the vc-relax outputs (ideal → relaxed SQS) available as extra training data?
- Target accuracy for C_ij (provisional: MAE ≤ 5–10 GPa vs DFT)?
- Is LAMMPS export required, or is an ASE/PyTorch calculator enough?
