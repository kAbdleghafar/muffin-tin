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

Relaxation data (user's Drive, `relax.rar`, id `1esbO_so0KSDYCu4tX8pQlpnJKMtVH_uR`, 226 MB →
1.2 GB; archive files `relax/Alloy_<id>.out`), stored as:
- `data/raw/relax_<id>.out` (203): pw.x **vc-relax** from the ideal SQS (bcc sites) to the
  relaxed cell, plus pw.x's final SCF at the relaxed structure (fresh G-vectors). 11,982 SCF
  steps. Settings vary: cutoffs 55/550 (110 runs), 60/500 (93); Gaussian 0.007 Ry (197) or
  0.009 (18) — absolute energies are **not** comparable with the elastic runs or across
  settings; use energy differences within a run. `180` failed before any SCF; `179` crashed
  after 10 SCF steps; 10 runs stopped at 100 BFGS steps.
- `data/raw/elastprev_<id>.out` (12: `81 83 84 85 99–106`): earlier thermo_pw elastic runs
  (Feb–Apr 2026, 60/550 Ry, Gaussian 0.007) on the same reference cells; `101 102 105 106` were
  interrupted (1–3 works). Parse with `parse_thermo_pw`.
- No relax run for 52 alloys: `81`–`85`, `99`–`106`, `110`–`134`, `219`–`232` (for the 12
  `elastprev` IDs the archive held the earlier elastic run instead of a relaxation).
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

`src/mtpot/io/pwscf.py`: `parse_pw_relax(path)` → `(frames, info)` for pw.x (vc-)relax outputs
(per SCF: E, −TS, F, S, cell, positions, `kind` = `bfgs`/`final_scf`; info: settings,
converged, maxsteps, final_scf, done, initial/relaxed structures). Agrees with ASE's pw.x
reader (F 1e-7 eV/Å, S 2e-9 eV/Å³, cell/positions 5e-5 Å). `scripts/qc_relax.py` →
`data/processed/relax_qc.csv` (settings, convergence, trajectory span, relaxed vs elastic
reference; ~50 s with `-j 4`).

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

`src/mtpot/potential/` (Phase 2): `splines` (torch cubic B-splines; cutoff, linear
continuation, monotone option), `eam` (multi-species EAM, batched; forces and ASE-sign stress by
autograd through positions and a strain tensor; `frame_graph`, `collate`; `ADP` = EAM +
½|μᵢ|² + ½‖λᵢ‖² − νᵢ²/6 with dipole u and quadrupole w splines per pair, Mishin 2005;
`make_model(cfg)` with `kind` eam/adp, configs without `kind` are EAM), `export`
(`write_setfl`, LAMMPS eam/alloy; `write_adp`, LAMMPS adp, `.adp` suffix; both match ASE's EAM
calculator), `data` (per-alloy training sets:
first + final ionic step per work, E0, neighbour lists, reference energies μ_Z), `train`
(losses on E/atom, F, S and C_ij from stress slopes at the 24 final frames; Adam and
full-batch L-BFGS; `load(path, kind="adp")` upgrades an EAM checkpoint, u and w seeded small
because u = w = 0 is a stationary point of every loss term). `src/mtpot/splits.py`: shared
alloy splits (`binaries`, `test`, `rest`, `all`), minus `EXCLUDED` (`63`, `64`, `88`).
Scripts: `build_dataset.py` (→ `data/processed/train/<id>.npz`, ~6 min, 21 MB),
`train_eam.py` (→ `models/<tag>.pt/.log/.json`; `--init` warm start; `--model adp`). Torch: CPU wheel from
`https://download.pytorch.org/whl/cpu` (the PyPI wheel pulls GBs of CUDA libraries).

Relax trajectories: `data.build_relax` → `data/processed/train_relax/<id>.npz` (202 runs,
~11 frames each: log-spaced BFGS steps incl. the ideal start and last step, plus the final
SCF; 7 MB, ~20 s). Energies enter only as differences within `energy_group` (BFGS frames;
the final SCF is its own group), so mixed cut-offs/smearing do not matter. `train_eam.py
--relax` adds the relax runs of the *training* alloys; those of validation alloys are only
reported (`val_relax_*`): their relaxed structures are the elastic reference cells, so
training on them would leak. Loss weights `rE/rF/rS` (default rS = 0.25, i.e. σ 0.2 GPa:
relax stresses carry setting-dependent offsets of ~0.2–0.3 GPa).

```bash
python scripts/build_dataset.py data/raw -j 4
python scripts/build_dataset.py data/raw --relax -j 4
python scripts/train_eam.py --train rest --val test --relax --optimizer lbfgs --epochs 300 \
    --eval-every 50 --tag eam_v1
```

`src/mtpot/elastic/workflow.py` (Phase 3): `TorchCalculator` (ASE calculator for the torch
model), `elastic_constants(reference, calc)` (24 thermo_pw strains, fractional coordinates
fixed, BFGS ionic relaxation at fixed cell, stress-strain slopes, zero-strain intercept),
`predict(atoms, model, mode)`: `dft_cell` (strain the DFT reference cell, as the labels),
`relax_ions`, `own` (relax cell + ions from the ideal SQS first: deployment).
`scripts/elastic_eval.py model.pt --ids test --mode dft_cell -j 4` → `models/<tag>_elastic_<mode>.csv`
(1 torch thread per alloy; minutes per alloy when the model is near an instability).
`src/mtpot/io/thermo_pw.py`: `tpw_reference` (reference cell from the head of an elast file).

Tests (`python -m pytest`; `-m "not data"` for the data-free subset):
- `tests/test_parser_synthetic.py`: parser units, signs, final flags, C_ij refit and CLI round trip
  on synthetic thermo_pw outputs with known answers (`tests/synthetic_tpw.py`), including a
  run killed mid-relaxation (its last work must not be `final`) and BFGS stopped at `nstep`.
- `tests/test_qc.py`: QC sweep, flags and cache on synthetic alloys and cubes.
- `tests/test_density.py`: neighbour pairs vs brute force, basis cutoff, recovery of known
  f_Z from synthetic triclinic superpositions (incl. monotone), harmonic analysis.
- `tests/test_eam.py`: autograd forces/stress vs finite differences, batching, monotone
  splines, setfl export vs ASE's EAM calculator.
- `tests/test_potential_data.py`: training-set builder and C_ij-from-stress on synthetic
  thermo_pw outputs (caught a swapped e4/e6 column).
- `tests/test_relax.py`: relax parser vs ASE's reader, relaxed structure = elastic reference,
  crashed run (data tests).
- `tests/test_elastic.py`: workflow with the torch calculator vs ASE's EAM calculator (setfl),
  perfect-bcc symmetry (no relaxation, cubic tensor), calculator vs model.
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
   ideal positions. **The vc-relax runs fix much of this** (ideal → relaxed: energy drop median
   78 meV/atom, max 274; initial max |F| median 1.1, max 2.0 eV/Å; pressure −20…+28 GPa; volume
   −12…+18 %; displacements up to 1.4 Å). ±3–5 % volume scans would still help.
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
   **Cause (from the relax runs):** each reference cell was relaxed with different settings than
   its elastic run. Mean residual pressure by (relax → elastic) settings: 55/550 G0.009 → same
   +0.23 GPa (Pulay-like offset of a fresh basis), 55/550 G0.007 → G0.009 +0.08, 60/500 G0.007
   → 55/550 G0.009 −0.16, 60/500 G0.007 → MV −0.32 GPa. A systematic offset, not a data error.
8. **Large non-cubic part (> 8%) in 17 alloys, all Ti-rich (Ti ≥ 31%)**, up to 20% (`88`
   Ta4Ti28): physical (soft bcc Ti), raises the C_ij error floor for those alloys.

9. **Relaxations vs elastic reference cells** (`data/processed/relax_qc.csv`): in 193 of 202
   vc-relax runs the relaxed structure *is* the elastic reference (positions 5e-7 Å, cell
   3e-6 Å, same species order everywhere). Exceptions: `91 170 173 174 195 200` (stopped at
   100 steps) and `179` (crashed) differ by 1–24 mÅ (relaxation finished elsewhere);
   **`87` (Nb4Ti28) and `88` (Ta4Ti28) differ completely** (positions 5 Å, cell 0.44–0.48 Å):
   their relaxations left bcc while the elastic runs used another cell — ask the user.
10. **DFT label precision**: 8 alloys have complete elastic runs at two settings (60/550 G0.007
   vs 55/550 G0.009 or MV). |ΔC11| mean 3.9 GPa (max 11.6, `100`), |ΔC12| 1.5 (max 3.3),
   |ΔC44| 2.2 (max 4.8), |ΔC′| 2.6 (max 7.5). A 5–10 GPa target is close to this floor.
11. **Reference cells that left bcc** (deviatoric strain of the elastic reference cell vs the
   ideal bcc supercell, `|dev ε|`): 217 alloys ≤ 3 %, but **all five group-IV-only alloys**
   (no Nb/Mo/Ta: `04` Ti20Zr12, `07` Ti16Zr16, `110` Hf4Ti14Zr14, `114` Hf8Ti12Zr12,
   `118` Hf12Ti10Zr10) have 12.4–13.9 % — cubic-cell edges e.g. 3.06/3.59/3.58 Å and a 94.5°
   angle (`07`), a Burgers/α″-type distortion: bcc is mechanically unstable there and the
   vc-relax went to another structure. Their DFT C tensors are positive definite (min
   eigenvalue 27–36 GPa): the labels are for that structure, not bcc. Next tier, all with only
   12.5 % Nb/Ta: `170` 9.5 %, `197` 8.3 %, `88` 6.9 %, `180` 5.9 %, `87` 5.8 %, `185` 5.3 %.
   Only `88` has a negative DFT C eigenvalue (−7.1 GPa).
   **Excluded from all splits (user decision, mechanically unstable): `63` Mo8Ta8Ti8Zr8,
   `64` Nb8Ta8Ti8Zr8, `88` Ta4Ti28** (`splits.EXCLUDED`; none is in `test`, so `test` keeps 23
   alloys). `63`/`64` look unremarkable in these QC numbers (|dev ε| 1.5/0.8 %, C positive
   definite); `87` and the group-IV-only alloys stay in.

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

## Phase 2 status (pure EAM: unstable at DFT structures; see below)

- Data checks: DFT forces are consistent with energy changes between ionic steps
  (corr 1.000); C_ij from stress slopes at final frames = thermo_pw within 0.008 GPa.
  DFT forces are tiny (RMS 0.014 eV/Å, ionic steps move atoms 0.001–0.003 Å): the force
  information is "these SQS displacements are in equilibrium".
- One epoch over 232 alloys: 16 s (Adam batch) / one full-batch L-BFGS evaluation: ~16 s.
- Adam (lr 1e-2) is a poor fit for this ~640-parameter smooth problem; L-BFGS works better.
- Single alloy (`01`, 300 L-BFGS iterations): E 0.4 meV/atom, S 0.08 GPa, C_ij within
  2–6 GPa, but F 0.11 eV/Å — forces are the hardest target even for one alloy.
- 23 binaries (300 it.): train E 1.0 meV/atom, F 0.24 eV/Å, S 0.32 GPa, C11/C12/C44 MAE
  19/24/22 GPa; no transfer to multi-element alloys (E ~1 eV/atom off): binaries train only
  6 of the 15 unlike pair functions.
- 60 random alloys (100 it.): held-out C11/C12/C44 MAE 115/68/31 GPa, F 0.47 eV/Å — far
  from converged.
- **C_ij at DFT positions is not the target**: the binaries model had C_ij errors of ~20 GPa at
  DFT positions on alloy `01`, but its own relaxed-ion C′ is −80 GPa (DFT +42) and 5 of 24
  strained relaxations do not converge. Judge models with `elastic_eval.py`, not the training C.
- Long run `eam_v1` (user-approved; 232 training alloys + 180 relax runs, 87/88 excluded, warm
  start `bin300`, ~40 s/iteration) **stopped at iteration ~100**: held-out C_ij at DFT positions
  65/38/28 GPa, but the Phase 3 workflow gave C11 = −220 GPa for `140`. At the DFT reference the
  model's force-constant matrix has 15–41 negative eigenvalues (down to −15 eV/Å²); its own
  ionic relaxation moves atoms ~1.1 Å and gains 0.9 eV/atom (mostly pair energy): bcc is a
  saddle point of the fitted EAM.
- `mtpot.potential.hessian` adds a stability penalty (eigenvalues below kmin) and relaxed-ion C
  by implicit differentiation (= workflow within 0.5 %); `train_eam.py --hessian-train N
  --hessian-val`. ~3 s per 40-atom alloy (1 thread), so only a subset gets it.
- **Pure EAM cannot fit these data and stay stable.** The untrained model is ~stable (0–5
  modes ≈ −0.05 eV/Å²); training makes it unstable. The best single-alloy EAM fit (`01`, C at
  DFT positions within 2–6 GPa) has 36/96 negative modes (−5.8 eV/Å²); with the stability
  penalty from the start, `01` still has 13 negative modes after 200 iterations and C11/C12/C44
  128/68/21 vs DFT 197/113/30. Consistent with Phase 1's directional (l = 4, group-dependent)
  residual: angular terms are needed. Proposed next: ADP (EAM + dipole/quadrupole terms;
  LAMMPS `pair_style adp`, ASE's EAM calculator reads ADP files for validation).

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
- ~~Are the vc-relax outputs available?~~ Yes for 203 alloys (+12 earlier elastic runs); see
  "Data". (`87`/`88` relax runs end off their elastic reference cells, data issue 9; `88` is
  excluded, `87` stays in.)
- Target accuracy for C_ij (provisional: MAE ≤ 5–10 GPa vs DFT)?
- Is LAMMPS export required, or is an ASE/PyTorch calculator enough?
