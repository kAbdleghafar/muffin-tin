# Phase 1: atom-centred density functions from the 23 binaries

Code: `src/mtpot/density/`, `scripts/density_phase1.py` (prepare / fit / angular / report).
Data: rho and ELF cubes of the 23 binary alloys (training) and 23 held-out alloys
(6 random each with 3, 4, 5 elements + all 5 six-element alloys, incl. `140`, `150`).

## 1. Radial profiles and core mask (`profiles_rho.png`, `profiles.csv`)

- For r ≤ 1.0 Å every element's spherically averaged density is the same in every alloy
  and on every atom (spread < 1.3 %; < 0.5 % for r ≤ 0.8 Å), in training and held-out
  alloys alike: the core is atomic and transferable.
- Beyond 1.2 Å the profiles depend on the environment (neighbours' tails).
- ELF has its minimum at 1.0–1.2 Å for all six elements (semicore / valence boundary).
- **Core mask: 1.0 Å for all species.** Ta's 4f shell peaks at 5.7 e/bohr³ (r ≈ 0.3 Å).

## 2. Fitting ρ ≈ Σᵢ f_Z(|r − Rᵢ|) (`fit_metrics.csv`, `f_Z_binaries.png`)

20,000 random grid points per alloy, errors outside the core mask, relative to the local
mean density (medians over alloys):

| f_Z form | train (23 binaries) | held-out (23) | held-out, fitted on itself |
|---|---|---|---|
| cubic spline, monotone, r_c = 6 Å (adopted) | 3.2 % | 3.4 % | 3.1 % |
| cubic spline, unconstrained | 3.1 % | 3.4–3.5 % | |
| sums of exponentials (smooth, convex) | 5.0 % | 5.3 % | |

- Transfer from binaries to 3–6-element alloys costs ~0.3 percentage points; R² ≈ 0.99.
  Interstitial charge (> 1.6 Å from any atom) is reproduced within ~0.5 % (−2 % for
  Zr-rich alloys: Zr appears in only two binaries).
- The cut-off radius barely matters (5.5–6.5 Å give the same errors).
- A uniform background term Σ_Z (N_Z/V) β_Z comes out ≈ 0 (≤ 0.05 e/atom): the missing
  interstitial charge is not uniform.

## 3. The decomposition is ill-posed at neighbour distances (`f_Z_compare.png`)

Fits of near-equal quality agree for r < ~2.3 Å but differ completely at the bcc neighbour
shells (2.8, 3.25, 4.6, 5.4 Å). The EAM embedding density they imply at atom sites,
ρ̄ᵢ = Σⱼ f_Zj(rᵢⱼ), has mean 0.0037 (monotone), −0.0073 (unconstrained) and 0.0122
(exponential) e/bohr³, with correlations of 0.36 and 0.02 between fits. ρ̄ᵢ is the
neighbours' density at a nucleus, i.e. inside the masked core, which the grid never
constrains. **The DFT density does not determine f_Z at the distances EAM uses.**

## 4. Angular content (`angular_summary.csv`)

Real Y_lm (l ≤ 4) least-squares fits on thin shells (1.2, 1.4, 1.6 Å) around every atom,
with a radial-slope term; K4 = x⁴+y⁴+z⁴−3/5 (> 0 toward ⟨100⟩, < 0 toward ⟨111⟩):

- DFT ρ is strongly cubic-anisotropic (l = 4: 4–12 % of the shell mean; K4 −0.2 to −0.7,
  i.e. toward the ⟨111⟩ nearest neighbours). The spherical superposition reproduces this
  almost exactly (median K4 −0.48 vs −0.48 at 1.6 Å): it comes from bcc geometry.
- The residual is small (1–3 % of the local density) but mostly angular: l = 4 ≈ 1.2–1.7 %,
  l = 1, 2 ≈ 0.6–0.9 %, spherical part ≈ 0.3–0.4 %.
- Its cubic sign is **systematic by group**: Mo, Nb, Ta carry extra density toward ⟨111⟩
  (residual K4 < 0 in ≥ 88 % of atoms), Ti and Hf toward ⟨100⟩ (> 0 in ≥ 93 %); Zr leans the
  same way as Ti and Hf (65–87 %). This is the t2g/eg (d-band filling) difference between
  bcc-stable group 5–6 and bcc-soft group 4 metals, the physics behind C′ and C44.
- ELF: l = 4 content 5–11 %; K4 < 0 at 1.2 Å for Mo/Nb/Ta, > 0 for all at 1.6 Å.

## Recommendations for Phase 2

1. Do not freeze f_Z from the density fit; learn it from energies, forces and stresses.
   Use rho only where it is informative: as a soft prior on f_Z for r < 2.3 Å, and/or as
   per-atom targets such as the shell-mean density around each atom (an effective-medium
   style local density).
2. Start from pure EAM (Cauchy pressure is positive in all 255 alloys) and test one angular
   channel as an ablation: the group-dependent cubic signal is small but systematic and
   tied to C′/C44. MEAM's l = 2 partial density vanishes in ideal bcc but appears under
   shear strain, which is where it would act.
