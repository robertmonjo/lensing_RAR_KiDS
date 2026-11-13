# Weak-lensing radial acceleration relation of KiDS-1000 isolated galaxies: a test of Hyperconical Modified Gravity

Code and data to reproduce every figure and table of the paper (R. Monjo and R. Ullah, A&A, in press, manuscript aa62683-26). The paper compares Hyperconical Modified Gravity (HMG), MOND and ΛCDM with the weak-lensing radial acceleration relation (RAR) of isolated KiDS-1000 galaxies (Brouwer et al. 2021).

The repository is self-contained: the scripts read only `./data` and write only to `./outputs`.

## Quick start

```bash
pip install -r requirements.txt      # Python >= 3.10
python run_all.py                    # about one hour on one core
```

## Scripts

| Script | Output (`./outputs`) | Paper |
|---|---|---|
| `hmg_model.py` | library: constants, HMG, MOND and CDM models, KiDS data, shared fits | — |
| `make_tables.py` | `tab_model_comparison.csv`, `tab_morph_pairs.csv` | Table 1, morphological pairs of Sect. 3.1 |
| `morph_threshold_sensitivity.py` | `morph_threshold_sensitivity.csv` | proxy thresholds, Sect. 3.1 |
| `make_tab_mice_comparison.py` | `tab_mice_comparison.csv` | Table 2 |
| `xcop_s_rnei_joint.py` | `xcop_s_rnei_grid.csv`, `xcop_s_rnei_ridge.csv`, `xcop_s_rnei_profile.png` | X-COP row of Table A.1, Fig. A.2 |
| `fit_external_udg.py` (uses `fit_hmg_udg.py`) | `udg_systems.csv`, `mightee_per_galaxy.csv` | UDG rows of Table A.1, Fig. A.2 |
| `make_reference_systems.py` | `reference_systems.csv`, per-object CSV files | external systems of Table A.1 and Fig. A.2 |
| `make_tab_regimes.py` | `tab_regimes.csv`, `tab_regimes_body.tex` | Table A.1 |
| `fit_gastaldello_s.py` | `gastaldello_s_fit.csv` | X-ray groups of Fig. A.2 |
| `make_fig1_kids_rar.py` | `fig1_kids_rar.{png,pdf}` | Fig. 1 |
| `make_figA1_regimes.py` | `figA1_regimes.{png,pdf}` | Fig. A.1 |
| `make_figA2_landscape.py` | `figA2_landscape.{png,pdf}` | Fig. A.2 |
| `forward_lensing.py` | `forward_lensing_table_A2.csv`, `forward_lensing_kernel.csv` | Appendix A.4, Table A.2 |
| `forward_projection_convergence.py` | `forward_projection_convergence.json` | Appendix A.4 (numerical convergence) |
| `run_all.py` | runs all of the above in order | — |

## Statistics

All reduced chi-squared values follow Sect. 2.4 of the paper: diagonal KiDS uncertainties and, for each data point, the one-sided width (16th or 84th percentile) on the side facing the prediction, in log10 g (`chi2_nu_log`) or in g (`chi2_nu_lin`); `chi2_nu = chi2 / (N - k)` with k the number of parameters fitted to the KiDS data. Model parameters are the minima of `chi2_nu_log`.

## Morphological subsets

The morphological and colour profiles of Brouwer et al. (2021, Fig. 8; split at Sérsic index n = 2 and at u − r = 2.5) are tabulated against g_bar = G M_bar / R², with M_bar = M* (1 + f_cold) (M_gal in Brouwer et al. 2021) and the cold-gas relation of Boselli et al. (2014). The two subsets of each split have the same stellar-mass distribution. To evaluate HMG, whose Hubble term depends on R, g_bar is converted to R with the isolated MICE2 lenses: the two MICE subsets of each split (bulge fraction B/T = 0.5; rest-frame g − r = 0.6) are reduced to the same stellar-mass distribution, and both share the map g_bar(R) = (⟨√(G M_bar)⟩ / R)² (`data/morph_gbar_r_mice.txt`, median masses in `data/morph_mstar_mice.csv`). `tab_morph_pairs.csv` gives, for each split, the separation of the two basin minima, the quiescent to star-forming ratio of the observed and predicted accelerations, and the star-forming fit evaluated on the quiescent data. `morph_threshold_sensitivity.py` repeats the fits with the maps obtained by moving the MICE2 cuts (B/T = 0.45; g − r = 0.55, 0.65) and with other seeds of the random mass matching (`data/morph_map_variants/`).

## External systems (Table A.1, Fig. A.2)

* SPARC galaxies (McGaugh 2007), HIFLUGCS clusters (Monjo & Banik 2025) and the galaxy–galaxy lensing RAR of Mistele et al. (2024): both HMG branches (s < 1 and s > 1) are fitted; samples of objects are summarised by the median and the 16th–84th percentiles.
* X-COP clusters (Eckert et al. 2022): s and the neighbourhood radius r_nei are fitted jointly per cluster. With r_nei free, chi2(s) is shallow, so s is weakly constrained.
* Gas-rich UDGs (Mancera Piña et al. 2022) and MIGHTEE UDGs (Ponomareva et al. 2021): one s fitted jointly to each sample, n_σ at that s; MIGHTEE also without J022350.6−045045, the one of the four that Adams et al. (2026) find off the baryonic Tully–Fisher relation. No per-galaxy MIGHTEE velocity uncertainty is published; ±8 km/s is assumed, which leaves the best-fit s unchanged and scales n_σ as its inverse. Fig. A.2 also shows the s that matches each MIGHTEE galaxy alone.
* Milky Way: values of the radial and vertical analysis (`data/external_reference/mw_chi2nu_MI.txt`).
* X-ray groups (Gastaldello et al. 2007): s per group from the NFW reconstruction (`data/gastaldello2007.csv`).

## Data (`./data`)

KiDS-1000 lensing RAR of isolated galaxies (Brouwer et al. 2021): the four stellar-mass bins are in `hmg_model.py`, the morphological profiles in `Fig-8_RAR-KiDS-isolated_*.txt`. Every file in `external_reference/` starts with a header giving its source. The MICE2 files (`rar_band_b21_bin*.txt`, `morph_*_allbins.txt`, `morph_gbar_r_mice.txt`, `morph_mstar_mice.csv`, `morph_map_variants/`) are produced by the MICE pipeline below.

## MICE2 pipeline (`MICE_n-body/`)

The ΛCDM comparison uses the MICE2 simulation (Crocce et al. 2015; Carretero et al. 2015) over 1024 deg², with the isolation criteria of Brouwer et al. (2021). `MICE_n-body/scripts/run_mice_pipeline.sh` downloads the catalogues from CosmoHub (free account; about 25 GB), selects the isolated lenses with true and perturbed redshifts, stacks the tangential shear of background galaxies, builds the isolation band and the morphological subsets, and copies the products to `data/`. See `MICE_n-body/README_pipeline.md`.

## License and citation

Code: MIT (`LICENSE`). Data files are redistributed from the cited sources and remain subject to their terms. Please cite the paper and the archived version of this repository (`CITATION.cff`).
