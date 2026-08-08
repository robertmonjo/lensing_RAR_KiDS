# Weak-lensing radial acceleration relation of KiDS-1000 isolated galaxies (HMG)

Reproduction pipeline for the paper testing **Hyperconical Modified Gravity (HMG)** against the weak-lensing radial acceleration relation (RAR) of isolated galaxies in KiDS-1000, compared with MOND and ΛCDM.

Everything regenerates **from scratch with no hidden dependencies**: all code reads only from `./data` and writes only to `./outputs` — no absolute paths, no external directories, no cluster-specific setup.

## Quick start

```bash
cd scripts_replicable
pip install numpy scipy matplotlib      # Python >= 3.9
python run_all.py
```

That single command produces every figure (PNG + PDF) and every CSV/TeX table in `./outputs`.

## What each script does

| Script | Output | Paper |
|---|---|---|
| `run_all.py` | orchestrates everything below | — |
| `hmg_model.py` | *library*: constants, HMG/MOND/CDM models, KiDS data, `./data` loaders | — |
| `make_fig1_kids_rar.py` | `fig1_kids_rar.{png,pdf}` | **Fig. 1** (body) |
| `make_figA1_regimes.py` | `figA1_regimes.{png,pdf}` | **Fig. A.1** |
| `make_reference_systems.py` | `reference_systems.csv` (+ per-object CSVs) | gold stars of Fig. A.2 |
| `make_figA2_landscape.py` | `figA2_landscape.{png,pdf}` | **Fig. A.2** |
| `make_tables.py` | `tab_model_comparison.csv`, `tab_mice_comparison.csv` | model-comparison + MICE tables |
| `xcop_rnei_nsig.py` | `xcop_rnei_nsig.txt` | X-COP n_σ for Table A.1 |
| `make_tab_regimes.py` | `tab_regimes.csv`, `tab_regimes_body.tex` | **Table A.1** |

Figure and script names follow the paper numbering (Fig. 1 in the body; Fig. A.1 and A.2 in the appendix); see `SCRIPT_NAMING.md`.

- **Fig. 1** — KiDS-1000 RAR: four stellar-mass bins (top) plus morphological subsets (bottom), with HMG, MOND and CDM curves.
- **Fig. A.1** — the two competing terms of ξ²ₛ (regime decomposition).
- **Fig. A.2** — (a) the `(r_eq, s)` dynamical-regime map with a `g_s/a₀` colour bar; (b) the χ²ᵥ(s) landscape, sharing the `s` axis. Gold stars are the external reference systems from `reference_systems.csv`; **both** branch stars are filled where the `s ↔ 1/s` branches are statistically indistinguishable (Wilks Δχ² < 4), otherwise the disfavoured branch is drawn open.

### Reference systems (Fig. A.2 gold stars)

`make_reference_systems.py` fits **both** the `s < 1` (neighbourhood) and `s > 1` (Hubble) HMG branches for each external system. The mirror branch is **fitted**, never assumed = 1/s (that equality holds only in the deep limit; e.g. gal-gal WL `s = 2.54` → mirror `0.36`, not `0.39`):

- **SPARC** — 50 of 61 McGaugh (2007) galaxies (≥ 4 points)
- **HIFLUGCS** — clusters (Monjo & Banik 2025)
- **X-COP** — clusters (Eckert 2022)
- **gal-gal WL** — Mistele (2024) stacked circular velocities
- **Milky Way** — two-branch fit (radial + vertical constraints)

## Data (`./data`)

Small committed text files (self-contained). Each `external_reference/*` file carries a provenance header. Sources: Brouwer et al. (2021, KiDS-1000), McGaugh (2007, SPARC), Mistele et al. (2024), Eckert et al. (2022, X-COP), Mancera Piña et al. (2022, UDGs), and the MICE ΛCDM mocks (produced by the separate `MICE_n-body` pipeline). The four-bin KiDS RAR and the MICE fallback band are hard-coded in `hmg_model.py` from the Brouwer+2021 published values.

## The model (`hmg_model.py`)

- **HMG** — Hyperconical Modified Gravity (Monjo 2025, ApJ 982, 70; Monjo & Banik 2025, ApJ 992, 35). `gobs_hmg`: `ξ²ₛ = 1/s³ + v_H²/(12 v_N²)`; morphological deep limit via `hmg_pred_dl` (`x = s^{3/2}`).
- **MOND** — McGaugh interpolation, `a₀ = 1.20×10⁻¹⁰ m/s²` fixed.
- **CDM** — truncated NFW (Moster+2013 SHMR, Dutton & Macciò 2014 concentration) + a two-halo power law (Tinker+2010 bias), one global amplitude.

## Notes

- This pipeline is the source of truth for the primary HMG/MOND/CDM χ² values (they match the manuscript). The MICE ΛCDM χ² entries in `tab_mice_comparison.csv` are literature values (flagged in the `source` column), not re-derived here.
- Table A.1's last column `n_σ = √(reduced χ²)` of the HMG prediction vs `g_obs`, computed uniformly in log-acceleration space at each row's `s`. Single stacked curves (KiDS, MW, gal-gal WL, UDGs) are fit jointly; multi-object samples (SPARC, HIFLUGCS, X-COP) are fit per object and the median is reported.
- `make_fig_gs_fields.py` and `make_fig_xi_req.py` are **auxiliary/exploratory** diagnostics — not part of the paper and not in `run_all.py`. Run them by hand if wanted.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data files under `data/` are redistributed from their original published sources and remain subject to those sources' terms.
