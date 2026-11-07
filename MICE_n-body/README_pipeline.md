# MICE2 ΛCDM predictions

This directory reproduces the MICE2 products used in the paper (`../data/rar_band_b21_bin*.txt`, `../data/morph_*_allbins.txt`, `../data/morph_gbar_r_mice.txt`, `../data/morph_mstar_mice.csv`, `../data/morph_map_variants/`) and the reduced χ² of the MICE prediction in Table 2. The products are already in the repository; this pipeline is only needed to regenerate them.

MICE2 (Crocce et al. 2015; Fosalba et al. 2015; Carretero et al. 2015) is distributed by CosmoHub (https://cosmohub.pic.es, free registration). The selection follows Brouwer et al. (2021, Sects. 2 and 5.3).

## Run

Requirements: `pip install -r requirements.txt` (NumPy, SciPy, Astropy and Requests; the versions listed are those of the environment that produced the files in `data/`).

```bash
cd MICE_n-body/scripts
bash run_mice_pipeline.sh path/to/cosmohub_credentials.txt   # user name on line 1, password on line 2
```

Steps 3 and 4 below take several hours each on one core; the downloads need about 25 GB.

## Steps

1. `cosmohub_download_b21.py` — lenses and sources over 1024 deg², z ≤ 1.4, with stellar and halo masses, shear and r-band magnitude.
2. `download_morph_fullsky.py` — bulge fraction and rest-frame g − r of all MICE2 galaxies.
3. `02_select_isolated.py --b21` — isolated lenses in the four stellar-mass bins of Brouwer et al. (2021) (r < 20, 0.1 < z < 0.5, no neighbour with more than 10 per cent of the lens stellar mass within 3 Mpc in projection and 1000 km/s along the line of sight), once with true redshifts and once with redshifts perturbed by N(0, 0.02 (1 + z)) (`--photoz_sigma 0.02 --suffix _photoz`).
4. `04_lensing_stack.py` — stacked excess surface density of each isolated sample from the tangential shear of background galaxies (inverse Σ_cr² weights, z_l < 0.50), converted to g_obs = 4 G ΔΣ; g_bar = G M_gal / R² with M_gal = M* (1 + f_cold) and the Boselli et al. (2014) cold-gas relation, averaged over the lenses.
5. `08_b21_band.py` — isolation band: envelope of the two stacks at each common radius (r ≥ 0.05 Mpc). Fig. 1 shows it over 0.30–1.50 Mpc.
6. `07_morphology_split.py` — morphological (B/T = 0.5) and colour (g − r = 0.6) subsets of the isolated lenses, reduced to a common stellar-mass distribution within each split (`mass_matching.py`), stacked with NFW haloes per lens.
7. `09_morph_gbar_r.py` — g_bar–radius map of the mass-matched subsets, used to evaluate HMG on the KiDS morphological profiles, and their median stellar masses; the map is also written with the cuts moved (B/T = 0.45; g − r = 0.55, 0.65) and with seeds 2–4 of the mass matching (`../data/morph_map_variants/`).
8. Copy of the products to `data/` and `../data/`.
9. `11_mice_lin_chi2.py` — χ²_ν of the z_true stack against KiDS on the outer radii (r ≥ 0.164 Mpc) of mass bins 2–4 (N = 30) and 1–4 (N = 40), with the statistics of Sect. 2.4 (Table 2).
10. `mice_morph_threshold_sensitivity.py` — change of the morphological stacks, per mass bin, when B/T is set to 0.45 or 0.55 and g − r to 0.55 or 0.65, beside the spread between seeds 1–4 of the random mass matching (`data/mice_morph_threshold_sensitivity.csv`).

## Files

`data/README_data.txt` describes every file in `data/`.
