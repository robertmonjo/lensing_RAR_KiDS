MICE_n-body/data
================

KiDS-1000 reference data (Brouwer et al. 2021)
  Fig-9_RAR-KiDS-isolated_Massbin-{1..4}.txt
    columns: g_bar  g_obs  g_obs_16  g_obs_84  [m/s^2]  r_Mpc
    rows from large to small radius. g_bar = G M_gal / r^2 with the mean M_gal of each
    stellar-mass bin (M* (1 + f_cold), Boselli et al. 2014), so r = sqrt(G M_gal / g_bar).

MICE2 products (copied here and to ../../data by scripts/run_mice_pipeline.sh)
  rar_stacked_b21_ztrue_bin{1..4}.txt     stacked lensing RAR, isolation with true redshifts
  rar_stacked_b21_photoz_bin{1..4}.txt    the same with redshifts perturbed by N(0, 0.02 (1 + z))
    columns: r_Mpc  log10_gbar  log10_gobs                (04_lensing_stack.py)
  rar_band_b21_bin{1..4}.txt              isolation band (envelope of the two stacks)
    columns: r_Mpc  log10_gbar  log10_gobs_lo  log10_gobs_hi   (08_b21_band.py)
  morph_{sersic_late,sersic_early,color_blue,color_red}_bin{b}.txt
    columns: log10_gbar  log10_gobs  log10_gobs_16  log10_gobs_84  r_Mpc   (07_morphology_split.py)
    mass-matched subsets; a file is absent when the subset is empty in that mass bin
    (after mass matching, the colour subsets contain lenses of the first two mass bins only)
  morph_*_allbins.txt                     the four mass bins of each subset together
  morph_gbar_r_mice.txt                   g_bar-radius map of the subsets (09_morph_gbar_r.py)
  morph_mstar_mice.csv                    median log10 M* of the mass-matched subsets
  mice_morph_threshold_sensitivity.csv    change of the subset stacks with the B/T and g-r cuts, per
                                          mass bin, and spread between mass-matching seeds 1-4

g_bar follows Brouwer et al. (2021): g_bar = G M_gal / R^2 with M_gal = M* (1 + f_cold),
log10 f_cold = -0.69 log10(M*/Msun) + 6.63.

The MICE2 catalogues themselves (about 25 GB) are downloaded from CosmoHub into
MICE2_b21/ by the pipeline and are not part of the repository.
