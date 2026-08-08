===============================================================================
 REPRODUCIBLE PIPELINE  --  Weak-lensing RAR of KiDS-1000 isolated galaxies (HMG)
===============================================================================

This folder regenerates, FROM SCRATCH and with NO hidden dependencies, every
figure and table used in the paper (main.tex on Overleaf).  All code reads only
from ./data and writes only to ./outputs.  No absolute paths, no external
directories, no cluster-specific setup.


-------------------------------------------------------------------------------
1. QUICK START
-------------------------------------------------------------------------------

    cd scripts_replicable
    python run_all.py

Requirements: Python >= 3.9 with numpy, scipy, matplotlib.

    pip install numpy scipy matplotlib

That single command produces every figure (PNG + PDF) and every CSV/TeX table in
./outputs (see section 3).  Nothing else needs to be run.


-------------------------------------------------------------------------------
2. WHICH SCRIPT DOES WHAT   (run order: just run_all.py)
-------------------------------------------------------------------------------

    run_all.py            <-- RUN THIS.  Orchestrates the four producers below.

    hmg_model.py              Shared library (imported, never run on its own):
                              physical constants, the HMG / MOND / CDM model
                              functions, the KiDS-1000 data (Brouwer+2021), and
                              the ./data loaders.

    make_fig1_kids_rar.py     -> outputs/fig1_kids_rar.{png,pdf}      [paper Fig. 1, body]
                              KiDS-1000 RAR: four stellar-mass bins (top) plus
                              morphological subsets (bottom, with HMG and CDM curves).

    make_figA1_regimes.py     -> outputs/figA1_regimes.{png,pdf}      [paper Fig. A.1]
                              The two competing terms of xi_s^2 (regime figure).

    make_reference_systems.py -> outputs/reference_systems.csv
                              -> outputs/{sparc,xcop,hiflugcs}_per_object CSVs
                              External reference systems drawn as gold stars in
                              Fig. 3(a).  For each system it FITS BOTH s<->1/s HMG
                              branches (the s<1 neighbourhood branch and the s>1
                              Hubble branch) and r_eq:
                                SPARC     = 50 of 61 McGaugh (2007) galaxies (>=4 pts)
                                HIFLUGCS  = 10 clusters (Li 2023 / Monjo & Banik 2025)
                                X-COP     = 12 clusters (Eckert 2022)
                                gal-gal WL= Mistele (2024) stacked velocities
                                Milky Way = ../vertical_gravity_hmg two-branch fit
                              The mirror branch is FITTED, never assumed = 1/s (exact
                              only in the deep limit; e.g. gal-gal WL s=2.54 -> mirror
                              0.36, not 0.39).  A `degenerate` flag marks whether the
                              two branches are statistically separable (Wilks Delta chi2
                              test; see the DELTA_DEG block in the script).  Reads
                              data/external_reference/.  MUST run before make_figA2.

    make_figA2_landscape.py   -> outputs/figA2_landscape.{png,pdf}   [paper Fig. A.2]
                              Panel (a) s_equal(r,M) regime map (with a g_s/a_0 colour
                              bar, capped at g_s^max) + panel (b) chi2_nu(s) landscape,
                              sharing the s axis.  Panel (b) also overlays a faint
                              per-object chi2_nu(s) haze for the two cluster samples.
                              Gold stars from reference_systems.csv etc.: BOTH branch
                              stars are filled where the branches are statistically
                              indistinguishable (Delta chi2 < 4, i.e. < 2 sigma:
                              Milky Way, gal-gal WL), otherwise the disfavoured branch
                              is drawn open (SPARC, HIFLUGCS, X-COP).

    NOTE: figure and script names follow the paper numbering (Fig. 1 in the body;
    Fig. A.1 and A.2 in the appendix); see SCRIPT_NAMING.md.  data/morph_gbar_r_mice.txt
    (per-morphology r(g_bar) from the isolated MICE lenses) is a required input for
    make_fig1_kids_rar/make_tables.

    make_tables.py            -> outputs/tab_model_comparison.csv
                              -> outputs/tab_mice_comparison.csv
                              Both paper tables, computed from ./data.

    xcop_rnei_nsig.py         -> outputs/xcop_rnei_nsig.txt
                              X-COP cluster n_sigma = 0.918 from the CORRECT HMG
                              "r_nei" model (eps0(r) varies radially with the gas
                              density), reproduced from the committed FITS extract
                              data/external_reference/xcop_profiles.txt.  A single
                              scalar s is NOT the valid model for the X-COP hydrostatic
                              profiles (it gives 2.731); see the script's header for
                              the full explanation.  make_tab_regimes.py reads this
                              value for the X-COP row of Table A.1.

Dependency order (handled by run_all.py): make_fig1_kids_rar and make_figA1_regimes
are standalone; make_reference_systems must run before make_figA2_landscape (which
reads reference_systems.csv); make_tab_regimes must run after make_tables and
make_reference_systems (it reads tab_model_comparison.csv and reference_systems.csv).
make_fig1_kids_rar/figA2_landscape/tables/tab_regimes all import hmg_model.py for the
shared model and the KiDS data.  Old (uninformative) script names are mapped in SCRIPT_NAMING.md.


-------------------------------------------------------------------------------
3. OUTPUTS  (./outputs)  and their main.tex references
-------------------------------------------------------------------------------

    fig1_kids_rar.png / .pdf         Fig. 1   (label fig:kids_rar)
    figA1_regimes.png / .pdf         Fig. A.1 (label fig:hmg_regimes)
    figA2_landscape.png / .pdf       Fig. A.2 (label fig:combined_fig45)
    reference_systems.csv           Fig. 3(a) gold stars: SPARC (N=50), HIFLUGCS,
                                    X-COP (r_eq, s, and mirror-branch s_alt)
    sparc_per_galaxy.csv            per-galaxy (r_eq, s_fill, s_open) for the 50
                                    SPARC galaxies (the small stars in Fig. 3a)
    xcop_per_cluster.csv            per-cluster r_eq for the 5 X-COP clusters
    tab_model_comparison.csv        Table tab:model_comparison
    tab_mice_comparison.csv         Table tab:mice_comparison
    tab_regimes.csv                 Table tab:regimes (Table A.1) as data
    tab_regimes_body.tex            Table tab:regimes LaTeX body (rows)


-------------------------------------------------------------------------------
4. INPUT DATA  (./data)  -- all small text files, committed for self-containment
-------------------------------------------------------------------------------

    Fig-8_RAR-KiDS-isolated_{Sersic,Color}bin_{1,2}.txt
        KiDS-1000 morphological excess-surface-density profiles
        (Brouwer+2021, Fig. 8, public data release).

    rar_band_b21_bin{1..4}.txt
        MICE LambdaCDM RAR band per stellar-mass bin (z_true / photo-z
        isolation limits).  PRODUCED BY ../MICE_n-body (see its README);
        copied here as a committed input so the figures reproduce without
        re-running the N-body pipeline.

    morph_{sersic_late,sersic_early,color_blue,color_red}_allbins.txt
        MICE morphological RAR (same provenance as the band files).

    external_reference/McGaugh2007.txt
        SPARC rotation curves (McGaugh 2007): Name, R[kpc], Vobs, Vgas, Vst,
        Qmax, Qpop.  61 galaxies; make_reference_systems.py fits the 50 with
        >=4 valid points.  Input for the SPARC reference star in Fig. 3(a).
    external_reference/clusterRAR.dat
        HIFLUGCS cluster RAR (Monjo & Banik 2025): Name, z, Radius, log(gbar),
        log(gtot), err_low, err_up.  Input for the HIFLUGCS reference star.
        Both files copied from ../cluster_galaxy_relation_apj/repo/data.

    The next four files feed ONLY the n_sigma (goodness-of-fit) column of Table A.1;
    each carries a comment header with its exact provenance:
    external_reference/udg_manceraPina2022.txt
        6 gas-rich UDGs (Mancera Pina 2022) as used in Monjo (2026);
        agc, log10_mbar, r_kpc, v_obs, sig_plus, sig_minus, v_newton.
    external_reference/mistele_gg_wl.txt
        Galaxy-galaxy weak-lensing rotation curves (Mistele 2024, Table 1) as used
        in Monjo (2025); 4 mass bins x 5 radii.
    external_reference/mw_chi2nu_nbar4.txt
        Milky Way HMG reduced chi2 (nbar4), from ../vertical_gravity_hmg.  The MW fit combines
        radial + vertical constraints (outside the rotation-RAR framework), so its n_sigma is
        taken from this published chi2nu rather than recomputed; see the file header.
    external_reference/xcop_rar_perpoint.txt
        X-COP clusters per-point RAR (Eckert 2022) from ../hydrostatic_equilibrium_xcop,
        R>=1000 kpc window; cluster, r_kpc, gbar_si, gobs_si, sigma_gobs_si (12 clusters).

The KiDS-1000 stellar-mass-bin RAR (four bins) and the MICE NFW fallback band
are hard-coded in hmg_model.py from the Brouwer+2021 published values.


-------------------------------------------------------------------------------
5. THE MODEL  (hmg_model.py)
-------------------------------------------------------------------------------

    HMG   Hyperconical Modified Gravity (Monjo 2025, ApJ 982, 70;
          Monjo & Banik 2025, PASA).
          - gobs_hmg: full formula, xi_s^2 = 1/s^3 + v_H^2 / (12 v_N^2).
          - g_s_dl / hmg_pred_dl: morphological deep limit, xi_s^2 = 1/s^3,
            q argument x = s^{3/2}  (no spurious 1/sqrt(3) factor).
    MOND  McGaugh interpolation, a0 = 1.20e-10 m/s^2 fixed.
    CDM   truncated NFW (Moster+2013 SHMR, Dutton & Maccio 2014 concentration)
          + two-halo power law (Tinker+2010 bias), one global amplitude xi0.
    const-g0  reduced 1-param submodel  gobs = sqrt(gbar (gbar + g0)).


-------------------------------------------------------------------------------
6. NOTES ON THE TABLES
-------------------------------------------------------------------------------

- All primary HMG / MOND / CDM chi2 values are computed here and match main.tex.
- The MICE LambdaCDM chi2 entries in tab_mice_comparison.csv are LITERATURE
  values (Brouwer+2021 / Crocce+2015), flagged in the 'source' column; they are
  not re-derived here (that is the job of ../MICE_n-body).
- chi2_nu convention follows the paper: chi2/N per subsample, except the single
  free-parameter outer-radii HMG fit, where the paper reports chi2/(N-1).
- Table A.1 last column n_sigma = sqrt(reduced chi2) of the HMG prediction vs g_obs,
  computed uniformly in log-acceleration space at the row's s from each system's own
  RAR data (external_reference/).  Single stacked curves (KiDS, MW, gal-gal WL, UDGs)
  are fit jointly; multi-object samples (SPARC, HIFLUGCS, X-COP) are fit per object and
  the median n_sigma is reported.  All systems land within ~0.6-2.7 sigma.
- A few SECONDARY entries (const_g0 column; morphological MOND) may differ from
  the current main.tex printout by < 10%.  This pipeline is the source of truth;
  those manuscript cells should be synced to the CSV.


-------------------------------------------------------------------------------
7. THE MICE N-BODY SIDE  (../MICE_n-body)
-------------------------------------------------------------------------------

The MICE band and morphological text files in ./data are outputs of the
independent, also-reproducible pipeline in ../MICE_n-body (scripts 01-08).
Its only manual step is downloading the MICE2 catalogue from CosmoHub; then
`bash scripts/run_photoz_pipeline.sh` regenerates the band/morph files, which
can be copied back into ./data to refresh the figures.  See
../MICE_n-body/README_pipeline.md.
