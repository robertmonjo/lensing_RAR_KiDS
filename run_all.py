"""
run_all.py -- reproduce every figure and table of the paper from scratch.

Runs (in ./, writing to ./outputs):
  make_fig1_kids_rar.py     -> fig1_kids_rar.png/.pdf          (Fig. 1)
  make_figA1_regimes.py     -> figA1_regimes.png/.pdf          (Fig. A.1)
  make_reference_systems.py -> reference_systems.csv + sparc_per_galaxy.csv
                               + xcop_per_cluster.csv  (Fig. A.2 reference stars;
                               SPARC = 50 of 61 McGaugh 2007 galaxies, >=4 points)
  make_figA2_landscape.py   -> figA2_landscape.png/.pdf        (Fig. A.2; reads the CSVs above)
  make_tables.py            -> tab_model_comparison.csv, tab_mice_comparison.csv
  xcop_rnei_nsig.py         -> xcop_rnei_nsig.txt  (X-COP n_sigma=0.918 from the radially-varying
                               r_nei model; a scalar s is NOT the valid model for X-COP -- read
                               the script header.  make_tab_regimes uses this value for X-COP)
  make_tab_regimes.py       -> tab_regimes.csv + tab_regimes_body.tex  (Table A.1; all columns
                               from baryonic mass M_bar=M_star(1+f_cold), with 1-sigma intervals
                               on s, xi_s^2, g_s/a_0 and a final n_sigma=sqrt(reduced chi2) column,
                               the HMG fit's deviation from each system's RAR data at the row's s)

Everything reads only from ./data (self-contained; no absolute paths, no
external directories). Just run:  python run_all.py
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

STEPS = [
    "make_fig1_kids_rar.py",       # Fig. 1  (main body)
    "make_figA1_regimes.py",       # Fig. A.1 (appendix)
    "make_reference_systems.py",   # must precede figA2: writes the CSVs it reads
    "make_figA2_landscape.py",     # Fig. A.2 (appendix)
    "make_tables.py",
    "xcop_rnei_nsig.py",           # X-COP r_nei-model n_sigma=0.918 (scalar-s is invalid there)
    "make_tab_regimes.py",         # Table A.1; reads tab_model_comparison + reference_systems + xcop_rnei
]

# Auxiliary/exploratory scripts deliberately NOT in STEPS (run by hand if needed):
#   forward_lensing_delta_sigma.py
#                           -- Appendix A.4 validation: Abel-projection vs SIS kernel.
#                              Reproduces the g_fwd/g_SIS ratios and chi2_nu values
#                              reported in Appendix A.4.  Writes forward_lensing_results.csv.
#                              Runtime ~5 min (Abel quadrature, 4 bins x 100 grid pts).
#                              Run: python forward_lensing_delta_sigma.py
#   10_lin_chi2_table2.py   -- recomputes chi2_nu_lin for all Table 2 rows (N=30/N=40,
#                              HMG/MICE/MOND); prints to stdout only (no output files).
#                              Values cross-check the LIN column of tab:mice_comparison.
#   make_fig_gs_fields.py   -- diagnostic; not in manuscript
#   make_fig_xi_req.py      -- diagnostic; not in manuscript
#   xcop_s_rnei_joint.py    -- robustness test: 2D (s,r_nei) joint fit + chi2(s) profile for X-COP
#                              Outputs: xcop_s_rnei_profile.png, xcop_s_rnei_grid.csv,
#                              xcop_s_rnei_ridge.csv.  Conclusion: s is unconstrained when
#                              r_nei is free (flat chi2 profile for s in [0.7,1.6]); the r_nei
#                              model (s=1) is the parsimonious correct parametrisation.

def main():
    print("=" * 70)
    print("Reproducing all paper figures (png/pdf) and tables (csv)")
    print("output directory:", OUT)
    print("=" * 70)
    for step in STEPS:
        print(f"\n>>> {step}")
        r = subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=HERE)
        if r.returncode != 0:
            print(f"!!! {step} FAILED (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\n" + "=" * 70)
    print("DONE. Files in ./outputs:")
    for f in sorted(os.listdir(OUT)):
        print("  ", f)

if __name__ == "__main__":
    main()
