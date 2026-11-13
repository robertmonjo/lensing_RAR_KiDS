"""
run_all.py -- reproduce every figure and table of the paper from the files in ./data.

  python run_all.py            all steps below (about one hour on one core)

Steps, in order (outputs in ./outputs):
  make_tables.py                     Table 1                      tab_model_comparison.csv,
                                                                  tab_morph_pairs.csv
  morph_threshold_sensitivity.py     Sect. 2.1 proxy thresholds   morph_threshold_sensitivity.csv
  make_tab_mice_comparison.py        Table 2                      tab_mice_comparison.csv
  xcop_s_rnei_joint.py               X-COP joint (s, r_nei) fits  xcop_s_rnei_grid.csv
  fit_external_udg.py                gas-rich and MIGHTEE UDGs    udg_systems.csv, mightee_per_galaxy.csv
  make_reference_systems.py          external systems             reference_systems.csv, *_per_*.csv
  make_tab_regimes.py                Table A.1                    tab_regimes.csv, tab_regimes_body.tex
  fit_gastaldello_s.py               X-ray groups of Fig. A.2     gastaldello_s_fit.csv
  make_fig1_kids_rar.py              Fig. 1                       fig1_kids_rar.png/.pdf
  make_figA1_regimes.py              Fig. A.1                     figA1_regimes.png/.pdf
  make_figA2_landscape.py            Fig. A.2                     figA2_landscape.png/.pdf
  forward_lensing.py                 Appendix A.4, Table A.2      forward_lensing_table_A2.csv,
                                                                  forward_lensing_kernel.csv
  forward_projection_convergence.py  Appendix A.4 convergence     forward_projection_convergence.json

The MICE2 products in ./data are produced from the MICE2 catalogue by
MICE_n-body/scripts/run_mice_pipeline.sh (requires a CosmoHub account).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

STEPS = [
    "make_tables.py",
    "morph_threshold_sensitivity.py",
    "make_tab_mice_comparison.py",
    "xcop_s_rnei_joint.py",
    "fit_external_udg.py",
    "make_reference_systems.py",
    "make_tab_regimes.py",
    "fit_gastaldello_s.py",
    "make_fig1_kids_rar.py",
    "make_figA1_regimes.py",
    "make_figA2_landscape.py",
    "forward_lensing.py",
    "forward_projection_convergence.py",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for step in STEPS:
        print(f"\n>>> {step}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=HERE,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
        if r.returncode != 0:
            print(f"!!! {step} failed (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\nDone. Files in ./outputs:")
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
