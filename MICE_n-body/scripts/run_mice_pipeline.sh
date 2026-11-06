#!/usr/bin/env bash
# run_mice_pipeline.sh -- MICE2 products used in the paper, from the CosmoHub download to
# the files read by run_all.py.
#
#   1. cosmohub_download_b21.py   MICE2 lens and source catalogue, 1024 deg^2, z <= 1.4 (~25 GB)
#   2. download_morph_fullsky.py  MICE2 bulge fraction and rest-frame g-r (full sky)
#   3. 02_select_isolated.py      isolated lenses (Brouwer et al. 2021 criteria), with true and
#                                 with perturbed redshifts z + N(0, 0.02 (1 + z))
#   4. 04_lensing_stack.py        stacked lensing RAR of both isolated samples (z_l < 0.50)
#   5. 08_b21_band.py             isolation band (envelope of the two stacks)
#   6. 07_morphology_split.py     morphological and colour subsets (B/T = 0.5, g-r = 0.6)
#   7. 09_morph_gbar_r.py         g_bar-radius map of the subsets, also with the thresholds moved
#                                 (B/T = 0.45, g-r = 0.55, 0.65) and with matching seeds 2-4
#   8. copy of the products to MICE_n-body/data and to the top-level data/
#   9. 11_mice_lin_chi2.py        reduced chi^2 of the MICE stack against KiDS (Table 2)
#  10. mice_morph_threshold_sensitivity.py  B/T and g-r threshold sensitivity of the stacks
#
# Requires a CosmoHub account (https://cosmohub.pic.es); put the user name on line 1 and the
# password on line 2 of a credentials file. Steps 3 and 4 take several hours each on one core.
#
# Usage (from MICE_n-body/scripts):
#   bash run_mice_pipeline.sh path/to/cosmohub_credentials.txt
set -euo pipefail

CRED="${1:?usage: bash run_mice_pipeline.sh CREDENTIALS_FILE}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICE="$(dirname "$HERE")"
REPO="$(dirname "$MICE")"
BASE="$MICE/data/MICE2_b21"
ISO="$BASE/isolated"
PY="${PYTHON:-python3}"
mkdir -p "$ISO" "$BASE/stacked_ztrue" "$BASE/stacked_photoz" "$BASE/band" "$BASE/morph"

[ -f "$BASE/mice2_b21_full.fits" ] || \
  $PY "$HERE/cosmohub_download_b21.py" --credentials-file "$CRED" --output "$BASE/mice2_b21_full.fits"
[ -f "$BASE/mice2_morph_b21_fullsky.fits" ] || \
  $PY "$HERE/download_morph_fullsky.py" --credentials-file "$CRED" --output "$BASE/mice2_morph_b21_fullsky.fits"

$PY "$HERE/02_select_isolated.py" --b21 --input "$BASE/mice2_b21_full.fits" --outdir "$ISO"
$PY "$HERE/02_select_isolated.py" --b21 --photoz_sigma 0.02 --suffix _photoz \
    --input "$BASE/mice2_b21_full.fits" --outdir "$ISO"

$PY "$HERE/04_lensing_stack.py" --bin 1 2 3 4 --z-lens-max 0.50 --raw-fits "$BASE/mice2_b21_full.fits" \
    --lens-pattern "$ISO/mice2_isolated_bin{b}.fits" --outdir "$BASE/stacked_ztrue"
$PY "$HERE/04_lensing_stack.py" --bin 1 2 3 4 --z-lens-max 0.50 --raw-fits "$BASE/mice2_b21_full.fits" \
    --lens-pattern "$ISO/mice2_isolated_photoz_bin{b}.fits" --outdir "$BASE/stacked_photoz"

$PY "$HERE/08_b21_band.py" --indir_ztrue "$BASE/stacked_ztrue" --indir_photoz "$BASE/stacked_photoz" \
    --outdir "$BASE/band"
$PY "$HERE/07_morphology_split.py" --morphfile "$BASE/mice2_morph_b21_fullsky.fits" --isodir "$ISO" \
    --outdir "$BASE/morph"
$PY "$HERE/09_morph_gbar_r.py" --isodir "$ISO" --morphfile "$BASE/mice2_morph_b21_fullsky.fits" \
    --mstar-out "$REPO/data/morph_mstar_mice.csv" > "$BASE/morph_gbar_r_mice.txt"
VAR="$REPO/data/morph_map_variants"
mkdir -p "$VAR"
for v in "bt045_gr060 --bt-split 0.45" "bt050_gr055 --gr-split 0.55" "bt050_gr065 --gr-split 0.65" \
         "seed2 --seed 2" "seed3 --seed 3" "seed4 --seed 4"; do
  set -- $v
  name="$1"; shift
  $PY "$HERE/09_morph_gbar_r.py" --isodir "$ISO" --morphfile "$BASE/mice2_morph_b21_fullsky.fits" "$@" \
      > "$VAR/map_$name.txt"
done

for b in 1 2 3 4; do
  cp "$BASE/stacked_ztrue/rar_stacked_bin$b.txt"  "$MICE/data/rar_stacked_b21_ztrue_bin$b.txt"
  cp "$BASE/stacked_photoz/rar_stacked_bin$b.txt" "$MICE/data/rar_stacked_b21_photoz_bin$b.txt"
  cp "$BASE/band/rar_band_b21_bin$b.txt" "$MICE/data/"
  cp "$BASE/band/rar_band_b21_bin$b.txt" "$REPO/data/"
done
cp "$BASE"/morph/morph_*.txt "$MICE/data/"
cp "$BASE"/morph/morph_*_allbins.txt "$REPO/data/"
cp "$BASE/morph_gbar_r_mice.txt" "$MICE/data/"
cp "$BASE/morph_gbar_r_mice.txt" "$REPO/data/"

$PY "$HERE/11_mice_lin_chi2.py"
$PY "$HERE/mice_morph_threshold_sensitivity.py" --morphfile "$BASE/mice2_morph_b21_fullsky.fits" \
    --isodir "$ISO" --outdir "$BASE/sensitivity"
cp "$BASE/sensitivity/mice_morph_threshold_sensitivity.csv" "$MICE/data/"
