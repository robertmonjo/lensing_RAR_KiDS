#!/usr/bin/env bash
# run_photoz_pipeline.sh
#
# Reproduce the B21-style MICE band (Brouwer+2021 Sect. 5.3):
#   - Run 1 (z_true): isolation with true simulation redshifts  -> lower limit
#   - Run 2 (z_photo): isolation with Gaussian photo-z noise    -> upper limit
#     sigma_z = 0.02 * (1 + z), emulating KiDS photo-z errors
#
# The two stacked mean signals are combined by 08_b21_band.py into
# rar_band_b21_bin{1..4}.txt for plotting.
#
# Usage:
#   conda activate mice
#   bash run_photoz_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-ztrue-isolation  Skip step 02 for z_true (isolation already done;
#                           re-run steps 03-05 with updated scripts)
#   --skip-ztrue            Skip ALL z_true steps (02-05); only run photo-z
#
# Working directory: the MICE_n-body/ folder (override with $MICE_WORKDIR)
# Conda env       : mice
#
# TYPICAL USE after uploading updated scripts (isolation already done):
#   bash run_photoz_pipeline.sh --skip-ztrue-isolation

set -euo pipefail

# WORKDIR defaults to this repo (parent of scripts/); override with $MICE_WORKDIR
WORKDIR="${MICE_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATADIR="${WORKDIR}/data/MICE2_isolated"
SCRIPTDIR="${WORKDIR}/scripts"

INPUT_RAW="${DATADIR}/mice2_raw_test.fits"

SKIP_ZTRUE_ISO=0
SKIP_ZTRUE_ALL=0
for arg in "$@"; do
    [[ "$arg" == "--skip-ztrue-isolation" ]] && SKIP_ZTRUE_ISO=1
    [[ "$arg" == "--skip-ztrue" ]]           && SKIP_ZTRUE_ALL=1
done

echo "========================================================"
echo "  B21 MICE band pipeline"
echo "  Working dir : ${WORKDIR}"
echo "  Data dir    : ${DATADIR}"
echo "  Input raw   : ${INPUT_RAW}"
echo "========================================================"
echo ""

cd "${WORKDIR}"

# ── Run 1: z_true isolation (lower limit of the band) ──────────────────────
if [[ "${SKIP_ZTRUE_ALL}" -eq 1 ]]; then
    echo ">>> Skipping ALL z_true steps (--skip-ztrue)."
elif [[ "${SKIP_ZTRUE_ISO}" -eq 1 ]]; then
    echo ">>> Skipping step 02 z_true isolation (already done); re-running 03-05."

    echo ""
    echo ">>> Run 1: compute g_bar (z_true, updated script)"
    python "${SCRIPTDIR}/03_compute_gbar.py" \
        --indir "${DATADIR}" \
        --outdir "${DATADIR}" \
        --suffix ""

    echo ""
    echo ">>> Run 1: compute g_obs (NFW mock ESD, z_true)"
    python "${SCRIPTDIR}/04_mock_esd.py" \
        --indir "${DATADIR}" \
        --outdir "${DATADIR}" \
        --suffix ""

    echo ""
    echo ">>> Run 1: stack RAR (z_true) -> rar_stacked_bin{1..4}.txt"
    python "${SCRIPTDIR}/05_stack_rar.py" \
        --indir "${DATADIR}" \
        --outdir "${DATADIR}" \
        --suffix ""

    echo ""
    echo "Run 1 (z_true, isolation skipped) complete."
else
    echo ">>> Run 1: isolation with TRUE redshifts (no photo-z noise)"
    python "${SCRIPTDIR}/02_select_isolated.py" \
        --input "${INPUT_RAW}" \
        --outdir "${DATADIR}" \
        --suffix "" \
        --photoz_sigma 0.0 \
        --seed 42

    echo ""
    echo ">>> Run 1: compute g_bar"
    python "${SCRIPTDIR}/03_compute_gbar.py" \
        --indir "${DATADIR}" \
        --outdir "${DATADIR}" \
        --suffix ""

    echo ""
    echo ">>> Run 1: compute g_obs (NFW mock ESD)"
    python "${SCRIPTDIR}/04_mock_esd.py" \
        --indir "${DATADIR}" \
        --outdir "${DATADIR}" \
        --suffix ""

    echo ""
    echo ">>> Run 1: stack RAR"
    python "${SCRIPTDIR}/05_stack_rar.py" \
        --indir "${DATADIR}" \
        --outdir "${DATADIR}" \
        --suffix ""

    echo ""
    echo "Run 1 (z_true) complete."
fi

echo ""

# ── Run 2: photo-z isolation (upper limit of the band) ─────────────────────
echo ">>> Run 2: isolation with PHOTO-Z redshifts (sigma=0.02)"
python "${SCRIPTDIR}/02_select_isolated.py" \
    --input "${INPUT_RAW}" \
    --outdir "${DATADIR}" \
    --suffix "_photoz" \
    --photoz_sigma 0.02 \
    --seed 42

echo ""
echo ">>> Run 2: compute g_bar"
python "${SCRIPTDIR}/03_compute_gbar.py" \
    --indir "${DATADIR}" \
    --outdir "${DATADIR}" \
    --suffix "_photoz"

echo ""
echo ">>> Run 2: compute g_obs (NFW mock ESD)"
python "${SCRIPTDIR}/04_mock_esd.py" \
    --indir "${DATADIR}" \
    --outdir "${DATADIR}" \
    --suffix "_photoz"

echo ""
echo ">>> Run 2: stack RAR"
python "${SCRIPTDIR}/05_stack_rar.py" \
    --indir "${DATADIR}" \
    --outdir "${DATADIR}" \
    --suffix "_photoz"

echo ""

# ── Step 8: combine into B21 band ──────────────────────────────────────────
echo ">>> Step 8: build B21 band (ztrue lower, zphotoz upper)"
python "${SCRIPTDIR}/08_b21_band.py" \
    --indir "${DATADIR}" \
    --outdir "${DATADIR}" \
    --suffix_ztrue "" \
    --suffix_zphotoz "_photoz"

echo ""
echo "========================================================"
echo "  Done.  Results:"
ls -lh "${DATADIR}"/rar_band_b21_bin*.txt 2>/dev/null || echo "  (no rar_band_b21 files found)"
echo ""
echo "  Download to local machine:"
echo "    scp <host>:${DATADIR}/rar_band_b21_bin*.txt \\"
echo "        ./MICE_n-body/data/"
echo "========================================================"
