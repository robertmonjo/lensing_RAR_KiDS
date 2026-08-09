# MICE2 Isolated Lensing RAR Pipeline

Reproduces the isolated-galaxy lensing RAR from MICE2 following Brouwer et al. (2021).
Contact: robert.monjo@cunef.edu

## Reproducibility (any user, any machine)

All scripts use **paths relative to this `MICE_n-body/` directory** — no absolute
or cluster-specific paths. The only manual step is downloading the MICE2 catalogue
from CosmoHub (Step 01; free account, ~2 GB), placed at
`data/MICE2_isolated/mice2_raw_test.fits`. Then:

```bash
pip install numpy scipy astropy          # or: conda activate mice
bash scripts/run_photoz_pipeline.sh      # runs steps 02–08
```

Optionally override the working directory with `export MICE_WORKDIR=/path/to/repo`
(defaults to this `MICE_n-body/` folder).

**Outputs → the main pipeline.** This produces `data/rar_band_b21_bin{1..4}.txt`
and `data/morph_*_allbins.txt`, which are the MICE inputs consumed by
`../scripts_replicable/data/` to draw the paper figures. After regenerating them,
copy them into `../scripts_replicable/data/` to refresh the figures.

---

## Environment

Requires Python ≥ 3.9 with `numpy`, `scipy`, `astropy`. Reference environment used
for the published run (a Linux cluster with conda):

```bash
conda activate mice          # Python 3.11 + astropy, scipy, numpy
```

---

## Pipeline steps

### Step 01 — Download MICE2 from CosmoHub

**Script:** `scripts/01_download_mice.py`

This step is done manually via the CosmoHub web interface (https://cosmohub.pic.es).
The downloaded FITS file must be placed at:
`data/MICE2_isolated/mice2_raw_test.fits`

**CosmoHub SQL query used:**
```sql
SELECT unique_gal_id, ra_gal, dec_gal, z_cgal,
       lmstellar, lmhalo, kappa, gamma1, gamma2
FROM micecatv2_0_view
WHERE z_cgal BETWEEN 0.01 AND 0.50
  AND lmstellar >= 9.20
  AND dec_gal BETWEEN -5.0 AND 5.0
  AND ra_gal BETWEEN 0.0 AND 60.0
ORDER BY unique_gal_id
```

Table: `micecatv2_0_view` (MICECAT v2.0, Crocce et al. 2015)
Sky coverage used: dec ∈ [−5°, 5°], ra ∈ [0°, 60°] (1/6 of the full MICE2 lightcone)
Redshift: z ∈ [0.01, 0.50]
Mass cut: log10(M*/h⁻²M☉) ≥ 9.20

Result: ~2.3 million galaxies (~150 MB FITS)

---

### Step 02 — Select isolated lenses

**Script:** `scripts/02_select_isolated.py`
**Input:** `data/MICE2_isolated/mice2_raw_test.fits`
**Output:** `data/MICE2_isolated/mice2_isolated_bin{1..4}.fits`

Isolation criterion (Brouwer+2021 Sec. 2.2):
- r_perp < 3 Mpc (physical, at lens redshift)
- |Δv| < 1000 km/s  →  Δz < 0.00334
- M_neighbour > 0.25 × M_lens

Stellar mass bins (in M☉):
| Bin | log10(M*/M☉) range | N_isolated |
|-----|---------------------|------------|
|  1  | [9.5, 10.3)         |   13,656   |
|  2  | [10.3, 10.6)        |    2,320   |
|  3  | [10.6, 10.8)        |    1,019   |
|  4  | [10.8, 11.2)        |      883   |

```bash
python scripts/02_select_isolated.py \
    --input  data/MICE2_isolated/mice2_raw_test.fits \
    --outdir data/MICE2_isolated/
```

---

### Step 03 — Compute g_bar (baryonic acceleration)

**Script:** `scripts/03_compute_gbar.py`
**Output:** `data/MICE2_isolated/gbar_bin{1..4}.txt`

g_bar = 4G ΔΣ_bar, with baryonic correction from Baldry+2012 gas fraction.
Point-mass approximation (valid for r ≫ r_half).

```bash
python scripts/03_compute_gbar.py \
    --indir  data/MICE2_isolated/ \
    --outdir data/MICE2_isolated/
```

---

### Step 04 — Compute mock g_obs (NFW approach)

**Script:** `scripts/04_mock_esd.py`
**Output:** `data/MICE2_isolated/gobs_mock_bin{1..4}.txt`
           `data/MICE2_isolated/gobs_full_bin{1..4}.npy`

g_obs from NFW profile using MICE halo masses (lmhalo).
NFW concentration from Duffy+2008.

**Note:** This is Approach A (fast, approximate). The exact B21 method
uses full shear stacking from the MICE2 source catalogue (~10× larger
download). The NFW approach overestimates g_obs by ~0.3–0.4 dex at
small radii due to the inner NFW cusp.

```bash
python scripts/04_mock_esd.py \
    --indir  data/MICE2_isolated/ \
    --outdir data/MICE2_isolated/
```

---

### Step 05 — Stack RAR and generate band figure

**Script:** `scripts/05_stack_rar.py`
**Output:** `data/MICE2_isolated/rar_stacked_bin{1..4}.txt`
           `data/MICE2_isolated/rar_band.txt`
           `data/MICE2_isolated/fig_mice_band.pdf`

```bash
python scripts/05_stack_rar.py \
    --indir  data/MICE2_isolated/ \
    --outdir data/MICE2_isolated/
```

---

### Step 06 — Chi² comparison with KiDS-1000

**Script:** `scripts/06_chi2_comparison.py`
**Output:** `data/MICE2_isolated/chi2_comparison_table.txt`

Compares MICE stacked g_obs with KiDS-1000 data (Brouwer+2021 Fig. 9),
bins 2–4, outer 10 radial points.

Result: χ²_ν = 56.0 (vs B21 reference: 1.66).
Expected discrepancy: NFW approach vs full shear stacking.

```bash
python scripts/06_chi2_comparison.py \
    --micesdir data/MICE2_isolated/ \
    --kidsdir  data/MICE2_isolated/ \
    --outdir   data/MICE2_isolated/
```

---

### Step 07 — Morphological split (optional)

**Script:** `scripts/07_morphology_split.py`

Requires an additional CosmoHub download with morphology and colour:
```sql
SELECT unique_gal_id, ra_gal, dec_gal, z_cgal,
       bulge_fraction, gr_cos
FROM micecatv2_0_view
WHERE z_cgal BETWEEN 0.01 AND 0.50
  AND lmstellar >= 9.20
  AND dec_gal BETWEEN -5.0 AND 5.0
  AND ra_gal BETWEEN 0.0 AND 60.0
ORDER BY unique_gal_id
```

Save as: `data/MICE2_isolated/07_morph_raw.fits`

Then run:
```bash
python scripts/07_morphology_split.py \
    --morphfile data/MICE2_isolated/07_morph_raw.fits \
    --isodir    data/MICE2_isolated/ \
    --outdir    data/MICE2_isolated/
```

Split criteria:
- Sérsic proxy: `bulge_fraction` (= B/T directly in MICE2)
  - Late (disc): B/T < 0.5  ~  Sérsic n < 2.5
  - Early (bulge): B/T ≥ 0.5  ~  Sérsic n ≥ 2.5
- Colour: `gr_cos` (rest-frame g−r from MICE2 SED fitting)
  - Blue (late): g−r < 0.6
  - Red  (early): g−r ≥ 0.6

---

## Quick run (steps 02–06)

```bash
cd MICE_n-body
bash run_mice_pipeline.sh
```

---

## Key results

| Quantity | Value |
|----------|-------|
| Isolated galaxies (Bin 1, log M* ∈ [9.5,10.3)) | 13,656 |
| Isolated galaxies (Bin 2, log M* ∈ [10.3,10.6)) |  2,320 |
| Isolated galaxies (Bin 3, log M* ∈ [10.6,10.8)) |  1,019 |
| Isolated galaxies (Bin 4, log M* ∈ [10.8,11.2)) |    883 |
| log10(g_bar) at r=0.03 Mpc (Bin 1)  | −11.6 m/s² |
| χ²_ν (MICE NFW vs KiDS, N=30)       |  56.0 (NFW approx.) |
| χ²_ν expected (B21 full shear)      |   1.66 |

---

## References

- Brouwer et al. (2021), A&A 650, A113 — KiDS-1000 lensing RAR
- Crocce et al. (2015), MNRAS 453, 1513 — MICE2 simulation
- Duffy et al. (2008), MNRAS 390, L64 — NFW concentration
- Baldry et al. (2012), MNRAS 421, 621 — gas fraction relation
