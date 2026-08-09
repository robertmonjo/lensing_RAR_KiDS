"""Select isolated lenses from the MICE2 catalogue following Brouwer+2021 Sec. 2.2.

Isolation criterion: a lens galaxy is isolated if it has no neighbour with
  (i)   projected separation r_perp < 3 Mpc (physical at lens redshift)
  (ii)  line-of-sight |Delta_v| < 1000 km/s
  (iii) neighbour stellar mass M_nb > 0.25 * M_lens

All distances are physical (not comoving).

Stellar mass bins (in M_sun, not h^-2 M_sun):
  Bin 1: [10^9.5,  10^10.3)
  Bin 2: [10^10.3, 10^10.6)
  Bin 3: [10^10.6, 10^10.8)
  Bin 4: [10^10.8, 10^11.2)

Requirements:
    pip install astropy numpy scipy

Usage:
    python 02_select_isolated.py --input ../data/MICE2_isolated/mice2_raw.fits \
                                  --outdir ../data/MICE2_isolated/
"""

import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree

# MICE2 cosmology
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.25)
H0_KMS = 70.0  # km/s/Mpc

# B21 isolation parameters
R_ISO_MPC = 3.0          # physical Mpc
DV_ISO_KMS = 1000.0      # km/s
FRAC_MASS = 0.25         # neighbour mass fraction

# Stellar mass bin edges in log10(M_sun)  [NOT h^-2 M_sun]
BIN_EDGES_MSUN = [9.5, 10.3, 10.6, 10.8, 11.2]

# lmstellar in MICE is log10(M_star / h^-2 M_sun); h=0.70 => offset = 2*log10(0.7)
H_MICE = 0.70
LOG10_H2 = 2.0 * np.log10(H_MICE)          # = -0.3010
# M_star [M_sun] = 10^(lmstellar) * h^2
# log10(M_star/M_sun) = lmstellar + LOG10_H2
# => lmstellar = log10(M_star/M_sun) - LOG10_H2

# Minimum lmstellar for neighbours (M_star > 10^9.5 M_sun)
LM_MIN_NEIGHBOUR = 9.5 - LOG10_H2  # = 9.801 in MICE units


def parse_args():
    p = argparse.ArgumentParser(description="Select isolated MICE2 lenses")
    p.add_argument("--input", default="../data/MICE2_isolated/mice2_raw.fits")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    p.add_argument("--nchunks", type=int, default=10,
                   help="Number of redshift chunks for neighbour search")
    p.add_argument("--photoz_sigma", type=float, default=0.0,
                   help="Add Gaussian photo-z noise sigma*(1+z) to z before isolation search"
                        " (B21 Sect. 5.3: sigma=0.02 emulates KiDS photo-z errors)")
    p.add_argument("--suffix", default="",
                   help="Output file suffix, e.g. '_photoz' -> mice2_isolated_photoz_bin{b}.fits")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for photo-z noise (reproducibility)")
    return p.parse_args()


def comoving_to_physical_distance_mpc(z, cosmo):
    """Comoving distance to redshift z in Mpc."""
    return cosmo.comoving_distance(z).value  # Mpc (comoving)


def angular_separation_rad(ra1, dec1, ra2, dec2):
    """Haversine angular separation in radians (vectorised)."""
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2)**2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def find_isolated(ra, dec, z, lm_msun, chunk_dz=0.02, photoz_sigma=0.0, rng=None):
    """
    Returns boolean mask: True = isolated galaxy.

    Uses a KD-tree on (x, y, z) Cartesian coordinates on the unit sphere,
    with a conservative angular search radius and then fine-grained checks.

    photoz_sigma > 0: add Gaussian noise sigma*(1+z) to z before the |Δz|
    line-of-sight criterion, emulating KiDS photometric redshift errors (B21 Sect. 5.3).
    """
    n = len(ra)
    isolated = np.ones(n, dtype=bool)

    # Photo-z noise applied to ALL galaxies (lenses + neighbours)
    if photoz_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(42)
        noise = rng.normal(0.0, photoz_sigma * (1.0 + z))
        z_search = np.clip(z + noise, 0.0, None)
    else:
        z_search = z

    # Cartesian coordinates on unit sphere for fast angular search
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    xyz = np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])

    # Process in comoving redshift chunks to limit memory
    z_bins = np.arange(z.min(), z.max() + chunk_dz, chunk_dz)

    # We need all potential neighbours (lmstellar > LM_MIN_NEIGHBOUR)
    nb_mask = lm_msun >= 9.5  # log10(M/M_sun) > 9.5

    tree = cKDTree(xyz[nb_mask])
    idx_nb = np.where(nb_mask)[0]

    print(f"  Total galaxies: {n:,}, potential neighbours: {nb_mask.sum():,}")

    for i, (z_lo, z_hi) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        lens_mask = (z >= z_lo) & (z < z_hi)
        if not lens_mask.any():
            continue

        z_mid = 0.5 * (z_lo + z_hi)
        d_A = COSMO.angular_diameter_distance(z_mid).value  # Mpc
        # Conservative angular radius: R_ISO_MPC / d_A (physical Mpc / Mpc)
        theta_max = R_ISO_MPC / d_A * 1.05  # 5% margin
        chord_max = 2.0 * np.sin(theta_max / 2.0)  # chord on unit sphere

        # Max redshift separation for |Delta_v| < DV_ISO_KMS: c*|dz| = |dv|
        dz_max = DV_ISO_KMS / 299792.458

        lens_idx = np.where(lens_mask)[0]

        for li in lens_idx:
            # Fast ball query on unit sphere
            candidates = tree.query_ball_point(xyz[li], chord_max)
            if not candidates:
                continue

            for ci in candidates:
                ni = idx_nb[ci]
                if ni == li:
                    continue
                # Use photo-z (z_search) for the line-of-sight criterion
                if abs(z_search[ni] - z_search[li]) > dz_max:
                    continue
                if lm_msun[ni] < lm_msun[li] + np.log10(FRAC_MASS):
                    continue

                # Precise projected physical separation check (use true z for d_A)
                theta = angular_separation_rad(ra[li], dec[li], ra[ni], dec[ni])
                r_perp = d_A * theta  # physical Mpc
                if r_perp < R_ISO_MPC:
                    isolated[li] = False
                    break  # one neighbour is enough to fail isolation

        print(f"  z chunk [{z_lo:.2f},{z_hi:.2f}): {lens_mask.sum():,} lenses, "
              f"{isolated[lens_mask].sum():,} isolated so far")

    return isolated


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.input} ...")
    with fits.open(args.input, memmap=True) as hdul:
        t = hdul[1].data
        ra = t["ra_gal"].astype(float)
        dec = t["dec_gal"].astype(float)
        z = t["z_cgal"].astype(float)
        lm = t["lmstellar"].astype(float)  # log10(M_star/h^-2 M_sun)
        lm_msun = lm + LOG10_H2            # log10(M_star/M_sun)
        lmhalo = t["lmhalo"].astype(float)

    print(f"Loaded {len(ra):,} galaxies")

    # Stellar mass bins in log10(M_sun)
    bin_edges = BIN_EDGES_MSUN
    n_bins = len(bin_edges) - 1

    # Build mass-bin masks (lenses only; all galaxies are neighbour candidates)
    bin_masks = []
    for b in range(n_bins):
        m = (lm_msun >= bin_edges[b]) & (lm_msun < bin_edges[b + 1])
        bin_masks.append(m)
        print(f"  Bin {b+1} [{bin_edges[b]:.1f},{bin_edges[b+1]:.1f}): {m.sum():,} galaxies")

    # Union of all lens candidates
    lens_candidate = np.zeros(len(ra), dtype=bool)
    for m in bin_masks:
        lens_candidate |= m

    print(f"\nRunning isolation selection on {lens_candidate.sum():,} lens candidates ...")
    if args.photoz_sigma > 0.0:
        print(f"  Photo-z noise: sigma={args.photoz_sigma}, seed={args.seed}")
    rng = np.random.default_rng(args.seed) if args.photoz_sigma > 0.0 else None
    iso = find_isolated(ra, dec, z, lm_msun,
                        photoz_sigma=args.photoz_sigma, rng=rng)

    for b in range(n_bins):
        sel = bin_masks[b] & iso
        print(f"Bin {b+1}: {sel.sum():,} isolated lenses (out of {bin_masks[b].sum():,})")

        outfile = os.path.join(args.outdir, f"mice2_isolated{args.suffix}_bin{b+1}.fits")
        cols = fits.BinTableHDU.from_columns([
            fits.Column("ra_gal",   "D", array=ra[sel]),
            fits.Column("dec_gal",  "D", array=dec[sel]),
            fits.Column("z_cgal",   "D", array=z[sel]),
            fits.Column("lmstellar","D", array=lm[sel]),
            fits.Column("lm_msun",  "D", array=lm_msun[sel]),
            fits.Column("lmhalo",   "D", array=lmhalo[sel]),
        ])
        cols.writeto(outfile, overwrite=True)
        print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
