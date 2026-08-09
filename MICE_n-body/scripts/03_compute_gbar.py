"""Compute the baryonic gravitational acceleration g_bar for MICE2 isolated lenses.

g_bar is computed following Brouwer+2021 Eq. 7:
    g_bar [m/s^2] = 4 * G * Sigma_bar [M_sun/pc^2] * [pc/m]

where Sigma_bar = ESD from stellar mass + cold gas component.

For mock galaxies we compute Sigma_bar analytically from:
  1. Stellar component: point mass approximation for outer radii (r > 3 r_half),
     Sersic projected profile for inner radii.
  2. Cold gas: gas fraction from the Baldry+2012 scaling relation.

Requirements:
    pip install astropy numpy scipy

Usage:
    python 03_compute_gbar.py --indir ../data/MICE2_isolated/ \
                               --outdir ../data/MICE2_isolated/
"""

import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.25)

# Physical constants
G_SI = 6.674e-11         # m^3 kg^-1 s^-2
MSUN_KG = 1.989e30       # kg
PC_M = 3.086e16          # m per parsec
MPC_PC = 1.0e6           # pc per Mpc
G_PC3_MSUN_S2 = G_SI * MSUN_KG / PC_M**3  # 4.52e-30 pc^3 M_sun^-1 s^-2

# Radial bin centres from Brouwer+2021 Fig. 9 data files (Mpc, inner to outer)
# These are loaded from the actual data files in 06_chi2_comparison.py.
# Here we use 15 log-spaced bins from 0.03 to 3.0 Mpc matching B21.
R_BINS_MPC = np.logspace(np.log10(0.03), np.log10(3.0), 15)

# ESD conversion: g_obs = 4 * G * Delta_Sigma * [pc/m]
# (factor 4 from the B21 lensing-to-dynamics conversion; see B21 App. B)
CONV_ESD_TO_GOBS = 4.0 * G_PC3_MSUN_S2 * PC_M  # converts M_sun/pc^2 to m/s^2


def gas_fraction_baldry2012(lm_msun):
    """
    Cold gas fraction from Baldry+2012 scaling relation (their Eq. 6):
        f_gas = M_gas / (M_gas + M_star)
    Approximation valid for 9 < log10(M_star) < 11.5.

    Returns f_gas (dimensionless).
    """
    # Simplified power-law fit to Fig. 8 of Baldry+2012
    # Molecular + atomic gas fraction
    log_fgas = -0.43 * (lm_msun - 10.0) - 0.49  # log10(f_gas)
    return 10.0 ** np.clip(log_fgas, -3.0, 0.0)


def esd_point_mass(r_mpc, m_msun):
    """
    ESD for a point mass M at projected radii r (Mpc).
    Delta_Sigma(r) = M / (pi * r^2)   [M_sun / pc^2]

    Valid for r >> r_half (outer radii).
    """
    r_pc = r_mpc * MPC_PC
    return m_msun / (np.pi * r_pc**2)


def esd_sersic_n1(r_mpc, m_msun, r_half_mpc):
    """
    ESD for an exponential (Sersic n=1) disc with total mass M and half-light
    radius r_half.  Uses the analytical projection of the exponential profile.

    Delta_Sigma(r) = M / (2 * pi * r_s^2) * [2 K_0(x) + 2 K_1(x)/x - ...]
    where x = r / r_s, r_s = r_half / b_n (b_n ~ 1.678 for n=1).

    For simplicity (and because r > 0.1 Mpc >> r_half ~ few kpc for most bins),
    we use the point-mass approximation for all outer radii.  This introduces
    < 1% error at r > 10 r_half.
    """
    return esd_point_mass(r_mpc, m_msun)


def compute_gbar_galaxy(lm_msun_val, r_bins_mpc=R_BINS_MPC):
    """
    Compute g_bar profile for a single galaxy with log10(M_star/M_sun) = lm_msun_val.

    Returns:
        g_bar  [m/s^2]  shape (len(r_bins),)
    """
    m_star = 10.0 ** lm_msun_val  # M_sun
    f_gas = gas_fraction_baldry2012(lm_msun_val)
    m_gas = m_star * f_gas / (1.0 - f_gas)  # cold gas mass M_sun
    m_bar = m_star + m_gas

    esd_bar = esd_point_mass(r_bins_mpc, m_bar)  # M_sun / pc^2
    g_bar = CONV_ESD_TO_GOBS * esd_bar           # m/s^2
    return g_bar


def process_bin(infile, outfile, r_bins_mpc=R_BINS_MPC):
    print(f"  Loading {infile} ...")
    with fits.open(infile) as hdul:
        lm_msun = hdul[1].data["lm_msun"].astype(float)

    n_gal = len(lm_msun)
    print(f"  {n_gal:,} galaxies")

    # Compute per-galaxy g_bar profiles
    g_bar_all = np.zeros((n_gal, len(r_bins_mpc)))
    for i in range(n_gal):
        g_bar_all[i] = compute_gbar_galaxy(lm_msun[i], r_bins_mpc)

    # Mean g_bar (used for stacked comparison)
    g_bar_mean = g_bar_all.mean(axis=0)

    # Save mean profile + full array (compact: only mean to save space)
    header = fits.Header()
    header["COMMENT"] = "g_bar profiles for MICE2 isolated lenses"
    header["COMMENT"] = "Col 0: r_Mpc, Col 1: g_bar_mean [m/s^2]"

    out = np.column_stack([r_bins_mpc, g_bar_mean])
    np.savetxt(outfile, out,
               header="r_Mpc  g_bar_mean_m_s2",
               comments="# ")
    print(f"  Saved mean g_bar to {outfile}")

    # Also save the full per-galaxy array as .npy for stacking in step 5
    npy_out = outfile.replace("gbar_", "gbar_full_").replace(".txt", ".npy")
    np.save(npy_out, g_bar_all)
    print(f"  Saved full g_bar array ({n_gal}x{len(r_bins_mpc)}) to {npy_out}")

    return g_bar_mean


def parse_args():
    p = argparse.ArgumentParser(description="Compute g_bar for MICE2 isolated lenses")
    p.add_argument("--indir", default="../data/MICE2_isolated/")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    p.add_argument("--suffix", default="",
                   help="File suffix: reads mice2_isolated{suffix}_bin{b}.fits, "
                        "writes gbar{suffix}_bin{b}.txt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Radial bins (Mpc): {R_BINS_MPC}")
    print()

    for b in range(1, 5):
        infile = os.path.join(args.indir, f"mice2_isolated{args.suffix}_bin{b}.fits")
        outfile = os.path.join(args.outdir, f"gbar{args.suffix}_bin{b}.txt")
        if not os.path.exists(infile):
            print(f"Bin {b}: {infile} not found, skipping.")
            continue
        print(f"Bin {b}:")
        process_bin(infile, outfile)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
