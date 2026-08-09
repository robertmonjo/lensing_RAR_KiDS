"""Stack g_obs vs g_bar for MICE2 isolated lenses and construct the RAR band.

The RAR band is the 16th--84th percentile of the INDIVIDUAL galaxy g_obs
distribution at each g_bar value, following Brouwer+2021 Fig. 9.

The stacked MEAN signal is what B21 use for chi^2 comparison.

Requirements:
    pip install numpy matplotlib astropy

Usage:
    python 05_stack_rar.py --indir ../data/MICE2_isolated/ \
                            --outdir ../data/MICE2_isolated/ \
                            --plot
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Radial bins (same as previous steps)
R_BINS_MPC = np.logspace(np.log10(0.03), np.log10(3.0), 15)

# B21 bin mean stellar masses (M_sun)
MSTAR_BINS = [1.5e10, 3.2e10, 4.6e10, 8.9e10]

# Physical constants for g_bar formula
G_PC3_MSUN_S2 = 4.52e-30
PC_M = 3.086e16
MPC_PC = 1.0e6
CONV_ESD_TO_GOBS = 4.0 * G_PC3_MSUN_S2 / PC_M


def load_kids_data(data_dir, b):
    """Load KiDS B21 data for mass bin b (1-indexed)."""
    fname = os.path.join(data_dir,
                         f"Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt")
    if not os.path.exists(fname):
        return None
    d = np.loadtxt(fname, comments="#")
    d = d[::-1]  # reverse: files are outer->inner, we want inner->outer
    return d  # columns: [gbar, gobs, gobs_lo, gobs_hi, r_Mpc]


def load_mice_data(indir, b, suffix=""):
    """Load precomputed MICE g_obs and g_bar arrays."""
    gbar_file = os.path.join(indir, f"gbar{suffix}_bin{b}.txt")
    gobs_file = os.path.join(indir, f"gobs_mock{suffix}_bin{b}.txt")
    gbar_full_file = os.path.join(indir, f"gbar_full{suffix}_bin{b}.npy")
    gobs_full_file = os.path.join(indir, f"gobs_full{suffix}_bin{b}.npy")

    if not os.path.exists(gbar_file) or not os.path.exists(gobs_file):
        return None, None, None, None

    gbar_mean_data = np.loadtxt(gbar_file, comments="#")
    gobs_mean_data = np.loadtxt(gobs_file, comments="#")

    g_bar_mean = gbar_mean_data[:, 1]   # m/s^2
    g_obs_mean = gobs_mean_data[:, 1]   # m/s^2

    g_bar_full = np.load(gbar_full_file) if os.path.exists(gbar_full_file) else None
    g_obs_full = np.load(gobs_full_file) if os.path.exists(gobs_full_file) else None

    return g_bar_mean, g_obs_mean, g_bar_full, g_obs_full


def build_rar_band(g_bar_all, g_obs_all, n_slices=50):
    """
    Construct the MICE RAR band: 16th/50th/84th percentile of g_obs
    in narrow g_bar slices.

    g_bar_all, g_obs_all: (n_gal, n_rbins) arrays in m/s^2
    Returns:
        gb_centres: log10(g_bar) bin centres
        go_p16, go_p50, go_p84: log10(g_obs) percentiles
    """
    log_gb = np.log10(g_bar_all.ravel())
    log_go = np.log10(g_obs_all.ravel())

    valid = np.isfinite(log_gb) & np.isfinite(log_go)
    log_gb = log_gb[valid]
    log_go = log_go[valid]

    gb_min, gb_max = log_gb.min(), log_gb.max()
    edges = np.linspace(gb_min, gb_max, n_slices + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    p16 = np.full(n_slices, np.nan)
    p50 = np.full(n_slices, np.nan)
    p84 = np.full(n_slices, np.nan)

    for i in range(n_slices):
        mask = (log_gb >= edges[i]) & (log_gb < edges[i + 1])
        if mask.sum() < 5:
            continue
        p16[i], p50[i], p84[i] = np.percentile(log_go[mask], [16, 50, 84])

    return centres, p16, p50, p84


def save_band(outdir, centres, p16, p50, p84):
    out = os.path.join(outdir, "rar_band.txt")
    np.savetxt(out,
               np.column_stack([centres, p16, p50, p84]),
               header="log10_gbar  log10_gobs_p16  log10_gobs_p50  log10_gobs_p84",
               comments="# ")
    print(f"Saved band: {out}")


def make_plot(outdir, kids_data, mice_means, mice_bands):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
    axes = axes.ravel()

    colors = ["#1a4068", "#2475b0", "#c55a11", "#7f1d1d"]

    for b in range(4):
        ax = axes[b]
        ax.set_title(f"Bin {b+1}, $M_\\star = {MSTAR_BINS[b]/1e10:.1f}\\times10^{{10}}M_\\odot$",
                     fontsize=10)

        # KiDS data
        kd = kids_data[b]
        if kd is not None:
            log_gb_k = np.log10(np.abs(kd[:, 0]))
            log_go_k = np.log10(np.abs(kd[:, 1]))
            log_go_lo = np.log10(np.abs(kd[:, 2]))
            log_go_hi = np.log10(np.abs(kd[:, 3]))
            ax.errorbar(log_gb_k, log_go_k,
                        yerr=[log_go_k - log_go_lo, log_go_hi - log_go_k],
                        fmt="o", color=colors[b], ms=4, label="KiDS-1000")

        # MICE mean
        mm = mice_means[b]
        if mm is not None:
            log_gb_m, log_go_m = mm
            ax.plot(log_gb_m, log_go_m, "k-", lw=1.5, label="MICE mean")

        # MICE band
        mb = mice_bands[b]
        if mb is not None:
            centres, p16, p50, p84 = mb
            valid = np.isfinite(p16) & np.isfinite(p84)
            ax.fill_between(centres[valid], p16[valid], p84[valid],
                            color="salmon", alpha=0.4, label="MICE 16-84%")

        ax.set_xlim(-15.5, -11.0)
        ax.set_ylim(-13.5, -9.5)
        ax.set_xlabel(r"$\log_{10}(g_\mathrm{bar})$ [m s$^{-2}$]", fontsize=9)
        ax.set_ylabel(r"$\log_{10}(g_\mathrm{obs})$ [m s$^{-2}$]", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("MICE RAR band vs KiDS-1000 data (Brouwer+2021)", fontsize=12)
    fig.tight_layout()
    outfile = os.path.join(outdir, "fig_mice_band.pdf")
    fig.savefig(outfile, dpi=150)
    print(f"Saved figure: {outfile}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Stack MICE RAR and build band")
    p.add_argument("--indir", default="../data/MICE2_isolated/")
    p.add_argument("--kidsdir",
                   default="../../data/brouwer2021_rar/",
                   help="Directory containing Fig-9_RAR-KiDS-isolated_Massbin-N.txt")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    p.add_argument("--suffix", default="",
                   help="File suffix: reads gbar{suffix}_bin{b}.txt and "
                        "gobs_mock{suffix}_bin{b}.txt; writes rar_stacked{suffix}_bin{b}.txt")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    kids_data = []
    mice_means = []
    mice_bands = []

    for b in range(1, 5):
        kd = load_kids_data(args.kidsdir, b)
        kids_data.append(kd)

        g_bar_mean, g_obs_mean, g_bar_full, g_obs_full = load_mice_data(args.indir, b, args.suffix)

        if g_bar_mean is None:
            mice_means.append(None)
            mice_bands.append(None)
            print(f"Bin {b}: MICE data not found, skipping.")
            continue

        # Stacked mean (per radial bin)
        log_gb_m = np.log10(np.abs(g_bar_mean))
        log_go_m = np.log10(np.abs(g_obs_mean))
        mice_means.append((log_gb_m, log_go_m))

        # Save stacked mean
        out_stack = os.path.join(args.outdir, f"rar_stacked{args.suffix}_bin{b}.txt")
        np.savetxt(out_stack,
                   np.column_stack([R_BINS_MPC, log_gb_m, log_go_m]),
                   header="r_Mpc  log10_gbar  log10_gobs",
                   comments="# ")
        print(f"Bin {b}: saved stacked mean -> {out_stack}")

        # RAR band from individual galaxies
        if g_bar_full is not None and g_obs_full is not None:
            centres, p16, p50, p84 = build_rar_band(g_bar_full, g_obs_full)
            mice_bands.append((centres, p16, p50, p84))
            save_band(args.outdir, centres, p16, p50, p84)
        else:
            mice_bands.append(None)
            print(f"Bin {b}: full arrays not found; band not computed.")

    if args.plot:
        make_plot(args.outdir, kids_data, mice_means, mice_bands)

    print("\nDone.  Next: run 06_chi2_comparison.py")


if __name__ == "__main__":
    main()
