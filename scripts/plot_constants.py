#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse

COLUMNS = [
    "iteration",      # 0  d.iteration
    "ttot",           # 1  d.ttot
    "minDt",          # 2  d.minDt
    "etot",           # 3  d.etot
    "ecin",           # 4  d.ecin
    "eint",           # 5  d.eint
    "egrav",          # 6  d.egrav
    "linmom",         # 7  d.linmom
    "angmom",         # 8  d.angmom
    "eMag",           # 9  md.eMag
    "meanDivBError",  # 10 md.meanDivBError
    "maxDivBError",   # 11 md.maxDivBError
    "resHeating",     # 12 md.resHeating: total resistive heating rate
    "extra",          # 13 test-specific scalar: KH growth rate (kh) or RMS Mach number (turb)
]

# label + panel title for the trailing test-specific column (index 12), by --kind
EXTRA_LABELS = {
    "turb": "RMS Mach number",
    "kh":   "KH growth rate",
}


def load(fname):
    data = np.loadtxt(fname)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    ncols = data.shape[1]
    return {col: data[:, i] for i, col in enumerate(COLUMNS[:ncols])}


def plot_constants(fname, show=False, kind="turb"):
    d = load(fname)
    it = d["iteration"]
    final_time = d["ttot"][-1]

    extra_label = EXTRA_LABELS.get(kind, "extra (col 13)")
    has_extra = "extra" in d
    has_res = "resHeating" in d
    n_panels = 6 + int(has_res) + int(has_extra)

    # cumulative magnetic energy removed by artificial resistivity
    cum = None
    if has_res:
        res = d["resHeating"]
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (res[1:] + res[:-1]) * np.diff(d["ttot"]))))

    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows + 1), sharex=True)
    ax_list = axes.flatten()
    fig.subplots_adjust(hspace=0.12, wspace=0.22, top=0.92, bottom=0.07, left=0.08, right=0.97)

    fig.suptitle(
        f"Simulation diagnostics — {os.path.basename(os.path.dirname(os.path.abspath(fname)))}",
        fontsize=13,
    )
    fig.text(
        0.97, 0.955, f"t_final = {final_time:.6g}    N_iter = {int(it[-1])}",
        ha='right', va='top', fontsize=10, color='gray',
    )

    # --- Panel 1: Energy components (log scale) ---
    ax = ax_list[0]
    ax.set_ylabel("Energy (log)")
    ax.set_yscale("log")

    for key, label, ls in [
        ("etot", "etot (total)",    "-"),
        ("ecin", "ecin (kinetic)",  "--"),
        ("eint", "eint (thermal)",  "-."),
        ("eMag", "eMag (magnetic)", ":"),
    ]:
        vals = d[key]
        pos = vals > 0
        if pos.any():
            ax.plot(it[pos], vals[pos], ls=ls, label=label, linewidth=1.2)

    # |egrav| with dashed style and marker in legend
    egrav = d["egrav"]
    abs_egrav = np.abs(egrav)
    pos = abs_egrav > 0
    if pos.any():
        ax.plot(it[pos], abs_egrav[pos], ls="--", color="gray",
                label="|egrav| (gravity)", linewidth=1.2)

    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 2: Relative energy drift ---
    ax = ax_list[1]
    ax.set_ylabel("(etot - etot₀) / |etot₀|")
    etot = d["etot"]
    etot0 = etot[0]
    if etot0 != 0:
        drift = (etot - etot0) / abs(etot0)
        ax.plot(it, drift, color="tab:red", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Momentum conservation ---
    ax = ax_list[2]
    ax.set_ylabel("Momentum")
    ax.plot(it, d["linmom"], label="linmom", linewidth=1.2)
    ax.plot(it, d["angmom"], label="angmom", linewidth=1.2, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Magnetic energy ---
    ax = ax_list[3]
    ax.set_ylabel("eMag / eMag₀")
    eMag = d["eMag"] / d["eMag"][0]
    pos = eMag > 0
    if pos.any():
        ax.plot(it[pos], eMag[pos], color="tab:purple", linewidth=1.2, label="eMag")
    # dissipation-only expectation. eMag above it = magnetic energy created by the
    # ideal induction term acting on velocity noise; the gap is that source, integrated.
    if has_res:
        floor = 1.0 - cum / d["eMag"][0]
        ax.plot(it, floor, color="tab:gray", linewidth=1.0, linestyle="--",
                label="1 − ∫resistive / eMag₀")
        ax.fill_between(it, floor, eMag, where=eMag > floor, color="tab:purple",
                        alpha=0.12, linewidth=0, label="ideal-induction source")
        ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 5: div(B) errors (log scale) ---
    ax = ax_list[4]
    ax.set_ylabel("div(B) error (log)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6, top=1e2)
    mean_b = d["meanDivBError"]
    max_b  = d["maxDivBError"]
    pos_mean = mean_b > 0
    pos_max  = max_b  > 0
    if pos_mean.any():
        ax.plot(it[pos_mean], mean_b[pos_mean], label="mean div(B) error", linewidth=1.2)
    if pos_max.any():
        ax.plot(it[pos_max],  max_b[pos_max],  label="max div(B) error",  linewidth=1.2, linestyle="--")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 6: Minimum timestep ---
    ax = ax_list[5]
    ax.set_ylabel("minDt")
    ax.plot(it, d["minDt"], color="tab:green", linewidth=1.2)
    ax.grid(True, alpha=0.3)

    next_panel = 6

    # --- Panel: resistive heating ---
    # Left axis: instantaneous total resistive heating rate dE_int/dt. 
    # Right axis: cumulative time-integral (= total magnetic energy dissipated so far)
    if has_res:
        ax = ax_list[next_panel]
        next_panel += 1
        res = d["resHeating"]
        ax.set_ylabel("resistive heating rate")
        ax.plot(it, res, color="tab:brown", linewidth=1.2, label="dE_int/dt (resistive)")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(it, cum, color="tab:gray", linewidth=1.2, linestyle="--",
                 label="cumulative dissipated energy")
        ax2.set_ylabel("cumulative")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

    # --- Panel: test-specific scalar (turbulence RMS Mach number, or magnetic KH growth rate) ---
    if has_extra:
        ax = ax_list[next_panel]
        next_panel += 1
        ax.set_ylabel(extra_label)
        ax.plot(it, d["extra"], color="tab:orange", linewidth=1.2)
        ax.grid(True, alpha=0.3)

    for ax in ax_list[n_panels:]:
        ax.set_visible(False)

    # with sharex, only the global bottom row shows tick labels; re-enable them on
    # the lowest populated panel of each column (the short column ends one row early)
    for col in range(ncols):
        col_idx = [r * ncols + col for r in range(nrows) if r * ncols + col < n_panels]
        bottom = ax_list[col_idx[-1]]
        bottom.set_xlabel("Iteration")
        bottom.tick_params(labelbottom=True)

    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, "constants_diagnostics.pdf")
    fig.savefig(outname, bbox_inches='tight')
    print(f"Saved: {outname}")

    if show:
        matplotlib.use('TkAgg')
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation diagnostics from constants.txt")
    parser.add_argument("file", help="Path to constants.txt")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    parser.add_argument("--kind", choices=["turb", "kh"], default="turb",
                        help="meaning of the trailing column 13: 'turb' = RMS Mach number (default), "
                             "'kh' = magnetic Kelvin-Helmholtz growth rate")
    args = parser.parse_args()

    plot_constants(args.file, show=args.show, kind=args.kind)
