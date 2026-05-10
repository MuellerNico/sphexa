#!/usr/bin/env python3

"""Post-run timing dashboard.

Parses the `# <substep>: <time>s` lines that the SPH-EXA Timer prints to
stdout (captured in the SLURM .out log), groups them into iterations using
`domain::sync` as the iteration boundary, and produces a multi-panel PDF.

profile.h5 is not used: the HDF5 file stores only the flat sequence of
substep durations and the set of unique names, without per-call name pairing
or iteration boundaries (the per-iteration substep count varies between
branches), so it cannot be reshaped unambiguously. The .out log has the
paired (name, time) data needed for an accurate breakdown.
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LINE_RE = re.compile(r'^# (.+?): ([0-9.eE+-]+)s\s*$')
ITER_BOUNDARY = 'domain::sync'


def parse_log(log_path):
    iterations = []
    current = None
    name_order = []
    seen = set()

    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            name, t = m.group(1), float(m.group(2))

            # Skip the totalTimer summary line ("# Total execution time of N iterations ... : Xs")
            if name.startswith("Total "):
                continue

            if name == ITER_BOUNDARY:
                if current is not None:
                    iterations.append(current)
                current = {}

            if current is None:
                continue

            current[name] = current.get(name, 0.0) + t
            if name not in seen:
                seen.add(name)
                name_order.append(name)

    if current is not None:
        iterations.append(current)

    return iterations, name_order


def to_matrix(iterations, name_order):
    n_iter = len(iterations)
    n_names = len(name_order)
    mat = np.zeros((n_iter, n_names), dtype=np.float64)
    for i, it in enumerate(iterations):
        for j, name in enumerate(name_order):
            mat[i, j] = it.get(name, 0.0)
    return mat


def plot_dashboard(log_path, out_dir=None, show=False):
    iterations, name_order = parse_log(log_path)
    if not iterations or not name_order:
        print(f"No substep timing lines found in {log_path}", file=sys.stderr)
        sys.exit(1)

    timings = to_matrix(iterations, name_order)
    n_iter, n_sub = timings.shape
    iters = np.arange(1, n_iter + 1)
    per_iter_total = timings.sum(axis=1)
    per_sub_total = timings.sum(axis=0)
    grand_total = float(per_sub_total.sum())

    order = np.argsort(per_sub_total)[::-1]
    names_sorted = [name_order[i] for i in order]
    per_sub_total_sorted = per_sub_total[order]
    timings_sorted = timings[:, order]

    n_panels = 4
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n_sub)]

    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4 * n_panels))
    fig.subplots_adjust(hspace=0.35, top=0.95, bottom=0.05, left=0.10, right=0.97)

    run_id = os.path.basename(out_dir) if out_dir else os.path.splitext(os.path.basename(log_path))[0]
    fig.suptitle(f"Profile dashboard — {run_id}", fontsize=13)
    fig.text(
        0.97, 0.975,
        f"iterations = {n_iter}    substeps = {n_sub}    total = {grand_total:.2f} s",
        ha='right', va='top', fontsize=10, color='gray',
    )

    # --- Panel 1: stacked area, time per substep per iteration ---
    ax = axes[0]
    ax.stackplot(iters, timings_sorted.T, labels=names_sorted, colors=colors, alpha=0.9)
    ax.set_ylabel("Wall time per iteration (s)")
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, n_iter)
    ax.legend(fontsize=7, ncol=min(4, n_sub), loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-substep wall time (stacked, sorted by total share)")

    # --- Panel 2: total wall time per iteration ---
    ax = axes[1]
    ax.plot(iters, per_iter_total, color='black', linewidth=1.0)
    median = float(np.median(per_iter_total))
    ax.axhline(median, color='tab:red', linewidth=0.8, linestyle='--',
               label=f"median = {median:.4f} s")
    ax.set_ylabel("Total time per iteration (s)")
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, n_iter)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title("Total wall time per iteration (spike detector)")

    # --- Panel 3: cumulative wall time ---
    ax = axes[2]
    ax.plot(iters, np.cumsum(per_iter_total), color='tab:blue', linewidth=1.2)
    ax.set_ylabel("Cumulative wall time (s)")
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, n_iter)
    ax.grid(True, alpha=0.3)
    ax.set_title("Cumulative wall time")

    # --- Panel 4: total share per substep (horizontal bar) ---
    ax = axes[3]
    y = np.arange(n_sub)
    pcts = 100 * per_sub_total_sorted / grand_total if grand_total > 0 else per_sub_total_sorted
    bars = ax.barh(y, per_sub_total_sorted, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total time (s)")
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title("Total time per substep")
    for bar, t, p in zip(bars, per_sub_total_sorted, pcts):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {t:.2f}s  ({p:.1f}%)", va='center', fontsize=8)

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(log_path))
    outname = os.path.join(out_dir, "profile_dashboard.pdf")
    fig.savefig(outname, bbox_inches='tight')
    print(f"Saved: {outname}")

    if show:
        matplotlib.use('TkAgg')
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a per-substep timing dashboard from an SPH-EXA SLURM .out log."
    )
    parser.add_argument("log", help="Path to SLURM .out log (e.g. logs/sphexa-12345.out)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write profile_dashboard.pdf (default: dirname of log)")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    args = parser.parse_args()

    plot_dashboard(args.log, out_dir=args.out_dir, show=args.show)
