#!/usr/bin/env python3
"""Particle-pairing / tensile-instability diagnostics from SPHEXA HDF5 dumps.

For each input file, builds a KD-tree on the particle positions and measures:

  (a) the nearest-neighbor distance distribution d_NN/h -- pairing shows up
      as mass far left of the expected glass spacing,
  (b) a kernel-neighbor scan: the locally normalized radial distribution g(q)
      of all neighbors inside the kernel support (q = r_ij/h_i <= 2). A
      healthy glass reads g ~ 1 at large q with an exclusion zone below the
      mean spacing; pairing is a spike at q << 1,
  (c) for MHD dumps (u + magneto::B* present): the same statistics binned in
      plasma beta, to check clumping against the tensile-correction limiter
      window (full strength beta < 2, ramp to zero at beta = 10).

Pass several files to overlay them (e.g. a magneto run against its pure-hydro
control). Plots are saved next to the first input file.

Examples:
  plot_pairing.py out/123_Sedov_magneto/dump.h5
  plot_pairing.py out/123_Sedov_magneto/dump.h5 out/456_Sedov_hydro/dump.h5
  plot_pairing.py dump.h5 --step 3
  plot_pairing.py dump.h5 --all --stride 2       pairing fraction vs time
  plot_pairing.py dump.h5 -i                     print metadata and exit
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse
from scipy.spatial import cKDTree

from _h5_common import print_metadata, get_nsteps

NN_BINS = np.linspace(0.0, 1.5, 151)   # d_NN/h histogram range
Q_BINS = np.linspace(0.0, 2.0, 101)    # kernel-scan range (kernel support)


def resolve_step(fname, step):
    nsteps = get_nsteps(fname)
    if nsteps == 0:
        sys.exit(f"No steps found in {fname}")
    if step is None:
        return nsteps - 1
    if step < 0:
        step += nsteps
    if not 0 <= step < nsteps:
        sys.exit(f"{fname}: step {step} out of range (0..{nsteps - 1})")
    return step


# Read positions/h plus what's needed for the beta diagnostics. Returns a dict;
# 'beta' is None for pure-hydro dumps. Positions are wrapped into [0, L) when
# all three boundaries are periodic so the KD-tree can use toroidal topology.
def read_step(fname, step):
    with h5py.File(fname, "r") as f:
        s = f[f"Step#{step}"]
        pos = np.column_stack([np.asarray(s[k]) for k in ("x", "y", "z")])
        h = np.asarray(s["h"])

        attrs = s.attrs
        time_val = float(np.atleast_1d(attrs.get("time", [np.nan]))[0])
        ng0 = float(np.atleast_1d(attrs.get("ng0", [100]))[0])

        boxsize = None
        box = attrs.get("box")
        btype = attrs.get("boundaryType")
        if box is not None and btype is not None and all(int(b) == 1 for b in np.atleast_1d(btype)):
            lo = np.asarray(box, dtype=float)[[0, 2, 4]]
            L = np.asarray(box, dtype=float)[[1, 3, 5]] - lo
            pos = (pos - lo) % L
            pos = np.where(pos >= L, pos - L, pos)  # guard rounding onto the upper edge
            boxsize = L

        beta = None
        bkeys = ("magneto::Bx", "magneto::By", "magneto::Bz")
        if "u" in s and all(k in s for k in bkeys):
            gamma = float(np.atleast_1d(attrs.get("gamma", [5.0 / 3.0]))[0])
            mu0 = float(np.atleast_1d(attrs.get("mu_0", [1.0]))[0])
            p = (gamma - 1.0) * np.asarray(s["rho"]) * np.asarray(s["u"])
            norm2B = sum(np.asarray(s[k]) ** 2 for k in bkeys)
            with np.errstate(divide='ignore'):
                beta = 2.0 * mu0 * p / norm2B

    return {"step": step, "time": time_val, "ng0": ng0,
            "pos": pos, "h": h, "boxsize": boxsize, "beta": beta}


# Nearest-neighbor distance of every particle (k=2 query; column 0 is self).
def nn_distances(tree, pos):
    d, _ = tree.query(pos, k=2, workers=-1)
    return d[:, 1]


# Kernel-neighbor scan on a random subsample: histogram of q = r_ij/h_i over
# all neighbors inside the support, normalized by the local number density so
# a locally uniform distribution gives g(q) ~ 1. With nc_i neighbors inside
# 2h_i, n_i = 3*nc_i/(32*pi*h_i^3) and dN/dq = 4*pi*n_i*h_i^3*q^2
# = (3/8)*nc_i*q^2, hence the per-pair weight 8/(3*nc_i).
def kernel_scan(tree, pos, h, nsample, kmax, seed=42):
    n = len(h)
    if n > nsample:
        idx = np.random.default_rng(seed).choice(n, size=nsample, replace=False)
    else:
        idx = np.arange(n)

    d, _ = tree.query(pos[idx], k=kmax, workers=-1)
    q = d[:, 1:] / h[idx, None]
    inside = q <= 2.0
    nc = inside.sum(axis=1)

    capped = float((nc >= kmax - 1).mean())
    if capped > 0.01:
        print(f"  WARNING: {capped * 100:.1f}% of sampled particles have >= {kmax - 1} "
              f"neighbors inside 2h; the scan undercounts there. Increase --kmax.")

    w = np.broadcast_to(8.0 / (3.0 * np.maximum(nc, 1))[:, None], q.shape)[inside]
    hist, _ = np.histogram(q[inside], bins=Q_BINS, weights=w)
    qc = 0.5 * (Q_BINS[:-1] + Q_BINS[1:])
    dq = np.diff(Q_BINS)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = hist / (len(idx) * dq * qc ** 2)
    return qc, g, nc


# Per-beta-bin statistics of d_NN/h: median, 5th percentile, pairing fraction.
def beta_binned(beta, dnn_h, q_pair, nbins=24, min_count=500):
    lb = np.log10(np.clip(beta, 1e-4, 1e6))
    lb = lb[np.isfinite(lb)]
    edges = np.linspace(max(lb.min(), -2), min(lb.max(), 4), nbins + 1)
    which = np.digitize(np.log10(np.clip(beta, 1e-4, 1e6)), edges) - 1

    centers, med, p05, frac = [], [], [], []
    for b in range(nbins):
        sel = which == b
        if sel.sum() < min_count:
            continue
        v = dnn_h[sel]
        centers.append(10 ** (0.5 * (edges[b] + edges[b + 1])))
        med.append(np.median(v))
        p05.append(np.percentile(v, 5))
        frac.append(float((v < q_pair).mean()))
    return np.array(centers), np.array(med), np.array(p05), np.array(frac)


# Full single-step analysis of one file: NN histogram, kernel scan, beta stats.
def analyze(fname, step, label, nsample, kmax, pair_threshold):
    data = read_step(fname, step)
    n = len(data["h"])
    print(f"{label}: step {step}, t={data['time']:.6f}, N={n}"
          + ("" if data["boxsize"] is not None else " (non-periodic tree)"))

    tree = cKDTree(data["pos"], boxsize=data["boxsize"])
    dnn_h = nn_distances(tree, data["pos"]) / data["h"]

    # expected glass NN spacing in units of h for ng0 neighbors inside 2h
    dx_exp = (32.0 * np.pi / (3.0 * data["ng0"])) ** (1.0 / 3.0)
    q_pair = pair_threshold * dx_exp
    frac = float((dnn_h < q_pair).mean())

    nn_hist, _ = np.histogram(dnn_h, bins=NN_BINS, density=True)
    qc, g, nc = kernel_scan(tree, data["pos"], data["h"], nsample, kmax)
    del tree

    print(f"  d_NN/h: median={np.median(dnn_h):.3f}  p5={np.percentile(dnn_h, 5):.3f}  "
          f"min={dnn_h.min():.3f}   (expected glass spacing {dx_exp:.3f})")
    print(f"  pairing fraction (d_NN/h < {q_pair:.3f}): {frac:.3e}  ({int(frac * n)} particles)")
    print(f"  neighbors inside 2h (sample of {len(nc)}): median={int(np.median(nc))}  max={nc.max()}")

    res = {"label": label, "time": data["time"], "step": step,
           "nn_hist": nn_hist, "dx_exp": dx_exp, "q_pair": q_pair, "frac": frac,
           "qc": qc, "g": g, "beta_stats": None}

    if data["beta"] is not None:
        beta = data["beta"]
        for lim in (1, 2, 10):
            print(f"  beta < {lim:>2}: {float((beta < lim).mean()) * 100:.2f}% of particles")
        res["beta_stats"] = beta_binned(beta, dnn_h, q_pair)
    return res


def mark_limiter_window(ax):
    ax.axvspan(2, 10, color='gray', alpha=0.12, lw=0)
    ax.axvline(2, color='gray', lw=0.8, ls='--')
    ax.axvline(10, color='gray', lw=0.8, ls='--')
    ax.axvline(1, color='gray', lw=0.8, ls=':')


def plot_comparison(results, outname):
    with_beta = [r for r in results if r["beta_stats"] is not None]
    if with_beta:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax_nn, ax_g, ax_bmed, ax_bfrac = axes.flat
    else:
        fig, (ax_nn, ax_g) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax_bmed = ax_bfrac = None

    centers = 0.5 * (NN_BINS[:-1] + NN_BINS[1:])
    for i, r in enumerate(results):
        c = f"C{i}"
        lbl = f"{r['label']} (t={r['time']:.3f})"
        ax_nn.stairs(r["nn_hist"], NN_BINS, color=c, label=lbl)
        ax_g.plot(r["qc"], r["g"], color=c, label=lbl)

    r0 = results[0]
    ax_nn.axvline(r0["dx_exp"], color='k', lw=0.8, ls='--', label='expected glass spacing')
    ax_nn.axvline(r0["q_pair"], color='r', lw=0.8, ls=':', label='pairing threshold')
    ax_nn.set_yscale('log')
    ax_nn.set_xlabel(r'$d_\mathrm{NN}/h$')
    ax_nn.set_ylabel('probability density')
    ax_nn.set_title('nearest-neighbor distance')
    ax_nn.legend(fontsize=8)

    ax_g.axhline(1.0, color='k', lw=0.8, ls='--')
    ax_g.set_yscale('log')
    ax_g.set_ylim(bottom=1e-3)
    ax_g.set_xlabel(r'$q = r_{ij}/h_i$')
    ax_g.set_ylabel(r'$g(q)$')
    ax_g.set_title('kernel-neighbor scan (local RDF)')
    ax_g.legend(fontsize=8)

    if with_beta:
        for i, r in enumerate(results):
            if r["beta_stats"] is None:
                continue
            c = f"C{i}"
            bc, med, p05, frac = r["beta_stats"]
            ax_bmed.plot(bc, med, color=c, marker='.', label=r["label"])
            ax_bmed.fill_between(bc, p05, med, color=c, alpha=0.25, lw=0)
            ax_bfrac.plot(bc, frac, color=c, marker='.', label=r["label"])

        for ax in (ax_bmed, ax_bfrac):
            mark_limiter_window(ax)
            ax.set_xscale('log')
            ax.set_xlabel(r'plasma $\beta$')
            ax.legend(fontsize=8)
        ax_bmed.axhline(r0["dx_exp"], color='k', lw=0.8, ls='--')
        ax_bmed.set_ylim(bottom=0)
        ax_bmed.set_ylabel(r'$d_\mathrm{NN}/h$ (median, 5th pct band)')
        ax_bmed.set_title(r'NN distance vs $\beta$  (shaded: $\hat H$ ramp $2<\beta<10$)')
        ax_bfrac.set_yscale('log')
        ax_bfrac.set_ylim(bottom=1e-7)
        ax_bfrac.set_ylabel('pairing fraction')
        ax_bfrac.set_title(r'pairing fraction vs $\beta$')

    plt.tight_layout()
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


# --all mode: pairing fraction and d_NN/h percentiles vs time (NN query only,
# no kernel scan -- steps are processed serially to bound memory).
def evolution(files, labels, stride, pair_threshold, outname):
    fig, (ax_frac, ax_d) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, (fname, label) in enumerate(zip(files, labels)):
        c = f"C{i}"
        steps = range(0, get_nsteps(fname), stride)
        times, fracs, meds, p05s, mins = [], [], [], [], []
        for step in steps:
            data = read_step(fname, step)
            tree = cKDTree(data["pos"], boxsize=data["boxsize"])
            dnn_h = nn_distances(tree, data["pos"]) / data["h"]
            del tree
            dx_exp = (32.0 * np.pi / (3.0 * data["ng0"])) ** (1.0 / 3.0)
            q_pair = pair_threshold * dx_exp
            times.append(data["time"])
            fracs.append(float((dnn_h < q_pair).mean()))
            meds.append(np.median(dnn_h))
            p05s.append(np.percentile(dnn_h, 5))
            mins.append(dnn_h.min())
            print(f"{label} step {step}: t={data['time']:.4f}  "
                  f"pairing={fracs[-1]:.3e}  min d_NN/h={mins[-1]:.3f}")

        ax_frac.plot(times, fracs, color=c, marker='.', label=label)
        ax_d.plot(times, meds, color=c, label=f"{label} median")
        ax_d.plot(times, p05s, color=c, ls='--', label=f"{label} p5")
        ax_d.plot(times, mins, color=c, ls=':', label=f"{label} min")

    ax_frac.set_yscale('log')
    ax_frac.set_ylim(bottom=1e-7)
    ax_frac.set_xlabel('time')
    ax_frac.set_ylabel('pairing fraction')
    ax_frac.legend(fontsize=8)
    ax_d.set_xlabel('time')
    ax_d.set_ylabel(r'$d_\mathrm{NN}/h$')
    ax_d.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Particle-pairing / tensile-instability diagnostics from SPHEXA HDF5 dumps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1])
    parser.add_argument("files", nargs="+", help="HDF5 dump file(s) to overlay")
    parser.add_argument("--step", type=int, default=None,
                        help="Step to analyze in every file (negative counts from the "
                             "end; default: last step of each file)")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata and exit")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Time series of pairing fraction / d_NN percentiles over all steps")
    parser.add_argument("--stride", type=int, default=1,
                        help="Step stride with --all (default: 1)")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated curve labels (default: parent directory names)")
    parser.add_argument("--sample", type=int, default=100000,
                        help="Subsample size for the kernel-neighbor scan (default: 100000)")
    parser.add_argument("--kmax", type=int, default=200,
                        help="Neighbors fetched per sampled particle in the kernel scan; "
                             "must exceed the max neighbor count inside 2h (default: 200)")
    parser.add_argument("--pair-threshold", type=float, default=0.5,
                        help="Pairing threshold as a fraction of the expected glass "
                             "spacing (default: 0.5)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: next to the first input file)")
    args = parser.parse_args()

    if args.info:
        for fname in args.files:
            print_metadata(fname)
        sys.exit(0)

    if args.labels is not None:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(args.files):
            parser.error(f"--labels needs {len(args.files)} entries, got {len(labels)}")
    else:
        labels = [os.path.basename(os.path.dirname(os.path.abspath(f))) or os.path.basename(f)
                  for f in args.files]

    outdir = os.path.dirname(os.path.abspath(args.files[0]))

    if args.all:
        outname = args.out or os.path.join(outdir, "pairing_evolution.png")
        evolution(args.files, labels, args.stride, args.pair_threshold, outname)
        sys.exit(0)

    results = [analyze(f, resolve_step(f, args.step), lbl,
                       args.sample, args.kmax, args.pair_threshold)
               for f, lbl in zip(args.files, labels)]
    outname = args.out or os.path.join(outdir, f"pairing_step{results[0]['step']}.png")
    plot_comparison(results, outname)
