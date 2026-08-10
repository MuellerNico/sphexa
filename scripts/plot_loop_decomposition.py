#!/usr/bin/env python3
"""Split the MHD-loop magnetic energy into the coherent loop and the numerical
garbage that masks its decay in the plain eMag diagnostic.

eMag alone is not a quality metric for this test: velocity noise drives a
small-scale dynamo through the ideal induction term, so tangled field is created
at the same time the loop is dissipated and the two partly cancel in the total.

    E_tot = E_coh + E_fluct + E_out

  E_coh   energy of the best-fit ideal loop: volume-weighted mean B_phi inside
          r < R, squared. This is the quantity that tracks visual structure.
  E_fluct scatter about that mean inside the loop
  E_out   field that has leaked outside r < R
  E_z     B_z energy, cross-cutting the above; identically zero in the exact
          solution, so it is a pure numerical-error metric

Usage:
    plot_loop_decomposition.py out/<id>/dump.h5
    plot_loop_decomposition.py out/*/dump.h5 --labels AV,AVSLR,AVSW
"""

import os
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

from _h5_common import (get_nsteps, loop_frame, CATEGORICAL_COLORS,
                        apply_clean_style)


def decompose(s, radius):
    mu0 = float(np.atleast_1d(s.attrs.get('mu_0', 1.0))[0])
    B = [np.asarray(s[f'magneto::B{c}']) for c in 'xyz']
    V = np.asarray(s['m']) / np.asarray(s['rho'])          # == xm/kx
    r, ex, ey = loop_frame(s)

    B2 = sum(b * b for b in B)
    E_tot = 0.5 * np.sum(V * B2) / mu0
    E_z = 0.5 * np.sum(V * B[2]**2) / mu0

    inside = r < radius
    Vin = V[inside]
    Vsum = Vin.sum()
    Bphi = ex[inside] * B[0][inside] + ey[inside] * B[1][inside]
    Bmean = np.sum(Vin * Bphi) / Vsum

    E_coh = 0.5 * Vsum * Bmean**2 / mu0
    E_in = 0.5 * np.sum(Vin * B2[inside]) / mu0
    return dict(t=float(np.atleast_1d(s.attrs['time'])[0]), E_tot=E_tot,
                E_coh=E_coh, E_fluct=E_in - E_coh, E_out=E_tot - E_in, E_z=E_z,
                Bmean=Bmean)


def series(fname, radius):
    out = []
    with h5py.File(fname, 'r') as f:
        for i in range(get_nsteps(fname)):
            out.append(decompose(f[f'Step#{i}'], radius))
    E0 = out[0]['E_tot']
    for row in out:
        for k in ('E_tot', 'E_coh', 'E_fluct', 'E_out', 'E_z'):
            row[k] /= E0
    return out


def plot(files, labels, radius, outname, clean=False):
    if clean:
        apply_clean_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.subplots_adjust(wspace=0.26, bottom=0.14, top=0.90)

    for n, (fname, label) in enumerate(zip(files, labels)):
        rows = series(fname, radius)
        color = CATEGORICAL_COLORS[n % len(CATEGORICAL_COLORS)]
        t = [row['t'] for row in rows]
        g = lambda k: [row[k] for row in rows]

        print(f"\n=== {label}  ({fname})")
        print(f"{'t':>8}{'E_tot':>9}{'E_coh':>9}{'E_fluct':>9}{'E_out':>9}{'E_z':>9}")
        for row in rows:
            print(f"{row['t']:8.3f}{row['E_tot']:9.4f}{row['E_coh']:9.4f}"
                  f"{row['E_fluct']:9.4f}{row['E_out']:9.4f}{row['E_z']:9.4f}")

        axes[0].plot(t, g('E_tot'), color=color, linewidth=1.4, linestyle='--')
        axes[0].plot(t, g('E_coh'), color=color, linewidth=1.8, label=label)
        axes[1].plot(t, np.add(g('E_fluct'), g('E_out')), color=color,
                     linewidth=1.8, label=label)
        axes[2].plot(t, g('E_z'), color=color, linewidth=1.8, label=label)

    axes[0].set_ylabel(r"$E / E_0$")
    axes[0].set_title("solid: coherent loop    dashed: total eMag", fontsize=10)
    axes[1].set_ylabel(r"$(E_\mathrm{fluct} + E_\mathrm{out}) / E_0$")
    axes[1].set_title("tangled + leaked field", fontsize=10)
    axes[2].set_ylabel(r"$E_z / E_0$")
    axes[2].set_title(r"$B_z$ energy (zero in the exact solution)", fontsize=10)

    for ax in axes:
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
    print(f"\nSaved: {outname}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MHD-loop magnetic energy decomposition (coherent / fluctuating / leaked)")
    parser.add_argument("files", nargs="+", help="one or more dump.h5")
    parser.add_argument("--labels", default=None,
                        help="comma-separated legend labels, one per file "
                             "(default: parent directory name)")
    parser.add_argument("--radius", type=float, default=0.3,
                        help="loop radius R0 (default: 0.3)")
    parser.add_argument("-o", "--out", default=None,
                        help="output file (default: loop_decomposition.pdf next to the first dump)")
    parser.add_argument("--clean", action="store_true",
                        help="serif publication style")
    args = parser.parse_args()

    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(args.files):
            parser.error(f"got {len(labels)} labels for {len(args.files)} files")
    else:
        labels = [os.path.basename(os.path.dirname(os.path.abspath(f))) for f in args.files]

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.files[0])),
                                   "loop_decomposition.pdf")
    plot(args.files, labels, args.radius, out, clean=args.clean)
