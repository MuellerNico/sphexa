#!/usr/bin/env python3
"""Plot 1D line cuts of any (raw or derived) field, overlaying multiple files.

Overlaid files take their color by argument position from the shared
categorical scheme palette (_h5_common.SCHEME_ORDER: a05, SLR, SLRB, SLRB2),
so a scheme keeps one color across every plotting script.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse

from _h5_common import (print_metadata, get_nsteps, resolve_field,
                        CLEAN_FONT, apply_clean_style, scheme_colors,
                        scheme_label, apply_sci_ticks, apply_log_ticks,
                        SCHEME_ORDER)
from plot_slice import cubic_spline_3d

# cut axis -> the two perpendicular axes (in x<y<z order, matching --pos)
_PERP_AXES = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}


# SPH-interpolate a per-particle field onto n_samples points along a line
# parallel to `axis` through pos=(p1, p2) on the perpendicular axes. Same
# normalized 3D-kernel interpolation as plot_slice restricted to particles
# whose kernel support reaches the line. Returns a dict with the sampled line
# and (optionally) the raw particles within one smoothing length of it.
def read_line_data(fname, step, field, axis, pos, n_samples, scatter):
    a1, a2 = _PERP_AXES[axis]
    with h5py.File(fname, "r") as f:
        key = f"Step#{step}"
        if key not in f:
            print(f"Error: {key} not found in {fname}")
            print_metadata(fname)
            sys.exit(1)
        s = f[key]
        if "h" not in s:
            sys.exit(f"Error: 'h' not in {fname}:{key}; line cuts need smoothing lengths")
        xs = np.asarray(s[axis])
        d1 = np.asarray(s[a1]) - pos[0]
        d2 = np.asarray(s[a2]) - pos[1]
        h = np.asarray(s["h"])
        values, label = resolve_field(s, field)
        time_val = s.attrs["time"][0]

    perp2 = d1**2 + d2**2
    mask = perp2 < (2.0 * h)**2
    print(f"{fname} Step#{step}: t={time_val:.6f}, "
          f"{mask.sum()}/{len(xs)} particles reach the line, "
          f"{field}: [{np.nanmin(values):.6f}, {np.nanmax(values):.6f}]")
    if not mask.any():
        sys.exit(f"Error: no particles within kernel support of "
                 f"{a1}={pos[0]}, {a2}={pos[1]} in {fname}")
    xs, perp2, h, values = xs[mask], perp2[mask], h[mask], values[mask]

    si = np.linspace(xs.min(), xs.max(), n_samples)
    num = np.zeros(n_samples)
    den = np.zeros(n_samples)
    for i, x0 in enumerate(si):
        r = np.sqrt((xs - x0)**2 + perp2)
        w = cubic_spline_3d(r / h)
        den[i] = w.sum()
        num[i] = (w * values).sum()
    line = np.full(n_samples, np.nan)
    ok = den > 0
    line[ok] = num[ok] / den[ok]

    out = {'time': time_val, 'label': label, 'si': si, 'line': line}
    if scatter:
        # tighter slab than the interpolation mask so perpendicular structure
        # doesn't masquerade as noise in the raw samples
        near = perp2 < h**2
        out['sx'] = xs[near]
        out['sv'] = values[near]
    return out


# Default legend label: last two path components, so identical dump names in
# different out/<jobid>/ dirs stay distinguishable.
def _default_label(fname):
    parts = os.path.normpath(os.path.abspath(fname)).split(os.sep)
    return os.path.splitext(os.sep.join(parts[-2:]))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 1D SPH-interpolated line cuts from SPHEXA HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s dump.h5                                rho along x through (y,z)=(0,0), final step\n"
            "  %(prog)s dump.h5 -i                              Print metadata + available fields\n"
            "  %(prog)s dump.h5 --field curlBmag --pos 0.3125 0 --step 10\n"
            "  %(prog)s a.h5 b.h5 --field magneto::By --labels 'low AR','high AR'\n"
            "  %(prog)s a.h5 a.h5 --step 0,20 --scatter         Same file, two steps, raw particles\n"
            "\nRun colors are positional: pass the dumps in the order "
            f"{', '.join(SCHEME_ORDER)}.\n"
        ),
    )
    parser.add_argument("files", nargs="+", help="HDF5 input file(s); multiple files overlay")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata + available fields and exit")
    parser.add_argument("--step", default=None,
                        help="Step number, or comma-separated list (one per file). "
                             "Default: final step of each file.")
    parser.add_argument("--field", default="rho",
                        help="Field to plot (raw dataset name or derived; default: rho)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x",
                        help="Axis the cut runs along (default: x)")
    parser.add_argument("--pos", nargs=2, type=float, default=[0.0, 0.0],
                        metavar=("P1", "P2"),
                        help="Cut position on the two perpendicular axes, in x<y<z "
                             "order (default: 0 0)")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated legend labels, one per file (default: "
                             f"the scheme name for that position, "
                             f"{'/'.join(SCHEME_ORDER)}, then dir/filename; + time)")
    parser.add_argument("-n", "--samples", type=int, default=512,
                        help="Sample points along the line (default: 512)")
    parser.add_argument("--scatter", action="store_true",
                        help="Also draw raw particles within one smoothing length "
                             "of the line (shows particle noise the interpolation smooths over)")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Log-scale y axis")
    parser.add_argument("--title", default=None,
                        help="Plot title (default: field label + cut position)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path (default: next to the first input file)")
    parser.add_argument("--clean", action="store_true",
                        help="Publish mode for thesis figures: save PDF instead of PNG, "
                             "drop the title (it goes in the caption), and render text "
                             f"in {CLEAN_FONT} (CLEAN_FONT in _h5_common.py).")

    args = parser.parse_args()

    if args.clean:
        apply_clean_style()

    if args.info:
        for f in args.files:
            print_metadata(f)
        sys.exit(0)

    if args.step is None:
        steps = [get_nsteps(f) - 1 for f in args.files]
    else:
        steps = [int(p) for p in args.step.split(",")]
        if len(steps) == 1:
            steps *= len(args.files)
        if len(steps) != len(args.files):
            parser.error(f"--step got {len(steps)} values for {len(args.files)} files")

    if args.labels is not None:
        labels = [l.strip() for l in args.labels.split(",")]
        if len(labels) != len(args.files):
            parser.error(f"--labels got {len(labels)} values for {len(args.files)} files")
    else:
        labels = None

    runs = [read_line_data(f, st, args.field, args.axis, args.pos,
                           args.samples, args.scatter)
            for f, st in zip(args.files, steps)]
    if labels is None:
        # the time stays in the label: the same file at two steps is a valid
        # overlay, and there the positional scheme name doesn't distinguish them
        labels = [f"{(scheme_label(k) if len(runs) > 1 else None) or _default_label(f)}"
                  f" (t={r['time']:.4f})"
                  for k, (f, r) in enumerate(zip(args.files, runs))]

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, lbl, c in zip(runs, labels, scheme_colors(len(runs))):
        ax.plot(r['si'], r['line'], lw=1.2, color=c, label=lbl)
        if args.scatter:
            ax.scatter(r['sx'], r['sv'], s=2, color=c,
                       alpha=0.3, linewidths=0, rasterized=True)
    if args.log:
        ax.set_yscale('log')
    a1, a2 = _PERP_AXES[args.axis]
    ax.set_xlabel(args.axis)
    ax.set_ylabel(runs[0]['label'])
    apply_sci_ticks(ax)
    apply_log_ticks(ax)
    if not args.clean:
        header = args.title if args.title is not None else runs[0]['label']
        ax.set_title(f"{header} along {args.axis} "
                     f"({a1}={args.pos[0]:+.4f}, {a2}={args.pos[1]:+.4f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if args.output is not None:
        outname = args.output
    else:
        outdir = os.path.dirname(os.path.abspath(args.files[0]))
        short = args.field.split('::')[-1]
        ext = 'pdf' if args.clean else 'png'
        outname = os.path.join(outdir, f"linecut_{short}_step{steps[0]}_{args.axis}"
                                       f"_{a1}{args.pos[0]:+.4f}_{a2}{args.pos[1]:+.4f}.{ext}")
    fig.savefig(outname, dpi=300 if args.clean else 150, bbox_inches='tight')
    print(f"Saved: {outname}")
