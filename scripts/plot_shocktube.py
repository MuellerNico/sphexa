#!/usr/bin/env python3
"""Plot 1D shocktube diagnostics (e.g. Brio-Wu) from SPHEXA MHD output.

Selects particles inside a thin tube along the x-axis (centered at the box
midpoint by default) and produces an 8-panel scatter figure versus x. Each
panel is resolved through `_h5_common.resolve_field`, so panels may be raw
HDF5 dataset names or derived quantities like `log_divBerr`.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse

from _h5_common import print_metadata, get_nsteps, resolve_field, resolution_label


# Default panel layout: top row vx, vy, Bx, By; bottom row rho, u, P, log divBerr.
# Each entry is just a field name -- resolve_field supplies the y-axis label.
_DEFAULT_PANELS = (
    'vx',
    'vy',
    'magneto::Bx',
    'magneto::By',
    'rho',
    'u',
    'p',
    'log_divBerr',
)


def compute_tube_fields(fname, step, y0=None, z0=None, thickness=None,
                        panels=_DEFAULT_PANELS):
    """Read a step and select particles in a tube along x.

    Tube center (y0, z0) defaults to the box midpoint; thickness is the
    half-width in each transverse direction, default 2 * median(h).
    Returns per-particle arrays for each requested panel, masked to the tube.
    """
    print(f"Reading step {step} from {fname}...")
    with h5py.File(fname, "r") as f:
        key = f"Step#{step}"
        if key not in f:
            print(f"Error: {key} not found in {fname}")
            print_metadata(fname)
            sys.exit(1)
        s = f[key]

        x = np.array(s['x'])
        y = np.array(s['y'])
        z = np.array(s['z'])
        h = np.array(s['h'])
        time_val = s.attrs['time'][0]

        # Resolve each panel before masking so the resolver sees a coherent
        # full-step group (matters for derived fields that need multiple
        # raw arrays). Missing fields are filled with zeros so the figure
        # layout stays comparable across dumps.
        values = {}
        labels = {}
        for name in panels:
            try:
                v, lbl = resolve_field(s, name)
            except KeyError:
                print(f"  field '{name}' unavailable in this dump -- filling with zeros")
                v = np.zeros(len(x))
                lbl = f"{name} (missing)"
            values[name] = v
            labels[name] = lbl

    n_particles = len(x)
    if y0 is None:
        y0 = 0.5 * (y.min() + y.max())
    if z0 is None:
        z0 = 0.5 * (z.min() + z.max())
    if thickness is None:
        thickness = 2.0 * float(np.median(h))

    mask = (np.abs(y - y0) < thickness) & (np.abs(z - z0) < thickness)
    n_in = int(mask.sum())
    extents = [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]
    res_label = resolution_label(extents, n_particles)

    print(f"Step {step}: time={time_val:.8f}, N={n_particles} ({res_label})")
    print(f"  tube center: (y={y0:.4f}, z={z0:.4f}), half-width: {thickness:.4f}")
    print(f"  particles in tube: {n_in} / {n_particles} ({100.0 * n_in / n_particles:.2f}%)")
    if n_in == 0:
        print("Error: tube is empty -- try a larger --thickness or different --y0/--z0")
        sys.exit(1)

    return {
        'step':    step,
        'time':    time_val,
        'res_label': res_label,
        'n_tube':  n_in,
        'x':       x[mask],
        'panels':  tuple(panels),
        'data':    {name: values[name][mask] for name in panels},
        'labels':  labels,
    }


def render_shocktube(grids, title="Brio-Wu", limits=None, xlim=None,
                     ms=2.0, color='k'):
    """Plot precomputed tube selection; returns a figure."""
    panels = grids['panels']
    n = len(panels)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True,
                             squeeze=False)
    flat = axes.flatten()

    for i, name in enumerate(panels):
        ax = flat[i]
        ax.plot(grids['x'], grids['data'][name], '.', ms=ms, color=color)
        ax.set_ylabel(grids['labels'][name])
        if limits and name in limits:
            ax.set_ylim(limits[name])
        if xlim is not None:
            ax.set_xlim(xlim)
        if i // cols == rows - 1:
            ax.set_xlabel('x')

    for j in range(n, len(flat)):
        flat[j].axis('off')

    fig.suptitle(f"{title}, t={grids['time']:.4f}  (tube N={grids['n_tube']})")
    fig.text(0.98, 0.005, f"Resolution: {grids['res_label']}", fontsize=10, ha='right')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def shared_limits(grids):
    """Per-panel (ymin, ymax) spanning every step, so frames share scales."""
    limits = {}
    for name in grids[0]['panels']:
        lo = min(np.nanmin(g['data'][name]) for g in grids)
        hi = max(np.nanmax(g['data'][name]) for g in grids)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            continue
        pad = 0.05 * (hi - lo)
        limits[name] = (lo - pad, hi + pad)
    return limits


def _save_png(fig, fname, step):
    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"shocktube_step{step}.png")
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


def plot_shocktube(fname, step, y0=None, z0=None, thickness=None,
                   title="Brio-Wu", xlim=None):
    g = compute_tube_fields(fname, step, y0, z0, thickness)
    fig = render_shocktube(g, title=title, xlim=xlim)
    _save_png(fig, fname, step)


def plot_all_steps(fname, steps, y0=None, z0=None, thickness=None,
                   title="Brio-Wu", xlim=None):
    """One PNG per step, with one set of y-limits shared across all frames."""
    grids = [compute_tube_fields(fname, s, y0, z0, thickness) for s in steps]
    limits = shared_limits(grids)
    for g in grids:
        fig = render_shocktube(g, title=title, limits=limits, xlim=xlim)
        _save_png(fig, fname, g['step'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Brio-Wu shocktube diagnostics from SPHEXA MHD HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s data.h5                PNG of the final step (default)\n"
            "  %(prog)s data.h5 -i             Print metadata + available fields\n"
            "  %(prog)s data.h5 5              PNG for step 5\n"
            "  %(prog)s data.h5 --all          PNG for every step in the file\n"
            "  %(prog)s data.h5 5 --thickness 0.05\n"
            "\nUse plot_gif.py for animated GIFs over a step range.\n"
        ),
    )
    parser.add_argument("file", help="HDF5 input file")
    parser.add_argument("step", nargs="?", type=int,
                        help="Step number. Omit to plot the final step.")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata + available fields and exit")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--y0", type=float, default=None,
                        help="y coordinate of tube center (default: box midpoint)")
    parser.add_argument("--z0", type=float, default=None,
                        help="z coordinate of tube center (default: box midpoint)")
    parser.add_argument("--thickness", type=float, default=None,
                        help="Tube half-width in y and z (default: 2 * median(h))")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                        help="Restrict the plotted x range")
    parser.add_argument("--title", default="Brio-Wu",
                        help="Plot title prefix (default: 'Brio-Wu')")

    args = parser.parse_args()

    if args.info:
        print_metadata(args.file)
        sys.exit(0)

    common = dict(y0=args.y0, z0=args.z0, thickness=args.thickness,
                  title=args.title, xlim=tuple(args.xlim) if args.xlim else None)

    if args.all:
        nsteps = get_nsteps(args.file)
        if nsteps == 0:
            print(f"No steps found in {args.file}")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        plot_all_steps(args.file, list(range(nsteps)), **common)
    else:
        if args.step is None:
            nsteps = get_nsteps(args.file)
            if nsteps == 0:
                print(f"No steps found in {args.file}")
                sys.exit(1)
            step = nsteps - 1
        else:
            step = args.step
        plot_shocktube(args.file, step, **common)
