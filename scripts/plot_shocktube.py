#!/usr/bin/env python3
"""Plot 1D shocktube diagnostics

Selects particles inside a thin tube along the x-axis (centered at the box
midpoint by default) and produces an 8-panel scatter figure versus x. Each
panel is resolved through `_h5_common.resolve_field`

Following Wissing & Shen (2020) Fig. 4, only the middle of the tube is shown
by default (--xlim, default [-1, 1]): the box spans [-2, 2] and a second
discontinuity sits at the periodic boundary. A high-resolution 1D reference
solution (briowu_reference.py) is overlaid at the dump time; --no-ref
disables it for non-Brio-Wu shocktubes.

Panels use fixed y-ranges (_DEFAULT_LIMITS) so figures from different runs
are directly comparable; --auto-ylim restores data-driven scaling. Passing
several dump files overlays them in one figure with --labels naming the runs 
in the legend.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

import os
import sys
import argparse

from _h5_common import (print_metadata, get_nsteps, resolve_field,
                        resolution_label, CLEAN_FONT, apply_clean_style)
import briowu_reference


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

# panel name -> column of the briowu_reference solution dict
_REF_COLUMNS = {
    'vx': 'vx', 'vy': 'vy', 'vz': 'vz',
    'rho': 'rho', 'u': 'u', 'p': 'p',
    'magneto::Bx': 'Bx', 'magneto::By': 'By', 'magneto::Bz': 'Bz',
}

# Fixed y-ranges so frames and runs share axes at t <= 0.2 including typical scatter spikes.
# --auto-ylim restores scaling.
_DEFAULT_LIMITS = {
    'vx':          (-0.35, 0.9),
    'vy':          (-2.4, 0.2),
    'magneto::Bx': (0.65, 0.85),
    'magneto::By': (-1.1, 1.1),
    'rho':         (0.0, 1.1),
    'u':           (0.5, 4.5),
    'p':           (0.0, 1.1),
    'log_divBerr': (-10.0, -1.0),
}

# scatter colors per run when several dumps are overlaid, in fixed order
_RUN_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')


def compute_tube_fields(fname, step, y0=None, z0=None, thickness=None,
                        panels=_DEFAULT_PANELS, xlim=None, ref=True):
    """Read a step and select particles in a tube along x.

    Tube center (y0, z0) defaults to the box midpoint; thickness is the
    half-width in each transverse direction, default 2 * median(h).
    Returns per-particle arrays for each requested panel, masked to the tube
    and to xlim (so shared y-limits ignore the periodic-boundary region).
    With ref=True the 1D reference solution is computed at the dump time.
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
        gamma = float(np.atleast_1d(s.attrs.get('gamma', briowu_reference.GAMMA))[0])

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
    if xlim is not None:
        mask &= (x >= xlim[0]) & (x <= xlim[1])
    n_in = int(mask.sum())
    extents = [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]
    res_label = resolution_label(extents, n_particles)

    print(f"Step {step}: time={time_val:.8f}, N={n_particles} ({res_label})")
    print(f"  tube center: (y={y0:.4f}, z={z0:.4f}), half-width: {thickness:.4f}")
    print(f"  particles in tube: {n_in} / {n_particles} ({100.0 * n_in / n_particles:.2f}%)")
    if n_in == 0:
        print("Error: tube is empty -- try a larger --thickness or different --y0/--z0")
        sys.exit(1)

    reference = None
    if ref:
        print(f"  computing 1D reference solution at t={time_val:.6f}...")
        reference = briowu_reference.get(time_val, gamma=gamma)

    return {
        'step':    step,
        'time':    time_val,
        'res_label': res_label,
        'n_tube':  n_in,
        'x':       x[mask],
        'panels':  tuple(panels),
        'data':    {name: values[name][mask] for name in panels},
        'labels':  labels,
        'ref':     reference,
    }


def render_shocktube(grids, title="Brio-Wu", limits=None, xlim=None,
                     ms=1.0, color='tab:blue', clean=False):
    """Plot precomputed tube selections; a list of grids is overlaid per panel."""
    gs = list(grids) if isinstance(grids, (list, tuple)) else [grids]
    g0 = gs[0]
    panels = g0['panels']
    n = len(panels)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True,
                             squeeze=False)
    flat = axes.flatten()

    colors = (color,) if len(gs) == 1 \
        else tuple(_RUN_COLORS[k % len(_RUN_COLORS)] for k in range(len(gs)))
    ref = g0.get('ref')
    for i, name in enumerate(panels):
        ax = flat[i]
        for g, c in zip(gs, colors):
            ax.plot(g['x'], g['data'][name], '.', ms=ms, color=c, alpha=0.3,
                    zorder=2, rasterized=True, label=g.get('label', 'SPHEXA'))
        if ref is not None and name in _REF_COLUMNS:
            ax.plot(ref['x'], ref[_REF_COLUMNS[name]], '-', color="black",
                    lw=1.5, zorder=3, label='reference')
        ax.set_ylabel(g0['labels'][name])
        if i == 0 and (ref is not None or len(gs) > 1):
            ax.legend(loc='best', fontsize=9, framealpha=0.9, markerscale=3)
        if limits and name in limits:
            ax.set_ylim(limits[name])
        if xlim is not None:
            ax.set_xlim(xlim)
        if i // cols == rows - 1:
            ax.set_xlabel('x')

    for j in range(n, len(flat)):
        flat[j].axis('off')

    if clean:
        plt.tight_layout()
    else:
        tube_n = "/".join(str(g['n_tube']) for g in gs)
        res = " | ".join(dict.fromkeys(g['res_label'] for g in gs))
        fig.suptitle(f"{title}, t={g0['time']:.4f}  (tube N={tube_n})")
        fig.text(0.98, 0.005, f"Resolution: {res}", fontsize=10, ha='right')
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


def _save_fig(fig, fname, step, suffix="", clean=False):
    outdir = os.path.dirname(os.path.abspath(fname))
    ext = 'pdf' if clean else 'png'
    outname = os.path.join(outdir, f"shocktube{suffix}_step{step}.{ext}")
    fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


def _as_list(fnames):
    return [fnames] if isinstance(fnames, str) else list(fnames)


def _last_step(fname):
    nsteps = get_nsteps(fname)
    if nsteps == 0:
        print(f"No steps found in {fname}")
        sys.exit(1)
    return nsteps - 1


def _run_label(fname):
    d = os.path.basename(os.path.dirname(os.path.abspath(fname)))
    return d or os.path.basename(fname)


def _gather(fnames, step, y0, z0, thickness, xlim, ref, labels):
    """One tube selection per file; the reference is computed only once."""
    grids = []
    for k, fn in enumerate(fnames):
        s = step if step is not None else _last_step(fn)
        g = compute_tube_fields(fn, s, y0, z0, thickness, xlim=xlim,
                                ref=ref and k == 0)
        g['label'] = labels[k] if labels \
            else ('SPHEXA' if len(fnames) == 1 else _run_label(fn))
        grids.append(g)
    return grids


def plot_shocktube(fnames, step, y0=None, z0=None, thickness=None,
                   title="Brio-Wu", xlim=None, ref=True, labels=None,
                   auto_ylim=False, clean=False):
    fnames = _as_list(fnames)
    grids = _gather(fnames, step, y0, z0, thickness, xlim, ref, labels)
    limits = shared_limits(grids) if auto_ylim else _DEFAULT_LIMITS
    fig = render_shocktube(grids, title=title, limits=limits, xlim=xlim, clean=clean)
    _save_fig(fig, fnames[0], grids[0]['step'],
              suffix="_compare" if len(fnames) > 1 else "", clean=clean)


def plot_all_steps(fnames, steps, y0=None, z0=None, thickness=None,
                   title="Brio-Wu", xlim=None, ref=True, labels=None,
                   auto_ylim=False, clean=False):
    """One PNG per step; fixed default y-limits keep frames and runs comparable."""
    fnames = _as_list(fnames)
    per_step = [_gather(fnames, s, y0, z0, thickness, xlim, ref, labels)
                for s in steps]
    limits = shared_limits([g for gs in per_step for g in gs]) if auto_ylim \
        else _DEFAULT_LIMITS
    suffix = "_compare" if len(fnames) > 1 else ""
    for gs in per_step:
        fig = render_shocktube(gs, title=title, limits=limits, xlim=xlim, clean=clean)
        _save_fig(fig, fnames[0], gs[0]['step'], suffix=suffix, clean=clean)


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
            "  %(prog)s a/dump.h5 b/dump.h5 --labels ARSLR noSLR\n"
            "                                  Overlay two runs in one figure\n"
            "\nUse plot_gif.py for animated GIFs over a step range.\n"
        ),
    )
    parser.add_argument("files", nargs="+", metavar="file",
                        help="HDF5 input file(s); extra files are overlaid "
                             "for run comparison")
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
    parser.add_argument("--xlim", type=float, nargs=2, default=(-1.0, 1.0),
                        metavar=("XMIN", "XMAX"),
                        help="Plotted x range (default: -1 1, the middle of the "
                             "[-2,2] tube, away from the periodic boundary)")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip the 1D Brio-Wu reference overlay")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend label per input file "
                             "(default: run directory name)")
    parser.add_argument("--auto-ylim", action="store_true",
                        help="Data-driven y-limits instead of the fixed "
                             "comparison ranges")
    parser.add_argument("--title", default="Brio-Wu",
                        help="Plot title prefix (default: 'Brio-Wu')")
    parser.add_argument("--clean", action="store_true",
                        help="Publish mode for thesis figures: save PDF instead of PNG, "
                             "drop the title and resolution label (those go in the "
                             f"caption), and render text in {CLEAN_FONT} "
                             "(CLEAN_FONT in _h5_common.py).")

    args = parser.parse_args()

    if args.clean:
        apply_clean_style()

    # nargs='+' swallows a trailing step number; pull it back out
    if args.step is None and len(args.files) > 1 \
            and args.files[-1].lstrip('+-').isdigit():
        args.step = int(args.files.pop())

    if args.labels and len(args.labels) != len(args.files):
        parser.error("--labels needs one label per input file")

    if args.info:
        for f in args.files:
            print_metadata(f)
        sys.exit(0)

    common = dict(y0=args.y0, z0=args.z0, thickness=args.thickness,
                  title=args.title, xlim=tuple(args.xlim) if args.xlim else None,
                  ref=not args.no_ref, labels=args.labels,
                  auto_ylim=args.auto_ylim, clean=args.clean)

    if args.all:
        nsteps = min(get_nsteps(f) for f in args.files)
        if nsteps == 0:
            print("No common steps found")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        plot_all_steps(args.files, list(range(nsteps)), **common)
    else:
        plot_shocktube(args.files, args.step, **common)
