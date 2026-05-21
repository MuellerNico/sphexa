#!/usr/bin/env python3
"""Animated GIFs over a step range, for slice or shocktube plots.

Reuses the `compute_*` / `render_*` / `shared_*` helpers from `plot_slice.py`
and `plot_shocktube.py`. Slice grids (the expensive step) are computed in
parallel; frame rendering is cheap and serial. Ranges longer than `--max-frames`
(default 100) are subsampled by striding.
"""

import argparse
import io
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import plot_slice
import plot_shocktube


def _stride(start, end, max_frames):
    all_steps = list(range(start, end + 1))
    if len(all_steps) > max_frames:
        stride = len(all_steps) // max_frames
        steps = all_steps[::stride]
        print(f"Range has {len(all_steps)} steps, using stride={stride} -> {len(steps)} frames")
        return steps
    return all_steps


def _fig_to_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    img = Image.open(io.BytesIO(buf.getvalue())).copy()
    buf.close()
    return img


def _save_gif(frames, outname, duration=100):
    frames[0].save(
        outname,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved GIF ({len(frames)} frames): {outname}")


# --- slice ----------------------------------------------------------------

def _slice_worker(args):
    return plot_slice.compute_slice_grids(*args)


def make_slice_gif(fname, start, end, field='rho', resolution=256,
                   slice_axis='z', slice_pos=0.0, title=None,
                   vmin=None, vmax=None, cmap='bone_r',
                   n_workers=None, max_frames=100):
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    steps = _stride(start, end, max_frames)
    work = [(fname, s, field, resolution, slice_axis, slice_pos) for s in steps]

    print(f"Computing {len(steps)} slice grids using {n_workers} workers...")
    grids_by_step = {}
    with Pool(n_workers) as pool:
        for i, g in enumerate(pool.imap_unordered(_slice_worker, work)):
            grids_by_step[g['step']] = g
            print(f"  [{i + 1}/{len(steps)}] Step {g['step']} done", flush=True)

    ordered = [grids_by_step[s] for s in steps]
    vmin, vmax = plot_slice.shared_ranges(ordered, vmin, vmax)
    print(f"Shared {field} scale: [{vmin:.6f}, {vmax:.6f}]")

    print("Rendering frames and assembling GIF...")
    frames = [_fig_to_frame(plot_slice.render_slice(g, slice_axis, slice_pos,
                                                    title, vmin, vmax, cmap))
              for g in ordered]

    outdir = os.path.dirname(os.path.abspath(fname))
    short = field.split('::')[-1]
    outname = os.path.join(outdir,
                           f"slice_{short}_steps{start}-{end}_{slice_axis}{slice_pos:+.4f}.gif")
    _save_gif(frames, outname)


# --- shocktube ------------------------------------------------------------

def _tube_worker(args):
    return plot_shocktube.compute_tube_fields(*args)


def make_shocktube_gif(fname, start, end, y0=None, z0=None, thickness=None,
                       title="Brio-Wu", xlim=None,
                       n_workers=None, max_frames=100):
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    steps = _stride(start, end, max_frames)
    work = [(fname, s, y0, z0, thickness) for s in steps]

    print(f"Computing {len(steps)} tube selections using {n_workers} workers...")
    grids_by_step = {}
    with Pool(n_workers) as pool:
        for i, g in enumerate(pool.imap_unordered(_tube_worker, work)):
            grids_by_step[g['step']] = g
            print(f"  [{i + 1}/{len(steps)}] Step {g['step']} done", flush=True)

    ordered = [grids_by_step[s] for s in steps]
    limits = plot_shocktube.shared_limits(ordered)

    print("Rendering frames and assembling GIF...")
    frames = [_fig_to_frame(plot_shocktube.render_shocktube(g, title=title,
                                                             limits=limits, xlim=xlim))
              for g in ordered]

    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"shocktube_steps{start}-{end}.gif")
    _save_gif(frames, outname)


# --- CLI ------------------------------------------------------------------

def _parse_range(s):
    start, end = s.split('-', 1)
    return int(start), int(end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animated GIF over a step range, for slice or shocktube plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s slice data.h5 0-20 --field rho\n"
            "  %(prog)s slice data.h5 0-100 --field Bmag --cmap viridis\n"
            "  %(prog)s shocktube data.h5 0-50\n"
        ),
    )
    sub = parser.add_subparsers(dest='kind', required=True)

    # slice subcommand
    ps = sub.add_parser('slice', help='2D slice GIF')
    ps.add_argument('file')
    ps.add_argument('range', help='Step range, e.g. 0-20')
    ps.add_argument('--field', default='rho',
                    help='Field to plot (raw or derived; default: rho)')
    ps.add_argument('--axis', choices=['x', 'y', 'z'], default='z',
                    help='Axis normal to the slice plane (default: z)')
    ps.add_argument('--pos', type=float, default=0.0,
                    help='Position along the slice axis (default: 0.0)')
    ps.add_argument('-r', '--resolution', type=int, default=256,
                    help='Interpolation grid resolution per side (default: 256)')
    ps.add_argument('--title', default=None,
                    help='Plot title prefix (default: field label)')
    ps.add_argument('--vmin', type=float, default=None)
    ps.add_argument('--vmax', type=float, default=None)
    ps.add_argument('--cmap', default='bone_r')
    ps.add_argument('-j', '--workers', type=int, default=None,
                    help='Number of worker processes (default: min(cpu_count, 16))')
    ps.add_argument('--max-frames', type=int, default=100,
                    help='Cap on frame count; longer ranges are strided (default: 100)')

    # shocktube subcommand
    pt = sub.add_parser('shocktube', help='1D tube-scatter GIF')
    pt.add_argument('file')
    pt.add_argument('range', help='Step range, e.g. 0-20')
    pt.add_argument('--y0', type=float, default=None)
    pt.add_argument('--z0', type=float, default=None)
    pt.add_argument('--thickness', type=float, default=None)
    pt.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('XMIN', 'XMAX'))
    pt.add_argument('--title', default='Brio-Wu')
    pt.add_argument('-j', '--workers', type=int, default=None)
    pt.add_argument('--max-frames', type=int, default=100)

    args = parser.parse_args()
    start, end = _parse_range(args.range)

    if args.kind == 'slice':
        make_slice_gif(args.file, start, end,
                       field=args.field, resolution=args.resolution,
                       slice_axis=args.axis, slice_pos=args.pos,
                       title=args.title, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
                       n_workers=args.workers, max_frames=args.max_frames)
    else:
        make_shocktube_gif(args.file, start, end,
                           y0=args.y0, z0=args.z0, thickness=args.thickness,
                           title=args.title, xlim=tuple(args.xlim) if args.xlim else None,
                           n_workers=args.workers, max_frames=args.max_frames)
