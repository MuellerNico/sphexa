#!/usr/bin/env python3
"""Plot 2D SPH-interpolated slices of any (raw or derived) field. """

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

import os
import sys
import argparse
import functools
from multiprocessing import Pool

from _h5_common import (print_metadata, get_nsteps, resolve_field,
                        resolution_label, CLEAN_FONT, apply_clean_style)


# M4 cubic-spline SPH kernel shape in 3D (q = r/h). The 1/h**3 factor is left
# off; it cancels in the normalized interpolation in sph_scatter_to_grid.
def cubic_spline_3d(q):
    sigma = 1.0 / np.pi
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    w[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * w


# SPH-interpolate a per-particle field onto a 2D slice grid. Each particle
# uses the full 3D kernel attenuated by its offset from the plane (not a thin
# slab), normalized as sum(v*w)/sum(w) so a constant field is reproduced
# exactly. Returns (xi, yi, field) with xi/yi the cell edges.
def sph_scatter_to_grid(xs, ys, zoff, hs, values, resolution):
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution

    xc = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, resolution)
    yc = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, resolution)

    value_grid = np.zeros((resolution, resolution))
    weight_grid = np.zeros((resolution, resolution))

    for i in range(len(xs)):
        hi = hs[i]
        support = 2.0 * hi
        inplane = np.sqrt(max(support * support - zoff[i] * zoff[i], 0.0))

        ix_lo = max(int((xs[i] - inplane - xmin) / dx), 0)
        ix_hi = min(int((xs[i] + inplane - xmin) / dx) + 1, resolution)
        iy_lo = max(int((ys[i] - inplane - ymin) / dy), 0)
        iy_hi = min(int((ys[i] + inplane - ymin) / dy) + 1, resolution)
        if ix_lo >= ix_hi or iy_lo >= iy_hi:
            continue

        gx = xc[ix_lo:ix_hi]
        gy = yc[iy_lo:iy_hi]
        gxx, gyy = np.meshgrid(gx, gy, indexing='ij')

        r = np.sqrt((gxx - xs[i])**2 + (gyy - ys[i])**2 + zoff[i]**2)
        w = cubic_spline_3d(r / hi)

        value_grid[iy_lo:iy_hi, ix_lo:ix_hi] += (w * values[i]).T
        weight_grid[iy_lo:iy_hi, ix_lo:ix_hi] += w.T

    valid = weight_grid > 0
    field = np.full((resolution, resolution), np.nan)
    field[valid] = value_grid[valid] / weight_grid[valid]

    xi, yi = np.meshgrid(
        np.linspace(xmin, xmax, resolution + 1),
        np.linspace(ymin, ymax, resolution + 1),
    )
    return xi, yi, field


# map slice axis -> plotting axes
_PLOT_AXES = {
    'z': ('x', 'y'),
    'y': ('x', 'z'),
    'x': ('y', 'z'),
}

# index of the vector component aligned with each spatial axis, used to pick the
# two in-plane components of a --fieldlines vector for the streamplot.
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


# Short stem for a component name, e.g. 'magneto::Bx' -> 'B', 'vy' -> 'v'.
def _vector_stem(name):
    short = name.split('::')[-1]
    return short[:-1] if short[-1:] in 'xyz' else short


# Read one step and return a dict of metadata plus the slice data: either an
# interpolated grid (xi, yi, values) or, with scatter=True, the raw per-particle
# samples in the slab (xs, ys, values). No plotting here.
def compute_slice_grids(fname, step, field, resolution,
                        slice_axis, slice_pos, scatter=False,
                        fieldlines=None):
    print(f"Reading step {step} from {fname}...")
    ha, va = _PLOT_AXES[slice_axis]
    with h5py.File(fname, "r") as f:
        key = f"Step#{step}"
        if key not in f:
            print(f"Error: {key} not found in {fname}")
            print_metadata(fname)
            sys.exit(1)
        s = f[key]

        coords = {
            'x': np.array(s["x"]),
            'y': np.array(s["y"]),
            'z': np.array(s["z"]),
        }
        values, label = resolve_field(s, field)
        time_val = s.attrs["time"][0]

        # Resolve the two vector components lying in the slice plane.
        stream = None
        if fieldlines is not None:
            uf = fieldlines[_AXIS_INDEX[ha]]
            vf = fieldlines[_AXIS_INDEX[va]]
            u_vals, _ = resolve_field(s, uf)
            v_vals, _ = resolve_field(s, vf)
            stream = (u_vals, v_vals, _vector_stem(uf))

        if "h" in s:
            h = np.array(s["h"])
        else:
            x, y, z = coords['x'], coords['y'], coords['z']
            n_particles = len(coords['x'])
            vol = (x.max() - x.min()) * (y.max() - y.min()) * (z.max() - z.min())
            h_est = 1.2 * (vol / n_particles) ** (1.0 / 3.0)
            h = np.full(n_particles, h_est)
            print(f"  h not in file, using estimate h={h_est:.6f}")

    n_particles = len(coords['x'])
    extents = [coords[ax].max() - coords[ax].min() for ax in ('x', 'y', 'z')]
    res_label = resolution_label(extents, n_particles)
    print(f"Step {step}: time={time_val:.8f}, N={n_particles} ({res_label})")
    for ax, vals in coords.items():
        print(f"  {ax}: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  {field}: [{np.nanmin(values):.6f}, {np.nanmax(values):.6f}]")

    mask = np.abs(coords[slice_axis] - slice_pos) < 2.0 * h
    print(f"  Particles in slice: {mask.sum()} / {n_particles} "
          f"({mask.sum() / n_particles * 100:.2f}%)")

    xs   = coords[ha][mask]
    ys   = coords[va][mask]
    zoff = coords[slice_axis][mask] - slice_pos
    hs   = h[mask]

    if scatter:
        return {'step': step, 'time': time_val, 'res_label': res_label,
                'field': field, 'label': label, 'mode': 'scatter',
                'xs': xs, 'ys': ys, 'values': values[mask],
                'fieldlines': fieldlines is not None}

    print(f"  Interpolating onto {resolution}x{resolution} grid...")
    xi, yi, di = sph_scatter_to_grid(xs, ys, zoff, hs, values[mask], resolution)

    out = {'step': step, 'time': time_val, 'res_label': res_label,
           'field': field, 'label': label, 'mode': 'grid',
           'xi': xi, 'yi': yi, 'values': di}

    if stream is not None:
        u_vals, v_vals, stem = stream
        _, _, ug = sph_scatter_to_grid(xs, ys, zoff, hs, u_vals[mask], resolution)
        _, _, vg = sph_scatter_to_grid(xs, ys, zoff, hs, v_vals[mask], resolution)
        out.update(stream_u=ug, stream_v=vg, stream_stem=stem)

    return out


# Plot a precomputed slice (grid or scatter) and return the figure.
def render_slice(grids, slice_axis, slice_pos, title,
                 vmin, vmax, cmap, log=False, point_size=1.0,
                 n_contours=0, contour_color='black',
                 fieldline_color='black', fieldline_density=1.0,
                 fieldline_broken=False, xlim=None, ylim=None, clean=False):
    ha, va = _PLOT_AXES[slice_axis]
    time_val = grids['time']
    header = title if title is not None else grids['label']

    # Log colormap: pass via norm= (and don't also pass vmin/vmax).
    color_kw = ({'norm': LogNorm(vmin=vmin, vmax=vmax)} if log
                else {'vmin': vmin, 'vmax': vmax})

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    if grids.get('mode') == 'scatter':
        im = ax.scatter(grids['xs'], grids['ys'], c=grids['values'],
                        s=point_size, cmap=cmap, linewidths=0,
                        rasterized=True, **color_kw)
        if n_contours > 0:
            print("  (contours skipped: not supported in --scatter mode)")
        if grids.get('fieldlines'):
            print("  (field lines skipped: not supported in --scatter mode)")
    else:
        xi, yi, di = grids['xi'], grids['yi'], grids['values']
        im = ax.pcolormesh(xi, yi, di, cmap=cmap, shading='auto',
                           rasterized=True, **color_kw)
        xc = 0.5 * (xi[0, :-1] + xi[0, 1:])
        yc = 0.5 * (yi[:-1, 0] + yi[1:, 0])
        if n_contours > 0:
            lo = vmin if vmin is not None else np.nanmin(di)
            hi = vmax if vmax is not None else np.nanmax(di)
            if log and lo > 0:
                levels = np.geomspace(lo, hi, n_contours)
            else:
                levels = np.linspace(lo, hi, n_contours)
            ax.contour(xc, yc, di, levels=levels,
                       colors=contour_color, linewidths=0.4, alpha=0.7)
        if grids.get('stream_u') is not None:
            ax.streamplot(xc, yc,
                          np.nan_to_num(grids['stream_u']),
                          np.nan_to_num(grids['stream_v']),
                          color=fieldline_color, density=fieldline_density,
                          linewidth=0.7, arrowsize=0.7,
                          broken_streamlines=fieldline_broken)
            ax.text(0.02, 0.98, f"{grids['stream_stem']} field lines",
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    color=fieldline_color)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # append_axes tracks the fixed-aspect host axes, so the colorbar stays
    # flush with the plot edge (fig.colorbar(ax=ax) leaves a gap there).
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    # Small/large tick values switch to an offset in scientific notation
    # (e.g. 0.2..1.2 with a x10^-3 above) instead of 0.0002, 0.0004, ...
    if not log:
        cbar.formatter.set_powerlimits((-3, 3))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()
    cbar.set_label(grids['label'])
    ax.set_xlabel(ha)
    ax.set_ylabel(va)
    if not clean:
        ax.set_title(f"{header}, t=[{time_val}]  ({slice_axis}={slice_pos:+.4f})")
        fig.text(0.98, 0.02, f"Resolution: {grids['res_label']}", fontsize=10, ha='right')
    return fig


# Common colormap limits across all grids; explicit vmin/vmax always win. With
# log=True, auto vmin is the smallest positive value (LogNorm needs vmin > 0).
def shared_ranges(grids, vmin, vmax, log=False):
    if vmin is None:
        if log:
            mins = []
            for g in grids:
                v = g['values']
                pos = v[(v > 0) & np.isfinite(v)]
                if pos.size:
                    mins.append(pos.min())
            vmin = min(mins) if mins else None
        else:
            vmin = min(np.nanmin(g['values']) for g in grids)
    if vmax is None:
        vmax = max(np.nanmax(g['values']) for g in grids)
    return vmin, vmax


def _save_fig(fig, fname, step, field, slice_axis, slice_pos, scatter, clean):
    outdir = os.path.dirname(os.path.abspath(fname))
    short = field.split('::')[-1]
    suffix = '_scatter' if scatter else ''
    ext = 'pdf' if clean else 'png'
    outname = os.path.join(outdir, f"slice_{short}_step{step}_{slice_axis}{slice_pos:+.4f}{suffix}.{ext}")
    fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


# Compute and plot a single slice, saved as PNG (PDF with clean=True).
def plot_slice(fname, step, field, resolution, slice_axis,
               slice_pos, title, vmin, vmax, cmap,
               log, scatter, point_size,
               n_contours, contour_color,
               fieldlines, fieldline_color, fieldline_density,
               fieldline_broken, xlim, ylim, clean):
    g = compute_slice_grids(fname, step, field, resolution, slice_axis, slice_pos,
                            scatter, fieldlines)
    fig = render_slice(g, slice_axis, slice_pos, title, vmin, vmax, cmap, log, point_size,
                       n_contours, contour_color, fieldline_color, fieldline_density,
                       fieldline_broken, xlim, ylim, clean)
    _save_fig(fig, fname, step, field, slice_axis, slice_pos, scatter, clean)


# Map func over items, fanning out across `jobs` worker processes (serial when
# jobs <= 1). Steps are independent, so plotting parallelizes cleanly.
def _pmap(jobs, func, items):
    items = list(items)
    if jobs > 1 and len(items) > 1:
        with Pool(min(jobs, len(items))) as pool:
            return pool.map(func, items)
    return [func(x) for x in items]


# Module-level (picklable) render+save of one precomputed grid, for the shared-
# scale path where ranges are known only after every grid is computed.
def _render_save(g, fname, field, slice_axis, slice_pos, title, vmin, vmax,
                 cmap, log, point_size, n_contours, contour_color, scatter,
                 fieldline_color, fieldline_density, fieldline_broken, xlim, ylim,
                 clean):
    fig = render_slice(g, slice_axis, slice_pos, title, vmin, vmax, cmap, log,
                       point_size, n_contours, contour_color,
                       fieldline_color, fieldline_density, fieldline_broken,
                       xlim, ylim, clean)
    _save_fig(fig, fname, g['step'], field, slice_axis, slice_pos, scatter, clean)


# One PNG per step. With shared_scale (default) a single colormap range spans
# all of them; otherwise each frame is auto-scaled to its own min/max. Explicit
# vmin/vmax always apply in either mode. jobs > 1 fans steps out over processes.
def plot_all_steps(fname, steps, field, resolution, slice_axis,
                   slice_pos, title, vmin, vmax, cmap,
                   log, scatter, point_size,
                   n_contours, contour_color,
                   fieldlines, fieldline_color, fieldline_density,
                   fieldline_broken, xlim, ylim, clean, shared_scale, jobs):
    # No shared range needed: compute+render+save each step independently.
    if not shared_scale:
        worker = functools.partial(plot_slice, fname, field=field, resolution=resolution,
                                   slice_axis=slice_axis, slice_pos=slice_pos, title=title,
                                   vmin=vmin, vmax=vmax, cmap=cmap, log=log, scatter=scatter,
                                   point_size=point_size, n_contours=n_contours,
                                   contour_color=contour_color, fieldlines=fieldlines,
                                   fieldline_color=fieldline_color,
                                   fieldline_density=fieldline_density,
                                   fieldline_broken=fieldline_broken, xlim=xlim, ylim=ylim,
                                   clean=clean)
        _pmap(jobs, worker, steps)
        return

    # Shared range: compute every grid first, then render against common limits.
    compute = functools.partial(compute_slice_grids, fname, field=field, resolution=resolution,
                                slice_axis=slice_axis, slice_pos=slice_pos, scatter=scatter,
                                fieldlines=fieldlines)
    grids = _pmap(jobs, compute, steps)
    vmin, vmax = shared_ranges(grids, vmin, vmax, log)
    print(f"Shared {field} scale: [{vmin:.6f}, {vmax:.6f}]" + (" (log)" if log else ""))
    render = functools.partial(_render_save, fname=fname, field=field, slice_axis=slice_axis,
                               slice_pos=slice_pos, title=title, vmin=vmin, vmax=vmax, cmap=cmap,
                               log=log, point_size=point_size, n_contours=n_contours,
                               contour_color=contour_color, scatter=scatter,
                               fieldline_color=fieldline_color,
                               fieldline_density=fieldline_density,
                               fieldline_broken=fieldline_broken, xlim=xlim, ylim=ylim,
                               clean=clean)
    _pmap(jobs, render, grids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 2D SPH-interpolated slices from SPHEXA HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s data.h5                            PNG of rho, final step (default)\n"
            "  %(prog)s data.h5 -i                         Print metadata + available fields\n"
            "  %(prog)s data.h5 5 --field Bmag             PNG of |B| at step 5\n"
            "  %(prog)s data.h5 --all --field magneto::alpha_B\n"
            "  %(prog)s data.h5 5 --axis x --pos 0.5\n"
            "  %(prog)s data.h5 --field rho --fieldlines          rho slice + B field lines\n"
            "  %(prog)s data.h5 --field rho --fieldlines vx,vy,vz  ... + velocity field lines\n"
        ),
    )
    parser.add_argument("file", help="HDF5 input file")
    parser.add_argument("step", nargs="?", type=int,
                        help="Step number. Omit to plot the final step.")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata + available fields and exit")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--field", default="rho",
                        help="Field to plot (raw dataset name or derived; default: rho)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z",
                        help="Axis normal to the slice plane (default: z)")
    parser.add_argument("--pos", type=float, default=0.0,
                        help="Position along the slice axis (default: 0.0)")
    parser.add_argument("-r", "--resolution", type=int, default=256,
                        help="Interpolation grid resolution per side (default: 256)")
    parser.add_argument("--title", default=None,
                        help="Plot title prefix (default: field label)")
    parser.add_argument("--clean", action="store_true",
                        help="Publish mode for thesis figures: save PDF instead of PNG, "
                             "drop the title and resolution label (those go in the "
                             f"caption), and render text in {CLEAN_FONT} "
                             "(CLEAN_FONT in _h5_common.py).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Lower colormap limit (default: auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Upper colormap limit (default: auto)")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Crop the horizontal plot axis to [LO, HI]. Useful for the "
                             "2:1 advection-loop box to show only the central square.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Crop the vertical plot axis to [LO, HI] (default: auto)")
    parser.add_argument("--shared-scale", action=argparse.BooleanOptionalAction, default=True,
                        help="With --all, share one colormap range across every frame "
                             "(default: on). Use --no-shared-scale for per-frame auto-scaling.")
    parser.add_argument("--cmap", default="RdBu",
                        help="Matplotlib colormap name (default: RdBu)")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Use a log-scale colormap (LogNorm). Best for positive "
                             "fields like rho, Bmag, vmag, Emag.")
    parser.add_argument("--scatter", action="store_true",
                        help="Skip SPH interpolation; render raw particle scatter (fast)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Scatter marker size (only used with --scatter; default: 1.0)")
    parser.add_argument("--contours", type=int, default=0, metavar="N",
                        help="Overlay N isocontours on grid plots (default: 0 = off)")
    parser.add_argument("--contour-color", default="black",
                        help="Contour line color (default: black)")
    parser.add_argument("--fieldlines", nargs="?", default=None,
                        const="magneto::Bx,magneto::By,magneto::Bz", metavar="FX,FY,FZ",
                        help="Overlay a streamplot of a vector field on grid plots. "
                             "Bare flag uses the B field; pass 3 comma-separated "
                             "components for another (e.g. --fieldlines vx,vy,vz). "
                             "The two components in the slice plane are used.")
    parser.add_argument("--fieldline-color", default="black",
                        help="Field-line streamplot color (default: black)")
    parser.add_argument("--fieldline-density", type=float, default=1.0,
                        help="Field-line streamplot density (default: 1.0)")
    parser.add_argument("--fieldline-broken", action="store_true",
                        help="Break streamlines when they crowd (matplotlib default). "
                             "Off by default here so closed loops (e.g. MHD loop test) "
                             "run as full circles.")
    parser.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                        help="Worker processes for --all (default: 1 = serial). "
                             "In SLURM, pass -j \"$SLURM_CPUS_PER_TASK\".")

    args = parser.parse_args()

    if args.clean:
        apply_clean_style()

    if args.info:
        print_metadata(args.file)
        sys.exit(0)

    fieldlines = None
    if args.fieldlines is not None:
        fieldlines = [c.strip() for c in args.fieldlines.split(",")]
        if len(fieldlines) != 3:
            parser.error("--fieldlines needs exactly 3 comma-separated components "
                         f"(x,y,z), got {len(fieldlines)}: {args.fieldlines!r}")

    common = dict(field=args.field, resolution=args.resolution, slice_axis=args.axis,
                  slice_pos=args.pos, title=args.title, vmin=args.vmin, vmax=args.vmax,
                  cmap=args.cmap, log=args.log, scatter=args.scatter,
                  point_size=args.point_size,
                  n_contours=args.contours, contour_color=args.contour_color,
                  fieldlines=fieldlines, fieldline_color=args.fieldline_color,
                  fieldline_density=args.fieldline_density,
                  fieldline_broken=args.fieldline_broken,
                  xlim=args.xlim, ylim=args.ylim, clean=args.clean)

    if args.all:
        nsteps = get_nsteps(args.file)
        if nsteps == 0:
            print(f"No steps found in {args.file}")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        plot_all_steps(args.file, list(range(nsteps)),
                       shared_scale=args.shared_scale, jobs=args.jobs, **common)
    else:
        if args.step is None:
            nsteps = get_nsteps(args.file)
            if nsteps == 0:
                print(f"No steps found in {args.file}")
                sys.exit(1)
            step = nsteps - 1
        else:
            step = args.step
        plot_slice(args.file, step, **common)
