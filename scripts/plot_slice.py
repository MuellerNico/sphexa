#!/usr/bin/env python3
"""Plot 2D SPH-interpolated slices of any (raw or derived) field. """

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse

from _h5_common import print_metadata, get_nsteps, resolve_field, resolution_label


def cubic_spline_3d(q):
    """Cubic spline (M4) SPH kernel shape in 3D. q = r/h.

    Returns sigma_3D * f(q). The 1/h**3 factor of the full kernel is omitted:
    it cancels in the normalized SPH interpolation (see sph_scatter_to_grid).
    """
    sigma = 1.0 / np.pi
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    w[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * w


def sph_scatter_to_grid(xs, ys, zoff, hs, values, resolution=256):
    """Interpolate a per-particle field onto a 2D slice grid (3D SPH kernel).

    Each particle contributes through the full 3D kernel attenuated by its
    perpendicular offset from the slice plane (not a thin-slab projection).
    Returns the normalized SPH estimate
        field(x) = sum_j values_j f(q_j) / sum_j f(q_j)
    which reproduces a constant field exactly. With SPHEXA's equal-mass
    particles and h ~ (m/rho)^(1/3) this is the standard volume-normalized
    SPH interpolant for density and matches a kernel average for any other
    field.
    """
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


# Axes plotted for each slice axis: (horizontal, vertical)
_PLOT_AXES = {
    'z': ('x', 'y'),
    'y': ('x', 'z'),
    'x': ('y', 'z'),
}


def compute_slice_grids(fname, step, field='rho', resolution=256,
                        slice_axis='z', slice_pos=0.0, scatter=False):
    """Read a step; either SPH-interpolate `field` onto a 2D slice grid,
    or (scatter=True) return the raw per-particle samples in the slab.

    Returns a dict with metadata plus either (xi, yi, values) for grid mode
    or (xs, ys, values) for scatter mode. Does no plotting.
    """
    print(f"Reading step {step} from {fname}...")
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

    ha, va = _PLOT_AXES[slice_axis]
    xs   = coords[ha][mask]
    ys   = coords[va][mask]
    zoff = coords[slice_axis][mask] - slice_pos
    hs   = h[mask]

    if scatter:
        return {'step': step, 'time': time_val, 'res_label': res_label,
                'field': field, 'label': label, 'mode': 'scatter',
                'xs': xs, 'ys': ys, 'values': values[mask]}

    print(f"  Interpolating onto {resolution}x{resolution} grid...")
    xi, yi, di = sph_scatter_to_grid(xs, ys, zoff, hs, values[mask], resolution)

    return {'step': step, 'time': time_val, 'res_label': res_label,
            'field': field, 'label': label, 'mode': 'grid',
            'xi': xi, 'yi': yi, 'values': di}


def render_slice(grids, slice_axis='z', slice_pos=0.0, title=None,
                 vmin=None, vmax=None, cmap='bone_r', point_size=1.0,
                 n_contours=0, contour_color='black'):
    """Plot a precomputed slice (grid or scatter); returns a figure."""
    ha, va = _PLOT_AXES[slice_axis]
    time_val = grids['time']
    header = title if title is not None else grids['label']

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    if grids.get('mode') == 'scatter':
        im = ax.scatter(grids['xs'], grids['ys'], c=grids['values'],
                        s=point_size, cmap=cmap, linewidths=0,
                        vmin=vmin, vmax=vmax, rasterized=True)
        if n_contours > 0:
            print("  (contours skipped: not supported in --scatter mode)")
    else:
        xi, yi, di = grids['xi'], grids['yi'], grids['values']
        im = ax.pcolormesh(xi, yi, di, cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)
        if n_contours > 0:
            lo = vmin if vmin is not None else np.nanmin(di)
            hi = vmax if vmax is not None else np.nanmax(di)
            xc = 0.5 * (xi[0, :-1] + xi[0, 1:])
            yc = 0.5 * (yi[:-1, 0] + yi[1:, 0])
            ax.contour(xc, yc, di, levels=np.linspace(lo, hi, n_contours),
                       colors=contour_color, linewidths=0.4, alpha=0.7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(grids['label'])
    ax.set_xlabel(ha)
    ax.set_ylabel(va)
    ax.set_title(f"{header}, t=[{time_val}]  ({slice_axis}={slice_pos:+.4f})")

    fig.text(0.98, 0.02, f"Resolution: {grids['res_label']}", fontsize=10, ha='right')
    plt.tight_layout()
    return fig


def shared_ranges(grids, vmin=None, vmax=None):
    """Global colormap limits across all grids; user-supplied values win."""
    if vmin is None:
        vmin = min(np.nanmin(g['values']) for g in grids)
    if vmax is None:
        vmax = max(np.nanmax(g['values']) for g in grids)
    return vmin, vmax


def _save_png(fig, fname, step, field, slice_axis, slice_pos, scatter=False):
    outdir = os.path.dirname(os.path.abspath(fname))
    short = field.split('::')[-1]
    suffix = '_scatter' if scatter else ''
    outname = os.path.join(outdir, f"slice_{short}_step{step}_{slice_axis}{slice_pos:+.4f}{suffix}.png")
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


def plot_slice(fname, step, field='rho', resolution=256, slice_axis='z',
               slice_pos=0.0, title=None, vmin=None, vmax=None, cmap='bone_r',
               scatter=False, point_size=1.0,
               n_contours=0, contour_color='black'):
    """Compute and plot a single 2D slice, saved as PNG."""
    g = compute_slice_grids(fname, step, field, resolution, slice_axis, slice_pos, scatter)
    fig = render_slice(g, slice_axis, slice_pos, title, vmin, vmax, cmap, point_size,
                       n_contours, contour_color)
    _save_png(fig, fname, step, field, slice_axis, slice_pos, scatter)


def plot_all_steps(fname, steps, field='rho', resolution=256, slice_axis='z',
                   slice_pos=0.0, title=None, vmin=None, vmax=None, cmap='bone_r',
                   scatter=False, point_size=1.0,
                   n_contours=0, contour_color='black'):
    """One PNG per step, sharing one colormap range across all steps."""
    grids = [compute_slice_grids(fname, s, field, resolution, slice_axis, slice_pos, scatter)
             for s in steps]
    vmin, vmax = shared_ranges(grids, vmin, vmax)
    print(f"Shared {field} scale: [{vmin:.6f}, {vmax:.6f}]")
    for g in grids:
        fig = render_slice(g, slice_axis, slice_pos, title, vmin, vmax, cmap, point_size,
                           n_contours, contour_color)
        _save_png(fig, fname, g['step'], field, slice_axis, slice_pos, scatter)


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
    parser.add_argument("--vmin", type=float, default=None,
                        help="Lower colormap limit (default: auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Upper colormap limit (default: auto)")
    parser.add_argument("--cmap", default="RdBu",
                        help="Matplotlib colormap name (default: RdBu)")
    parser.add_argument("--scatter", action="store_true",
                        help="Skip SPH interpolation; render raw particle scatter (fast)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Scatter marker size (only used with --scatter; default: 1.0)")
    parser.add_argument("--contours", type=int, default=0, metavar="N",
                        help="Overlay N isocontours on grid plots (default: 0 = off)")
    parser.add_argument("--contour-color", default="black",
                        help="Contour line color (default: black)")

    args = parser.parse_args()

    if args.info:
        print_metadata(args.file)
        sys.exit(0)

    common = dict(field=args.field, resolution=args.resolution, slice_axis=args.axis,
                  slice_pos=args.pos, title=args.title, vmin=args.vmin, vmax=args.vmax,
                  cmap=args.cmap, scatter=args.scatter, point_size=args.point_size,
                  n_contours=args.contours, contour_color=args.contour_color)

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
        plot_slice(args.file, step, **common)
