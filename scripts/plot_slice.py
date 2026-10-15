#!/usr/bin/env python3

"""Plot 2D SPH-interpolated slices of any raw or derived field from an H5 dump. 
Each input is a file with an optional step (final step by default)

Help:    plot_slice.py --help
Example: plot_slice.py dump.h5 --field magneto::Bx
Example: plot_slice.py a.h5 0 b.h5 0 a.h5 5 b.h5 5 --field Bmag --layout 2 2 \ 
            --labels nx=128 nx=256 --label-time --cmap viridis --
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

from _h5_common import (print_metadata, get_nsteps, resolve_field,
                        resolution_label, read_domain, CLEAN_FONT,
                        apply_clean_style)


# All compute/render options in one object, so every function takes (data, opt).
# This is also where the option defaults live: the parser below reads them off
# DEFAULT, so each one is written once. Frozen, so panels can share an instance.
@dataclass(frozen=True)
class Opts:
    field: str = 'rho'
    resolution: int = 256
    slice_axis: str = 'z'
    slice_pos: Optional[float] = None
    smooth: float = 1.0
    scatter: bool = False
    # colors
    cmap: str = 'RdBu'
    log: bool = False
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    shared_scale: bool = True
    # overlays
    point_size: float = 1.0
    n_contours: int = 0
    contour_color: str = 'black'
    fieldlines: Optional[Sequence[str]] = None
    fieldline_color: str = 'black'
    fieldline_density: float = 1.0
    fieldline_broken: bool = False
    # layout and annotation
    title: Optional[str] = None
    labels: Optional[Sequence[str]] = None
    layout: Optional[Sequence[int]] = None
    label_color: str = 'white'
    label_time: bool = False
    xlim: Optional[Sequence[float]] = None
    ylim: Optional[Sequence[float]] = None
    clean: bool = False


DEFAULT = Opts()


# M4 cubic spline in 3D (q = r/h). The 1/h**3 factor cancels in the normalized
# interpolation below, so it is left off.
def cubic_spline_3d(q):
    sigma = 1.0 / np.pi
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    w[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * w


# SPH-interpolate particles onto a 2D slice grid, each using the full 3D kernel
# attenuated by its offset from the plane (not a thin slab). Normalized as
# sum(v*w)/sum(w), so a constant field is reproduced exactly. Returns
# (xi, yi, field) with xi/yi the cell edges.
def sph_scatter_to_grid(xs, ys, zoff, hs, values, resolution, bounds):
    xmin, xmax, ymin, ymax = bounds
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


# slice axis -> in-plane plot axes
_PLOT_AXES = {
    'z': ('x', 'y'),
    'y': ('x', 'z'),
    'x': ('y', 'z'),
}

# spatial axis -> index of the matching --fieldlines vector component
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


# 'magneto::Bx' -> 'B', 'vy' -> 'v'
def _vector_stem(name):
    short = name.split('::')[-1]
    return short[:-1] if short[-1:] in 'xyz' else short


# Read one step: metadata plus either an interpolated grid (xi, yi, values) or,
# with opt.scatter, the raw particles in the slab (xs, ys, values). No plotting.
def compute_slice_grids(fname, step, opt):
    print(f"Reading step {step} from {fname}...")
    slice_axis = opt.slice_axis
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
        values, label = resolve_field(s, opt.field)
        time_val = s.attrs["time"][0]

        # the two vector components lying in the slice plane
        stream = None
        if opt.fieldlines is not None:
            uf = opt.fieldlines[_AXIS_INDEX[ha]]
            vf = opt.fieldlines[_AXIS_INDEX[va]]
            u_vals, _ = resolve_field(s, uf)
            v_vals, _ = resolve_field(s, vf)
            stream = (u_vals, v_vals, _vector_stem(uf))

        dom = read_domain(s)
        slice_pos = opt.slice_pos
        if slice_pos is None:
            slice_pos = float(dom.centre[dom.index(slice_axis)])
        if "h" in s:
            h = np.array(s["h"])
        else:
            n_particles = len(coords['x'])
            h_est = 1.2 * (float(np.prod(dom.length)) / n_particles) ** (1.0 / 3.0)
            h = np.full(n_particles, h_est)
            print(f"  h not in file, using estimate h={h_est:.6f}")

    # wider render kernel: same interpolation, more neighbours per pixel
    if opt.smooth != 1.0:
        h = h * opt.smooth
        print(f"  Render smoothing: h x {opt.smooth:g}")

    n_particles = len(coords['x'])
    res_label = resolution_label(dom.length, n_particles)
    print(f"Step {step}: time={time_val:.8f}, N={n_particles} ({res_label})")
    print(f"  box: {dom.describe()}, slicing at {slice_axis}={slice_pos:+.6f}")
    print(f"  {opt.field}: [{np.nanmin(values):.6f}, {np.nanmax(values):.6f}]")

    # periodic wrap: a cut on a box boundary keeps the full kernel support
    off  = dom.offset(coords[slice_axis] - slice_pos, slice_axis)
    mask = np.abs(off) < 2.0 * h
    print(f"  Particles in slice: {mask.sum()} / {n_particles} "
          f"({mask.sum() / n_particles * 100:.2f}%)")

    i_ha, i_va = dom.index(ha), dom.index(va)
    bounds = (dom.lo[i_ha], dom.hi[i_ha], dom.lo[i_va], dom.hi[i_va])
    xs   = coords[ha][mask]
    ys   = coords[va][mask]
    zoff = off[mask]
    hs   = h[mask]

    panel = {'step': step, 'time': time_val, 'res_label': res_label,
             'field': opt.field, 'label': label, 'slice_pos': slice_pos,
             'bounds': bounds}

    if opt.scatter:
        panel.update(mode='scatter', xs=xs, ys=ys, values=values[mask],
                     fieldlines=opt.fieldlines is not None)
        return panel

    print(f"  Interpolating onto {opt.resolution}x{opt.resolution} grid...")
    xi, yi, di = sph_scatter_to_grid(xs, ys, zoff, hs, values[mask],
                                     opt.resolution, bounds)
    panel.update(mode='grid', xi=xi, yi=yi, values=di)

    if stream is not None:
        u_vals, v_vals, stem = stream
        _, _, ug = sph_scatter_to_grid(xs, ys, zoff, hs, u_vals[mask],
                                       opt.resolution, bounds)
        _, _, vg = sph_scatter_to_grid(xs, ys, zoff, hs, v_vals[mask],
                                       opt.resolution, bounds)
        panel.update(stream_u=ug, stream_v=vg, stream_stem=stem)

    return panel


# Draw one panel into an axes and return the mappable. vmin/vmax are resolved
# per figure, so they come in separately from opt.
def _draw_panel(ax, panel, opt, vmin, vmax):
    # a log colormap goes in via norm=, not vmin/vmax
    color_kw = ({'norm': LogNorm(vmin=vmin, vmax=vmax)} if opt.log
                else {'vmin': vmin, 'vmax': vmax})

    if panel['mode'] == 'scatter':
        im = ax.scatter(panel['xs'], panel['ys'], c=panel['values'],
                        s=opt.point_size, cmap=opt.cmap, linewidths=0,
                        rasterized=True, **color_kw)
        if opt.n_contours > 0:
            print("  (contours skipped: not supported in --scatter mode)")
        if panel.get('fieldlines'):
            print("  (field lines skipped: not supported in --scatter mode)")
    else:
        xi, yi, di = panel['xi'], panel['yi'], panel['values']
        im = ax.pcolormesh(xi, yi, di, cmap=opt.cmap, shading='auto',
                           rasterized=True, **color_kw)
        xc = 0.5 * (xi[0, :-1] + xi[0, 1:])
        yc = 0.5 * (yi[:-1, 0] + yi[1:, 0])
        if opt.n_contours > 0:
            lo = vmin if vmin is not None else np.nanmin(di)
            hi = vmax if vmax is not None else np.nanmax(di)
            if opt.log and lo > 0:
                levels = np.geomspace(lo, hi, opt.n_contours)
            else:
                levels = np.linspace(lo, hi, opt.n_contours)
            ax.contour(xc, yc, di, levels=levels,
                       colors=opt.contour_color, linewidths=0.4, alpha=0.7)
        if panel.get('stream_u') is not None:
            ax.streamplot(xc, yc,
                          np.nan_to_num(panel['stream_u']),
                          np.nan_to_num(panel['stream_v']),
                          color=opt.fieldline_color, density=opt.fieldline_density,
                          linewidth=0.7, arrowsize=0.7,
                          broken_streamlines=opt.fieldline_broken)
            ax.text(0.02, 0.98, f"{panel['stream_stem']} field lines",
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    color=opt.fieldline_color)
    if opt.xlim is not None:
        ax.set_xlim(opt.xlim)
    if opt.ylim is not None:
        ax.set_ylim(opt.ylim)
    return im


# Small/large tick values get a shared power-of-ten offset (2, 4, ... x10^-4)
# instead of 0.0002, 0.0004, ...
def _format_cbar(cbar, label, log):
    if not log:
        cbar.formatter.set_powerlimits((-3, 3))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()
    cbar.set_label(label)


# one row up to 3 panels, two rows beyond (2x2, 2x3, ...)
def auto_layout(n):
    return (1, n) if n <= 3 else (2, -(-n // 2))


# Panel height/width in data units, to size the figure to the data so the tiled
# panels pack without gaps.
def _panel_aspect(panel, opt):
    x0, x1, y0, y1 = panel['bounds']
    if opt.xlim is not None:
        x0, x1 = opt.xlim
    if opt.ylim is not None:
        y0, y1 = opt.ylim
    return (y1 - y0) / (x1 - x0) if x1 > x0 else 1.0


# corner tag naming a panel (resolution, scheme, ...)
def _corner_label(ax, text, color, right=True):
    ax.text(0.97 if right else 0.03, 0.96, text, transform=ax.transAxes,
            va='top', ha='right' if right else 'left', color=color)


# Tile panels into one figure on a shared color scale and colorbar; a single
# plot is the 1x1 case. Panels fill row-major, tagged with opt.labels (one per
# panel or one per column) top right and, with opt.label_time, their time top
# left.
def render(panels, opt, vmin, vmax):
    ha, va = _PLOT_AXES[opt.slice_axis]
    nrows, ncols = opt.layout if opt.layout is not None else auto_layout(len(panels))
    labels = opt.labels
    if labels is not None:
        labels = [labels[k % len(labels)] for k in range(len(panels))]

    # ImageGrid keeps fixed-aspect panels flush and sizes the shared colorbar to
    # their combined height; plain subplots leave ragged gaps.
    pw = 3.6
    fig = plt.figure(figsize=(ncols * pw + 1.2,
                              nrows * pw * _panel_aspect(panels[0], opt)
                              + (0.4 if opt.clean else 1.0)))
    axes = ImageGrid(fig, (0.06, 0.06, 0.88, 0.90 if opt.clean else 0.86),
                     nrows_ncols=(nrows, ncols),
                     axes_pad=(0.08, 0.08 if opt.clean else 0.34), share_all=True,
                     cbar_mode='single', cbar_location='right',
                     cbar_size='3%', cbar_pad=0.08)

    im = None
    for k, ax in enumerate(axes):
        if k >= len(panels):
            ax.set_axis_off()
            continue
        im = _draw_panel(ax, panels[k], opt, vmin, vmax)
        if labels is not None:
            _corner_label(ax, labels[k], opt.label_color)
        if opt.label_time:
            _corner_label(ax, f"t={panels[k]['time']:.1f}", opt.label_color, right=False)
        if not opt.clean:
            ax.set_title(f"t={panels[k]['time']:.4g}  ({panels[k]['res_label']})",
                         fontsize=9)
        if k % ncols == 0:
            ax.set_ylabel(va)
        # bottom-most filled panel of its column, which is not the bottom row
        # when the last row is partly empty; ImageGrid already hid those labels,
        # so undo it explicitly
        if k + ncols >= len(panels):
            ax.set_xlabel(ha)
            ax.xaxis.label.set_visible(True)
            ax.tick_params(labelbottom=True)

    _format_cbar(fig.colorbar(im, cax=axes.cbar_axes[0]), panels[0]['label'], opt.log)
    if not opt.clean:
        header = opt.title if opt.title is not None else panels[0]['label']
        fig.suptitle(f"{header}  ({_pos_label(panels, opt.slice_axis)})",
                     y=0.99, va='top')
    return fig


# Panels can sit at different planes (each file's own midplane), so name the
# plane only when they agree.
def _pos_label(panels, slice_axis):
    pos = {p['slice_pos'] for p in panels}
    return (f"{slice_axis}={pos.pop():+.4f}" if len(pos) == 1
            else f"{slice_axis}=box midplane")


def _pos_tag(panels, slice_axis):
    pos = {p['slice_pos'] for p in panels}
    return (f"{slice_axis}{pos.pop():+.4f}" if len(pos) == 1
            else f"{slice_axis}mid")


# Common color limits over all panels; explicit vmin/vmax win. With opt.log the
# auto vmin is the smallest positive value (LogNorm needs vmin > 0).
def shared_ranges(panels, opt):
    vmin, vmax = opt.vmin, opt.vmax
    if vmin is None:
        if opt.log:
            mins = []
            for p in panels:
                v = p['values']
                pos = v[(v > 0) & np.isfinite(v)]
                if pos.size:
                    mins.append(pos.min())
            vmin = min(mins) if mins else None
        else:
            vmin = min(np.nanmin(p['values']) for p in panels)
    if vmax is None:
        vmax = max(np.nanmax(p['values']) for p in panels)
    return vmin, vmax


def _range_label(vmin, vmax, log):
    def fmt(v):
        return 'auto' if v is None else f"{v:.6f}"
    return f"[{fmt(vmin)}, {fmt(vmax)}]" + (" (log)" if log else "")


# Figures land next to their input file, named after field, step and plane.
def save_fig(fig, fname, step, opt, pos_tag, tag=''):
    outdir = os.path.dirname(os.path.abspath(fname))
    short = opt.field.split('::')[-1]
    suffix = (('_scatter' if opt.scatter else '')
              + ('' if opt.smooth == 1.0 else f"_h{opt.smooth:g}"))
    ext = 'pdf' if opt.clean else 'png'
    outname = os.path.join(outdir,
                           f"slice_{short}_step{step}_{pos_tag}{tag}{suffix}.{ext}")
    fig.savefig(outname, dpi=300 if opt.clean else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


# One figure per panel set, each set a list of (file, step) panels. Panels of a
# figure always share a color scale; opt.shared_scale extends that range across
# every figure, so an --all series stays comparable frame to frame.
def plot_panel_sets(panel_sets, opt):
    computed = [[compute_slice_grids(f, s, opt) for f, s in ps]
                for ps in panel_sets]

    flat = [p for ps in computed for p in ps]
    if opt.shared_scale:
        vmin, vmax = shared_ranges(flat, opt)
        if len(flat) > 1:
            print(f"Shared {opt.field} scale: {_range_label(vmin, vmax, opt.log)}")

    for panels in computed:
        lo, hi = (vmin, vmax) if opt.shared_scale else shared_ranges(panels, opt)
        fig = render(panels, opt, lo, hi)
        save_fig(fig, panel_sets[0][0][0], panels[0]['step'], opt,
                 _pos_tag(panels, opt.slice_axis),
                 tag='_panels' if len(panels) > 1 else '')


# "a.h5 3 b.h5 c.h5 -2" -> [('a.h5', 3), ('b.h5', None), ('c.h5', -2)]: a bare
# integer sets the step of the file in front of it. Repeating a file is how two
# steps of one dump land side by side.
def _parse_inputs(tokens, parser):
    panels = []
    for tok in tokens:
        if not tok.lstrip('+-').isdigit():
            panels.append((tok, None))
        elif not panels:
            parser.error(f"step {tok} given before any input file")
        elif panels[-1][1] is not None:
            parser.error(f"{panels[-1][0]} got two step numbers ({panels[-1][1]}, {tok})")
        else:
            panels[-1] = (panels[-1][0], int(tok))
    return panels


# Absolute step per input: None is that file's last step, so runs with different
# dump counts still line up on t_end; negative counts back from the end.
def _resolve_steps(panels, parser):
    nsteps = {f: get_nsteps(f) for f, _ in panels}
    for f, n in nsteps.items():
        if n == 0:
            parser.error(f"no steps found in {f}")
    return [(f, nsteps[f] - 1 if s is None else s + nsteps[f] if s < 0 else s)
            for f, s in panels]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 2D SPH-interpolated slices from SPHEXA HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s data.h5                                 Plot density rho, final step (default)\n"
            "  %(prog)s data.h5 -i                              Print metadata + available fields\n"
            "  %(prog)s data.h5 5 --field Bmag                  |B| at step 5\n"
            "  %(prog)s data.h5 --all --field magneto::alpha_B  Plot all steps in dump \n"
            "  %(prog)s data.h5 5 --axis x --pos 0.5            Manual slicing\n"
            "  %(prog)s data.h5 --fieldlines vx,vy,vz           rho slice + velocity field lines\n"
            "  %(prog)s a.h5 b.h5 c.h5 d.h5 --labels a b c d    2x2 panels, one shared color scale\n"
            "  %(prog)s a.h5 0 a.h5 5 --label-time              Two steps of the same dump\n"
            "  %(prog)s data.h5 --clean -resolution 512         Clean PDF output with increased resolution\n"
        ),
    )
    parser.add_argument("files", nargs="+", metavar="FILE [STEP]",
                        help="HDF5 input file(s), each optionally followed by a step "
                             "number (default: final step). Several panels are tiled "
                             "into one figure with shared color scale")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata + available fields and exit")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--field", default=DEFAULT.field,
                        help="Field to plot, raw dataset name or derived "
                             "(default: %(default)s)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT.slice_axis,
                        help="Axis normal to the slice plane (default: %(default)s)")
    parser.add_argument("--pos", type=float, default=DEFAULT.slice_pos,
                        help="Position along the slice axis (default: box midplane)")
    parser.add_argument("-r", "--resolution", type=int, default=DEFAULT.resolution,
                        help="Interpolation grid resolution (default: %(default)s)")
    parser.add_argument("--title", default=DEFAULT.title,
                        help="Plot title prefix (default: field label)")
    parser.add_argument("--labels", nargs="+", default=DEFAULT.labels, metavar="LABEL",
                        help="Per panel annotation. One per column (repeating) also works.")
    parser.add_argument("--label-time", action="store_true",
                        help="Annotate each panel with its simulation time")
    parser.add_argument("--label-color", default=DEFAULT.label_color,
                        help="Panel label color (default: %(default)s)")
    parser.add_argument("--layout", type=int, nargs=2, default=DEFAULT.layout,
                        metavar=("ROWS", "COLS"),
                        help="Panel grid")
    parser.add_argument("--clean", action="store_true",
                        help="Publish mode for figures: save PDF instead of PNG, "
                             "drop the title and resolution label, and render text in "
                             f"{CLEAN_FONT} (CLEAN_FONT in _h5_common.py).")
    parser.add_argument("--vmin", type=float, default=DEFAULT.vmin,
                        help="Lower colormap limit (default: auto)")
    parser.add_argument("--vmax", type=float, default=DEFAULT.vmax,
                        help="Upper colormap limit (default: auto)")
    parser.add_argument("--xlim", type=float, nargs=2, default=DEFAULT.xlim,
                        metavar=("LO", "HI"),
                        help="Crop horizontal plot axis to [LO, HI]")
    parser.add_argument("--ylim", type=float, nargs=2, default=DEFAULT.ylim,
                        metavar=("LO", "HI"),
                        help="Crop vertical plot axis to [LO, HI]")
    parser.add_argument("--shared-scale", action=argparse.BooleanOptionalAction,
                        default=DEFAULT.shared_scale,
                        help="With --all, share one colormap range across every frame "
                             "(default: on). Use --no-shared-scale for per-frame auto-scaling. "
                             "Panels of one figure always share their scale.")
    parser.add_argument("--cmap", default=DEFAULT.cmap,
                        help="Matplotlib colormap name (default: %(default)s)")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Use log-scale colormap (LogNorm)")
    parser.add_argument("--scatter", action="store_true",
                        help="Skip SPH interpolation; render raw particle scatter (fast)")
    parser.add_argument("--smooth", type=float, default=DEFAULT.smooth, metavar="FACTOR",
                        help="Scale every particle's h by FACTOR in the render kernel "
                             "(default: %(default)s = the simulation's own h). >1 averages "
                             "over more neighbours per pixel.")
    parser.add_argument("--point-size", type=float, default=DEFAULT.point_size,
                        help="Scatter marker size, only used with --scatter "
                             "(default: %(default)s)")
    parser.add_argument("--contours", type=int, default=DEFAULT.n_contours, metavar="N",
                        help="Overlay N isocontours on grid plots (default: %(default)s)")
    parser.add_argument("--contour-color", default=DEFAULT.contour_color,
                        help="Contour line color (default: %(default)s)")
    parser.add_argument("--fieldlines", nargs="?", default=None,
                        const="magneto::Bx,magneto::By,magneto::Bz", metavar="FX,FY,FZ",
                        help="Overlay a streamplot of a vector field on grid plots 
                             "(e.g. --fieldlines vx,vy,vz). Only the two components 
                             "in the slice plane are used (default: B field)")
    parser.add_argument("--fieldline-color", default=DEFAULT.fieldline_color,
                        help="Field-line streamplot color (default: %(default)s)")
    parser.add_argument("--fieldline-density", type=float, default=DEFAULT.fieldline_density,
                        help="Field-line streamplot density (default: %(default)s)")
    parser.add_argument("--fieldline-broken", action="store_true",
                        help="Break streamlines when they crowd (matplotlib default). "
                             "Off by default")

    args = parser.parse_args()
    panels = _parse_inputs(args.files, parser)

    if args.clean:
        apply_clean_style()

    if args.info:
        for f in dict.fromkeys(f for f, _ in panels):
            print_metadata(f)
        sys.exit(0)

    # --all sweeps the steps itself, one figure per step over all files
    if args.all:
        if any(s is not None for _, s in panels):
            parser.error("--all plots every step; drop the explicit step numbers")
        fnames = [f for f, _ in panels]
        nsteps = min(get_nsteps(f) for f in fnames)
        if nsteps == 0:
            parser.error(f"no steps found in {fnames}")
        panel_sets = [[(f, s) for f in fnames] for s in range(nsteps)]
    else:
        panel_sets = [_resolve_steps(panels, parser)]

    n_panels = len(panel_sets[0])
    nrows, ncols = args.layout if args.layout is not None else auto_layout(n_panels)
    if nrows * ncols < n_panels:
        parser.error(f"--layout {nrows} {ncols} holds {nrows * ncols} panels, "
                     f"but {n_panels} panels were given")
    if args.labels is not None and len(args.labels) not in (n_panels, ncols):
        parser.error(f"--labels got {len(args.labels)} values; expected one per "
                     f"panel ({n_panels}) or one per column ({ncols})")

    fieldlines = None
    if args.fieldlines is not None:
        fieldlines = [c.strip() for c in args.fieldlines.split(",")]
        if len(fieldlines) != 3:
            parser.error("--fieldlines needs exactly 3 comma-separated components "
                         f"(x,y,z), got {len(fieldlines)}: {args.fieldlines!r}")

    # the one place argparse names are mapped onto Opts fields
    opt = Opts(field=args.field, resolution=args.resolution, slice_axis=args.axis,
               slice_pos=args.pos, smooth=args.smooth, scatter=args.scatter,
               cmap=args.cmap, log=args.log, vmin=args.vmin, vmax=args.vmax,
               shared_scale=args.shared_scale, point_size=args.point_size,
               n_contours=args.contours, contour_color=args.contour_color,
               fieldlines=fieldlines, fieldline_color=args.fieldline_color,
               fieldline_density=args.fieldline_density,
               fieldline_broken=args.fieldline_broken,
               title=args.title, labels=args.labels, layout=(nrows, ncols),
               label_color=args.label_color, label_time=args.label_time,
               xlim=args.xlim, ylim=args.ylim, clean=args.clean)

    if args.all:
        print(f"Plotting all {nsteps} steps as {n_panels}-panel figures..."
              if n_panels > 1 else f"Plotting all {nsteps} steps...")
    plot_panel_sets(panel_sets, opt)
