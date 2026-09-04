#!/usr/bin/env python3
"""Plot 2D SPH-interpolated slices of any (raw or derived) field.

Several inputs are tiled into one figure (--layout, --labels), all panels drawn
against a single shared color scale and colorbar. Each input is a file with an
optional step, so repeating a file shows two of its steps side by side:

    plot_slice.py a.h5 2 b.h5 2 a.h5 5 b.h5 5 --labels 128 256 --label-time
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

import os
import sys
import argparse
import functools
from multiprocessing import Pool

from _h5_common import (print_metadata, get_nsteps, resolve_field,
                        resolution_label, read_domain, CLEAN_FONT,
                        apply_clean_style)


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
                        fieldlines=None, smooth=1.0):
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

        dom = read_domain(s)
        if slice_pos is None:
            slice_pos = float(dom.centre[dom.index(slice_axis)])
        if "h" in s:
            h = np.array(s["h"])
        else:
            n_particles = len(coords['x'])
            h_est = 1.2 * (float(np.prod(dom.length)) / n_particles) ** (1.0 / 3.0)
            h = np.full(n_particles, h_est)
            print(f"  h not in file, using estimate h={h_est:.6f}")

    # Wider render kernel: same interpolation, more neighbours per pixel.
    if smooth != 1.0:
        h = h * smooth
        print(f"  Render smoothing: h x {smooth:g}")

    n_particles = len(coords['x'])
    res_label = resolution_label(dom.length, n_particles)
    print(f"Step {step}: time={time_val:.8f}, N={n_particles} ({res_label})")
    print(f"  box: {dom.describe()}, slicing at {slice_axis}={slice_pos:+.6f}")
    print(f"  {field}: [{np.nanmin(values):.6f}, {np.nanmax(values):.6f}]")

    # periodic wrap on the slice axis: a cut on the boundary of a [0, L] box
    # keeps the full kernel support instead of half of it
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

    if scatter:
        return {'step': step, 'time': time_val, 'res_label': res_label,
                'field': field, 'label': label, 'mode': 'scatter',
                'slice_pos': slice_pos, 'bounds': bounds,
                'xs': xs, 'ys': ys, 'values': values[mask],
                'fieldlines': fieldlines is not None}

    print(f"  Interpolating onto {resolution}x{resolution} grid...")
    xi, yi, di = sph_scatter_to_grid(xs, ys, zoff, hs, values[mask], resolution, bounds)

    out = {'step': step, 'time': time_val, 'res_label': res_label,
           'field': field, 'label': label, 'mode': 'grid',
           'slice_pos': slice_pos, 'bounds': bounds, 'xi': xi, 'yi': yi, 'values': di}

    if stream is not None:
        u_vals, v_vals, stem = stream
        _, _, ug = sph_scatter_to_grid(xs, ys, zoff, hs, u_vals[mask], resolution, bounds)
        _, _, vg = sph_scatter_to_grid(xs, ys, zoff, hs, v_vals[mask], resolution, bounds)
        out.update(stream_u=ug, stream_v=vg, stream_stem=stem)

    return out


# Draw one precomputed slice (grid or scatter) into an existing axes; returns
# the mappable for the colorbar. Shared by the single- and multi-panel figures.
def _draw_panel(ax, grids, vmin, vmax, cmap, log=False, point_size=1.0,
                n_contours=0, contour_color='black',
                fieldline_color='black', fieldline_density=1.0,
                fieldline_broken=False, xlim=None, ylim=None):
    # Log colormap: pass via norm= (and don't also pass vmin/vmax).
    color_kw = ({'norm': LogNorm(vmin=vmin, vmax=vmax)} if log
                else {'vmin': vmin, 'vmax': vmax})

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
    return im


# Small/large tick values switch to an offset in scientific notation
# (e.g. 0.2..1.2 with a x10^-3 above) instead of 0.0002, 0.0004, ...
def _format_cbar(cbar, label, log):
    if not log:
        cbar.formatter.set_powerlimits((-3, 3))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()
    cbar.set_label(label)


# Plot a precomputed slice (grid or scatter) and return the figure.
def render_slice(grids, slice_axis, title,
                 vmin, vmax, cmap, log=False, point_size=1.0,
                 n_contours=0, contour_color='black',
                 fieldline_color='black', fieldline_density=1.0,
                 fieldline_broken=False, xlim=None, ylim=None, clean=False):
    ha, va = _PLOT_AXES[slice_axis]
    time_val = grids['time']
    slice_pos = grids['slice_pos']
    header = title if title is not None else grids['label']

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    im = _draw_panel(ax, grids, vmin, vmax, cmap, log, point_size,
                     n_contours, contour_color, fieldline_color,
                     fieldline_density, fieldline_broken, xlim, ylim)
    # append_axes tracks the fixed-aspect host axes, so the colorbar stays
    # flush with the plot edge (fig.colorbar(ax=ax) leaves a gap there).
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    _format_cbar(fig.colorbar(im, cax=cax), grids['label'], log)
    ax.set_xlabel(ha)
    ax.set_ylabel(va)
    if not clean:
        ax.set_title(f"{header}, t=[{time_val}]  ({slice_axis}={slice_pos:+.4f})")
        fig.text(0.98, 0.02, f"Resolution: {grids['res_label']}", fontsize=10, ha='right')
    return fig


# Panel layout for n files: a single row up to 3, two rows beyond (2x2, 2x3, ...).
def _auto_layout(n):
    return (1, n) if n <= 3 else (2, -(-n // 2))


# Panel height/width in data units, so the figure can be sized to the data and
# the tiled panels pack without gaps.
def _panel_aspect(g, xlim, ylim):
    x0, x1, y0, y1 = g['bounds']
    if xlim is not None:
        x0, x1 = xlim
    if ylim is not None:
        y0, y1 = ylim
    return (y1 - y0) / (x1 - x0) if x1 > x0 else 1.0


# Corner tag naming a panel (resolution, scheme, ...).
def _corner_label(ax, text, color, right=True):
    ax.text(0.97 if right else 0.03, 0.96, text, transform=ax.transAxes,
            va='top', ha='right' if right else 'left', color=color)


# Tile several precomputed slices into one figure, all drawn against the same
# color scale and served by a single colorbar. Panels fill row-major and are
# tagged with labels (one per panel, or one per column) top right and, with
# label_time, their time top left.
def render_panels(grids, slice_axis, title, labels, layout,
                  label_color, label_time, vmin, vmax, cmap, log=False, point_size=1.0,
                  n_contours=0, contour_color='black',
                  fieldline_color='black', fieldline_density=1.0,
                  fieldline_broken=False, xlim=None, ylim=None, clean=False):
    ha, va = _PLOT_AXES[slice_axis]
    nrows, ncols = layout if layout is not None else _auto_layout(len(grids))
    if nrows * ncols < len(grids):
        sys.exit(f"Error: --layout {nrows} {ncols} holds {nrows * ncols} panels, "
                 f"but {len(grids)} panels were given")
    if labels is not None:
        labels = [labels[k % len(labels)] for k in range(len(grids))]

    # ImageGrid keeps the panels flush and hands the shared colorbar an axes
    # that tracks their combined height (plain subplots leave ragged gaps
    # around fixed-aspect axes).
    pw = 3.6
    fig = plt.figure(figsize=(ncols * pw + 1.2,
                              nrows * pw * _panel_aspect(grids[0], xlim, ylim)
                              + (0.4 if clean else 1.0)))
    axes = ImageGrid(fig, (0.06, 0.06, 0.88, 0.86 if not clean else 0.90),
                     nrows_ncols=(nrows, ncols),
                     axes_pad=(0.08, 0.08 if clean else 0.34), share_all=True,
                     cbar_mode='single', cbar_location='right',
                     cbar_size='3%', cbar_pad=0.08)

    im = None
    for k, ax in enumerate(axes):
        if k >= len(grids):
            ax.set_axis_off()
            continue
        im = _draw_panel(ax, grids[k], vmin, vmax, cmap, log, point_size,
                         n_contours, contour_color, fieldline_color,
                         fieldline_density, fieldline_broken, xlim, ylim)
        if labels is not None:
            _corner_label(ax, labels[k], label_color)
        if label_time:
            _corner_label(ax, f"t={grids[k]['time']:.1f}", label_color, right=False)
        if not clean:
            ax.set_title(f"t={grids[k]['time']:.4g}  ({grids[k]['res_label']})",
                         fontsize=9)
        if k % ncols == 0:
            ax.set_ylabel(va)
        # bottom-most filled panel of its column, which is not the bottom row
        # when the last row is partly empty; ImageGrid's label_mode='L' already
        # hid those, so undo it explicitly
        if k + ncols >= len(grids):
            ax.set_xlabel(ha)
            ax.xaxis.label.set_visible(True)
            ax.tick_params(labelbottom=True)

    _format_cbar(fig.colorbar(im, cax=axes.cbar_axes[0]), grids[0]['label'], log)
    if not clean:
        header = title if title is not None else grids[0]['label']
        fig.suptitle(f"{header}  ({_pos_label(grids, slice_axis)})", y=0.99, va='top')
    return fig


# Panels of one figure can sit at different planes (each file's own box
# midplane), so name the plane only when they agree.
def _pos_label(grids, slice_axis):
    pos = {g['slice_pos'] for g in grids}
    return (f"{slice_axis}={pos.pop():+.4f}" if len(pos) == 1
            else f"{slice_axis}=box midplane")


def _pos_tag(grids, slice_axis):
    pos = {g['slice_pos'] for g in grids}
    return (f"{slice_axis}{pos.pop():+.4f}" if len(pos) == 1
            else f"{slice_axis}mid")


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


def _save_fig(fig, fname, step, field, pos_tag, scatter, clean, tag='',
              smooth=1.0):
    outdir = os.path.dirname(os.path.abspath(fname))
    short = field.split('::')[-1]
    suffix = ('_scatter' if scatter else '') + ('' if smooth == 1.0 else f"_h{smooth:g}")
    ext = 'pdf' if clean else 'png'
    outname = os.path.join(outdir, f"slice_{short}_step{step}_{pos_tag}{tag}{suffix}.{ext}")
    fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


# Compute and plot a single slice, saved as PNG (PDF with clean=True).
def plot_slice(fname, step, field, resolution, slice_axis,
               slice_pos, title, vmin, vmax, cmap,
               log, scatter, point_size,
               n_contours, contour_color,
               fieldlines, fieldline_color, fieldline_density,
               fieldline_broken, xlim, ylim, clean, smooth=1.0):
    g = compute_slice_grids(fname, step, field, resolution, slice_axis, slice_pos,
                            scatter, fieldlines, smooth)
    fig = render_slice(g, slice_axis, title, vmin, vmax, cmap, log, point_size,
                       n_contours, contour_color, fieldline_color, fieldline_density,
                       fieldline_broken, xlim, ylim, clean)
    _save_fig(fig, fname, step, field, _pos_tag([g], slice_axis), scatter, clean,
              smooth=smooth)


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
def _render_save(g, fname, field, slice_axis, title, vmin, vmax,
                 cmap, log, point_size, n_contours, contour_color, scatter,
                 fieldline_color, fieldline_density, fieldline_broken, xlim, ylim,
                 clean, smooth=1.0):
    fig = render_slice(g, slice_axis, title, vmin, vmax, cmap, log,
                       point_size, n_contours, contour_color,
                       fieldline_color, fieldline_density, fieldline_broken,
                       xlim, ylim, clean)
    _save_fig(fig, fname, g['step'], field, _pos_tag([g], slice_axis), scatter, clean,
              smooth=smooth)


# One PNG per step. With shared_scale (default) a single colormap range spans
# all of them; otherwise each frame is auto-scaled to its own min/max. Explicit
# vmin/vmax always apply in either mode. jobs > 1 fans steps out over processes.
def plot_all_steps(fname, steps, field, resolution, slice_axis,
                   slice_pos, title, vmin, vmax, cmap,
                   log, scatter, point_size,
                   n_contours, contour_color,
                   fieldlines, fieldline_color, fieldline_density,
                   fieldline_broken, xlim, ylim, clean, shared_scale, jobs,
                   smooth=1.0):
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
                                   clean=clean, smooth=smooth)
        _pmap(jobs, worker, steps)
        return

    # Shared range: compute every grid first, then render against common limits.
    compute = functools.partial(compute_slice_grids, fname, field=field, resolution=resolution,
                                slice_axis=slice_axis, slice_pos=slice_pos, scatter=scatter,
                                fieldlines=fieldlines, smooth=smooth)
    grids = _pmap(jobs, compute, steps)
    vmin, vmax = shared_ranges(grids, vmin, vmax, log)
    print(f"Shared {field} scale: [{vmin:.6f}, {vmax:.6f}]" + (" (log)" if log else ""))
    render = functools.partial(_render_save, fname=fname, field=field, slice_axis=slice_axis,
                               title=title, vmin=vmin, vmax=vmax, cmap=cmap,
                               log=log, point_size=point_size, n_contours=n_contours,
                               contour_color=contour_color, scatter=scatter,
                               fieldline_color=fieldline_color,
                               fieldline_density=fieldline_density,
                               fieldline_broken=fieldline_broken, xlim=xlim, ylim=ylim,
                               clean=clean, smooth=smooth)
    _pmap(jobs, render, grids)


# "a.h5 3 b.h5 c.h5 -2" -> [('a.h5', 3), ('b.h5', None), ('c.h5', -2)]: a bare
# integer sets the step of the file in front of it, absent means that file's
# final step. Repeating a file is how two steps of one dump land side by side.
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


# Module-level (picklable) compute of one (file, step) panel. step None means
# that file's own final step, so runs with different dump counts still line up
# on t_end; negative steps count back from the end.
def _compute_panel(item, **kw):
    fname, step = item
    if step is None:
        step = -1
    if step < 0:
        step += get_nsteps(fname)
    return compute_slice_grids(fname, step, **kw)


# One tiled figure per panel set, each set a list of (file, step) panels.
# Panels of a figure always share one color scale; with shared_scale that range
# additionally spans every figure, so an --all series stays comparable frame to
# frame.
def plot_panels(panel_sets, field, resolution, slice_axis,
                slice_pos, title, vmin, vmax, cmap,
                log, scatter, point_size,
                n_contours, contour_color,
                fieldlines, fieldline_color, fieldline_density,
                fieldline_broken, xlim, ylim, clean,
                labels, layout, label_color, label_time, shared_scale, jobs,
                smooth=1.0):
    compute = functools.partial(_compute_panel, field=field, resolution=resolution,
                                slice_axis=slice_axis, slice_pos=slice_pos,
                                scatter=scatter, fieldlines=fieldlines, smooth=smooth)
    grids = _pmap(jobs, compute, [item for ps in panel_sets for item in ps])

    figures, at = [], 0
    for ps in panel_sets:
        figures.append(grids[at:at + len(ps)])
        at += len(ps)

    if shared_scale:
        vmin, vmax = shared_ranges(grids, vmin, vmax, log)
        print(f"Shared {field} scale: [{vmin:.6f}, {vmax:.6f}]" + (" (log)" if log else ""))

    for panel in figures:
        lo, hi = ((vmin, vmax) if shared_scale
                  else shared_ranges(panel, vmin, vmax, log))
        fig = render_panels(panel, slice_axis, title, labels, layout,
                            label_color, label_time, lo, hi, cmap, log, point_size,
                            n_contours, contour_color, fieldline_color,
                            fieldline_density, fieldline_broken, xlim, ylim, clean)
        _save_fig(fig, panel_sets[0][0][0], panel[0]['step'], field,
                  _pos_tag(panel, slice_axis), scatter, clean, tag='_panels',
                  smooth=smooth)


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
            "  %(prog)s data.h5 --smooth 2                 smoother render (wider kernel)\n"
            "  %(prog)s data.h5 --field rho --fieldlines          rho slice + B field lines\n"
            "  %(prog)s data.h5 --field rho --fieldlines vx,vy,vz  ... + velocity field lines\n"
            "  %(prog)s a.h5 b.h5 c.h5 d.h5 --labels 128 256 512 1024\n"
            "                                              2x2 panels, one shared color scale\n"
            "  %(prog)s a.h5 b.h5 --layout 2 1 --labels a05 SLRB2 --clean\n"
            "  %(prog)s a.h5 2 b.h5 2 c.h5 2 a.h5 5 b.h5 5 c.h5 5 \\\n"
            "        --labels 128 256 512 --label-time\n"
            "                                              3 resolutions x 2 times (rows)\n"
        ),
    )
    parser.add_argument("files", nargs="+", metavar="FILE [STEP]",
                        help="HDF5 input file(s), each optionally followed by its step "
                             "number (negative counts from the end, absent = final "
                             "step). Several panels are tiled into one figure with a "
                             "shared color scale, filling the grid row-major; repeat a "
                             "file to show two of its steps (see --layout, --labels).")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata + available fields and exit")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--field", default="rho",
                        help="Field to plot (raw dataset name or derived; default: rho)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z",
                        help="Axis normal to the slice plane (default: z)")
    parser.add_argument("--pos", type=float, default=None,
                        help="Position along the slice axis (default: each file's "
                             "own box midplane, read from the 'box' attribute)")
    parser.add_argument("-r", "--resolution", type=int, default=256,
                        help="Interpolation grid resolution per side (default: 256)")
    parser.add_argument("--title", default=None,
                        help="Plot title prefix (default: field label)")
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                        help="Panel annotation drawn in the top-right corner, one per "
                             "panel (e.g. --labels '$n_x=128$' '$n_x=256$'). One per "
                             "column also works and repeats down the rows.")
    parser.add_argument("--label-time", action="store_true",
                        help="Annotate each panel with its time, top left")
    parser.add_argument("--label-color", default="white",
                        help="Panel label color (default: white)")
    parser.add_argument("--layout", type=int, nargs=2, default=None,
                        metavar=("ROWS", "COLS"),
                        help="Panel grid (default: one row up to 3 panels, two rows "
                             "beyond)")
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
                             "(default: on). Use --no-shared-scale for per-frame auto-scaling. "
                             "Panels of one figure always share their scale.")
    parser.add_argument("--cmap", default="RdBu",
                        help="Matplotlib colormap name (default: RdBu)")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Use a log-scale colormap (LogNorm). Best for positive "
                             "fields like rho, Bmag, vmag, Emag.")
    parser.add_argument("--scatter", action="store_true",
                        help="Skip SPH interpolation; render raw particle scatter (fast)")
    parser.add_argument("--smooth", type=float, default=1.0, metavar="FACTOR",
                        help="Scale every particle's h by FACTOR in the render kernel "
                             "(default: 1.0 = the simulation's own h). >1 averages over "
                             "more neighbours per pixel, giving the smooth look of "
                             "published rendered slices at the cost of a blurrier "
                             "contact discontinuity; try 1.5-3. Cost grows as FACTOR^2. "
                             "In --scatter mode it only widens the slab.")
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
                        help="Worker processes for --all and for multi-file panels "
                             "(default: 1 = serial). "
                             "In SLURM, pass -j \"$SLURM_CPUS_PER_TASK\".")

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
            print(f"No steps found in {fnames}")
            sys.exit(1)
        panel_sets = [[(f, s) for f in fnames] for s in range(nsteps)]
    else:
        panel_sets = [panels]

    n_panels = len(panel_sets[0])
    nrows, ncols = args.layout if args.layout is not None else _auto_layout(n_panels)
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

    common = dict(field=args.field, resolution=args.resolution, slice_axis=args.axis,
                  slice_pos=args.pos, title=args.title, vmin=args.vmin, vmax=args.vmax,
                  cmap=args.cmap, log=args.log, scatter=args.scatter,
                  point_size=args.point_size,
                  n_contours=args.contours, contour_color=args.contour_color,
                  fieldlines=fieldlines, fieldline_color=args.fieldline_color,
                  fieldline_density=args.fieldline_density,
                  fieldline_broken=args.fieldline_broken,
                  xlim=args.xlim, ylim=args.ylim, clean=args.clean,
                  smooth=args.smooth)

    # a lone panel with --labels/--label-time still goes through the panel
    # renderer, so the annotations are available on single-panel figures too
    if n_panels > 1 or args.labels is not None or args.label_time:
        if args.all:
            print(f"Plotting all {nsteps} steps as {n_panels}-panel figures...")
        plot_panels(panel_sets, labels=args.labels, layout=(nrows, ncols),
                    label_color=args.label_color, label_time=args.label_time,
                    shared_scale=args.shared_scale, jobs=args.jobs, **common)
    elif args.all:
        fname = panels[0][0]
        print(f"Plotting all {nsteps} steps...")
        plot_all_steps(fname, list(range(nsteps)),
                       shared_scale=args.shared_scale, jobs=args.jobs, **common)
    else:
        fname, step = panels[0]
        if step is None or step < 0:
            nsteps = get_nsteps(fname)
            if nsteps == 0:
                print(f"No steps found in {fname}")
                sys.exit(1)
            step = nsteps - 1 if step is None else step + nsteps
        plot_slice(fname, step, **common)
