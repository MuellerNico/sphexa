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
in the legend. Overlaid runs are drawn in argument order, so the first file
ends up at the bottom of the stack, and take their color from the shared
categorical scheme palette by position (_h5_common.SCHEME_ORDER: a05, SLR,
SLRB, SLRB2), so a scheme keeps one color across every plotting script.
--residual saves a second figure of
q - q_ref(x), where run-to-run differences fill the axis instead of sitting
sub-pixel on the full data range.
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
                        resolution_label, CLEAN_FONT, apply_clean_style,
                        scheme_colors, scheme_label, apply_sci_ticks, SCHEME_ORDER)
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

# --dissipation panel set: the default fields plus the artificial-resistivity
# fractional decay rate |dB_diss|/|B| and the resistive heating rate
_DISSIPATION_PANELS = _DEFAULT_PANELS + ('dB_diss_rel', 'magneto::du_diss')

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
    'u':           (0.0, 4.5),
    'p':           (0.0, 1.1),
    'log_divBerr': (-7.0, 0.0),
}

# default window and fields for the --zoom post-shock oscillation figure
_ZOOM_DEFAULT_RANGE = (-0.2, 0.4)
_ZOOM_DEFAULT_FIELDS = ('magneto::By', 'vx', 'rho', 'p')


# --- reference-comparison metrics ---

def _flat_intervals(xr, qr, slope_tol=0.1, min_width=0.06, margin=0.03):
    """x-intervals where the reference is a flat plateau: |dq/dx| below slope_tol
    times the global (range / domain length) scale, contiguous runs at least
    min_width wide after shrinking by margin on each side to stay clear of the
    smeared fronts. Restricting metrics to these windows isolates post-shock
    ringing from front-smearing error."""
    scale = (np.max(qr) - np.min(qr)) / (xr[-1] - xr[0])
    if scale == 0:
        return [(xr[0] + margin, xr[-1] - margin)]
    flat = np.abs(np.gradient(qr, xr)) < slope_tol * scale
    intervals = []
    i, n = 0, len(xr)
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        x0, x1 = xr[i] + margin, xr[j] - margin
        if x1 - x0 >= min_width:
            intervals.append((x0, x1))
        i = j + 1
    return intervals


def compute_ref_metrics(g, ref):
    """Per reference panel: L1 = mean |q - ref| over the tube selection, and
    osc = max |q - ref| restricted to flat-plateau reference intervals
    (the post-shock oscillation/overshoot amplitude)."""
    metrics = {}
    xs = g['x']
    for name in g['panels']:
        col = _REF_COLUMNS.get(name)
        if col is None:
            continue
        xr, qr = np.asarray(ref['x']), np.asarray(ref[col])
        qref = np.interp(xs, xr, qr)
        q = g['data'][name]
        L1 = float(np.mean(np.abs(q - qref)))
        sel = np.zeros(len(xs), dtype=bool)
        for x0, x1 in _flat_intervals(xr, qr):
            sel |= (xs >= x0) & (xs <= x1)
        osc = float(np.max(np.abs(q[sel] - qref[sel]))) if sel.any() else np.nan
        metrics[name] = {'L1': L1, 'osc': osc}
    return metrics


def _print_metrics(grids):
    print(f"{'run':<20s} {'panel':<14s} {'L1':>12s} {'plateau osc':>12s}")
    for g in grids:
        for name, m in g.get('metrics', {}).items():
            print(f"{g.get('label', ''):<20s} {name.split('::')[-1]:<14s} "
                  f"{m['L1']:>12.4e} {m['osc']:>12.4e}")


def median_h(fname, step):
    with h5py.File(fname, "r") as f:
        key = f"Step#{step}"
        if key not in f:
            print(f"Error: {key} not found in {fname}")
            print_metadata(fname)
            sys.exit(1)
        return float(np.median(np.array(f[key]['h'])))


def compute_tube_fields(fname, step, y0=None, z0=None, thickness=None,
                        panels=_DEFAULT_PANELS, xlim=None, ref=True, every=1):
    """Read a step and select particles in a tube along x.

    Tube center (y0, z0) defaults to the box midpoint; thickness is the
    absolute half-width in each transverse direction, default 2 * median(h)
    of this dump (the CLI resolves --thickness against the first file so
    overlaid runs share one tube).
    Returns per-particle arrays for each requested panel, masked to the tube
    and to xlim (so shared y-limits ignore the periodic-boundary region).
    With ref=True the 1D reference solution is computed at the dump time.
    'sub' decimates the selection for plotting only: metrics stay on the full
    set, so --every thins the scatter without biasing L1/osc.
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
    if every > 1:
        print(f"  plotting every {every}th: {len(range(0, n_in, every))} points")

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
        'sub':     slice(None, None, every if every > 1 else None),
    }


def render_shocktube(grids, title="Brio-Wu", limits=None, xlim=None,
                     ms=1.0, clean=False):
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

    colors = scheme_colors(len(gs))
    ref = g0.get('ref')
    for i, name in enumerate(panels):
        ax = flat[i]
        # argument order = stacking order, first file at the bottom
        for k, (g, c) in enumerate(zip(gs, colors)):
            sub = g.get('sub', slice(None))
            ax.plot(g['x'][sub], g['data'][name][sub], '.', ms=ms, color=c, alpha=1.0,
                    zorder=2 + k, rasterized=True, label=g.get('label', 'SPHEXA'))
        if ref is not None and name in _REF_COLUMNS:
            ax.plot(ref['x'], ref[_REF_COLUMNS[name]], '-', color="black",
                    lw=1.5, zorder=2 + len(gs), label='reference')
        ax.set_ylabel(g0['labels'][name])
        apply_sci_ticks(ax)
        for li, (g, c) in enumerate(zip(gs, colors)):
            m = g.get('metrics', {}).get(name)
            if m is None:
                continue
            txt = f"L1={m['L1']:.2e}"
            if np.isfinite(m['osc']):
                txt += f" osc={m['osc']:.2e}"
            # stack downwards from the top of the block so the reading order
            # matches the legend, which lists handles in plotting order
            ax.text(0.02, 0.02 + 0.06 * (len(gs) - 1 - li), txt, transform=ax.transAxes,
                    fontsize=7, color=c, ha='left', va='bottom')
        if i == 0 and (ref is not None or len(gs) > 1):
            ax.legend(loc='best', fontsize=9, framealpha=0.9, markerscale=3)
        if limits and name in limits:
            ax.set_ylim(limits[name])
        if xlim is not None:
            ax.set_xlim(xlim)

    for j in range(n, len(flat)):
        flat[j].axis('off')

    # x-label on the lowest populated panel of each column
    for col in range(cols):
        col_idx = [r * cols + col for r in range(rows) if r * cols + col < n]
        if col_idx:
            bottom = flat[col_idx[-1]]
            bottom.set_xlabel('x')
            bottom.tick_params(labelbottom=True)

    if clean:
        plt.tight_layout()
    else:
        tube_n = "/".join(str(g['n_tube']) for g in gs)
        res = " | ".join(dict.fromkeys(g['res_label'] for g in gs))
        fig.suptitle(f"{title}, t={g0['time']:.4f}  (tube N={tube_n})")
        fig.text(0.98, 0.005, f"Resolution: {res}", fontsize=10, ha='right')
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def render_zoom(grids, zoom, fields, title="Brio-Wu", ms=2.5, clean=False):
    """Zoomed view of the post-shock oscillation window; y-ranges auto-scale to
    the window data so sub-percent ringing is resolvable."""
    gs = list(grids) if isinstance(grids, (list, tuple)) else [grids]
    g0 = gs[0]
    n = len(fields)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharex=True,
                             squeeze=False)
    flat = axes.flatten()
    colors = scheme_colors(len(gs))
    ref = g0.get('ref')

    for i, name in enumerate(fields):
        ax = flat[i]
        lo, hi = np.inf, -np.inf
        for k, (g, c) in enumerate(zip(gs, colors)):
            sub = g.get('sub', slice(None))
            xs = g['x'][sub]
            m = (xs >= zoom[0]) & (xs <= zoom[1])
            if not m.any():
                continue
            v = g['data'][name][sub][m]
            ax.plot(xs[m], v, '.', ms=ms, color=c, alpha=0.4,
                    zorder=2 + k, rasterized=True, label=g.get('label', 'SPHEXA'))
            lo, hi = min(lo, np.nanmin(v)), max(hi, np.nanmax(v))
        if ref is not None and name in _REF_COLUMNS:
            ax.plot(ref['x'], ref[_REF_COLUMNS[name]], '-', color='black',
                    lw=1.5, zorder=2 + len(gs), label='reference')
        ax.set_xlim(zoom)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(g0['labels'].get(name, name))
        apply_sci_ticks(ax)
        if i == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9, markerscale=3)

    for j in range(n, len(flat)):
        flat[j].axis('off')
    for col in range(cols):
        col_idx = [r * cols + col for r in range(rows) if r * cols + col < n]
        if col_idx:
            flat[col_idx[-1]].set_xlabel('x')
            flat[col_idx[-1]].tick_params(labelbottom=True)

    if clean:
        plt.tight_layout()
    else:
        fig.suptitle(f"{title} zoom [{zoom[0]:g}, {zoom[1]:g}], t={g0['time']:.4f}")
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def render_residual(grids, ref, title="Brio-Wu", xlim=None, ms=1.0, clean=False):
    """q - q_ref(x) per panel. Same layout as render_shocktube but restricted to
    panels the reference covers. y-limits come from the 99.5th percentile of
    |residual| so isolated front-smearing spikes don't flatten the plateaus,
    where the run-to-run differences actually live."""
    gs = list(grids) if isinstance(grids, (list, tuple)) else [grids]
    g0 = gs[0]
    panels = tuple(p for p in g0['panels'] if p in _REF_COLUMNS)
    if not panels:
        return None
    n = len(panels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True,
                             squeeze=False)
    flat = axes.flatten()
    colors = scheme_colors(len(gs))

    xr = np.asarray(ref['x'])
    for i, name in enumerate(panels):
        ax = flat[i]
        qr = np.asarray(ref[_REF_COLUMNS[name]])
        span = 0.0
        for k, (g, c) in enumerate(zip(gs, colors)):
            sub = g.get('sub', slice(None))
            xs = g['x'][sub]
            d = g['data'][name][sub] - np.interp(xs, xr, qr)
            ax.plot(xs, d, '.', ms=ms, color=c, alpha=0.3, zorder=2 + k,
                    rasterized=True, label=g.get('label', 'SPHEXA'))
            if len(d):
                span = max(span, float(np.nanpercentile(np.abs(d), 99.5)))
        ax.axhline(0.0, color='black', lw=1.0, zorder=2 + len(gs))
        if span > 0:
            ax.set_ylim(-1.15 * span, 1.15 * span)
        ax.set_ylabel(f"$\\Delta$ {g0['labels'][name]}")
        apply_sci_ticks(ax)
        for li, (g, c) in enumerate(zip(gs, colors)):
            m = g.get('metrics', {}).get(name)
            if m is None:
                continue
            txt = f"L1={m['L1']:.2e}"
            if np.isfinite(m['osc']):
                txt += f" osc={m['osc']:.2e}"
            # stack downwards from the top of the block so the reading order
            # matches the legend, which lists handles in plotting order
            ax.text(0.02, 0.02 + 0.06 * (len(gs) - 1 - li), txt, transform=ax.transAxes,
                    fontsize=7, color=c, ha='left', va='bottom')
        if i == 0 and len(gs) > 1:
            ax.legend(loc='best', fontsize=9, framealpha=0.9, markerscale=3)
        if xlim is not None:
            ax.set_xlim(xlim)

    for j in range(n, len(flat)):
        flat[j].axis('off')
    for col in range(cols):
        col_idx = [r * cols + col for r in range(rows) if r * cols + col < n]
        if col_idx:
            flat[col_idx[-1]].set_xlabel('x')
            flat[col_idx[-1]].tick_params(labelbottom=True)

    if clean:
        plt.tight_layout()
    else:
        fig.suptitle(f"{title} residual vs reference, t={g0['time']:.4f}")
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
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


def _gather(fnames, step, y0, z0, hfac, xlim, ref, labels, panels=_DEFAULT_PANELS,
            every=1):
    """One tube selection per file; the reference is computed only once.

    The tube half-width is hfac * median(h) of the *first* file, applied to all
    of them: h drifts between runs, and a per-file width would compare
    different selections.
    """
    s0 = step if step is not None else _last_step(fnames[0])
    thickness = hfac * median_h(fnames[0], s0)

    grids = []
    for k, fn in enumerate(fnames):
        s = step if step is not None else _last_step(fn)
        g = compute_tube_fields(fn, s, y0, z0, thickness, panels=panels, xlim=xlim,
                                ref=ref and k == 0, every=every)
        if labels:
            g['label'] = labels[k]
        elif len(fnames) == 1:
            g['label'] = 'SPHEXA'
        else:
            g['label'] = scheme_label(k) or _run_label(fn)
        grids.append(g)

    t0 = grids[0]['time']
    for g in grids[1:]:
        if abs(g['time'] - t0) > 1e-6 * max(abs(t0), 1.0):
            print(f"  warning: '{g['label']}' is at t={g['time']:.8f} but the "
                  f"reference is evaluated at t={t0:.8f} (first file)")

    ref_sol = grids[0].get('ref')
    if ref_sol is not None:
        for g in grids:
            g['metrics'] = compute_ref_metrics(g, ref_sol)
        _print_metrics(grids)
    return grids


def _render_all(gs, fnames, limits, xlim, title, clean, ms, zoom, zoom_fields,
                residual, suffix):
    fig = render_shocktube(gs, title=title, limits=limits, xlim=xlim, ms=ms, clean=clean)
    _save_fig(fig, fnames[0], gs[0]['step'], suffix=suffix, clean=clean)
    if zoom is not None:
        fig = render_zoom(gs, zoom, zoom_fields, title=title, ms=2.5 * ms, clean=clean)
        _save_fig(fig, fnames[0], gs[0]['step'], suffix=suffix + "_zoom", clean=clean)
    if residual and gs[0].get('ref') is not None:
        fig = render_residual(gs, gs[0]['ref'], title=title, xlim=xlim, ms=ms,
                              clean=clean)
        if fig is not None:
            _save_fig(fig, fnames[0], gs[0]['step'], suffix=suffix + "_residual",
                      clean=clean)


def plot_shocktube(fnames, step, y0=None, z0=None, hfac=2.0,
                   title="Brio-Wu", xlim=None, ref=True, labels=None,
                   auto_ylim=False, clean=False, panels=_DEFAULT_PANELS,
                   zoom=None, zoom_fields=_ZOOM_DEFAULT_FIELDS, every=1, ms=1.0,
                   residual=False):
    fnames = _as_list(fnames)
    grids = _gather(fnames, step, y0, z0, hfac, xlim, ref, labels, panels, every)
    limits = shared_limits(grids) if auto_ylim else _DEFAULT_LIMITS
    suffix = "_compare" if len(fnames) > 1 else ""
    _render_all(grids, fnames, limits, xlim, title, clean, ms, zoom, zoom_fields,
                residual, suffix)


def plot_all_steps(fnames, steps, y0=None, z0=None, hfac=2.0,
                   title="Brio-Wu", xlim=None, ref=True, labels=None,
                   auto_ylim=False, clean=False, panels=_DEFAULT_PANELS,
                   zoom=None, zoom_fields=_ZOOM_DEFAULT_FIELDS, every=1, ms=1.0,
                   residual=False):
    """One PNG per step; fixed default y-limits keep frames and runs comparable."""
    fnames = _as_list(fnames)
    per_step = [_gather(fnames, s, y0, z0, hfac, xlim, ref, labels, panels, every)
                for s in steps]
    limits = shared_limits([g for gs in per_step for g in gs]) if auto_ylim \
        else _DEFAULT_LIMITS
    suffix = "_compare" if len(fnames) > 1 else ""
    for gs in per_step:
        _render_all(gs, fnames, limits, xlim, title, clean, ms, zoom, zoom_fields,
                    residual, suffix)


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
            "  %(prog)s data.h5 5 --thickness 1  Narrower tube: 1 * median(h)\n"
            "  %(prog)s a/dump.h5 b/dump.h5 --labels ARSLR noSLR\n"
            "                                  Overlay two runs in one figure\n"
            "  %(prog)s noisy.h5 a.h5 b.h5 --every 4 --residual\n"
            "                                  Noisiest run at the bottom of the\n"
            "                                  stack, thinned scatter, + residual figure\n"
            "  %(prog)s a/dump.h5 b/dump.h5 --labels SLR SLRB2 --dissipation --auto-ylim\n"
            "                                  ... + dissipation-rate & heating panels vs x\n"
            "\nRun colors are positional: pass the dumps in the order "
            f"{', '.join(SCHEME_ORDER)}.\n"
            "Use plot_gif.py for animated GIFs over a step range.\n"
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
    parser.add_argument("--thickness", type=float, default=2.0, metavar="FACTOR",
                        help="Tube half-width in y and z, in units of median(h) "
                             "(default: 2). Evaluated on the first file and reused "
                             "for the others so overlaid runs share one tube.")
    parser.add_argument("--every", type=int, default=1, metavar="N",
                        help="Plot only every Nth particle of the tube selection. "
                             "Thins dense overlays; metrics stay on the full set.")
    parser.add_argument("--ms", type=float, default=1.0,
                        help="Scatter marker size (default: 1.0). The --zoom figure "
                             "uses 2.5x this.")
    parser.add_argument("--residual", action="store_true",
                        help="Also save a q - q_ref(x) figure. Run-to-run differences "
                             "fill the y-axis instead of sitting sub-pixel on the "
                             "full data range. Needs the reference (not --no-ref).")
    parser.add_argument("--xlim", type=float, nargs=2, default=(-1.0, 1.0),
                        metavar=("XMIN", "XMAX"),
                        help="Plotted x range (default: -1 1, the middle of the "
                             "[-2,2] tube, away from the periodic boundary)")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip the 1D Brio-Wu reference overlay")
    parser.add_argument("--ref-cache", default=None, metavar="DIR",
                        help="Directory for the cached reference solutions (default: "
                             f"{briowu_reference._DEFAULT_CACHE_DIR}, or "
                             "$SPHEXA_REF_CACHE). The solution is solved once per "
                             "dump time and reloaded from an npz afterwards.")
    parser.add_argument("--no-ref-cache", action="store_true",
                        help="Always re-solve the reference instead of caching it")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend label per input file (default: the scheme "
                             f"name for that position, {'/'.join(SCHEME_ORDER)}, "
                             "then the run directory name)")
    parser.add_argument("--auto-ylim", action="store_true",
                        help="Data-driven y-limits instead of the fixed "
                             "comparison ranges")
    parser.add_argument("--dissipation", action="store_true",
                        help="Append two diagnostic panels: the artificial-resistivity "
                             "fractional decay rate |dB_diss|/|B| and the resistive "
                             "heating rate du_diss, projected onto x. Pair with "
                             "--auto-ylim for frame/run-comparable dissipation scales.")
    parser.add_argument("--panels", default=None, metavar="F1,F2,...",
                        help="Comma-separated panel field list (raw or derived names) "
                             "overriding the default 8-panel layout. Wins over "
                             "--dissipation. Example: --panels rho,p,dB_diss_rel,magneto::du_diss")
    parser.add_argument("--zoom", nargs="*", type=float, default=None,
                        metavar="XMIN XMAX",
                        help="Also save a zoomed post-shock oscillation figure "
                             "(auto y-scaling). Bare flag uses the window "
                             f"{_ZOOM_DEFAULT_RANGE[0]:g} {_ZOOM_DEFAULT_RANGE[1]:g}; "
                             "or pass XMIN XMAX.")
    parser.add_argument("--zoom-fields", default=",".join(_ZOOM_DEFAULT_FIELDS),
                        metavar="F1,F2,...",
                        help="Comma-separated fields for the --zoom figure "
                             f"(default: {','.join(_ZOOM_DEFAULT_FIELDS)})")
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

    if args.no_ref_cache:
        briowu_reference.set_cache_dir(False)
    elif args.ref_cache:
        briowu_reference.set_cache_dir(args.ref_cache)

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

    if args.panels:
        panels = tuple(p.strip() for p in args.panels.split(',') if p.strip())
    elif args.dissipation:
        panels = _DISSIPATION_PANELS
    else:
        panels = _DEFAULT_PANELS

    zoom = None
    if args.zoom is not None:
        if len(args.zoom) == 0:
            zoom = _ZOOM_DEFAULT_RANGE
        elif len(args.zoom) == 2:
            zoom = tuple(args.zoom)
        else:
            parser.error("--zoom takes zero or two values (XMIN XMAX)")
    zoom_fields = tuple(f.strip() for f in args.zoom_fields.split(',') if f.strip())

    if args.every < 1:
        parser.error("--every needs a positive integer")

    common = dict(y0=args.y0, z0=args.z0, hfac=args.thickness,
                  title=args.title, xlim=tuple(args.xlim) if args.xlim else None,
                  ref=not args.no_ref, labels=args.labels,
                  auto_ylim=args.auto_ylim, clean=args.clean, panels=panels,
                  zoom=zoom, zoom_fields=zoom_fields, every=args.every,
                  ms=args.ms, residual=args.residual)

    if args.all:
        nsteps = min(get_nsteps(f) for f in args.files)
        if nsteps == 0:
            print("No common steps found")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        plot_all_steps(args.files, list(range(nsteps)), **common)
    else:
        plot_shocktube(args.files, args.step, **common)
