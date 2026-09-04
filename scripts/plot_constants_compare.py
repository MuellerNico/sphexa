#!/usr/bin/env python3
"""Overlay constants.txt series from several runs.

Modes (pick the panels; --panels overrides any of them):
  heating   artificial-resistivity heating rate + cumulative dissipated energy
  emag      magnetic energy evolution, with the dissipation-only floor
  divb      mean/max div(B) error vs time + error-vs-resolution convergence
  conv      the error-vs-resolution panel alone
  kh        Kelvin-Helmholtz mode amplitude M, over the McNally reference
  custom    whatever --panels lists

Usage:
    plot_constants_compare.py heating out/9481246_Alfven_SLRB out/9440992_Alfven_SLR
    plot_constants_compare.py emag out/*_Loop_* --clean --smooth 21
    plot_constants_compare.py divb out/*_OT_* --clean --tmin 0.1
    plot_constants_compare.py conv out/*_Sedov_* --clean --legend fit --xticks 50,100,200,300,400
    plot_constants_compare.py kh out/12518185_KH_290c --clean
    plot_constants_compare.py custom out/a out/b --panels etot_drift,linmom --logy

Run arguments are constants.txt paths or the run directory holding one.
"""

import os
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

from plot_constants import COLUMNS, EXTRA_LABELS, load
from _h5_common import (CATEGORICAL_COLORS, apply_clean_style, apply_log_ticks,
                        apply_sci_ticks)

RAW_LABELS = {
    "iteration":     "iteration",
    "ttot":          r"$t$",
    "minDt":         r"$\Delta t_\mathrm{min}$",
    "etot":          r"$E_\mathrm{tot}$",
    "ecin":          r"$E_\mathrm{kin}$",
    "eint":          r"$E_\mathrm{int}$",
    "egrav":         r"$E_\mathrm{grav}$",
    "linmom":        r"$|\mathbf{p}|$",
    "angmom":        r"$|\mathbf{L}|$",
    "eMag":          r"$E_\mathrm{mag}$",
    "meanDivBError": r"$\langle\varepsilon_{\mathrm{div}B}\rangle$",
    "maxDivBError":  r"$\max\;\varepsilon_{\mathrm{div}B}$",
    "resHeating":    r"$\dot u_\mathrm{diss}$",
}

X_LABELS = {"ttot": r"$t$", "iteration": "iteration"}

# name -> (fn(d) -> values, y-label, required columns)
SERIES = {}


def _series(name, label, needs):
    def deco(fn):
        SERIES[name] = (fn, label, tuple(needs))
        return fn
    return deco


for _col in COLUMNS:
    SERIES[_col] = ((lambda c: lambda d: d[c])(_col), RAW_LABELS.get(_col, _col), (_col,))


def _cumtrapz(y, x):
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))


def _eMag0(d):
    if d["eMag"][0] == 0:
        raise ValueError("eMag_0 is 0")
    return d["eMag"][0]


@_series('eMag_norm', r"$E_\mathrm{mag}/E_\mathrm{mag,0}$", ['eMag'])
def _eMag_norm(d):
    return d["eMag"] / _eMag0(d)


@_series('etot_drift', r"$(E_\mathrm{tot}-E_\mathrm{tot,0})/|E_\mathrm{tot,0}|$", ['etot'])
def _etot_drift(d):
    return (d["etot"] - d["etot"][0]) / abs(d["etot"][0])


@_series('cumRes', r"$\int\dot u_\mathrm{diss}\,\mathrm{d}t$", ['resHeating', 'ttot'])
def _cumRes(d):
    return _cumtrapz(d["resHeating"], d["ttot"])


@_series('cumRes_norm', r"$\int\dot u_\mathrm{diss}\,\mathrm{d}t\,/\,E_\mathrm{mag,0}$",
         ['resHeating', 'ttot', 'eMag'])
def _cumRes_norm(d):
    return _cumRes(d) / _eMag0(d)


@_series('resHeating_norm', r"$\dot u_\mathrm{diss}/E_\mathrm{mag,0}$", ['resHeating', 'eMag'])
def _resHeating_norm(d):
    return d["resHeating"] / _eMag0(d)


# magnetic energy left if artificial resistivity were the only sink; eMag above it
# is field created by the ideal induction term (see plot_loop_decomposition)
@_series('eMag_floor', r"$E_\mathrm{mag}$ (dissipation only)", ['resHeating', 'ttot', 'eMag'])
def _eMag_floor(d):
    return _eMag0(d) - _cumRes(d)


@_series('eMag_floor_norm', r"$E_\mathrm{mag}/E_\mathrm{mag,0}$ (dissipation only)",
         ['resHeating', 'ttot', 'eMag'])
def _eMag_floor_norm(d):
    return 1.0 - _cumRes_norm(d)


# McNally et al. (2012) eq. 10-11 mode amplitude, written to the trailing column
# by the TimeEnergyGrowth observable ("growth rate" there, but it is the amplitude)
@_series('khM', r"mode amplitude $M$", ['extra'])
def _khM(d):
    return d["extra"]


# --norm swaps these in; everything else falls back to division by its first value
NORM_MAP = {
    "eMag":        "eMag_norm",
    "resHeating":  "resHeating_norm",
    "cumRes":      "cumRes_norm",
    "eMag_floor":  "eMag_floor_norm",
}

MODE_PANELS = {
    "conv":    [],
    "kh":      ["khM"],
    "heating": ["resHeating", "cumRes"],
    "emag":    ["eMag"],
    "divb":    ["meanDivBError", "maxDivBError"],
}

# (norm, logy) defaults per mode
MODE_DEFAULTS = {
    "conv":    (False, True),
    "kh":      (False, True),
    "heating": (True, False),
    "emag":    (True, False),
    "divb":    (False, True),
    "custom":  (False, False),
}


def series_value(d, name, kind="turb"):
    if name not in SERIES:
        raise SystemExit(f"unknown series '{name}'\n  available: {', '.join(sorted(SERIES))}")
    fn, label, needs = SERIES[name]
    missing = [c for c in needs if c not in d]
    if missing:
        return None, label, [f"needs {', '.join(missing)}"]
    if name == "extra":
        label = EXTRA_LABELS.get(kind, "extra (col 13)")
    try:
        return fn(d), label, []
    except ValueError as e:
        return None, label, [str(e)]


def moving_average(y, w):
    if w <= 1:
        return y
    k = np.ones(int(w))
    n = np.convolve(np.ones_like(y, dtype=float), k, 'same')
    return np.convolve(y, k, 'same') / n


# time-weighted mean over the whole run; falls back to the plain mean for a
# single-sample or zero-length series
def time_average(d, y):
    t = d["ttot"]
    if len(t) < 2 or t[-1] == t[0]:
        return float(np.mean(y))
    return float(_cumtrapz(y, t)[-1] / (t[-1] - t[0]))


# --- reference solutions ---

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
KH_REFERENCE = os.path.join(DATA_DIR, 'KH_McNally.txt')
KH_REF_LABEL = "McNally et al. 2012"


# (t, M) from the McNally reference table; its third column is a GCI uncertainty
# of order 1e-7, far below line width, so it is not drawn
def load_reference(path):
    ref = np.loadtxt(path)
    return ref[:, 0], ref[:, 1]


# relative L1 against the reference over its time span, resampled onto the run's times
def reference_error(t, y, tref, yref):
    inside = (t >= tref[0]) & (t <= tref[-1])
    if inside.sum() < 2:
        return np.nan
    interp = np.interp(t[inside], tref, yref)
    return float(np.mean(np.abs(y[inside] - interp)) / np.mean(np.abs(interp)))


# --- run loading ---

_PARTICLES_RE = re.compile(r"particles:\s*(\d+)(?:\s*\(~([\d.]+)x([\d.]+)x([\d.]+))?")


# linear resolution from info.log: the largest per-axis count for anisotropic
# boxes (a thin OT/shock slab is named after its in-plane resolution), else N^(1/3)
def detect_resolution(rundir):
    info = os.path.join(rundir, "info.log")
    if not os.path.exists(info):
        return None, None
    for line in open(info):
        m = _PARTICLES_RE.search(line)
        if m:
            n = int(m.group(1))
            if m.group(2):
                return max(float(m.group(i)) for i in (2, 3, 4)), n
            return n ** (1.0 / 3.0), n
    return None, None


def resolve_path(arg):
    if os.path.isdir(arg):
        return os.path.join(arg, "constants.txt")
    return arg


def default_label(fname):
    name = os.path.basename(os.path.dirname(os.path.abspath(fname)))
    return re.sub(r"^\d+_", "", name) or name


def load_run(arg, label, tmin=None, tmax=None):
    fname = resolve_path(arg)
    if not os.path.exists(fname):
        raise SystemExit(f"no constants.txt at {fname}")
    d = load(fname)
    if tmin is not None or tmax is not None:
        keep = np.ones_like(d["ttot"], dtype=bool)
        if tmin is not None:
            keep &= d["ttot"] >= tmin
        if tmax is not None:
            keep &= d["ttot"] <= tmax
        if not keep.any():
            raise SystemExit(f"--tmin/--tmax window is empty for {fname}")
        d = {k: v[keep] for k, v in d.items()}
    rundir = os.path.dirname(os.path.abspath(fname))
    res, npart = detect_resolution(rundir)
    return dict(file=fname, dir=rundir, d=d, label=label or default_label(fname),
                res=res, npart=npart)


# --- plotting ---

def draw_panel(ax, runs, name, colors, xaxis, kind, logx, logy, smooth, absval,
               floor, grid=True, ref=None):
    label = None
    for k, run in enumerate(runs):
        y, label, why = series_value(run["d"], name, kind)
        if y is None:
            print(f"  {run['label']}: no '{name}' ({'; '.join(why)})")
            continue
        x = run["d"][xaxis]
        if absval:
            y = np.abs(y)
        y = moving_average(y, smooth)
        if logy:
            pos = y > 0
            x, y = x[pos], y[pos]
        ax.plot(x, y, color=colors[k], linewidth=1.4, label=run["label"])

        if floor and name in ("eMag", "eMag_norm"):
            fy, _, _ = series_value(run["d"], name.replace("eMag", "eMag_floor"), kind)
            if fy is not None:
                ax.plot(run["d"][xaxis], moving_average(fy, smooth), color=colors[k],
                        linewidth=1.0, linestyle="--", alpha=0.85)

    if ref is not None and name == "khM":
        ax.plot(ref[0], ref[1], color="black", linewidth=1.5, zorder=1,
                label=KH_REF_LABEL)

    ax.set_ylabel(label or name)
    ax.set_xlabel(X_LABELS.get(xaxis, xaxis))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if grid:
        ax.grid(True, which="both", alpha=0.3)


# fixed decimal ticks on a log axis; the shared log-tick helper labels every
# minor tick below two decades, which is unreadable on a short resolution span
def set_log_xticks(ax, vals):
    ax.set_xticks(list(vals))
    ax.set_xticklabels([f"{v:g}" for v in vals])
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def draw_convergence(ax, runs, colors, series_names, kind, legend='all', grid=True):
    known = [(k, r) for k, r in enumerate(runs) if r["res"]]
    if len(known) < 2:
        print("convergence panel skipped: need >= 2 runs with a known resolution "
              "(info.log or --res)")
        return False

    markers = ('o', 's', '^', 'D')
    for si, name in enumerate(series_names):
        pts = []
        for k, run in known:
            y, _, _ = series_value(run["d"], name, kind)
            if y is None:
                continue
            pts.append((run["res"], time_average(run["d"], y), k))
        if len(pts) < 2:
            continue
        pts.sort()
        res = np.array([p[0] for p in pts])
        val = np.array([p[1] for p in pts])
        slope = np.polyfit(np.log10(res), np.log10(val), 1)[0] if (val > 0).all() else np.nan
        tag = rf"$\propto N^{{{slope:.2f}}}$" if np.isfinite(slope) else "fit"
        ax.plot(res, val, color="0.4", linewidth=1.0, linestyle="--", zorder=1,
                marker=markers[si % len(markers)], markersize=0, label=tag)
        for r, v, k in pts:
            ax.plot([r], [v], marker=markers[si % len(markers)], markersize=7,
                    color=colors[k], zorder=3,
                    label=runs[k]["label"] if si == 0 and legend == "all" else "_nolegend_")
        print(f"convergence fit for {name}: slope {slope:.3f} "
              f"over N = {', '.join(f'{r:g}' for r in res)}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"time-averaged $\langle\varepsilon_{\mathrm{div}B}\rangle$")
    if grid:
        ax.grid(True, which="both", alpha=0.3)
    if legend != "none":
        ax.legend(fontsize=9, loc="best")
    return True


def summarize(runs, kind):
    head = (f"{'run':>22}{'N':>8}{'t_final':>10}{'eMag_f/eMag_0':>15}"
            f"{'res/eMag_0':>12}{'<eps_divB>':>12}{'max eps_divB':>14}")
    print(head)
    print("-" * len(head))
    for run in runs:
        d = run["d"]
        g = lambda n: series_value(d, n, kind)[0]
        emag = g("eMag_norm")
        cum = g("cumRes_norm")
        mean_b = d.get("meanDivBError")
        max_b = d.get("maxDivBError")
        print(f"{run['label'][:22]:>22}"
              f"{(str(run['npart']) if run['npart'] else '-'):>8}"
              f"{d['ttot'][-1]:>10.4g}"
              f"{(f'{emag[-1]:.4f}' if emag is not None else '-'):>15}"
              f"{(f'{cum[-1]:.4f}' if cum is not None else '-'):>12}"
              f"{(f'{time_average(d, mean_b):.3e}' if mean_b is not None else '-'):>12}"
              f"{(f'{np.max(max_b):.3e}' if max_b is not None else '-'):>14}")
    print()


def plot(runs, panels, xaxis, kind, logx, logy, smooth, absval, floor,
         convergence, ylim, out, clean, show, title, legend='all', xticks=None,
         grid=True, ref=None):
    if clean:
        apply_clean_style()

    n = len(panels) + int(convergence)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.4 * rows), squeeze=False)
    flat = axes.flatten()
    colors = [CATEGORICAL_COLORS[k % len(CATEGORICAL_COLORS)] for k in range(len(runs))]

    for i, name in enumerate(panels):
        draw_panel(flat[i], runs, name, colors, xaxis, kind, logx, logy, smooth,
                   absval, floor, grid=grid, ref=ref)
        if ylim is not None:
            flat[i].set_ylim(*ylim)

    if convergence:
        # mean only: the max is set by a single particle near a field null and
        # does not converge, so it would just squash the mean on a shared log axis
        if not draw_convergence(flat[len(panels)], runs, colors,
                                ["meanDivBError"], kind, legend=legend, grid=grid):
            if not panels:
                plt.close(fig)
                return
            flat[len(panels)].axis('off')

    for ax in flat[n:]:
        ax.axis('off')

    for ax in flat[:n]:
        apply_log_ticks(ax, axis='both')
        apply_sci_ticks(ax, axis='both')
    if convergence and xticks:
        set_log_xticks(flat[len(panels)], xticks)

    if legend == "all" and panels:
        handles, labels = flat[0].get_legend_handles_labels()
        if floor and any(p in ("eMag", "eMag_norm") for p in panels):
            handles.append(Line2D([], [], color="0.4", linestyle="--", linewidth=1.0))
            labels.append("dissipation-only floor")
        if handles:
            flat[0].legend(handles, labels, fontsize=9, loc="best")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(out, dpi=300 if clean else 150, bbox_inches='tight')
    print(f"Saved: {os.path.abspath(out)}")

    if show:
        matplotlib.use('TkAgg')
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare constants.txt diagnostics across runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="series available for --panels:\n  " + "\n  ".join(sorted(SERIES)))
    p.add_argument("mode", choices=sorted(MODE_DEFAULTS),
                   help="which panels to draw (custom requires --panels)")
    p.add_argument("files", nargs="+",
                   help="constants.txt paths, or run directories containing one")
    p.add_argument("--labels", help="comma-separated legend labels, one per run "
                                    "(default: run directory name without the job id)")
    p.add_argument("--panels", help="comma-separated series names, overriding the mode default")
    p.add_argument("--x", dest="xaxis", default="ttot", choices=["ttot", "iteration"],
                   help="x-axis quantity (default: ttot)")
    p.add_argument("--norm", action=argparse.BooleanOptionalAction, default=None,
                   help="swap eMag / resHeating / cumRes for their eMag_0-normalized "
                        "variants, so runs are comparable (default: on for heating/emag)")
    p.add_argument("--logy", action=argparse.BooleanOptionalAction, default=None,
                   help="log y-axis (default: on for divb)")
    p.add_argument("--logx", action="store_true", help="log x-axis")
    p.add_argument("--abs", dest="absval", action="store_true",
                   help="plot |y| (resistive heating can dip negative in the conjugate form)")
    p.add_argument("--smooth", type=int, default=1, metavar="W",
                   help="centered moving average of width W on the plotted series")
    p.add_argument("--floor", action=argparse.BooleanOptionalAction, default=True,
                   help="in emag panels, overlay eMag_0 - int(resistive heating) (default: on)")
    p.add_argument("--convergence", action=argparse.BooleanOptionalAction, default=None,
                   help="error-vs-resolution panel (default: on for divb and conv)")
    p.add_argument("--res", help="comma-separated linear resolutions, one per run, "
                                 "overriding detection from info.log")
    p.add_argument("--tmin", type=float, help="drop rows before this time; cumulative "
                                              "integrals restart here and eMag_0 is taken here "
                                              "(e.g. skip the MHD-loop startup transient)")
    p.add_argument("--tmax", type=float, help="drop rows past this time (also shortens "
                                              "the cumulative integrals and time averages)")
    p.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"),
                   help="y-limits applied to every time-series panel")
    p.add_argument("--kind", choices=["turb", "kh"], default="turb",
                   help="meaning of the trailing column 13 when plotting 'extra'")
    p.add_argument("--ref", default=None, metavar="PATH",
                   help=f"reference table for the kh panel (default: {KH_REFERENCE})")
    p.add_argument("--no-ref", action="store_true", help="skip the reference overlay")
    p.add_argument("--grid", action=argparse.BooleanOptionalAction, default=None,
                   help="background grid (default: on, off under --clean, matching "
                        "the gridless paper figures from plot_shocktube)")
    p.add_argument("--legend", choices=["all", "fit", "none"], default="all",
                   help="'all' = run labels (+ the convergence fit), 'fit' = only the "
                        "convergence power-law entry, 'none' = no legend (default: all)")
    p.add_argument("--xticks", help="comma-separated x tick values for the convergence "
                                    "panel, e.g. 50,100,200,300,400")
    p.add_argument("--title", help="figure title (default: none in --clean, mode summary otherwise)")
    p.add_argument("--clean", action="store_true", help="serif publication style, 300 dpi pdf")
    p.add_argument("-o", "--out", help="output file "
                                       "(default: compare_<mode>.{pdf,png} next to the first run)")
    p.add_argument("--show", action="store_true", help="show interactive plot")
    args = p.parse_args()

    labels = args.labels.split(",") if args.labels else [None] * len(args.files)
    if len(labels) != len(args.files):
        p.error(f"got {len(labels)} labels for {len(args.files)} runs")

    runs = [load_run(f, lab, tmin=args.tmin, tmax=args.tmax)
            for f, lab in zip(args.files, labels)]

    if args.res:
        res = args.res.split(",")
        if len(res) != len(runs):
            p.error(f"got {len(res)} --res entries for {len(runs)} runs")
        for run, r in zip(runs, res):
            run["res"] = float(r)

    norm, logy = MODE_DEFAULTS[args.mode]
    norm = norm if args.norm is None else args.norm
    logy = logy if args.logy is None else args.logy
    convergence = args.mode in ("divb", "conv") if args.convergence is None else args.convergence
    if args.mode == "conv" and not convergence:
        p.error("mode 'conv' draws nothing with --no-convergence")

    if args.panels:
        panels = args.panels.split(",")
    elif args.mode == "custom":
        p.error("mode 'custom' needs --panels")
    else:
        panels = list(MODE_PANELS[args.mode])

    if norm:
        panels = [NORM_MAP.get(name, name) for name in panels]
    for name in panels:
        if name not in SERIES:
            p.error(f"unknown series '{name}'; available: {', '.join(sorted(SERIES))}")

    title = args.title
    if title is None and not args.clean:
        title = f"{args.mode}: " + " vs ".join(r["label"] for r in runs)

    ext = "pdf" if args.clean else "png"
    out = args.out or os.path.join(runs[0]["dir"], f"compare_{args.mode}.{ext}")

    xticks = [float(v) for v in args.xticks.split(",")] if args.xticks else None

    ref = None
    if "khM" in panels and not args.no_ref:
        refpath = args.ref or KH_REFERENCE
        if args.xaxis != "ttot":
            print(f"reference overlay skipped: it is tabulated in time, not {args.xaxis}")
        elif not os.path.exists(refpath):
            print(f"reference overlay skipped: no such file {refpath}")
        else:
            ref = load_reference(refpath)
            for run in runs:
                y, _, _ = series_value(run["d"], "khM", args.kind)
                if y is None:
                    continue
                err = reference_error(run["d"]["ttot"], y, *ref)
                print(f"{run['label']}: relative L1 vs {KH_REF_LABEL} over "
                      f"t <= {ref[0][-1]:g} = {err:.4f}")
            print()

    summarize(runs, args.kind)
    plot(runs, panels, args.xaxis, args.kind, args.logx, logy,
         args.smooth, args.absval, args.floor, convergence, args.ylim, out,
         args.clean, args.show, title, legend=args.legend, xticks=xticks,
         grid=(not args.clean) if args.grid is None else args.grid, ref=ref)
