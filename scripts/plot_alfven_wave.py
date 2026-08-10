#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse

from _h5_common import (print_metadata, get_nsteps, resolution_label,
                        CLEAN_FONT, apply_clean_style, scheme_colors,
                        scheme_color,scheme_label, apply_sci_ticks, 
                        apply_log_ticks, SCHEME_ORDER)


# Wave parameters from main/src/init/alfven_wave_init.hpp::ALfvenWaveConstants
SIN_A = 2.0 / 3.0
SIN_B = 2.0 / np.sqrt(5.0)
LAMBDA = 1.0
AMPLITUDE = 0.1
B_PARALLEL = 1.0
RHO0 = 1.0
MU_0 = 1.0
# delta_V == +delta_B in the initializer => wave travels in -x1 direction,
# so the analytic solution shifts as x1 + v_A * t.
WAVE_SIGN = +1.0


def rotation_to_rotated(sinA=SIN_A, sinB=SIN_B):
    """Rows are (x1, x2, x3) basis vectors expressed in cartesian coords.
    Matches coordinateTransformToRotated in alfven_wave_init.hpp."""
    cosA = np.sqrt(1.0 - sinA * sinA)
    cosB = np.sqrt(1.0 - sinB * sinB)
    return np.array([
        [cosA * cosB, cosA * sinB, sinA],
        [-sinB,       cosB,        0.0 ],
        [-sinA*cosB, -sinA*sinB,   cosA],
    ])


def read_step(fname, step):
    f = h5py.File(fname, "r")
    key = f"Step#{step}"
    if key not in f:
        print(f"Error: {key} not found in {fname}")
        print_metadata(fname)
        sys.exit(1)
    return f, f[key]


def compute_x1_b2(h5step):
    """Project particle coords onto x1 and B onto the rotated x2/x3 axes."""
    R = rotation_to_rotated()
    x = np.asarray(h5step["x"])
    y = np.asarray(h5step["y"])
    z = np.asarray(h5step["z"])
    Bx = np.asarray(h5step["magneto::Bx"])
    By = np.asarray(h5step["magneto::By"])
    Bz = np.asarray(h5step["magneto::Bz"])

    x1 = R[0, 0] * x  + R[0, 1] * y  + R[0, 2] * z
    B2 = R[1, 0] * Bx + R[1, 1] * By + R[1, 2] * Bz
    B3 = R[2, 0] * Bx + R[2, 1] * By + R[2, 2] * Bz
    return x1, B2, B3


def analytic_b2(x1, t, v_alfven):
    k = 2.0 * np.pi / LAMBDA
    return AMPLITUDE * np.sin(k * (x1 + WAVE_SIGN * v_alfven * t))


# --- mode-decomposition analysis (amplitude decay / noise power / phase error) ---

def fit_mode(x1, q, basis):
    """Least-squares fit q ~ a sin(k x1) + b cos(k x1). Returns (A, phi) with
    q = A sin(k x1 + phi) for basis='sin', q = A cos(k x1 + phi) for basis='cos',
    plus the fitted values, so the caller can form the mode-free residual."""
    k = 2.0 * np.pi / LAMBDA
    s, c = np.sin(k * x1), np.cos(k * x1)
    M = np.array([[s @ s, s @ c], [s @ c, c @ c]])
    a, b = np.linalg.solve(M, np.array([s @ q, c @ q]))
    fit = a * s + b * c
    if basis == 'sin':
        return float(np.hypot(a, b)), float(np.arctan2(b, a)), fit
    return float(np.hypot(a, b)), float(np.arctan2(-a, b)), fit


def analyze_run(fname, v_alfven=None, trange=None):
    """Per step: mean transverse mode amplitude A, wrapped phase error vs the
    analytic solution, and rms mode-free residual Pn (the noise power).
    trange=(tmin, tmax) restricts the analysis to dumps in that time window."""
    if v_alfven is None:
        v_alfven = np.sqrt(B_PARALLEL * B_PARALLEL / (MU_0 * RHO0))
    k = 2.0 * np.pi / LAMBDA

    out = {'t': [], 'A': [], 'phase_err': [], 'Pn': []}
    nsteps = get_nsteps(fname)
    print(f"Analyzing {fname} ({nsteps} steps)...")
    for step in range(nsteps):
        f, s = read_step(fname, step)
        t = float(s.attrs["time"][0])
        if trange is not None and not (trange[0] <= t <= trange[1]):
            f.close()
            continue
        x1, B2, B3 = compute_x1_b2(s)
        f.close()

        A2, p2, fit2 = fit_mode(x1, B2, 'sin')
        A3, p3, fit3 = fit_mode(x1, B3, 'cos')
        Pn = float(np.sqrt(np.mean((B2 - fit2)**2 + (B3 - fit3)**2)))
        expected = WAVE_SIGN * k * v_alfven * t
        perr = float(np.angle(np.exp(1j * (0.5 * (p2 + p3) - expected))))

        out['t'].append(float(t))
        out['A'].append(0.5 * (A2 + A3))
        out['phase_err'].append(perr)
        out['Pn'].append(Pn)
        print(f"  step {step:3d}: t={t:8.4f}  A/A0={0.5*(A2+A3)/AMPLITUDE:8.5f}  "
              f"Pn={Pn:.4e}  phase_err={perr:+.4f}")
    if not out['t']:
        sys.exit(f"Error: no steps of {fname} fall inside --trange")
    return {key: np.array(v) for key, v in out.items()}


def _fit_rate(t, q, tmin):
    """Exponential rate from a linear fit to ln q(t) for t >= tmin."""
    m = (t >= tmin) & (q > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(t[m], np.log(q[m]), 1)[0])


def _run_label(fname):
    d = os.path.basename(os.path.dirname(os.path.abspath(fname)))
    return d or os.path.basename(fname)


def analyze(fnames, labels=None, fit_tmin=2.0, v_alfven=None, clean=False,
            trange=None):
    """Multi-run overlay: A(t)/A0 with bleed rates, Pn(t) with cleaning rates,
    and phase error, each as a separate figure next to the first file.

    Runs are colored by argument position from the shared scheme palette."""
    runs = [analyze_run(f, v_alfven, trange) for f in fnames]
    if labels is None:
        labels = [(scheme_label(k) if len(fnames) > 1 else None) or _run_label(f)
                  for k, f in enumerate(fnames)]
    outdir = os.path.dirname(os.path.abspath(fnames[0]))

    print(f"\n{'run':<28s} {'bleed rate [1/t]':>18s} {'noise rate [1/t]':>18s}")
    rates = []
    for r, lbl in zip(runs, labels):
        rA = _fit_rate(r['t'], r['A'], fit_tmin)
        rP = _fit_rate(r['t'], r['Pn'], fit_tmin)
        rates.append((rA, rP))
        print(f"{lbl:<28s} {rA:>18.4e} {rP:>18.4e}")

    figures = (
        ('amp',   True,  r'$A/A_0$',          lambda r: r['A'] / AMPLITUDE, 0),
        ('noise', True,  r'$P_n$',            lambda r: r['Pn'],            1),
        ('phase', False, 'phase error [rad]', lambda r: r['phase_err'],     None),
    )
    colors = scheme_colors(len(runs))
    ext = 'pdf' if clean else 'png'
    for suffix, logy, ylabel, get, rate_idx in figures:
        fig, ax = plt.subplots(figsize=(7, 5))
        for (r, lbl), rr, c in zip(zip(runs, labels), rates, colors):
            lab = lbl if rate_idx is None or clean or not np.isfinite(rr[rate_idx]) \
                else f"{lbl} (rate {rr[rate_idx]:+.2e}/t)"
            ax.plot(r['t'], get(r), 'o-', ms=3, lw=1.0, color=c, label=lab)
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('t')
        ax.set_ylabel(ylabel)
        apply_sci_ticks(ax)
        apply_log_ticks(ax)
        ax.legend(fontsize=8)
        plt.tight_layout()
        outname = os.path.join(outdir, f"alfven_analysis_{suffix}.{ext}")
        fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outname}")


# --- resolution convergence ---

# Per-axis resolutions of the convergence lineup, positional over the input
# files. Hardcoded: single-use figure, and the dumps don't record nx.
_CONVERGENCE_NX = (60, 120, 180, 240)


def l1_error(fname, step=None, v_alfven=None):
    """L1 of B2 against the analytic solution at `step` (default: last)."""
    if v_alfven is None:
        v_alfven = np.sqrt(B_PARALLEL * B_PARALLEL / (MU_0 * RHO0))
    if step is None:
        step = get_nsteps(fname) - 1
    f, s = read_step(fname, step)
    x1, B2, _ = compute_x1_b2(s)
    t = float(s.attrs["time"][0])
    n = len(x1)
    f.close()
    return float(np.mean(np.abs(B2 - analytic_b2(x1, t, v_alfven)))), t, n


def convergence(fnames, step=None, v_alfven=None, clean=False):
    """L1(nx) over the input files in _CONVERGENCE_NX order, log-log, with an
    nx^-2 reference anchored to the coarsest run."""
    from matplotlib.ticker import NullLocator

    if len(fnames) > len(_CONVERGENCE_NX):
        sys.exit(f"Error: --convergence has {len(_CONVERGENCE_NX)} hardcoded "
                 f"resolutions {_CONVERGENCE_NX}, got {len(fnames)} files")
    nx = np.array(_CONVERGENCE_NX[:len(fnames)], dtype=float)

    L1, times = [], []
    print(f"{'nx':>6s} {'N':>12s} {'t':>10s} {'L1':>14s}")
    for fn, n_ax in zip(fnames, nx):
        e, t, n = l1_error(fn, step, v_alfven)
        L1.append(e)
        times.append(t)
        print(f"{n_ax:>6.0f} {n:>12d} {t:>10.4f} {e:>14.6e}")
    L1 = np.array(L1)

    if max(times) - min(times) > 1e-6 * max(abs(times[0]), 1.0):
        print(f"  warning: dumps are not at a common time: "
              f"{', '.join(f'{t:.6f}' for t in times)}")

    order = np.nan
    if len(fnames) >= 2:
        order = float(np.polyfit(np.log(nx), np.log(L1), 1)[0])
        print(f"fitted convergence order: {order:+.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    lab = r'$L_1$' if not np.isfinite(order) or clean else rf'$L_1$ (order {order:+.2f})'
    ax.plot(nx, L1, 'o-', color=scheme_color(1), lw=1.2, ms=5, label=lab)
    ax.plot(nx, L1[0] * (nx / nx[0])**-2.0, '--', color='black', lw=1.2,
            label=r'$\propto n_x^{-2}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n_x$')
    ax.set_ylabel(r'$L_1(B_2)$')
    # one labeled tick per run: the decade ticks of a 60..240 span are useless
    ax.set_xticks(nx)
    ax.set_xticklabels([f"{v:.0f}" for v in nx])
    ax.xaxis.set_minor_locator(NullLocator())
    apply_log_ticks(ax)
    ax.legend(fontsize=9)
    plt.tight_layout()

    outdir = os.path.dirname(os.path.abspath(fnames[0]))
    ext = 'pdf' if clean else 'png'
    outname = os.path.join(outdir, f"alfven_convergence.{ext}")
    fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


def render_alfven_wave(fname, step, v_alfven=None, clean=False):
    print(f"Reading step {step} from {fname}...")
    f, h5step = read_step(fname, step)

    if v_alfven is None:
        v_alfven = np.sqrt(B_PARALLEL * B_PARALLEL / (MU_0 * RHO0))

    x1, B2, _ = compute_x1_b2(h5step)
    time_val = h5step.attrs["time"][0]
    n_particles = len(x1)
    extents = [np.asarray(h5step[ax]).max() - np.asarray(h5step[ax]).min()
               for ax in ('x', 'y', 'z')]
    res_label = resolution_label(extents, n_particles)

    print(f"Step {step}: time={time_val:.8f}, N={n_particles} ({res_label})")
    print(f"  x1: [{x1.min():.4f}, {x1.max():.4f}]")
    print(f"  B2: [{B2.min():.6f}, {B2.max():.6f}]")

    B2_ref = analytic_b2(x1, time_val, v_alfven)
    L1 = float(np.mean(np.abs(B2 - B2_ref)))
    print(f"  v_alfven = {v_alfven:.6f}, L1 = {L1:.10g}")

    x1_line = np.linspace(x1.min(), x1.max(), 1024)
    B2_line = analytic_b2(x1_line, time_val, v_alfven)

    fig, ax = plt.subplots(figsize=(7, 5))
    # rasterized: one embedded image instead of a vector path per particle,
    # which is what makes the PDF slow to open at these particle counts
    ax.scatter(x1, B2, s=1, marker='.', color=scheme_color(1), alpha=0.5,
               linewidths=0, rasterized=True)
    ax.plot(x1_line, B2_line, color='black', linewidth=1.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("B2")
    apply_sci_ticks(ax)
    if not clean:
        ax.set_title(f"Alfvèn Wave, L1={L1}, time: [{time_val}]")
        fig.text(0.98, 0.02, f"Resolution: {res_label}", fontsize=10, ha='right')
    plt.tight_layout()

    f.close()
    return fig, time_val, L1


def plot_alfven_wave(fname, step, v_alfven=None, clean=False):
    fig, _, _ = render_alfven_wave(fname, step, v_alfven=v_alfven, clean=clean)
    outdir = os.path.dirname(os.path.abspath(fname))
    ext = 'pdf' if clean else 'png'
    outname = os.path.join(outdir, f"alfven_b2_step{step}.{ext}")
    fig.savefig(outname, dpi=300 if clean else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


def _render_frame(args):
    fname, step, v_alfven = args
    fig, _, _ = render_alfven_wave(fname, step, v_alfven=v_alfven)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    buf.close()
    return step, png_bytes


def make_gif(fname, start, end, n_workers=None, v_alfven=None):
    import io
    from multiprocessing import Pool, cpu_count
    from PIL import Image

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    all_steps = list(range(start, end + 1))
    max_frames = 100
    if len(all_steps) > max_frames:
        stride = len(all_steps) // max_frames
        steps = all_steps[::stride]
        print(f"Range has {len(all_steps)} steps, using stride={stride} -> {len(steps)} frames")
    else:
        steps = all_steps

    n_total = len(steps)
    print(f"Generating {n_total} frames using {n_workers} workers...")

    work = [(fname, step, v_alfven) for step in steps]
    results = {}
    with Pool(n_workers) as pool:
        for i, (step, png_bytes) in enumerate(pool.imap_unordered(_render_frame, work)):
            results[step] = png_bytes
            print(f"  [{i + 1}/{n_total}] Step {step} done", flush=True)

    print("Assembling GIF...")
    frames = []
    for step in steps:
        buf = io.BytesIO(results[step])
        frames.append(Image.open(buf).copy())
        buf.close()

    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"alfven_b2_steps{start}-{end}.gif")
    frames[0].save(
        outname,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved GIF ({len(frames)} frames): {outname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the travelling Alfvèn-wave test from SPH HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s data.h5 -p           Print metadata\n"
            "  %(prog)s data.h5 5            PNG of B2 vs x1 at step 5\n"
            "  %(prog)s data.h5 0-100        GIF of steps 0..100\n"
            "  %(prog)s data.h5 --all        PNG of B2 vs x1 for every step\n"
            "  %(prog)s a.h5 b.h5 --analyze --labels 'alphaB=0.5' SLR\n"
            "                               Mode amplitude / noise power / phase\n"
            "                               figures overlaying both runs\n"
            "  %(prog)s a.h5 b.h5 --analyze --trange 0 5 --clean\n"
            "                               ... over t in [0, 5] only, as PDF\n"
            "  %(prog)s 60/d.h5 120/d.h5 180/d.h5 --convergence\n"
            "                               L1 vs resolution, log-log, + nx^-2 line\n"
            "\nWith --analyze, run colors are positional: pass the dumps in the "
            f"order {', '.join(SCHEME_ORDER)}.\n"
        ),
    )
    parser.add_argument("files", nargs="+", metavar="file",
                        help="HDF5 input file(s); several files only with --analyze")
    parser.add_argument("step", nargs="?", help="Step number, range (e.g. 0-20), or omit for metadata")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--analyze", action="store_true",
                        help="Fit the transverse mode per step and plot A(t)/A0, "
                             "noise power Pn(t) and phase error as separate figures; "
                             "multiple files overlay for scheme comparison")
    parser.add_argument("--convergence", action="store_true",
                        help="L1 error of each input file vs resolution, log-log, "
                             "with an nx^-2 reference. Files are assumed to be in "
                             f"order of increasing resolution nx = {_CONVERGENCE_NX} "
                             "(hardcoded in the script). Uses the final step of each "
                             "file unless a step is given.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend label per input file with --analyze (default: "
                             f"the scheme name for that position, "
                             f"{'/'.join(SCHEME_ORDER)}, then the run directory name)")
    parser.add_argument("--trange", type=float, nargs=2, default=None,
                        metavar=("TMIN", "TMAX"),
                        help="Restrict --analyze to dumps with TMIN <= t <= TMAX, "
                             "setting the plotted x range. Past a few wave periods "
                             "the solution has drifted too far for the noise "
                             "quantification to mean much.")
    parser.add_argument("--fit-tmin", type=float, default=2.0,
                        help="Skip t < FIT_TMIN in the decay-rate fits, to exclude "
                             "the initial transient (default: 2.0)")
    parser.add_argument("--v-alfven", type=float, default=None,
                        help="Override Alfvèn speed (default: sqrt(B_par^2/(mu_0*rho))=1.0)")
    parser.add_argument("--clean", action="store_true",
                        help="Publish mode for thesis figures: save PDF instead of PNG, "
                             "drop the title and resolution label (those go in the "
                             f"caption), and render text in {CLEAN_FONT} "
                             "(CLEAN_FONT in _h5_common.py).")

    args = parser.parse_args()

    if args.clean:
        apply_clean_style()

    # nargs='+' swallows a trailing step number; pull it back out
    if args.step is None and len(args.files) > 1 and args.files[-1].lstrip('+-').isdigit():
        args.step = args.files.pop()

    if args.labels and len(args.labels) != len(args.files):
        parser.error("--labels needs one label per input file")

    if args.convergence:
        step = int(args.step) if args.step and args.step.isdigit() else None
        convergence(args.files, step=step,
                    v_alfven=args.v_alfven, clean=args.clean)
        sys.exit(0)

    if args.analyze:
        analyze(args.files, labels=args.labels, fit_tmin=args.fit_tmin,
                v_alfven=args.v_alfven, clean=args.clean,
                trange=tuple(args.trange) if args.trange else None)
        sys.exit(0)

    if len(args.files) > 1:
        parser.error("multiple input files are only supported with --analyze "
                     "or --convergence")
    fname = args.files[0]

    if args.all:
        nsteps = get_nsteps(fname)
        if nsteps == 0:
            print(f"No steps found in {fname}")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        for step in range(nsteps):
            plot_alfven_wave(fname, step, v_alfven=args.v_alfven, clean=args.clean)
        sys.exit(0)

    if args.step is None or args.step == "-p":
        print_metadata(fname)
        sys.exit(0)

    if "-" in args.step and not args.step.startswith("-"):
        start, end = args.step.split("-", 1)
        make_gif(fname, int(start), int(end), v_alfven=args.v_alfven)
    else:
        plot_alfven_wave(fname, int(args.step), v_alfven=args.v_alfven,
                         clean=args.clean)
