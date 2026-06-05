#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse

from _h5_common import print_metadata, get_nsteps, resolution_label


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
    """Project particle coords onto x1 and B onto the rotated x2 axis."""
    R = rotation_to_rotated()
    x = np.asarray(h5step["x"])
    y = np.asarray(h5step["y"])
    z = np.asarray(h5step["z"])
    Bx = np.asarray(h5step["magneto::Bx"])
    By = np.asarray(h5step["magneto::By"])
    Bz = np.asarray(h5step["magneto::Bz"])

    x1 = R[0, 0] * x  + R[0, 1] * y  + R[0, 2] * z
    B2 = R[1, 0] * Bx + R[1, 1] * By + R[1, 2] * Bz
    return x1, B2


def analytic_b2(x1, t, v_alfven):
    k = 2.0 * np.pi / LAMBDA
    return AMPLITUDE * np.sin(k * (x1 + WAVE_SIGN * v_alfven * t))


def render_alfven_wave(fname, step, v_alfven=None):
    print(f"Reading step {step} from {fname}...")
    f, h5step = read_step(fname, step)

    if v_alfven is None:
        v_alfven = np.sqrt(B_PARALLEL * B_PARALLEL / (MU_0 * RHO0))

    x1, B2 = compute_x1_b2(h5step)
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
    ax.scatter(x1, B2, s=4, facecolors='none', edgecolors='black', linewidths=0.4)
    ax.plot(x1_line, B2_line, color='red', linewidth=1.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("B2")
    ax.set_title(f"Alfvèn Wave, L1={L1}, time: [{time_val}]")
    fig.text(0.98, 0.02, f"Resolution: {res_label}", fontsize=10, ha='right')
    plt.tight_layout()

    f.close()
    return fig, time_val, L1


def plot_alfven_wave(fname, step, v_alfven=None):
    fig, _, _ = render_alfven_wave(fname, step, v_alfven=v_alfven)
    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"alfven_b2_step{step}.png")
    fig.savefig(outname, dpi=150, bbox_inches='tight')
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
        ),
    )
    parser.add_argument("file", help="HDF5 input file")
    parser.add_argument("step", nargs="?", help="Step number, range (e.g. 0-20), or omit for metadata")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--v-alfven", type=float, default=None,
                        help="Override Alfvèn speed (default: sqrt(B_par^2/(mu_0*rho))=1.0)")

    args = parser.parse_args()

    if args.all:
        nsteps = get_nsteps(args.file)
        if nsteps == 0:
            print(f"No steps found in {args.file}")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        for step in range(nsteps):
            plot_alfven_wave(args.file, step, v_alfven=args.v_alfven)
        sys.exit(0)

    if args.step is None or args.step == "-p":
        print_metadata(args.file)
        sys.exit(0)

    if "-" in args.step and not args.step.startswith("-"):
        start, end = args.step.split("-", 1)
        make_gif(args.file, int(start), int(end), v_alfven=args.v_alfven)
    else:
        plot_alfven_wave(args.file, int(args.step), v_alfven=args.v_alfven)
