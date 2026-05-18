#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import argparse


def print_metadata(fname):
    """Print all available fields, dimensions, attributes, and step info."""
    with h5py.File(fname, "r") as f:
        print(f"=== HDF5 Metadata: {fname} ===")

        nsteps = len([k for k in f.keys() if k.startswith("Step#")])
        print(f"Number of steps: {nsteps}")
        print(f"{'Step':>8s} {'Iteration':>12s} {'Time':>15s} {'N particles':>14s}")
        print("-" * 52)

        for i in range(nsteps):
            step = f[f"Step#{i}"]
            iteration = step.attrs.get("iteration", [None])[0]
            time = step.attrs.get("time", [None])[0]
            n = None
            for key in step.keys():
                ds = step[key]
                if hasattr(ds, 'shape') and len(ds.shape) > 0:
                    n = ds.shape[0]
                    break
            print(f"{i:>8d} {iteration:>12} {time:>15.8f} {n:>14}")

        if nsteps > 0:
            step0 = f["Step#0"]
            print(f"\nDatasets in Step#0:")
            for key in sorted(step0.keys()):
                ds = step0[key]
                print(f"  {key:>20s}  shape={ds.shape}  dtype={ds.dtype}")

            print(f"\nAttributes in Step#0:")
            for attr in sorted(step0.attrs.keys()):
                val = step0.attrs[attr]
                print(f"  {attr:>20s} = {val}")
        print()


def read_step(fname, step):
    f = h5py.File(fname, "r")
    key = f"Step#{step}"
    if key not in f:
        print(f"Error: {key} not found in {fname}")
        print_metadata(fname)
        sys.exit(1)
    return f, f[key]


def get_nsteps(fname):
    with h5py.File(fname, "r") as f:
        return len([k for k in f.keys() if k.startswith("Step#")])


def cubic_spline_3d(q):
    """Cubic spline (M4) SPH kernel shape in 3D. q = r/h.

    Returns sigma_3D * f(q). The 1/h**3 factor of the full kernel is omitted:
    it cancels for both fields plotted here (see sph_scatter_to_grid).
    """
    sigma = 1.0 / np.pi
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    w[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * w


def sph_scatter_to_grid(xs, ys, zoff, hs, deposit, resolution=256, normalize=False):
    """Scatter 3D SPH particles onto a 2D slice grid via kernel interpolation.

    The slice plane cuts through the 3D volume; each particle contributes
    through the full 3D kernel, so its weight is attenuated by zoff, its
    perpendicular distance from the plane (not a thin-slab projection).

    The kernel's 1/h**3 normalization is dropped throughout. For equal-mass
    SPH (every SPHEXA test case) the particle volume obeys V_j ~ h_j**3, so:
      - density:  rho(x) ~ sum_j rho_j f(q_j)         (h**3 cancels m_j W_j)
      - alpha_B:  a(x)   = sum_j a_j f(q_j) / sum_j f(q_j)  (h**3 and the
                  unknown coupling constant cancel in the ratio)
    Density is therefore returned in arbitrary units (true rho up to a global
    factor); alpha_B is exact since the normalization is a pure ratio.

    xs, ys:    in-plane particle coordinates
    zoff:      signed perpendicular distance of each particle from the slice
    hs:        smoothing lengths
    deposit:   per-particle quantity accumulated as sum_j deposit_j f(q_j)
    normalize: if True, divide by sum_j f(q_j) (Shepard average, for alpha_B);
               if False, return the raw kernel sum (for density)

    Returns (xi, yi, field) with xi, yi as grid-edge meshes for pcolormesh.
    """
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution

    # Grid cell centers
    xc = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, resolution)
    yc = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, resolution)

    value_grid = np.zeros((resolution, resolution))
    weight_grid = np.zeros((resolution, resolution)) if normalize else None

    for i in range(len(xs)):
        hi = hs[i]
        support = 2.0 * hi

        # In-plane reach shrinks with the perpendicular offset from the slice
        inplane = np.sqrt(max(support * support - zoff[i] * zoff[i], 0.0))

        # Grid index range affected by this particle
        ix_lo = max(int((xs[i] - inplane - xmin) / dx), 0)
        ix_hi = min(int((xs[i] + inplane - xmin) / dx) + 1, resolution)
        iy_lo = max(int((ys[i] - inplane - ymin) / dy), 0)
        iy_hi = min(int((ys[i] + inplane - ymin) / dy) + 1, resolution)
        if ix_lo >= ix_hi or iy_lo >= iy_hi:
            continue

        # Vectorized over the affected patch
        gx = xc[ix_lo:ix_hi]
        gy = yc[iy_lo:iy_hi]
        gxx, gyy = np.meshgrid(gx, gy, indexing='ij')

        r = np.sqrt((gxx - xs[i])**2 + (gyy - ys[i])**2 + zoff[i]**2)
        w = cubic_spline_3d(r / hi)

        value_grid[iy_lo:iy_hi, ix_lo:ix_hi] += (w * deposit[i]).T
        if weight_grid is not None:
            weight_grid[iy_lo:iy_hi, ix_lo:ix_hi] += w.T

    if weight_grid is not None:
        valid = weight_grid > 0
        field = np.full((resolution, resolution), np.nan)
        field[valid] = value_grid[valid] / weight_grid[valid]
    else:
        field = value_grid

    xi, yi = np.meshgrid(
        np.linspace(xmin, xmax, resolution + 1),
        np.linspace(ymin, ymax, resolution + 1),
    )
    return xi, yi, field


# Axes to plot for each slice axis (horizontal, vertical)
_PLOT_AXES = {
    'z': ('x', 'y'),
    'y': ('x', 'z'),
    'x': ('y', 'z'),
}


def render_density_slice(fname, step, resolution=256, slice_axis='z', slice_pos=0.0,
                         title="Density", vmin=None, vmax=None, cmap='bone_r'):
    """Render a 2D slice of density using SPH kernel interpolation.

    slice_axis: which axis is normal to the slice plane ('x', 'y', or 'z')
    slice_pos:  coordinate value at which to cut (default 0.0)

    Returns (fig, time_val) so callers can save as PNG or capture for GIF.
    """
    print(f"Reading step {step} from {fname}...")
    f, h5step = read_step(fname, step)

    coords = {
        'x': np.array(h5step["x"]),
        'y': np.array(h5step["y"]),
        'z': np.array(h5step["z"]),
    }
    rho = np.array(h5step["rho"])

    have_alpha_B = "magneto::alpha_B" in h5step
    alpha_B = np.array(h5step["magneto::alpha_B"]) if have_alpha_B else None

    time_val = h5step.attrs["time"][0]
    n_particles = len(coords['x'])
    n_cbrt = round(n_particles ** (1.0 / 3.0), 1)

    print(f"Step {step}: time={time_val:.8f}, N={n_particles} (~{n_cbrt}^3)")
    for ax, vals in coords.items():
        print(f"  {ax}: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  rho: [{rho.min():.6f}, {rho.max():.6f}]")
    if have_alpha_B:
        print(f"  alpha_B: [{alpha_B.min():.6f}, {alpha_B.max():.6f}]")

    # Read or estimate smoothing lengths
    if "h" in h5step:
        h = np.array(h5step["h"])
        print(f"  h: [{h.min():.6f}, {h.max():.6f}], median={np.median(h):.6f}")
    else:
        x, y, z = coords['x'], coords['y'], coords['z']
        vol = (x.max() - x.min()) * (y.max() - y.min()) * (z.max() - z.min())
        h_est = 1.2 * (vol / n_particles) ** (1.0 / 3.0)
        h = np.full(n_particles, h_est)
        print(f"  h not in file, using estimate h={h_est:.6f}")

    # Select particles within their own kernel support of the slice plane
    mask = np.abs(coords[slice_axis] - slice_pos) < 2.0 * h
    print(f"  Particles in slice: {mask.sum()} / {n_particles} ({mask.sum() / n_particles * 100:.2f}%)")

    ha, va = _PLOT_AXES[slice_axis]
    xs   = coords[ha][mask]
    ys   = coords[va][mask]
    zoff = coords[slice_axis][mask] - slice_pos
    hs   = h[mask]

    # SPH kernel scatter onto grid. Density is the bare kernel sum (true rho
    # up to a global factor); alpha_B is intensive, so it is normalized.
    print(f"  Interpolating onto {resolution}x{resolution} grid...")
    xi, yi, di = sph_scatter_to_grid(xs, ys, zoff, hs, rho[mask], resolution)
    if have_alpha_B:
        _, _, ai = sph_scatter_to_grid(xs, ys, zoff, hs, alpha_B[mask], resolution,
                                       normalize=True)

    # Plot
    if have_alpha_B:
        fig, (ax_rho, ax_alpha) = plt.subplots(1, 2, figsize=(15, 7))
    else:
        fig, ax_rho = plt.subplots(figsize=(8, 7))

    ax_rho.set_aspect('equal', adjustable='box')
    im = ax_rho.pcolormesh(xi, yi, di, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax_rho, shrink=0.8)
    cbar.set_label("rho [arb. units]")
    ax_rho.set_xlabel(ha)
    ax_rho.set_ylabel(va)
    ax_rho.set_title(f"{title}, t=[{time_val}]  ({slice_axis}={slice_pos:+.4f})")

    if have_alpha_B:
        ax_alpha.set_aspect('equal', adjustable='box')
        im_a = ax_alpha.pcolormesh(xi, yi, ai, cmap='viridis', shading='auto')
        cbar_a = fig.colorbar(im_a, ax=ax_alpha, shrink=0.8)
        cbar_a.set_label("alpha_B")
        ax_alpha.set_xlabel(ha)
        ax_alpha.set_ylabel(va)
        ax_alpha.set_title(f"alpha_B, t=[{time_val}]  ({slice_axis}={slice_pos:+.4f})")

    fig.text(0.78, 0.02, f"Resolution: {n_cbrt}^3", fontsize=10)
    plt.tight_layout()

    f.close()
    return fig, time_val


def plot_density_slice(fname, step, resolution=256, slice_axis='z', slice_pos=0.0,
                       title="Density", vmin=None, vmax=None, cmap='bone_r'):
    """Plot a single 2D slice and save as PNG."""
    fig, _ = render_density_slice(fname, step, resolution, slice_axis=slice_axis, slice_pos=slice_pos,
                                  title=title, vmin=vmin, vmax=vmax, cmap=cmap)
    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"slice_rho_step{step}_{slice_axis}{slice_pos:+.4f}.png")
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outname}")


def _render_frame(args):
    """Worker function for parallel GIF generation. Returns (step, png_bytes)."""
    fname, step, resolution, slice_axis, slice_pos, title, vmin, vmax, cmap = args
    fig, _ = render_density_slice(fname, step, resolution, slice_axis=slice_axis, slice_pos=slice_pos,
                                  title=title, vmin=vmin, vmax=vmax, cmap=cmap)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    buf.close()
    return step, png_bytes


def make_gif(fname, start, end, resolution=256, n_workers=None, slice_axis='z', slice_pos=0.0,
             title="Density", vmin=None, vmax=None, cmap='bone_r'):
    """Create an animated GIF from a range of steps.

    Limits to at most 100 frames by striding over the range.
    Uses multiprocessing to render frames in parallel.
    """
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

    work = [(fname, step, resolution, slice_axis, slice_pos, title, vmin, vmax, cmap) for step in steps]
    results = {}
    with Pool(n_workers) as pool:
        for i, (step, png_bytes) in enumerate(pool.imap_unordered(_render_frame, work)):
            results[step] = png_bytes
            print(f"  [{i + 1}/{n_total}] Step {step} done", flush=True)

    # Assemble in order
    print("Assembling GIF...")
    frames = []
    for step in steps:
        buf = io.BytesIO(results[step])
        frames.append(Image.open(buf).copy())
        buf.close()

    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"slice_rho_steps{start}-{end}_{slice_axis}{slice_pos:+.4f}.gif")
    frames[0].save(
        outname,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0,
    )
    print(f"Saved GIF ({len(frames)} frames): {outname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot density slices from SPH HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s data.h5                       PNG of the final step (default)\n"
            "  %(prog)s data.h5 -i                    Print metadata\n"
            "  %(prog)s data.h5 5                     PNG, z=0 slice at step 5\n"
            "  %(prog)s data.h5 --all                 PNG for every step in the file\n"
            "  %(prog)s data.h5 5 --axis x --pos 0.5  PNG, x=0.5 slice at step 5\n"
            "  %(prog)s data.h5 0-20 --axis y         GIF, y=0 slices for steps 0-20\n"
        ),
    )
    parser.add_argument("file", help="HDF5 input file")
    parser.add_argument("step", nargs="?",
                        help="Step number or range (e.g. 0-20). Omit to plot the final step.")
    parser.add_argument("-i", "--info", action="store_true",
                        help="Print HDF5 metadata and exit")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Plot every step in the file as an individual PNG")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z",
                        help="Axis normal to the slice plane (default: z)")
    parser.add_argument("--pos", type=float, default=0.0,
                        help="Position along the slice axis (default: 0.0)")
    parser.add_argument("-r", "--resolution", type=int, default=256,
                        help="Interpolation grid resolution per side (default: 256)")
    parser.add_argument("--title", default="Density",
                        help="Plot title prefix before the time/slice info (default: 'Density')")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Lower density limit for the colormap (default: auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Upper density limit for the colormap (default: auto)")
    parser.add_argument("--cmap", default="bone_r",
                        help="Matplotlib colormap name (default: bone_r)")

    args = parser.parse_args()

    if args.info:
        print_metadata(args.file)
        sys.exit(0)

    common = dict(resolution=args.resolution, slice_axis=args.axis, slice_pos=args.pos,
                  title=args.title, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)

    if args.all:
        nsteps = get_nsteps(args.file)
        if nsteps == 0:
            print(f"No steps found in {args.file}")
            sys.exit(1)
        print(f"Plotting all {nsteps} steps...")
        for step in range(nsteps):
            plot_density_slice(args.file, step, **common)
    elif args.step is None:
        nsteps = get_nsteps(args.file)
        if nsteps == 0:
            print(f"No steps found in {args.file}")
            sys.exit(1)
        plot_density_slice(args.file, nsteps - 1, **common)
    elif "-" in args.step and not args.step.startswith("-"):
        start, end = args.step.split("-", 1)
        make_gif(args.file, int(start), int(end), **common)
    else:
        plot_density_slice(args.file, int(args.step), **common)
