#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys


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


def cubic_spline_2d(q):
    """Cubic spline SPH kernel in 2D, normalized. q = r/h."""
    sigma = 10.0 / (7.0 * np.pi)
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    w[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * w


def sph_scatter_to_grid(xs, ys, ds, hs, resolution=256):
    """Scatter SPH particles onto a 2D grid using kernel-weighted interpolation.

    For each particle, deposits its field value weighted by the SPH kernel
    onto nearby grid cells within the kernel support radius (2h).
    """
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution

    # Grid cell centers
    xc = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, resolution)
    yc = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, resolution)

    weight_grid = np.zeros((resolution, resolution))
    value_grid = np.zeros((resolution, resolution))

    for i in range(len(xs)):
        hi = hs[i]
        support = 2.0 * hi

        # Grid index range affected by this particle
        ix_lo = max(int((xs[i] - support - xmin) / dx), 0)
        ix_hi = min(int((xs[i] + support - xmin) / dx) + 1, resolution)
        iy_lo = max(int((ys[i] - support - ymin) / dy), 0)
        iy_hi = min(int((ys[i] + support - ymin) / dy) + 1, resolution)

        # Vectorized over the affected patch
        gx = xc[ix_lo:ix_hi]
        gy = yc[iy_lo:iy_hi]
        gxx, gyy = np.meshgrid(gx, gy, indexing='ij')

        r = np.sqrt((gxx - xs[i])**2 + (gyy - ys[i])**2)
        q = r / hi
        w = cubic_spline_2d(q) / hi**2

        value_grid[iy_lo:iy_hi, ix_lo:ix_hi] += (w * ds[i]).T
        weight_grid[iy_lo:iy_hi, ix_lo:ix_hi] += w.T

    # Normalize: where we have contributions, divide by total weight
    valid = weight_grid > 0
    result = np.full((resolution, resolution), np.nan)
    result[valid] = value_grid[valid] / weight_grid[valid]

    xi, yi = np.meshgrid(
        np.linspace(xmin, xmax, resolution + 1),
        np.linspace(ymin, ymax, resolution + 1),
    )
    return xi, yi, result


def plot_density_slice(fname, step):
    """Plot a 2D xy-slice of density using SPH kernel interpolation."""
    print(f"Reading step {step} from {fname}...")
    f, h5step = read_step(fname, step)

    x = np.array(h5step["x"])
    y = np.array(h5step["y"])
    z = np.array(h5step["z"])
    rho = np.array(h5step["rho"])
    
    time = h5step.attrs["time"][0]
    n_particles = len(x)
    n_cbrt = round(n_particles ** (1.0 / 3.0), 1)

    print(f"Step {step}: time={time:.8f}, N={n_particles} (~{n_cbrt}^3)")
    print(f"  x: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  y: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  z: [{z.min():.4f}, {z.max():.4f}]")
    print(f"  rho: [{rho.min():.6f}, {rho.max():.6f}]")

    # Read or estimate smoothing lengths
    if "h" in h5step:
        h = np.array(h5step["h"])
        print(f"  h: [{h.min():.6f}, {h.max():.6f}], median={np.median(h):.6f}")
    else:
        # Estimate h from mean particle spacing: h ~ (V/N)^(1/3)
        vol = (x.max() - x.min()) * (y.max() - y.min()) * (z.max() - z.min())
        h_est = 1.2 * (vol / n_particles) ** (1.0 / 3.0)
        h = np.full(n_particles, h_est)
        print(f"  h not in file, using estimate h={h_est:.6f}")

    # Select thin z-slice around z=0
    z_threshold = 2.0 * np.median(h)
    print(f"  z_threshold = 2*median(h) = {z_threshold:.6f}")

    mask = np.abs(z) < z_threshold
    print(f"  Particles in slice: {mask.sum()} / {n_particles} ({mask.sum() / n_particles * 100:.2f}%)")

    xs, ys, ds, hs = x[mask], y[mask], rho[mask], h[mask]

    # SPH kernel scatter onto grid
    resolution = 512
    print(f"  Interpolating onto {resolution}x{resolution} grid...")
    xi, yi, di = sph_scatter_to_grid(xs, ys, ds, hs, resolution)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')

    im = ax.pcolormesh(xi, yi, di, cmap='bone_r', shading='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("rho")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Magnetic Sedov, Density, t=[{time}]")
    fig.text(0.78, 0.02, f"Resolution: {n_cbrt}^3", fontsize=10)

    outdir = os.path.dirname(os.path.abspath(fname))
    outname = os.path.join(outdir, f"slice_rho_step{step}.png")
    plt.tight_layout()
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    print(f"Saved: {outname}")
    f.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_density_slice.py <hdf5_file> [step | -p]")
        sys.exit(1)

    fname = sys.argv[1]

    if len(sys.argv) < 3 or sys.argv[2] == "-p":
        print_metadata(fname)
        sys.exit(0)

    step = int(sys.argv[2])
    plot_density_slice(fname, step)
