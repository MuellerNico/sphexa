#!/usr/bin/env python3

import argparse
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from _h5_common import resolution_label


def format_runtime(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_size(num_bytes: int) -> str:
    n = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def format_attr(val):
    if isinstance(val, bytes):
        return val.decode(errors="replace")
    arr = np.asarray(val)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.flatten()[0])
    if arr.size <= 8:
        return str(arr.tolist())
    return f"<{arr.dtype} shape={tuple(arr.shape)}>"


def parse_start(info_log: Path):
    for line in info_log.read_text().splitlines():
        if line.startswith("start:"):
            return datetime.fromisoformat(line.split(":", 1)[1].strip())
    return None


def collect_dump(h5file: Path):
    info = {"rows": [], "n_particles": None, "fields": [], "attrs": {},
            "box": None}
    with h5py.File(h5file, "r") as f:
        step_keys = sorted(
            (k for k in f.keys() if k.startswith("Step#")),
            key=lambda k: int(k.split("#")[1]),
        )
        for key in step_keys:
            step = f[key]
            iteration = step.attrs.get("iteration", [None])[0]
            time = step.attrs.get("time", [None])[0]
            n = None
            for ds_key in step.keys():
                ds = step[ds_key]
                if hasattr(ds, "shape") and len(ds.shape) > 0:
                    n = ds.shape[0]
                    break
            info["rows"].append((int(key.split("#")[1]), iteration, time, n))
            if info["n_particles"] is None:
                info["n_particles"] = n

        if step_keys:
            last = f[step_keys[-1]]
            info["fields"] = sorted(last.keys())

            s0 = f[step_keys[0]]
            info["attrs"] = {k: s0.attrs[k] for k in sorted(s0.attrs.keys())}
            if all(ax in s0 for ax in ("x", "y", "z")):
                info["box"] = {
                    ax: (float(np.min(s0[ax][...])), float(np.max(s0[ax][...])))
                    for ax in ("x", "y", "z")
                }
    return info


def detect_git():
    repo = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--long", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=5, cwd=repo,
        )
        if out.returncode == 0 and out.stdout.strip():
            desc = out.stdout.strip()
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=repo,
            )
            b = branch.stdout.strip()
            return f"{desc} ({b})" if branch.returncode == 0 and b else desc
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "n/a"


def detect_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            names = [n.strip() for n in out.stdout.strip().splitlines() if n.strip()]
            return ", ".join(names) if names else "n/a"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Append post-run metadata from an SPH HDF5 dump to info.log."
    )
    parser.add_argument("info_log", help="Path to info.log (will be appended to)")
    parser.add_argument("dump", help="Path to dump.h5")
    args = parser.parse_args()

    info_log = Path(args.info_log)
    dump = Path(args.dump)

    if not info_log.exists():
        print(f"Error: {info_log} not found", file=sys.stderr)
        sys.exit(1)
    if not dump.exists():
        print(f"Error: {dump} not found", file=sys.stderr)
        sys.exit(1)

    end_dt = datetime.now(timezone.utc).astimezone()
    start_dt = parse_start(info_log)
    runtime_str = "unknown"
    if start_dt is not None:
        runtime_str = format_runtime((end_dt - start_dt).total_seconds())

    info = collect_dump(dump)
    rows = info["rows"]
    n_particles = info["n_particles"]
    final_time = rows[-1][2] if rows else None

    extents = None
    if info["box"] is not None:
        extents = [hi - lo for lo, hi in (info["box"][ax] for ax in ("x", "y", "z"))]

    dump_size = dump.stat().st_size
    profile = dump.with_name("profile.h5")
    profile_size = profile.stat().st_size if profile.exists() else None

    skip_attrs = {"iteration", "time"}
    extra_attrs = {k: v for k, v in info["attrs"].items() if k not in skip_attrs}

    with info_log.open("a") as f:
        f.write(f"end: {end_dt.isoformat(timespec='seconds')}\n")
        f.write(f"runtime: {runtime_str}\n")
        f.write(f"hostname: {socket.gethostname()}\n")
        f.write(f"git: {detect_git()}\n")
        f.write(f"gpu: {detect_gpu()}\n")
        if n_particles is not None:
            f.write(f"particles: {n_particles} ({resolution_label(extents, n_particles)})\n")
        if final_time is not None:
            f.write(f"final time: {final_time:.8f}\n")
        f.write(f"writes: {len(rows)}\n")
        f.write(f"dump size: {format_size(dump_size)}\n")
        if profile_size is not None:
            f.write(f"profile size: {format_size(profile_size)}\n")
        if info["fields"]:
            f.write(f"fields: {', '.join(info['fields'])}\n")

        if info["box"] is not None:
            f.write("box (Step#0):\n")
            for ax, (lo, hi) in info["box"].items():
                f.write(f"  {ax}: [{lo:.6f}, {hi:.6f}]  (extent {hi - lo:.6f})\n")

        if extra_attrs:
            f.write("attrs (Step#0):\n")
            for k, v in extra_attrs.items():
                f.write(f"  {k} = {format_attr(v)}\n")

        f.write(f"\n{'Step':>8s} {'Iteration':>12s} {'Time':>15s}\n")
        f.write("-" * 38 + "\n")
        for step, iteration, time, _ in rows:
            f.write(f"{step:>8d} {iteration:>12} {time:>15.8f}\n")

    print(f"Appended run metadata to {info_log}")


if __name__ == "__main__":
    main()
