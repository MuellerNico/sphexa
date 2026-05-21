"""Shared helpers for SPHEXA HDF5 plotting scripts.

Provides metadata inspection and a derived-field resolver. Plot scripts call
`resolve_field(step_group, name)` to get a per-particle array plus a display
label for either a raw HDF5 dataset or a derived quantity (e.g. `Bmag`,
`log_divBerr`). New derived fields are added via the `@_derive` decorator
below -- no changes needed in the consumer scripts.
"""

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Field resolver
# ---------------------------------------------------------------------------

# Pretty labels for raw datasets that benefit from LaTeX rendering. Anything
# not in this map falls back to the dataset name verbatim.
_RAW_LABELS = {
    'rho':                  'density',
    'p':                    r'$P$',
    'u':                    r'$u$',
    'vx':                   r'$v_x$',
    'vy':                   r'$v_y$',
    'vz':                   r'$v_z$',
    'magneto::Bx':          r'$B_x$',
    'magneto::By':          r'$B_y$',
    'magneto::Bz':          r'$B_z$',
    'magneto::divB':        r'$\nabla\cdot B$',
    'magneto::alpha_B':     r'$\alpha_B$',
    'magneto::gradB_norm':  r'$|\nabla B|$',
}

# Derived-field registry: {name: (formula, label, required_raw_fields)}
_DERIVED = {}


def _derive(name, label, deps):
    def deco(fn):
        _DERIVED[name] = (fn, label, set(deps))
        return fn
    return deco


def _arr(s, k):
    return np.asarray(s[k])


@_derive('Bmag', r'$|B|$',
         ['magneto::Bx', 'magneto::By', 'magneto::Bz'])
def _Bmag(s):
    return np.sqrt(_arr(s, 'magneto::Bx')**2 +
                   _arr(s, 'magneto::By')**2 +
                   _arr(s, 'magneto::Bz')**2)


@_derive('divBerr', r'$h\,|\nabla\cdot B|/|B|$',
         ['magneto::Bx', 'magneto::By', 'magneto::Bz', 'magneto::divB', 'h'])
def _divBerr(s):
    with np.errstate(divide='ignore', invalid='ignore'):
        return _arr(s, 'h') * np.abs(_arr(s, 'magneto::divB')) / _Bmag(s)


@_derive('log_divBerr', r'$\log_{10}(h\,|\nabla\cdot B|/|B|)$',
         ['magneto::Bx', 'magneto::By', 'magneto::Bz', 'magneto::divB', 'h'])
def _log_divBerr(s):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.log10(_divBerr(s))
    v[~np.isfinite(v)] = np.nan
    return v


@_derive('Emag', 'magnetic energy density',
         ['magneto::Bx', 'magneto::By', 'magneto::Bz'])
def _Emag(s):
    return 0.5 * (_arr(s, 'magneto::Bx')**2 +
                  _arr(s, 'magneto::By')**2 +
                  _arr(s, 'magneto::Bz')**2)


@_derive('Pmag', r'$P_\mathrm{mag}$',
         ['magneto::Bx', 'magneto::By', 'magneto::Bz'])
def _Pmag(s):
    # magnetic pressure B^2/(2*mu_0); equals Emag for the default mu_0 = 1.
    # mu_0 is a step attribute, not a dataset -- fall back to 1.0 if absent.
    mu0 = float(np.atleast_1d(s.attrs.get('mu_0', 1.0))[0])
    return _Emag(s) / mu0


@_derive('KE', 'kinetic energy density',
         ['rho', 'vx', 'vy', 'vz'])
def _KE(s):
    return 0.5 * _arr(s, 'rho') * (_arr(s, 'vx')**2 +
                                   _arr(s, 'vy')**2 +
                                   _arr(s, 'vz')**2)


def resolve_field(s, name):
    """Return (values, label) for either a raw h5 dataset or a derived field.

    `s` is an open `h5py.Group` (a `Step#i` group).
    """
    if name in s:
        return _arr(s, name), _RAW_LABELS.get(name, name)
    if name in _DERIVED:
        fn, label, _ = _DERIVED[name]
        return fn(s), label
    raw = sorted([k for k in s.keys() if hasattr(s[k], 'shape')])
    raise KeyError(
        f"unknown field '{name}'\n"
        f"  raw datasets: {raw}\n"
        f"  derived: {sorted(_DERIVED)}"
    )


def available_fields(s):
    """Return (raw_dataset_names, derived_names_that_resolve_against_s)."""
    raw_keys = set(s.keys())
    raw = sorted(raw_keys)
    derived = [n for n, (_, _, deps) in _DERIVED.items() if deps.issubset(raw_keys)]
    return raw, derived


def derived_label(name):
    """Display label for a derived field name."""
    return _DERIVED[name][1]


# ---------------------------------------------------------------------------
# Step / metadata helpers
# ---------------------------------------------------------------------------

def get_nsteps(fname):
    with h5py.File(fname, "r") as f:
        return len([k for k in f.keys() if k.startswith("Step#")])


def print_metadata(fname):
    """Print step summary, raw datasets in Step#0, and derived fields that
    can be resolved from those datasets."""
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

            _, derived = available_fields(step0)
            print(f"\nDerived fields resolvable from these datasets:")
            for name in derived:
                print(f"  {name:>20s}  ({_DERIVED[name][1]})")
        print()
