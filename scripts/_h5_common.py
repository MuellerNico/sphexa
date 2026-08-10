# Shared helpers for the SPHEXA HDF5 plotting scripts: metadata inspection and
# a derived-field resolver. Scripts call resolve_field(step_group, name) to get
# a per-particle array + display label for a raw dataset or a derived quantity
# (Bmag, log_divBerr, ...). Add new derived fields with the @_derive decorator
# below; the consumer scripts need no changes.

import h5py
import numpy as np

# Pretty labels for datasets
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
    'magneto::curlB_x':     r'$(\nabla\times B)_x$',
    'magneto::curlB_y':     r'$(\nabla\times B)_y$',
    'magneto::curlB_z':     r'$(\nabla\times B)_z$',
    'magneto::dB_diss_x':   r'$\dot B_{\mathrm{diss},x}$',
    'magneto::dB_diss_y':   r'$\dot B_{\mathrm{diss},y}$',
    'magneto::dB_diss_z':   r'$\dot B_{\mathrm{diss},z}$',
    'magneto::du_diss':     r'$\dot u_\mathrm{diss}$',
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


@_derive('curlBmag', r'$|\nabla\times B|$',
         ['magneto::curlB_x', 'magneto::curlB_y', 'magneto::curlB_z'])
def _curlBmag(s):
    return np.sqrt(_arr(s, 'magneto::curlB_x')**2 +
                   _arr(s, 'magneto::curlB_y')**2 +
                   _arr(s, 'magneto::curlB_z')**2)


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


@_derive('vmag', r'$|v|$',
         ['vx', 'vy', 'vz'])
def _vmag(s):
    return np.sqrt(_arr(s, 'vx')**2 +
                   _arr(s, 'vy')**2 +
                   _arr(s, 'vz')**2)


@_derive('KE', 'kinetic energy density',
         ['rho', 'vx', 'vy', 'vz'])
def _KE(s):
    return 0.5 * _arr(s, 'rho') * (_arr(s, 'vx')**2 +
                                   _arr(s, 'vy')**2 +
                                   _arr(s, 'vz')**2)


@_derive('dB_diss', r'$|\dot B_\mathrm{diss}|$',
         ['magneto::dB_diss_x', 'magneto::dB_diss_y', 'magneto::dB_diss_z'])
def _dB_diss(s):
    # magnitude of the K-applied artificial-resistivity contribution to dB/dt
    return np.sqrt(_arr(s, 'magneto::dB_diss_x')**2 +
                   _arr(s, 'magneto::dB_diss_y')**2 +
                   _arr(s, 'magneto::dB_diss_z')**2)


@_derive('dB_diss_rel', r'$|\dot B_\mathrm{diss}|/|B|$',
         ['magneto::dB_diss_x', 'magneto::dB_diss_y', 'magneto::dB_diss_z',
          'magneto::Bx', 'magneto::By', 'magneto::Bz'])
def _dB_diss_rel(s):
    # fractional resistive decay rate [1/time]; blows up at field nulls (guarded to nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = _dB_diss(s) / _Bmag(s)
    v[~np.isfinite(v)] = np.nan
    return v


# --- MHD-loop (Gardiner-Stone) frame ---
# The exact solution is B = A0 phi_hat inside r < R0 and 0 outside, translated at
# constant velocity, so B_phi carries all the signal and B_perp is pure error.

def _minimum_image(delta, length):
    return delta - length * np.round(delta / length)


# B^2-weighted centroid of the loop in the xy-plane. Circular mean, so a loop
# straddling the periodic edge is not pulled toward the box centre.
def loop_centre(s):
    box = np.atleast_1d(s.attrs['box']).astype(float)
    w = _Emag(s) * _arr(s, 'm') / _arr(s, 'rho')
    centre = []
    for q, lo, hi in ((_arr(s, 'x'), box[0], box[1]), (_arr(s, 'y'), box[2], box[3])):
        length = hi - lo
        th = 2.0 * np.pi * (q - lo) / length
        ang = np.arctan2(np.sum(w * np.sin(th)), np.sum(w * np.cos(th))) % (2.0 * np.pi)
        centre.append(lo + ang * length / (2.0 * np.pi))
    return centre[0], centre[1]


# (r, phi_hat_x, phi_hat_y) per particle about the loop axis.
def loop_frame(s):
    box = np.atleast_1d(s.attrs['box']).astype(float)
    cx, cy = loop_centre(s)
    dx = _minimum_image(_arr(s, 'x') - cx, box[1] - box[0])
    dy = _minimum_image(_arr(s, 'y') - cy, box[3] - box[2])
    r = np.hypot(dx, dy)
    rsafe = np.maximum(r, 1e-30)
    return r, -dy / rsafe, dx / rsafe


_LOOP_DEPS = ['x', 'y', 'm', 'rho', 'magneto::Bx', 'magneto::By', 'magneto::Bz']


@_derive('Bphi', r'$B_\phi$', _LOOP_DEPS)
def _Bphi(s):
    _, ex, ey = loop_frame(s)
    return ex * _arr(s, 'magneto::Bx') + ey * _arr(s, 'magneto::By')


@_derive('Bperp', r'$|B_\perp|$', _LOOP_DEPS)
def _Bperp(s):
    # everything not along phi_hat: identically zero in the exact solution
    return np.sqrt(np.maximum(2.0 * _Emag(s) - _Bphi(s)**2, 0.0))


@_derive('Bperp_rel', r'$|B_\perp|/|B|$', _LOOP_DEPS)
def _Bperp_rel(s):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = _Bperp(s) / _Bmag(s)
    v[~np.isfinite(v)] = np.nan
    return v


@_derive('mod', r'$\mathrm{clamp}(h|\nabla B|/|B|)$',
         ['h', 'magneto::gradB_norm', 'magneto::Bx', 'magneto::By', 'magneto::Bz'])
def _mod(s):
    # Tricco-Price steepness modulator that gates the SLRB/SLRB2 schemes; guarded, clamped to [0,1]
    with np.errstate(divide='ignore', invalid='ignore'):
        v = _arr(s, 'h') * _arr(s, 'magneto::gradB_norm') / _Bmag(s)
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0), 0.0, 1.0)


# (values, label) for a raw dataset or a derived field; s is an open Step#i group.
def resolve_field(s, name):
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


# (raw dataset names, derived names whose deps are all present in s).
def available_fields(s):
    raw_keys = set(s.keys())
    raw = sorted(raw_keys)
    derived = [n for n, (_, _, deps) in _DERIVED.items() if deps.issubset(raw_keys)]
    return raw, derived


# Display label for a derived field name.
def derived_label(name):
    return _DERIVED[name][1]


# --- metadata helpers ---

def get_nsteps(fname):
    with h5py.File(fname, "r") as f:
        return len([k for k in f.keys() if k.startswith("Step#")])


# Print the step summary, the raw datasets in Step#0, and which derived fields
# resolve from them.
def print_metadata(fname):
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


# --- domain resolution ---

# Per-axis particle count for uniform glass tiling
# (simply particle count cbrt split by box aspect ratio)
# smears over any internal density jump
def effective_resolution(extents, n_particles):
    if extents is None or not n_particles:
        return None
    lx, ly, lz = (float(e) for e in extents)
    if min(lx, ly, lz) <= 0:
        return None
    spacing = (lx * ly * lz / n_particles) ** (1.0 / 3.0)
    return lx / spacing, ly / spacing, lz / spacing


# Compact resolution string for annotations
def resolution_label(extents, n_particles):
    n_cbrt = round(n_particles ** (1.0 / 3.0), 1) if n_particles else None
    res = effective_resolution(extents, n_particles)
    if res is None:
        return f"~{n_cbrt}^3"
    nx, ny, nz = res
    if abs(nx - ny) < 0.5 and abs(ny - nz) < 0.5:
        return f"~{n_cbrt}^3"
    return f"~{nx:.0f}x{ny:.0f}x{nz:.0f}, ~{n_cbrt}^3 total"


# Font used for --clean publish figures; change once here. The later entries
# are fallbacks for hosts without the first (Nimbus Roman is the
# metric-compatible Times clone shipped on most Linux systems).
CLEAN_FONT = "Times New Roman"


def apply_clean_style():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = [CLEAN_FONT, 'Nimbus Roman',
                                         'Liberation Serif', 'STIXGeneral']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'


# --- run colors ---
SCHEME_ORDER = ('a05', 'SLR', 'SLRB', 'SLRB2')

# seaborn's default "deep" palette (mby remove red/4th color)
CATEGORICAL_COLORS = ('#DD8452', '#4C72B0', '#55A868', '#C44E52', '#8172B3',
                      '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD')


def scheme_color(k):
    return CATEGORICAL_COLORS[k % len(CATEGORICAL_COLORS)]


# Colors for n overlaid runs, positional: file k is assumed to be SCHEME_ORDER[k].
def scheme_colors(n):
    return tuple(scheme_color(k) for k in range(n))


# Default legend label for overlaid run k; None past the known schemes, so
# callers fall back to their own naming.
def scheme_label(k):
    return SCHEME_ORDER[k] if k < len(SCHEME_ORDER) else None


# --- tick formatting ---

# Offset scientific notation on the tick labels (0.0002, 0.0004, ... -> 2, 4, ...
# with a shared x10^-4), matching the colorbar formatting in plot_slice.
def apply_sci_ticks(ax, axis='y'):
    import matplotlib.ticker
    names = ('x', 'y') if axis == 'both' else (axis,)
    for name in names:
        a = getattr(ax, f'{name}axis')
        if a.get_scale() != 'linear':
            continue
        fmt = a.get_major_formatter()
        if not isinstance(fmt, matplotlib.ticker.ScalarFormatter):
            continue
        fmt.set_powerlimits((-3, 3))
        fmt.set_useMathText(True)


def _log_tick_label(v, _pos=None):
    if v <= 0:
        return ''
    if 1e-3 <= v < 1e4:
        return f'{v:g}'
    e = int(np.floor(np.log10(v)))
    m = v / 10.0**e
    return rf'$10^{{{e}}}$' if abs(m - 1.0) < 1e-9 else rf'${m:g}\times10^{{{e}}}$'


# Plain decimal labels on log axes: matplotlib renders a narrow log range as
# 9.3x10^-1, 9.4x10^-1, ... which reads badly. Minor ticks get labeled below two
# decades, where the decade majors alone are too sparse (often a single label).
def apply_log_ticks(ax, axis='y'):
    import matplotlib.ticker as mticker
    names = ('x', 'y') if axis == 'both' else (axis,)
    for name in names:
        a = getattr(ax, f'{name}axis')
        if a.get_scale() != 'log':
            continue
        lo, hi = sorted(getattr(ax, f'get_{name}lim')())
        if lo <= 0 or hi <= 0:
            continue
        fmt = mticker.FuncFormatter(_log_tick_label)
        a.set_major_formatter(fmt)
        decades = np.log10(hi / lo)
        if decades >= 2.0:
            continue
        if decades >= 1.0:
            a.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))
        a.set_minor_formatter(fmt)
