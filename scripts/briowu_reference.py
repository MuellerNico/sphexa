#!/usr/bin/env python3
"""Reference solution for the Brio & Wu (1988) MHD shocktube.

This module computes a converged 1D grid solution: second-order MUSCL (minmod) + SSP-RK2 
finite volumes with the HLLD approximate Riemann solver of Miyoshi & Kusano (2005), 
on a grid fine enough that the profile is converged at plot scale.

Initial states (code units, mu0 = 1), discontinuity at x = 0:
    left:  rho=1,     P=1,   v=0, B=(0.75,  1, 0)
    right: rho=0.125, P=0.1, v=0, B=(0.75, -1, 0)

Use solve(t) from other scripts (plot_shocktube.py overlays it), or run as a
CLI to write a whitespace-separated table:  x rho vx vy vz p Bx By Bz u
"""

import argparse
import numpy as np

GAMMA = 2.0
BX    = 0.75
# (rho, vx, vy, vz, p, By, Bz)
STATE_L = (1.0,   0.0, 0.0, 0.0, 1.0, 1.0,  0.0)
STATE_R = (0.125, 0.0, 0.0, 0.0, 0.1, -1.0, 0.0)

_SMALL = 1e-12


def _cons(rho, vx, vy, vz, p, By, Bz, gamma):
    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2) \
        + 0.5 * (BX**2 + By**2 + Bz**2)
    return np.array([rho, rho * vx, rho * vy, rho * vz, By, Bz, E])


def _prim(U, gamma):
    rho = np.maximum(U[0], _SMALL)
    vx, vy, vz = U[1] / rho, U[2] / rho, U[3] / rho
    By, Bz = U[4], U[5]
    p = (gamma - 1.0) * (U[6] - 0.5 * rho * (vx**2 + vy**2 + vz**2)
                         - 0.5 * (BX**2 + By**2 + Bz**2))
    return rho, vx, vy, vz, np.maximum(p, _SMALL), By, Bz


def _flux(rho, vx, vy, vz, p, By, Bz, gamma):
    pT = p + 0.5 * (BX**2 + By**2 + Bz**2)
    E  = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2) \
        + 0.5 * (BX**2 + By**2 + Bz**2)
    vB = vx * BX + vy * By + vz * Bz
    return np.array([rho * vx,
                     rho * vx**2 + pT - BX**2,
                     rho * vx * vy - BX * By,
                     rho * vx * vz - BX * Bz,
                     By * vx - BX * vy,
                     Bz * vx - BX * vz,
                     (E + pT) * vx - BX * vB])


def _cfast(rho, p, By, Bz, gamma):
    a2 = gamma * p / rho
    b2 = (BX**2 + By**2 + Bz**2) / rho
    bx2 = BX**2 / rho
    s = a2 + b2
    return np.sqrt(0.5 * (s + np.sqrt(np.maximum(s * s - 4.0 * a2 * bx2, 0.0))))


def _hlld(WL, WR, gamma):
    """HLLD interface flux (Miyoshi & Kusano 2005) for constant Bx."""
    rhoL, vxL, vyL, vzL, pL, ByL, BzL = WL
    rhoR, vxR, vyR, vzR, pR, ByR, BzR = WR

    UL = _cons(*WL, gamma)
    UR = _cons(*WR, gamma)
    FL = _flux(*WL, gamma)
    FR = _flux(*WR, gamma)

    cmax = np.maximum(_cfast(rhoL, pL, ByL, BzL, gamma),
                      _cfast(rhoR, pR, ByR, BzR, gamma))
    SL = np.minimum(vxL, vxR) - cmax
    SR = np.maximum(vxL, vxR) + cmax

    pTL = pL + 0.5 * (BX**2 + ByL**2 + BzL**2)
    pTR = pR + 0.5 * (BX**2 + ByR**2 + BzR**2)

    dSL = SL - vxL
    dSR = SR - vxR
    SM = (dSR * rhoR * vxR - dSL * rhoL * vxL - pTR + pTL) \
        / (dSR * rhoR - dSL * rhoL)
    pTs = (dSR * rhoR * pTL - dSL * rhoL * pTR
           + rhoL * rhoR * dSR * dSL * (vxR - vxL)) \
        / (dSR * rhoR - dSL * rhoL)

    def star(rho, vx, vy, vz, p, By, Bz, S, dS, pT, U):
        rhos = rho * dS / (S - SM)
        den  = rho * dS * (S - SM) - BX**2
        deg  = np.abs(den) < _SMALL * rho * dS * dS + _SMALL
        den  = np.where(deg, 1.0, den)
        vys  = np.where(deg, vy, vy - BX * By * (SM - vx) / den)
        vzs  = np.where(deg, vz, vz - BX * Bz * (SM - vx) / den)
        Bys  = np.where(deg, By, By * (rho * dS * dS - BX**2) / den)
        Bzs  = np.where(deg, Bz, Bz * (rho * dS * dS - BX**2) / den)
        vB   = vx * BX + vy * By + vz * Bz
        vBs  = SM * BX + vys * Bys + vzs * Bzs
        Es   = (dS * U[6] - pT * vx + pTs * SM + BX * (vB - vBs)) / (S - SM)
        return np.array([rhos, rhos * SM, rhos * vys, rhos * vzs, Bys, Bzs, Es]), \
            vys, vzs, Bys, Bzs, vBs

    UsL, vysL, vzsL, BysL, BzsL, vBsL = star(*WL, SL, dSL, pTL, UL)
    UsR, vysR, vzsR, BysR, BzsR, vBsR = star(*WR, SR, dSR, pTR, UR)

    sqL = np.sqrt(UsL[0])
    sqR = np.sqrt(UsR[0])
    SsL = SM - np.abs(BX) / sqL
    SsR = SM + np.abs(BX) / sqR

    sgn = np.sign(BX) if BX != 0.0 else 1.0
    inv = 1.0 / (sqL + sqR)
    vyss = (sqL * vysL + sqR * vysR + (BysR - BysL) * sgn) * inv
    vzss = (sqL * vzsL + sqR * vzsR + (BzsR - BzsL) * sgn) * inv
    Byss = (sqL * BysR + sqR * BysL + sqL * sqR * (vysR - vysL) * sgn) * inv
    Bzss = (sqL * BzsR + sqR * BzsL + sqL * sqR * (vzsR - vzsL) * sgn) * inv
    vBss = SM * BX + vyss * Byss + vzss * Bzss

    def dstar(Us, vBs, sq, sign):
        Ess = Us[6] + sign * sq * (vBs - vBss) * sgn
        return np.array([Us[0], Us[0] * SM, Us[0] * vyss, Us[0] * vzss,
                         Byss, Bzss, Ess])

    UssL = dstar(UsL, vBsL, sqL, -1.0)
    UssR = dstar(UsR, vBsR, sqR, +1.0)

    FsL  = FL + SL[None] * (UsL - UL)
    FssL = FsL + SsL[None] * (UssL - UsL)
    FsR  = FR + SR[None] * (UsR - UR)
    FssR = FsR + SsR[None] * (UssR - UsR)

    F = np.where(SL > 0, FL,
        np.where(SsL >= 0, FsL,
        np.where(SM >= 0, FssL,
        np.where(SsR >= 0, FssR,
        np.where(SR > 0, FsR, FR)))))
    return F


def _minmod(a, b):
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def _rhs(U, dx, gamma):
    W = np.array(_prim(U, gamma))
    Wg = np.pad(W, ((0, 0), (2, 2)), mode='edge')
    slope = _minmod(Wg[:, 1:-1] - Wg[:, :-2], Wg[:, 2:] - Wg[:, 1:-1])
    Wm = Wg[:, 1:-1] - 0.5 * slope   # left face of each cell
    Wp = Wg[:, 1:-1] + 0.5 * slope   # right face of each cell
    Wm[0] = np.maximum(Wm[0], _SMALL)
    Wp[0] = np.maximum(Wp[0], _SMALL)
    Wm[4] = np.maximum(Wm[4], _SMALL)
    Wp[4] = np.maximum(Wp[4], _SMALL)
    # n+1 interfaces between the n+2 cells (n physical + 1 ghost each side)
    F = _hlld(Wp[:, :-1], Wm[:, 1:], gamma)
    return -(F[:, 1:] - F[:, :-1]) / dx


def solve(t, n=8192, xmin=-2.0, xmax=2.0, gamma=GAMMA, cfl=0.4):
    """Evolve the Brio-Wu problem to time t; outflow boundaries.

    Returns a dict of cell-centered arrays: x, rho, vx, vy, vz, p, Bx, By,
    Bz, u. Valid while no wave has reached the domain edge (t <~ 1 for the
    default domain).
    """
    dx = (xmax - xmin) / n
    x = xmin + (np.arange(n) + 0.5) * dx
    W0 = np.where(x < 0.0, np.array(STATE_L)[:, None], np.array(STATE_R)[:, None])
    U = _cons(*W0, gamma)

    tcur = 0.0
    while tcur < t:
        rho, vx, vy, vz, p, By, Bz = _prim(U, gamma)
        smax = np.max(np.abs(vx) + _cfast(rho, p, By, Bz, gamma))
        dt = min(cfl * dx / smax, t - tcur)
        U1 = U + dt * _rhs(U, dx, gamma)
        U = 0.5 * (U + U1 + dt * _rhs(U1, dx, gamma))
        tcur += dt

    rho, vx, vy, vz, p, By, Bz = _prim(U, gamma)
    return {'x': x, 'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz, 'p': p,
            'Bx': np.full(n, BX), 'By': By, 'Bz': Bz,
            'u': p / ((gamma - 1.0) * rho)}


_cache = {}


def get(t, gamma=GAMMA):
    """Memoized solve() at overlay resolution (4096 cells is converged at plot scale)."""
    key = (round(float(t), 9), float(gamma))
    if key not in _cache:
        _cache[key] = solve(t, n=4096, gamma=gamma)
    return _cache[key]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-t", "--time", type=float, default=0.2)
    ap.add_argument("-n", "--ncells", type=int, default=8192)
    ap.add_argument("-o", "--output", default=None,
                    help="write table to this path (default: stdout summary only)")
    ap.add_argument("--plot", default=None, metavar="PNG",
                    help="save a quick-look figure of the solution")
    args = ap.parse_args()

    sol = solve(args.time, n=args.ncells)
    cols = ('x', 'rho', 'vx', 'vy', 'vz', 'p', 'Bx', 'By', 'Bz', 'u')
    print(f"Brio-Wu reference at t={args.time} on {args.ncells} cells")
    for c in cols[1:]:
        print(f"  {c:>3s}: min={sol[c].min():+.4f} max={sol[c].max():+.4f}")

    if args.output:
        np.savetxt(args.output, np.column_stack([sol[c] for c in cols]),
                   header=" ".join(cols), fmt="%.8e")
        print(f"Saved: {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
        for ax, c in zip(axes.flat, ('vx', 'vy', 'Bx', 'By', 'rho', 'u', 'p', 'Bz')):
            ax.plot(sol['x'], sol[c], '-', color='tab:blue', lw=1.2)
            ax.set_ylabel(c)
            ax.set_xlim(-1, 1)
        for ax in axes[1]:
            ax.set_xlabel('x')
        fig.suptitle(f"Brio-Wu HLLD reference, t={args.time}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"Saved: {args.plot}")
