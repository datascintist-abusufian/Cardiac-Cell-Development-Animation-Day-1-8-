"""
Electrophysiology of the monolayer: a 2-D lattice of FitzHugh–Nagumo (FHN) elements,

    dv/dt = v - v^3/3 - w + I_i + sum_j g_ij (v_j - v_i)
    dw/dt = eps (v + a - b w)

Each lattice node belongs to the nearest living cell (Voronoi tessellation of the cell
agents from model.py) or is empty (no cell / dead cell, non-conducting). Pacemaker cells
hiPSC-CMs are spontaneously active, so every excitable node is an oscillator (I = 0.5,
inside the oscillatory regime). Its intrinsic rate is set through the recovery rate
eps_i = eps_0 * f_i (the FHN period scales ~ 1/eps), with f_i drawn around 1 with a
maturation-dependent spread; designated pacemaker cells are ~25 % faster. Gap-junction
conductance g_ij scales with the day's coupling, so maturation synchronises the
oscillators (Kuramoto-like entrainment) and damage fragments them again.
Model time is calibrated so that the reference oscillator beats at `rate_bpm`. The model
is phenomenological: trends in synchrony, rate dispersion and conduction are meaningful,
absolute conduction velocities are in model units.

Calcium and tension per node:
    dCa/dt = r (1 - Ca)/tau_on - Ca / tau_decay(cell)     r: release gate, set to 1 at each
                                                          upstroke and inactivating with tau_r
    F/F0   = 1 + A(cell) Ca                 (fluo-4-like signal)
    T      = SO(cell) * Ca^h / (Ca^h + K^h)  (active tension, scaled by sarcomere order)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .model import CultureParams, phenotypes

A_FHN, B_FHN, EPS0, I_OSC = 0.7, 0.8, 0.025, 0.5


@lru_cache(maxsize=2)
def reference_period(eps=EPS0, I=I_OSC, dt=0.05):
    """Period (model time units) of a single uncoupled FHN oscillator (cached)."""
    v, w = -1.2, -0.6
    ups = []
    for k in range(int(1500 / dt)):
        dv = v - v ** 3 / 3 - w + I
        vn = v + dt * dv
        if v < 0 <= vn:
            ups.append(k * dt)
        w, v = w + dt * eps * (v + A_FHN - B_FHN * w), vn
    return float(np.median(np.diff(np.array(ups)[2:])))


def build_tissue(cells, p: CultureParams, coupling: float, N=32, fov_um=276.0, seed=0, rate_cv=0.2):
    """Map living cells onto an N x N lattice and assign per-node parameters."""
    rng = np.random.default_rng(seed)
    node_um = fov_um / N
    yy, xx = (np.mgrid[0:N, 0:N] + 0.5) * node_um
    cell_id = -np.ones((N, N), int)
    if len(cells):
        cx, cy = cells.x_um.values, cells.y_um.values
        d2 = (xx[..., None] - cx) ** 2 + (yy[..., None] - cy) ** 2
        nearest = d2.argmin(-1)
        reach = (np.sqrt(cells.area_um2.values / np.pi) * 1.45)[nearest]
        cell_id = np.where(np.sqrt(d2.min(-1)) <= reach, cells.cell.values[nearest], -1)
    # per-cell lookups
    lut = {c.cell: c for c in cells.itertuples()}
    excit = cell_id >= 0
    I = np.zeros((N, N))
    eps = np.full((N, N), EPS0)
    tau = np.full((N, N), 400.0)
    amp = np.zeros((N, N))
    so = np.zeros((N, N))
    pace = np.zeros((N, N), bool)
    ref_period = reference_period()
    for cid in np.unique(cell_id[excit]):
        c = lut[cid]
        ph = phenotypes(c.M, c.D, p)
        m = cell_id == cid
        tau[m] = ph["ca_tau_decay_ms"]
        amp[m] = ph["ca_amplitude"]
        so[m] = ph["sarcomere_order"]
        # a fraction of damaged cells is inexcitable
        if c.dying or rng.random() > ph["excitable_fraction"]:
            excit[m] = False
            continue
        factor = max(0.35, 1 + rate_cv * c.rate_z) * (1.25 if c.pacemaker else 1.0)
        I[m] = I_OSC
        eps[m] = EPS0 * factor                # intrinsic rate ~ factor x reference
        pace[m] = bool(c.pacemaker)
    # gap-junction conductance on edges (lognormal heterogeneity, higher within a cell)
    hetero = 0.6 * (1 - min(coupling, 1)) + 0.15
    gx = coupling * 1.6 * rng.lognormal(0, hetero, (N, N - 1))
    gy = coupling * 1.6 * rng.lognormal(0, hetero, (N - 1, N))
    same_x = (cell_id[:, 1:] == cell_id[:, :-1]) & (cell_id[:, 1:] >= 0)
    same_y = (cell_id[1:, :] == cell_id[:-1, :]) & (cell_id[1:, :] >= 0)
    gx = np.where(same_x, 3.0, gx)
    gy = np.where(same_y, 3.0, gy)
    gx *= excit[:, 1:] & excit[:, :-1]
    gy *= excit[1:, :] & excit[:-1, :]
    return dict(N=N, node_um=node_um, cell_id=cell_id, excitable=excit, pacemaker=pace, I=I, eps=eps,
                tau=tau, amp=amp, so=so, gx=gx, gy=gy, ref_period_tu=ref_period)


def simulate(tissue, rate_bpm=46.0, duration_s=7.0, dt=0.05, frame_ms=10.0, seed=0):
    """Integrate the lattice. Returns global traces (every step) and field frames."""
    rng = np.random.default_rng(seed)
    N = tissue["N"]
    ms_per_tu = (60000.0 / rate_bpm) / tissue["ref_period_tu"]
    n_steps = int(duration_s * 1000 / ms_per_tu / dt)
    ex = tissue["excitable"]
    v = np.where(ex, -1.2 + 0.4 * rng.random((N, N)), -1.2)
    w = np.where(ex, -0.62 + 1.6 * rng.random((N, N)), -0.62)   # random initial phases
    eps = tissue["eps"]
    ca = np.zeros((N, N))
    rel = np.zeros((N, N))
    I, gx, gy = tissue["I"], tissue["gx"], tissue["gy"]
    tau_tu = tissue["tau"] / ms_per_tu
    ton_tu = 18.0 / ms_per_tu
    decay_r = np.exp(-dt / (70.0 / ms_per_tu))
    frame_every = max(1, int(round(frame_ms / ms_per_tu / dt)))
    t_ms, v_mean, ca_mean, f_frames, t_frames = [], [], [], [], []
    v_frames, ca_frames = [], []
    exf = ex.astype(float)
    nex = max(exf.sum(), 1)
    for k in range(n_steps):
        lap = np.zeros((N, N))
        fx = gx * (v[:, 1:] - v[:, :-1])
        lap[:, :-1] += fx
        lap[:, 1:] -= fx
        fy = gy * (v[1:, :] - v[:-1, :])
        lap[:-1, :] += fy
        lap[1:, :] -= fy
        dv = v - v ** 3 / 3 - w + I + lap
        dw = eps * (v + A_FHN - B_FHN * w)
        vnew = np.where(ex, v + dt * dv, -1.2)
        rel = np.where((v < 0.5) & (vnew >= 0.5), 1.0, rel * decay_r)
        v = vnew
        w = np.where(ex, w + dt * dw, -0.62)
        ca = ca + dt * (rel * (1 - ca) / ton_tu - ca / tau_tu)
        ca = np.where(ex, ca, 0.0)
        if k % 4 == 0:
            t_ms.append(k * dt * ms_per_tu)
            v_mean.append(float((v * exf).sum() / nex))
            ca_mean.append(float((ca * tissue["amp"] * exf).sum() / nex))
        if k % frame_every == 0:
            t_frames.append(k * dt * ms_per_tu)
            v_frames.append(v.astype(np.float32))
            ca_frames.append(ca.astype(np.float32))
    ca_frames = np.array(ca_frames)
    tension = (tissue["so"] * ca_frames ** 3 / (ca_frames ** 3 + 0.35 ** 3)).astype(np.float32)
    return dict(t_ms=np.array(t_ms), v_mean=np.array(v_mean), ca_dff=np.array(ca_mean),
                t_frames=np.array(t_frames), v=np.array(v_frames), ca=ca_frames, tension=tension,
                ms_per_tu=ms_per_tu)


# ------------------------------------------------------------------ functional readouts
def activation_times(v_frames, t_frames, excit, thr=0.0):
    """Upstroke times (ms) per node, as a list of arrays (row-major)."""
    above = v_frames > thr
    up = (~above[:-1]) & above[1:]
    N = v_frames.shape[1]
    out = []
    for i in range(N):
        for j in range(N):
            if not excit[i, j]:
                out.append(np.array([]))
                continue
            k = np.nonzero(up[:, i, j])[0]
            # linear interpolation of the crossing time
            v0, v1 = v_frames[k, i, j], v_frames[k + 1, i, j]
            frac = (thr - v0) / np.where(v1 - v0 == 0, 1, v1 - v0)
            out.append(t_frames[k] + frac * (t_frames[k + 1] - t_frames[k]))
    return out


def kuramoto(acts, t_grid, min_beats=2):
    """Kuramoto order parameter R(t) from per-node activation times (phase interpolation)."""
    phases = []
    for a in acts:
        if len(a) < min_beats:
            continue
        ph = np.interp(t_grid, a, 2 * np.pi * np.arange(len(a)), left=np.nan, right=np.nan)
        phases.append(ph)
    if not phases:
        return np.full_like(t_grid, np.nan), 0
    P = np.array(phases)
    valid = ~np.isnan(P)
    Z = np.where(valid, np.exp(1j * np.nan_to_num(P)), 0).sum(0)
    cnt = valid.sum(0)
    R = np.where(cnt > 0, np.abs(Z) / np.maximum(cnt, 1), np.nan)
    R[valid.mean(0) < 0.5] = np.nan
    return R, len(phases)


def activation_map(acts, N, t_ref, window_ms=None):
    """Activation time (ms) of each node for the first beat at/after t_ref; NaN if none in window."""
    m = np.full(N * N, np.nan)
    for k, a in enumerate(acts):
        a = a[a >= t_ref]
        if len(a) and (window_ms is None or a[0] - t_ref < window_ms):
            m[k] = a[0] - t_ref
    return m.reshape(N, N)


def conduction_velocity(amap, node_um):
    """Median local conduction velocity (um/ms, model units) from the activation-map gradient."""
    gy, gx = np.gradient(amap)
    g = np.hypot(gx, gy)               # ms per node
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = node_um / g
    cv = cv[np.isfinite(cv) & (g > 1e-3)]
    cv = cv[(cv > 0.01) & (cv < 50)]
    return (float(np.median(cv)) if len(cv) > 10 else np.nan), cv


def single_cell_calcium(tissue, sim, n=40, seed=0):
    """Per-cell (node) calcium kinetics: median over n randomly sampled excitable nodes."""
    from .analysis import calcium_metrics
    rng = np.random.default_rng(seed)
    ii, jj = np.nonzero(tissue["excitable"])
    if len(ii) == 0:
        return {}, None
    pick = rng.choice(len(ii), min(n, len(ii)), replace=False)
    rows = []
    for k in pick:
        i, j = ii[k], jj[k]
        trace = 1 + tissue["amp"][i, j] * sim["ca"][:, i, j]
        m, _ = calcium_metrics(sim["t_frames"], trace)
        if m.get("n_transients", 0) >= 2:
            rows.append(m)
    import pandas as pd
    df = pd.DataFrame(rows)
    if not len(df):
        return {}, df
    return {k: float(df[k].median()) for k in df.columns}, df


def functional_summary(tissue, sim):
    """Scalar functional readouts for one simulated well-day."""
    acts = activation_times(sim["v"], sim["t_frames"], tissue["excitable"])
    N = tissue["N"]
    t_end = sim["t_frames"][-1]
    t_grid = np.linspace(t_end * 0.35, t_end, 300)
    R, n_osc = kuramoto(acts, t_grid)
    n_ex = int(tissue["excitable"].sum())
    rates = [60000 / np.median(np.diff(a)) for a in acts if len(a) >= 3]
    # activation map of the first wave after mid-recording (window = 0.8 x beat period)
    allacts = np.concatenate([a for a in acts if len(a)]) if any(len(a) for a in acts) else np.array([])
    later = np.sort(allacts[allacts > t_end * 0.5])
    period = 60000 / np.median(rates) if rates else 1000
    amap = activation_map(acts, N, later[0], 0.8 * period) if len(later) else np.full((N, N), np.nan)
    cv, _ = conduction_velocity(amap, tissue["node_um"])
    active = np.array([len(a) >= 2 for a in acts]).reshape(N, N)
    return dict(
        excitable_nodes=n_ex,
        active_fraction=float(active.sum() / max(n_ex, 1)),
        kuramoto_R=float(np.nanmean(R)) if np.isfinite(R).any() else 0.0,
        beat_rate_bpm=float(np.median(rates)) if rates else 0.0,
        rate_dispersion=float(np.std(rates) / np.mean(rates)) if len(rates) > 3 else np.nan,
        conduction_velocity_um_ms=cv,
        activation_spread_ms=float(np.nanmax(amap) - np.nanmin(amap)) if np.isfinite(amap).any() else np.nan,
        activation_map=amap, R_t=R, R_tgrid=t_grid, acts=acts,
    )
