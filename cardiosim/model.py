"""
Mechanistic, stochastic model of an hiPSC-cardiomyocyte culture followed for eight days.

Two latent processes drive every phenotype:

    maturation  M_w(t) = 1 / (1 + exp(-k_m (t - t_m - u_w)))        u_w ~ N(0, s_w^2)
    damage      D_w(t) = 1 / (1 + exp(-k_d (t - t_d - v_w)))        v_w ~ N(0, s_w^2)

with well-level random effects (u_w, v_w) so replicate wells differ as real cultures do.
Cells attach with an exponential delay, die with a hazard that rises with damage,
h(t) = h_0 + h_1 D(t), and each carries its own maturation offset. Ground-truth
phenotypes (sarcomere length, alignment, coupling, calcium kinetics ...) are smooth
functions of M and D; the functional readouts are produced by the electrophysiology
model in ep.py and the image readouts by rendering (render.py) followed by the
analysis pipeline (analysis.py), so "measured" values carry realistic error.

Parameter values are illustrative and chosen to fall inside ranges reported for
hiPSC-CMs (see the Methods page); they are not fitted to a specific data set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CultureParams:
    n_wells: int = 6
    n_days: int = 8
    cells_seeded: int = 55            # cells in the imaged field of view
    attach_mean_days: float = 1.2     # mean attachment/spreading delay
    # maturation
    mat_midpoint: float = 3.4         # day of half-maximal maturation
    mat_rate: float = 1.4             # 1/day
    # damage / culture stress
    dmg_onset: float = 6.7            # day of half-maximal damage
    dmg_rate: float = 2.4             # 1/day
    # survival
    base_hazard: float = 0.015        # 1/day
    dmg_hazard: float = 0.55          # extra 1/day at full damage
    # coupling (relative gap-junction conductance)
    g_max: float = 1.0
    # pacemaking
    pacemaker_fraction: float = 0.10
    rate_bpm: float = 46.0            # mean intrinsic rate of pacemaker cells
    rate_cv_immature: float = 0.30    # cell-to-cell CV of intrinsic rate when immature
    # variability
    well_sd_days: float = 0.35        # SD of well random effects (days)
    seed: int = 11

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ latent states
def logistic(t, mid, rate):
    return 1.0 / (1.0 + np.exp(-rate * (np.asarray(t, float) - mid)))


def well_effects(p: CultureParams):
    rng = np.random.default_rng(p.seed)
    return rng.normal(0, p.well_sd_days, (p.n_wells, 2))


def latent(t, p: CultureParams, u=0.0, v=0.0):
    M = logistic(t, p.mat_midpoint + u, p.mat_rate)
    D = logistic(t, p.dmg_onset + v, p.dmg_rate)
    return M, D


# ------------------------------------------------------------------ ground-truth phenotypes
def phenotypes(M, D, p: CultureParams):
    """Smooth ground-truth phenotype functions of maturation M and damage D (arrays OK)."""
    M, D = np.asarray(M, float), np.asarray(D, float)
    healthy = 1 - D
    return dict(
        coupling=p.g_max * (0.08 + 0.92 * M) * (1 - 0.85 * D),         # relative Cx43 conductance
        sarcomere_length_um=1.62 + 0.30 * M - 0.22 * D,                  # immature ~1.6, mature ~1.9 um
        sarcomere_order=np.clip(0.12 + 0.80 * M * (1 - 0.85 * D), 0, 1),  # striation regularity
        aspect_ratio=1.25 + 2.6 * M * (1 - 0.55 * D),
        cell_area_um2=400 + 650 * M - 180 * D,
        alignment_kappa=0.15 + 4.5 * M * healthy,                        # von Mises concentration
        nuclear_area_um2=125 - 18 * M - 40 * D,                          # condensation -> smaller, brighter
        ca_amplitude=0.35 + 0.65 * M * (1 - 0.7 * D),                    # dF/F0 scale
        ca_tau_decay_ms=420 - 190 * M + 260 * D,                         # SERCA maturation, then failure
        rate_cv=p.rate_cv_immature * (1 - 0.65 * M) + 0.25 * D,          # heterogeneity of intrinsic rate
        excitable_fraction=np.clip(0.55 + 0.45 * M - 0.5 * D, 0.05, 1),  # electrically active fraction
    )


# ------------------------------------------------------------------ cell agents
@dataclass
class Well:
    index: int
    u: float
    v: float
    cells: pd.DataFrame = field(repr=False)


def make_cells(p: CultureParams, well: int, u: float, v: float, fov_um: float = 276.0):
    """Seed cell agents for one well: position, attachment and death times, per-cell offsets."""
    rng = np.random.default_rng(p.seed * 1000 + well)
    n = p.cells_seeded
    # Poisson-disc-like placement by rejection
    pts = []
    tries = 0
    min_d = fov_um / np.sqrt(n) * 0.55
    while len(pts) < n and tries < 20000:
        q = rng.uniform(8, fov_um - 8, 2)
        if all(np.hypot(*(q - r)) > min_d for r in pts):
            pts.append(q)
        tries += 1
    pts = np.array(pts)
    n = len(pts)
    attach = rng.exponential(p.attach_mean_days, n)
    # death times by inverse-transform sampling of the cumulative hazard on a fine grid
    tg = np.linspace(0, 14, 1401)
    _, Dg = latent(tg, p, u, v)
    H = np.cumsum((p.base_hazard + p.dmg_hazard * Dg) * (tg[1] - tg[0]))
    e = rng.exponential(1.0, n) * rng.lognormal(0, 0.35, n)   # frailty
    death = np.interp(e, H, tg, right=np.inf)
    cells = pd.DataFrame(dict(
        cell=np.arange(n), x_um=pts[:, 0], y_um=pts[:, 1], attach_day=attach, death_day=death,
        mat_offset=rng.normal(0, 0.45, n),                 # cell-intrinsic maturation offset (days)
        size_factor=rng.lognormal(0, 0.18, n),
        pacemaker=rng.random(n) < p.pacemaker_fraction,
        rate_z=rng.normal(0, 1, n),
        theta_noise=rng.random(n),
        phase=rng.random(n),
    ))
    return cells


def make_study(p: CultureParams):
    eff = well_effects(p)
    return [Well(w, eff[w, 0], eff[w, 1], make_cells(p, w, eff[w, 0], eff[w, 1])) for w in range(p.n_wells)]


def cell_states(well: Well, day: float, p: CultureParams, axis_deg: float = 25.0):
    """Per-cell state on a given day (only attached, living cells), incl. geometry in um."""
    c = well.cells
    alive = (c.attach_day <= day) & (c.death_day > day)
    dying = alive & (c.death_day - day < 0.9)   # apoptotic phase before detachment
    cs = c[alive].copy()
    M, D = latent(day - cs.mat_offset.values, p, well.u, well.v)
    D = np.clip(D + 0.6 * dying[alive].values, 0, 1)
    ph = phenotypes(M, D, p)
    rng = np.random.default_rng(int(p.seed * 7919 + well.index * 101 + day * 13))
    area = ph["cell_area_um2"] * cs.size_factor.values
    ar = np.maximum(1.05, ph["aspect_ratio"] * rng.lognormal(0, 0.12, len(cs)))
    major = np.sqrt(4 * area * ar / np.pi)     # full axis lengths (um)
    minor = major / ar
    kappa = float(np.mean(ph["alignment_kappa"]))
    theta = np.deg2rad(axis_deg) + vonmises_from_uniform(cs.theta_noise.values, kappa)
    cs = cs.assign(M=M, D=D, dying=dying[alive].values, area_um2=area, aspect_ratio=ar, major_um=major,
                   minor_um=minor, theta=theta, sarcomere_length_um=ph["sarcomere_length_um"],
                   sarcomere_order=ph["sarcomere_order"], nuclear_area_um2=ph["nuclear_area_um2"])
    return cs


def vonmises_from_uniform(u, kappa):
    """Map fixed uniforms to von Mises(0, kappa) angles on the axial circle (period pi)."""
    grid = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    pdf = np.exp(kappa * np.cos(2 * grid))
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(u, cdf, grid)


def ground_truth_table(p: CultureParams, wells=None, days=None):
    """Per well-day latent states and ground-truth phenotypes (what a perfect assay would read)."""
    wells = wells or make_study(p)
    days = days or list(range(1, p.n_days + 1))
    rows = []
    for w in wells:
        for d in days:
            cs = cell_states(w, d, p)
            M, D = latent(d, p, w.u, w.v)
            ph = phenotypes(M, D, p)
            rows.append(dict(well=w.index, day=d, M=float(M), D=float(D), n_cells=len(cs),
                             n_dying=int(cs.dying.sum()),
                             cum_deaths=int(((w.cells.death_day <= d) & (w.cells.attach_day <= d)).sum()),
                             **{f"true_{k}": float(v) for k, v in ph.items()},
                             true_orientation_order=orientation_order(cs.theta.values) if len(cs) else np.nan,
                             true_mean_aspect_ratio=float(cs.aspect_ratio.mean()) if len(cs) else np.nan,
                             true_mean_area_um2=float(cs.area_um2.mean()) if len(cs) else np.nan))
    return pd.DataFrame(rows)


def orientation_order(theta):
    """2D nematic order parameter S = |<exp(2i theta)>| (0 = isotropic, 1 = aligned)."""
    theta = np.asarray(theta)
    if len(theta) == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(2j * theta))))
