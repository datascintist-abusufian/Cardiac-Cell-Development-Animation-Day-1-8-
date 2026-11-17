"""
Statistics for longitudinal culture phenotyping and in-silico experiments.

day_summary          mean, SD and t-based 95 % CI per day over replicate wells
mixed_model_test     linear mixed model  y ~ C(day) + (1 | well), likelihood-ratio test of day
hedges_g             bias-corrected standardised effect vs a reference day, with CI
turning_point        continuous two-segment (hinge) regression; cluster bootstrap CI of the break
gp_forecast          Gaussian-process regression with predictive 95 % band (honest extrapolation)
state_space          PCA of standardised phenotypes + Gaussian mixture states selected by BIC
kaplan_meier/logrank cell survival from attachment to death; two-group log-rank test
morris               Morris elementary-effects global sensitivity screening
hill_fit             four-parameter Hill (dose-response) fit with IC50 confidence interval
bland_altman         agreement between pipeline measurement and ground truth
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def day_summary(df, col, by="day"):
    g = df.groupby(by)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    t = stats.t.ppf(0.975, np.maximum(out["count"] - 1, 1))
    out["ci_lo"] = out["mean"] - t * out["std"] / np.sqrt(out["count"])
    out["ci_hi"] = out["mean"] + t * out["std"] / np.sqrt(out["count"])
    return out


def mixed_model_test(df, col, group="well"):
    """LRT of the day effect in y ~ C(day) + (1|well). Returns dict(p, lr, icc)."""
    import warnings

    import statsmodels.formula.api as smf
    d = df[[col, "day", group]].dropna().rename(columns={col: "y"})
    if d["y"].std() == 0 or len(d) < 10:
        return dict(p=np.nan, lr=np.nan, icc=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = smf.mixedlm("y ~ C(day)", d, groups=d[group]).fit(reml=False)
        null = smf.mixedlm("y ~ 1", d, groups=d[group]).fit(reml=False)
    lr = 2 * (full.llf - null.llf)
    k = d["day"].nunique() - 1
    var_w = float(full.cov_re.iloc[0, 0]) if full.cov_re.size else 0.0
    icc = var_w / (var_w + full.scale) if (var_w + full.scale) > 0 else np.nan
    return dict(p=float(stats.chi2.sf(max(lr, 0), k)), lr=float(lr), icc=float(icc))


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (y.mean() - x.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = J * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se


def _hinge_fit(t, y, grid):
    best = (np.inf, None, None)
    for tau in grid:
        X = np.column_stack([np.ones_like(t), np.minimum(t, tau), np.maximum(t - tau, 0)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((y - X @ beta) ** 2)
        if sse < best[0]:
            best = (sse, tau, beta)
    return best


def turning_point(df, col, n_boot=400, seed=0, group="well"):
    """Day at which the trajectory switches slope (rise -> decline), with cluster-bootstrap 95 % CI."""
    d = df[["day", col, group]].dropna()
    grid = np.arange(2.0, d["day"].max() - 0.5 + 1e-9, 0.05)
    sse, tau, beta = _hinge_fit(d["day"].values.astype(float), d[col].values, grid)
    rng = np.random.default_rng(seed)
    wells = d[group].unique()
    taus = []
    for _ in range(n_boot):
        pick = rng.choice(wells, len(wells), replace=True)
        b = pd.concat([d[d[group] == w] for w in pick])
        taus.append(_hinge_fit(b["day"].values.astype(float), b[col].values, grid)[1])
    lo, hi = np.percentile(taus, [2.5, 97.5])
    return dict(tau=float(tau), ci=(float(lo), float(hi)), slope_before=float(beta[1]),
                slope_after=float(beta[2]), boot=np.array(taus))


def gp_forecast(days, values, horizon=11, n_grid=200):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    x = np.asarray(days, float)[:, None]
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    mu, sd = y.mean(), y.std() or 1.0
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.5, (0.6, 6.0)) + WhiteKernel(0.1, (1e-4, 1.0))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=3, random_state=0)
    gp.fit(x, (y - mu) / sd)
    xg = np.linspace(1, horizon, n_grid)[:, None]
    m, s = gp.predict(xg, return_std=True)
    return dict(x=xg.ravel(), mean=m * sd + mu, lo=(m - 1.96 * s) * sd + mu, hi=(m + 1.96 * s) * sd + mu,
                kernel=str(gp.kernel_))


def state_space(df, cols, k_max=5, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    d = df.dropna(subset=cols).copy()
    Z = StandardScaler().fit_transform(d[cols])
    pca = PCA(n_components=min(3, len(cols)), random_state=seed).fit(Z)
    S = pca.transform(Z)
    bics, models = [], []
    for k in range(1, k_max + 1):
        gm = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=5, reg_covar=1e-3).fit(S[:, :2])
        bics.append(gm.bic(S[:, :2]))
        models.append(gm)
    kbest = int(np.argmin(bics)) + 1
    gm = models[kbest - 1]
    d["PC1"], d["PC2"] = S[:, 0], S[:, 1]
    lab = gm.predict(S[:, :2])
    # order states by mean day so labels read as a progression
    order = np.argsort([d.loc[lab == s, "day"].mean() for s in range(kbest)])
    remap = {old: new for new, old in enumerate(order)}
    d["state"] = [remap[s] + 1 for s in lab]
    d["state_prob"] = gm.predict_proba(S[:, :2]).max(1)
    loadings = pd.DataFrame(pca.components_[:2].T, index=cols, columns=["PC1", "PC2"])
    return dict(data=d, bic=bics, k=kbest, explained=pca.explained_variance_ratio_, loadings=loadings)


def kaplan_meier(time, event):
    t, e = np.asarray(time, float), np.asarray(event, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    uniq = np.unique(t[e])
    S, s, out_t, out_s, lo, hi, var = 1.0, [], [0.0], [1.0], [1.0], [1.0], 0.0
    for u in uniq:
        n = (t >= u).sum()
        dd = ((t == u) & e).sum()
        if n == 0:
            continue
        S *= 1 - dd / n
        var += dd / (n * (n - dd)) if n > dd else 0
        se = S * np.sqrt(var)
        out_t.append(u)
        out_s.append(S)
        lo.append(max(0, S - 1.96 * se))
        hi.append(min(1, S + 1.96 * se))
    return pd.DataFrame(dict(t=out_t, S=out_s, lo=lo, hi=hi))


def logrank(t1, e1, t2, e2):
    t = np.r_[t1, t2]
    e = np.r_[e1, e2].astype(bool)
    g = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    O1 = E1 = V = 0.0
    for u in np.unique(t[e]):
        at = t >= u
        n, n1 = at.sum(), (at & (g == 0)).sum()
        d = ((t == u) & e).sum()
        d1 = ((t == u) & e & (g == 0)).sum()
        if n < 2:
            continue
        O1 += d1
        E1 += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = (O1 - E1) ** 2 / V if V > 0 else 0.0
    return dict(chi2=float(chi2), p=float(stats.chi2.sf(chi2, 1)))


def morris(fn, bounds, names, r=10, levels=4, seed=0):
    """Morris elementary effects. fn(x: dict) -> dict of outputs. Returns DataFrame(mu*, sigma)."""
    rng = np.random.default_rng(seed)
    k = len(bounds)
    delta = levels / (2 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    effects = {}
    runs = []
    for _ in range(r):
        x = rng.choice(grid[: levels // 2], k) if levels > 2 else rng.choice(grid, k)
        order = rng.permutation(k)
        sign = rng.choice([-1, 1], k)
        x = np.where(sign < 0, x + delta, x)  # start so that the step stays in [0, 1]
        cur = x.copy()
        y0 = fn({n: b[0] + v * (b[1] - b[0]) for n, b, v in zip(names, bounds, cur)})
        runs.append((cur.copy(), y0))
        for i in order:
            nxt = cur.copy()
            nxt[i] = cur[i] + sign[i] * delta
            y1 = fn({n: b[0] + v * (b[1] - b[0]) for n, b, v in zip(names, bounds, nxt)})
            runs.append((nxt.copy(), y1))
            for out in y1:
                effects.setdefault(out, {nm: [] for nm in names})
                effects[out][names[i]].append((y1[out] - y0[out]) / (sign[i] * delta))
            cur, y0 = nxt, y1
    rows = []
    for out, d in effects.items():
        for nm, ee in d.items():
            ee = np.array(ee, float)
            ee = ee[np.isfinite(ee)]
            rows.append(dict(output=out, parameter=nm, mu_star=float(np.mean(np.abs(ee))) if len(ee) else np.nan,
                             mu=float(np.mean(ee)) if len(ee) else np.nan,
                             sigma=float(np.std(ee, ddof=1)) if len(ee) > 1 else np.nan))
    return pd.DataFrame(rows), runs


def hill(c, bottom, top, ic50, h):
    return bottom + (top - bottom) / (1 + (np.maximum(c, 1e-12) / ic50) ** h)


def hill_fit(conc, resp):
    conc, resp = np.asarray(conc, float), np.asarray(resp, float)
    p0 = [resp.min(), resp.max(), np.median(conc[conc > 0]), 1.0]
    popt, pcov = optimize.curve_fit(hill, conc, resp, p0=p0, bounds=([-1, -1, 1e-3, 0.2], [2, 2, 1e3, 6]), maxfev=20000)
    se = np.sqrt(np.diag(pcov))
    # CI of IC50 on the log scale (delta method)
    lic, lse = np.log(popt[2]), se[2] / popt[2]
    return dict(bottom=popt[0], top=popt[1], ic50=popt[2], hill=popt[3],
                ic50_ci=(float(np.exp(lic - 1.96 * lse)), float(np.exp(lic + 1.96 * lse))), popt=popt)


def bland_altman(meas, truth):
    m, t = np.asarray(meas, float), np.asarray(truth, float)
    ok = np.isfinite(m) & np.isfinite(t)
    m, t = m[ok], t[ok]
    diff = m - t
    bias, sd = diff.mean(), diff.std(ddof=1)
    r = np.corrcoef(m, t)[0, 1] if len(m) > 2 else np.nan
    slope, intercept = np.polyfit(t, m, 1) if len(m) > 2 else (np.nan, np.nan)
    ccc = (2 * np.cov(m, t)[0, 1] / (m.var(ddof=1) + t.var(ddof=1) + (m.mean() - t.mean()) ** 2)) if len(m) > 2 else np.nan
    return dict(bias=float(bias), loa=(float(bias - 1.96 * sd), float(bias + 1.96 * sd)), r=float(r),
                ccc=float(ccc), slope=float(slope), intercept=float(intercept), mean=(m + t) / 2, diff=diff, n=len(m))
