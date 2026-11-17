"""Longitudinal phenotyping: trajectories, mixed models, effect sizes, turning points, forecasts, states, survival."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sst

from cardiosim import ui
from cardiosim.stats import (day_summary, gp_forecast, hedges_g, kaplan_meier, mixed_model_test, state_space,
                             turning_point)

ui.setup("Longitudinal phenotyping · Cardiomyocyte Lab", "📈")
ui.hero("📈 Longitudinal phenotyping", "Twenty-two pipeline-measured phenotypes in 6 replicate wells over 8 days — "
        "with the statistics a reviewer would expect.")
study, cells, *_ = ui.load_data()
feats = [f for f in ui.FEATURES if f in study.columns]

tabs = st.tabs(["Trajectories & mixed models", "Effect sizes", "Turning points", "Forecasting", "Cell-state space",
                "Survival", "Phenotype correlations"])

# --------------------------------------------------------------------------- trajectories
with tabs[0]:
    groups = sorted({ui.FEATURES[f][2] for f in feats})
    gsel = st.multiselect("Phenotype groups", groups, default=groups)
    shown = [f for f in feats if ui.FEATURES[f][2] in gsel]
    cols = st.columns(3)
    for i, f in enumerate(shown):
        s = day_summary(study, f)
        fig = go.Figure()
        for w, g in study.groupby("well"):
            fig.add_scatter(x=g.day, y=g[f], mode="lines", line=dict(color=ui.PALETTE[7], width=1), opacity=0.35, showlegend=False)
        fig.add_scatter(x=s.day, y=s["mean"], mode="lines+markers", line=dict(color=ui.PALETTE[0], width=3), showlegend=False,
                        error_y=dict(type="data", symmetric=False, array=s.ci_hi - s["mean"], arrayminus=s["mean"] - s.ci_lo))
        fig.update_layout(xaxis_title="day")
        with cols[i % 3]:
            ui.show(ui.style(fig, 240, legend=False, title=ui.flabel(f)))

    @st.cache_data(show_spinner="Fitting linear mixed models…")
    def lrt_table(feats_):
        rows = []
        for f in feats_:
            r = mixed_model_test(study, f)
            rows.append(dict(Phenotype=ui.flabel(f), Group=ui.FEATURES[f][2], LR=r["lr"], p=r["p"], ICC_well=r["icc"]))
        df = pd.DataFrame(rows)
        p = df.p.fillna(1).values
        order = np.argsort(p)
        q = np.empty_like(p)
        q[order] = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
        df["q (BH-FDR)"] = np.minimum(q, 1)
        return df
    st.markdown("##### Day effect: linear mixed model `y ~ C(day) + (1 | well)`, likelihood-ratio test (7 df)")
    t = lrt_table(tuple(feats))
    st.dataframe(t.style.format({"LR": "{:.1f}", "p": "{:.2e}", "ICC_well": "{:.2f}", "q (BH-FDR)": "{:.2e}"}),
                 hide_index=True, use_container_width=True)
    ui.finding(f"{(t['q (BH-FDR)'] < 0.05).sum()} of {len(t)} phenotypes change significantly over culture time after "
               "Benjamini–Hochberg correction. The intraclass correlation (ICC) shows how much variance is due to well-to-well "
               "differences — the reason wells, not images, are the experimental unit.")

# --------------------------------------------------------------------------- effect sizes
with tabs[1]:
    ref = st.selectbox("Reference day", list(range(1, 9)), index=0)
    G = pd.DataFrame(index=[ui.flabel(f) for f in feats], columns=range(1, 9), dtype=float)
    for f in feats:
        x = study.loc[study.day == ref, f]
        for d in range(1, 9):
            G.loc[ui.flabel(f), d] = hedges_g(x, study.loc[study.day == d, f])[0] if d != ref else 0.0
    lim = np.nanpercentile(np.abs(G.values.astype(float)), 95)
    fig = px.imshow(G.astype(float).clip(-lim, lim), color_continuous_scale="RdBu_r", zmin=-lim, zmax=lim, aspect="auto",
                    labels=dict(x="day", color="Hedges g"), text_auto=".1f")
    ui.show(ui.style(fig, 640, legend=False, title=f"Standardised change vs day {ref} (Hedges g, bias-corrected; n = 6 wells per day)"))
    st.caption("|g| ≈ 0.2 small, 0.5 medium, 0.8 large. With only 6 wells, g is imprecise: the 95 % CI half-width at g = 1 is ≈ 1.2.")

# --------------------------------------------------------------------------- turning points
with tabs[2]:
    st.markdown("A continuous two-segment (hinge) regression locates the day at which each trajectory changes slope — the "
                "end of maturation / onset of degeneration. Confidence intervals come from a cluster bootstrap over wells.")
    cand = ["kuramoto_R", "img_striated_fraction", "img_sarcomere_score", "img_orientation_order", "ca_amplitude_dff",
            "mot_amplitude_um", "mot_sync_index", "img_n_cells", "ca_tau_decay_ms", "rate_dispersion"]
    cand = [c for c in cand if c in study.columns]

    @st.cache_data(show_spinner="Bootstrapping turning points…")
    def tps(c_):
        return {f: turning_point(study, f, n_boot=300) for f in c_}
    T = tps(tuple(cand))
    df = pd.DataFrame([dict(Phenotype=ui.flabel(f), day=T[f]["tau"], lo=T[f]["ci"][0], hi=T[f]["ci"][1],
                            before=T[f]["slope_before"], after=T[f]["slope_after"]) for f in cand]).sort_values("day")
    fig = go.Figure()
    fig.add_scatter(x=df.day, y=df.Phenotype, mode="markers", marker=dict(size=11, color=ui.PALETTE[1]),
                    error_x=dict(type="data", symmetric=False, array=df.hi - df.day, arrayminus=df.day - df.lo), showlegend=False)
    fig.update_layout(xaxis_title="turning-point day (95 % CI)", xaxis_range=[1.5, 8])
    ui.show(ui.style(fig, 420, legend=False))
    f = st.selectbox("Inspect", cand, format_func=ui.flabel)
    d = study[["day", f, "well"]].dropna()
    tau = T[f]["tau"]
    X = np.column_stack([np.ones(len(d)), np.minimum(d.day, tau), np.maximum(d.day - tau, 0)])
    beta = np.linalg.lstsq(X, d[f], rcond=None)[0]
    xs = np.linspace(1, 8, 100)
    fig = go.Figure()
    fig.add_scatter(x=d.day + np.random.default_rng(0).uniform(-.12, .12, len(d)), y=d[f], mode="markers",
                    marker=dict(color=ui.PALETTE[0], opacity=0.6), name="well-days")
    fig.add_scatter(x=xs, y=beta[0] + beta[1] * np.minimum(xs, tau) + beta[2] * np.maximum(xs - tau, 0), name="hinge fit",
                    line=dict(color=ui.PALETTE[1], width=3))
    fig.add_vrect(x0=T[f]["ci"][0], x1=T[f]["ci"][1], fillcolor=ui.PALETTE[1], opacity=0.12, line_width=0)
    fig.update_layout(xaxis_title="day", yaxis_title=ui.flabel(f))
    ui.show(ui.style(fig, 340))
    ui.finding(f"Functional and structural maturation peak at different times: e.g. synchrony turns at day "
               f"{T['kuramoto_R']['tau']:.1f}, striated-cell fraction at day {T['img_striated_fraction']['tau']:.1f}. "
               "Ordering these breakpoints gives a testable sequence of degeneration events.")

# --------------------------------------------------------------------------- forecasting
with tabs[3]:
    st.markdown("Gaussian-process regression (RBF + white-noise kernel, hyper-parameters by marginal likelihood) fitted to all "
                "48 well-days. The predictive band widens beyond day 8 — the honest answer to “what happens next?”.")
    f = st.selectbox("Phenotype to forecast", feats, index=feats.index("kuramoto_R"), format_func=ui.flabel, key="gp")
    g = gp_forecast(study.day, study[f], horizon=11)
    fig = go.Figure()
    fig.add_scatter(x=np.r_[g["x"], g["x"][::-1]], y=np.r_[g["hi"], g["lo"][::-1]], fill="toself", fillcolor="rgba(42,120,214,0.15)",
                    line=dict(width=0), name="95 % predictive band")
    fig.add_scatter(x=g["x"], y=g["mean"], line=dict(color=ui.PALETTE[0], width=3), name="GP mean")
    fig.add_scatter(x=study.day, y=study[f], mode="markers", marker=dict(color=ui.PALETTE[7], size=6), name="observed well-days")
    fig.add_vrect(x0=8, x1=11, fillcolor="#eeeeee", opacity=0.5, line_width=0, annotation_text="extrapolation", annotation_position="top left")
    fig.update_layout(xaxis_title="day", yaxis_title=ui.flabel(f))
    ui.show(ui.style(fig, 400))
    st.caption(f"Fitted kernel: {g['kernel']}")
    ui.finding("Earlier versions of this app forecast days 9–13 with a random forest on eight hand-entered points; tree models cannot "
               "extrapolate, so that forecast was a flat copy of day 8 with an arbitrary ±10 % band. The GP makes its uncertainty explicit.", "warn")

# --------------------------------------------------------------------------- state space
with tabs[4]:
    sfeats = ["kuramoto_R", "rate_dispersion", "img_striated_fraction", "img_orientation_order", "img_mean_aspect_ratio",
              "img_n_fragments", "ca_amplitude_dff", "ca_tau_decay_ms", "mot_amplitude_um", "img_n_cells"]
    sfeats = [s for s in sfeats if s in study.columns]
    ss = state_space(study, sfeats)
    d = ss["data"]
    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig = go.Figure()
        for w, gw in d.sort_values("day").groupby("well"):
            fig.add_scatter(x=gw.PC1, y=gw.PC2, mode="lines", line=dict(color="#bbbbbb", width=1), showlegend=False)
        for s_ in sorted(d.state.unique()):
            m = d.state == s_
            fig.add_scatter(x=d.PC1[m], y=d.PC2[m], mode="markers+text", text=d.day[m].astype(str), textposition="top center",
                            marker=dict(size=11, color=ui.PALETTE[(s_ - 1) % 8]), name=f"state {s_}")
        L = ss["loadings"] * 3
        for f_, r in L.iterrows():
            fig.add_annotation(x=r.PC1, y=r.PC2, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowcolor="#aaa", text="")
        fig.add_scatter(x=L.PC1 * 1.12, y=L.PC2 * 1.12, mode="text", text=[ui.FEATURES[f_][0] for f_ in L.index],
                        textfont=dict(size=10, color="#777"), showlegend=False, hoverinfo="skip")
        fig.update_layout(xaxis_title=f"PC1 ({ss['explained'][0]:.0%})", yaxis_title=f"PC2 ({ss['explained'][1]:.0%})")
        ui.show(ui.style(fig, 520, title="Phenotype state space (numbers = day; grey lines = each well's path)"))
    with c2:
        fig = go.Figure(go.Scatter(x=list(range(1, len(ss["bic"]) + 1)), y=ss["bic"], mode="lines+markers", line=dict(color=ui.PALETTE[4])))
        fig.update_layout(xaxis_title="number of states k", yaxis_title="BIC")
        ui.show(ui.style(fig, 240, legend=False, title=f"Gaussian-mixture model selection → k = {ss['k']}"))
        tab = pd.crosstab(d.state, d.day)
        fig = px.imshow(tab, text_auto=True, color_continuous_scale="Blues", labels=dict(x="day", y="state"))
        ui.show(ui.style(fig, 260, legend=False, title="Well-days per state"))
    ui.finding("The culture follows a curved path through phenotype space: immature → mature/synchronous → degenerating. "
               "States are discovered without labels (GMM, BIC-selected) and ordered by mean day.")

# --------------------------------------------------------------------------- survival
with tabs[5]:
    st.markdown("Cell survival from attachment to death (Kaplan–Meier, Greenwood 95 % CI); cells alive at day 8 are censored.")
    c = cells[np.isfinite(cells.attach_day) & (cells.attach_day < 8) & (cells.death_day > cells.attach_day)].copy()
    c["time"] = np.minimum(c.death_day, 8) - c.attach_day
    c["event"] = c.death_day <= 8
    fig = go.Figure()
    km = kaplan_meier(c.time, c.event)
    fig.add_scatter(x=np.r_[km.t, km.t[::-1]], y=np.r_[km.hi, km.lo[::-1]], fill="toself", fillcolor="rgba(214,58,122,0.12)",
                    line=dict(width=0), name="pooled 95 % CI", line_shape="hv")
    fig.add_scatter(x=km.t, y=km.S, line=dict(color=ui.PALETTE[5], width=3, shape="hv"), name="pooled")
    for w, g in c.groupby("well"):
        k_ = kaplan_meier(g.time, g.event)
        fig.add_scatter(x=k_.t, y=k_.S, line=dict(width=1, shape="hv"), opacity=0.6, name=f"well {w + 1}")
    fig.update_layout(xaxis_title="days since attachment", yaxis_title="survival probability", yaxis_range=[0, 1.02])
    ui.show(ui.style(fig, 420))
    med = km.t[km.S <= 0.5].min() if (km.S <= 0.5).any() else np.nan
    ui.finding(f"{c.event.mean():.0%} of attached cells die by day 8; median survival time "
               f"{'not reached' if np.isnan(med) else f'{med:.1f} days'}. Deaths cluster after day 6, when the damage process "
               "raises the hazard — the same time at which synchrony and sarcomere order collapse.")

# --------------------------------------------------------------------------- correlations
with tabs[6]:
    R = study[feats].corr(method="spearman")
    R.index = R.columns = [ui.FEATURES[f][0] for f in feats]
    fig = px.imshow(R, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    ui.show(ui.style(fig, 680, legend=False, title="Spearman correlation between phenotypes across 48 well-days"))
    st.caption("Correlations across time mix maturation and degeneration; within-day (well-level) correlations would isolate "
               "biological covariation from shared time trends.")
    st.download_button("⬇️ Download the full study table (CSV)", study.to_csv(index=False).encode(), "cardiosim_study.csv", "text/csv")
