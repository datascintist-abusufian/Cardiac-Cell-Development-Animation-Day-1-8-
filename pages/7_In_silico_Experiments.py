"""In-silico experiments: gap-junction blocker dose–response, culture stress, global sensitivity."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cardiosim import ui
from cardiosim.model import CultureParams, latent, make_study, phenotypes
from cardiosim.stats import hill, hill_fit, kaplan_meier, logrank

ui.setup("In-silico experiments · Cardiomyocyte Lab", "🧪")
ui.hero("🧪 In-silico experiments", "Virtual pharmacology and stress tests on the culture model, with the same statistics "
        "you would use at the bench — and the advantage of knowing the true answer.")
study, cells, dose, morris, meta = ui.load_data()
tabs = st.tabs(["Gap-junction blocker dose–response", "Culture stress & survival", "Global sensitivity (Morris)"])

# --------------------------------------------------------------------------- dose response
with tabs[0]:
    st.markdown(f"A carbenoxolone-like uncoupler scales gap-junction conductance by 1 / (1 + (c / IC₅₀)ʰ) with a **hidden true "
                f"IC₅₀ = {meta['true_ic50_uM']} µM** and Hill slope {meta['true_hill']}. Day-5 cultures of 3 wells were "
                f"simulated at {len(meta['concentrations_uM'])} concentrations; the IC₅₀ is then estimated from the "
                "measured synchrony only, as in a real assay.")
    resp = st.selectbox("Response", ["kuramoto_R", "activation_spread_ms", "rate_dispersion"], format_func=ui.flabel)
    d = dose.copy()
    d["c_plot"] = d.conc_uM.replace(0, 0.03)
    fit = hill_fit(d.conc_uM.values, d[resp].values) if resp == "kuramoto_R" else None
    fig = go.Figure()
    fig.add_scatter(x=d.c_plot, y=d[resp], mode="markers", marker=dict(size=10, color=d.well, colorscale="Viridis"), name="wells")
    if fit:
        cs = np.logspace(-1.6, 1.6, 200)
        fig.add_scatter(x=cs, y=hill(cs, *fit["popt"]), line=dict(color=ui.PALETTE[1], width=3), name="4-parameter Hill fit")
        fig.add_vline(x=fit["ic50"], line_dash="dot", line_color=ui.PALETTE[1], annotation_text=f"functional IC₅₀ {fit['ic50']:.1f} µM", annotation_position="top right")
        fig.add_vline(x=meta["true_ic50_uM"], line_dash="dash", line_color=ui.INK2, annotation_text="molecular IC₅₀", annotation_position="bottom right")
    fig.update_layout(xaxis_type="log", xaxis_range=[np.log10(0.02), np.log10(60)],
                      xaxis_title="blocker concentration (µM; vehicle plotted at 0.03)", yaxis_title=ui.flabel(resp))
    ui.show(ui.style(fig, 420))
    if fit:
        k = st.columns(4)
        ui.kpi(k[0], "Functional IC₅₀ (synchrony)", f"{fit['ic50']:.2f} µM", f"95 % CI {fit['ic50_ci'][0]:.2f}–{fit['ic50_ci'][1]:.2f}")
        ui.kpi(k[1], "Molecular IC₅₀ (coupling)", f"{meta['true_ic50_uM']} µM", "hidden parameter of the simulation")
        ui.kpi(k[2], "Hill slope", f"{fit['hill']:.2f}", f"molecular {meta['true_hill']}")
        ui.kpi(k[3], "Dynamic range", f"{fit['top']:.2f} → {fit['bottom']:.2f}", "R at vehicle → max block")
        ratio = fit["ic50"] / meta["true_ic50_uM"]
        ui.finding(f"The functional IC₅₀ is <b>{ratio:.1f}×</b> the molecular IC₅₀ and the curve is steeper (Hill {fit['hill']:.1f} vs "
                   f"{meta['true_hill']}). Synchrony is a threshold read-out of coupling: a mature monolayer has a coupling reserve, "
                   "and entrainment only fails once conductance drops below a critical value, then collapses abruptly. Functional "
                   "assays therefore under-estimate potency and exaggerate cooperativity — important when hiPSC-CM synchrony is used "
                   "to rank gap-junction or conduction-modifying drugs.", "ok")
    else:
        ui.finding("Hill fitting is shown for synchrony; activation spread and rate dispersion rise non-monotonically at high block "
                   "because fragments beat independently.")

# --------------------------------------------------------------------------- stress
with tabs[1]:
    st.markdown("Oxidative or metabolic stress is modelled as an **earlier damage onset** and a higher damage-induced death "
                "hazard. Cells are simulated in replicate wells; survival is compared with Kaplan–Meier curves and a log-rank test.")
    c = st.columns(3)
    shift = c[0].slider("Damage onset shift (days)", -3.0, 1.0, -1.5, 0.1)
    hz = c[1].slider("Damage-hazard multiplier", 0.5, 3.0, 1.6, 0.1)
    nw = c[2].slider("Wells per arm", 2, 8, 4)

    @st.cache_data(show_spinner=False)
    def arm(shift_, hz_, nw_, seed):
        base = CultureParams()
        p = CultureParams(**{**base.to_dict(), "dmg_onset": base.dmg_onset + shift_, "dmg_hazard": base.dmg_hazard * hz_,
                             "n_wells": nw_, "seed": seed})
        ws = make_study(p)
        cc = pd.concat([w.cells.assign(well=w.index) for w in ws])
        cc = cc[cc.attach_day < 8]
        t = np.minimum(cc.death_day, 8) - cc.attach_day
        e = cc.death_day <= 8
        return t.values, e.values, p
    tc, ec, pc = arm(0.0, 1.0, nw, 101)
    ts, es, ps = arm(shift, hz, nw, 202)
    lr = logrank(tc, ec, ts, es)
    fig = go.Figure()
    for (t_, e_, name, col) in [(tc, ec, "control", ui.PALETTE[0]), (ts, es, "stressed", ui.PALETTE[5])]:
        km = kaplan_meier(t_, e_)
        fig.add_scatter(x=np.r_[km.t, km.t[::-1]], y=np.r_[km.hi, km.lo[::-1]], fill="toself", line=dict(width=0),
                        fillcolor=col, opacity=0.15, showlegend=False)
        fig.add_scatter(x=km.t, y=km.S, line=dict(color=col, width=3, shape="hv"), name=f"{name} (n = {len(t_)} cells)")
    fig.update_layout(xaxis_title="days since attachment", yaxis_title="survival probability", yaxis_range=[0, 1.02])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        ui.show(ui.style(fig, 400, title=f"Log-rank χ² = {lr['chi2']:.1f}, p = {lr['p']:.2g}"))
    with c2:
        tt = np.linspace(0, 10, 200)
        fig = go.Figure()
        for p_, name, col in [(pc, "control", ui.PALETTE[0]), (ps, "stressed", ui.PALETTE[5])]:
            M, D = latent(tt, p_)
            ph = phenotypes(M, D, p_)
            fig.add_scatter(x=tt, y=ph["coupling"], line=dict(color=col, width=2.5), name=f"{name}: coupling")
            fig.add_scatter(x=tt, y=ph["sarcomere_order"], line=dict(color=col, dash="dot"), name=f"{name}: sarcomere order")
        fig.update_layout(xaxis_title="day")
        ui.show(ui.style(fig, 400, title="Mean phenotype trajectories"))
    ui.finding("Earlier damage onset shortens the functional window (coupled, striated, synchronous) before cell loss begins — "
               "the model predicts that functional read-outs (synchrony, sarcomere order) fall <i>before</i> viability does, which "
               "makes them earlier biomarkers of culture stress than cell counts.")

# --------------------------------------------------------------------------- morris
with tabs[2]:
    st.markdown(f"Morris elementary-effects screening of {len(meta['morris']['names'])} biological parameters "
                f"({meta['morris']['trajectories']} trajectories, {meta['morris']['levels']} levels; each run simulates "
                "one well for 8 days including electrophysiology). μ* = mean absolute effect (importance); σ = spread of "
                "effects (non-linearity or interactions).")
    out = st.selectbox("Output", sorted(morris.output.unique()),
                       format_func=lambda s: s.replace("_", " "))
    m = morris[morris.output == out]
    fig = go.Figure()
    fig.add_scatter(x=m.mu_star, y=m.sigma, mode="markers+text", text=m.parameter.str.replace("_", " "), textposition="top center",
                    marker=dict(size=14, color=ui.PALETTE[4]))
    mx = max(m.mu_star.max(), m.sigma.max()) * 1.15
    fig.add_scatter(x=[0, mx], y=[0, mx], line=dict(dash="dot", color=ui.INK2), name="σ = μ*")
    fig.update_layout(xaxis_title="μ* (importance)", yaxis_title="σ (non-linearity / interaction)", xaxis_range=[0, mx], yaxis_range=[0, mx])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        ui.show(ui.style(fig, 420, legend=False, title=out.replace("_", " ")))
    with c2:
        piv = morris.pivot(index="parameter", columns="output", values="mu_star")
        piv = piv / piv.max()
        fig = px.imshow(piv, color_continuous_scale="Purples", text_auto=".2f", aspect="auto")
        ui.show(ui.style(fig, 420, legend=False, title="Normalised μ* (per output)"))
    top = m.sort_values("mu_star", ascending=False).parameter.iloc[0]
    ui.finding(f"For <b>{out.replace('_', ' ')}</b> the most influential parameter is <b>{top.replace('_', ' ')}</b>. "
               "Parameters near σ = μ* act non-linearly or through interactions; those near the origin can be fixed in "
               "future calibration, reducing the number of quantities that must be measured.")
