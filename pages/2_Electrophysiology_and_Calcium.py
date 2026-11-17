"""Electrophysiology and calcium handling of the monolayer."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cardiosim import ep, render, ui
from cardiosim.stats import day_summary

ui.setup("Electrophysiology & calcium · Cardiomyocyte Lab", "⚡")
ui.hero("⚡ Electrophysiology &amp; calcium handling",
        "Coupled FitzHugh–Nagumo oscillators on the real cell layout: excitation waves, synchronisation, "
        "and single-cell Ca²⁺ transient kinetics across maturation and degeneration.")
study, *_ = ui.load_data()

with st.sidebar:
    well = st.selectbox("Replicate well", list(range(6)), format_func=lambda w: f"Well {w + 1}")
    day = st.slider("Culture day", 1, 8, 5)
out, art = ui.well_day(well, day)
tis, sim, fs = art["tissue"], art["sim"], art["functional"]

k = st.columns(5)
ui.kpi(k[0], "Kuramoto R", f"{fs['kuramoto_R']:.2f}", "1 = perfectly synchronous")
ui.kpi(k[1], "Rate dispersion", f"{fs['rate_dispersion']:.2f}" if np.isfinite(fs['rate_dispersion']) else "—", "CV of cell beat rates")
ui.kpi(k[2], "Beat rate", f"{fs['beat_rate_bpm']:.0f} bpm", "median over cells")
ui.kpi(k[3], "Activation spread", f"{fs['activation_spread_ms']:.0f} ms", "first wave, model time")
ui.kpi(k[4], "Excitable tissue", f"{tis['excitable'].mean():.0%}", "of lattice nodes")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("##### Membrane potential (model units) — wave propagation")

    @st.cache_data(show_spinner="Rendering voltage movie…")
    def vmovie(w, d):
        o, a = ui.well_day(w, d)
        v = a["sim"]["v"]
        sel = v[len(v) // 3::3][:120]
        x = np.clip((sel + 2.1) / 4.1, 0, 1)
        rgb = np.stack([x ** 0.8, 0.25 + 0.5 * x * (1 - x) * 2, 1 - x], -1)
        rgb[:, ~a["tissue"]["excitable"]] = 0.12
        big = np.repeat(np.repeat((rgb * 255).astype(np.uint8), 10, 1), 10, 2)
        return render.to_gif(big, fps=15)
    st.image(vmovie(well, day), use_container_width=True,
             caption="Red = depolarised, blue = resting, dark = no cell / inexcitable. 30 ms per frame.")
with c2:
    st.markdown("##### Activation map (first wave after mid-recording)")
    amap = fs["activation_map"]
    fig = px.imshow(amap, color_continuous_scale="Turbo", labels=dict(color="ms"), aspect="equal")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    ui.show(ui.style(fig, 400, legend=False))

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Activation raster and synchrony")
    acts = fs["acts"]
    rng = np.random.default_rng(0)
    idx = [i for i, a in enumerate(acts) if len(a)]
    pick = sorted(rng.choice(idx, min(120, len(idx)), replace=False)) if idx else []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    for r, i in enumerate(pick):
        fig.add_scatter(x=acts[i] / 1000, y=np.full(len(acts[i]), r), mode="markers", marker=dict(size=3, color=ui.PALETTE[0]),
                        showlegend=False, row=1, col=1)
    fig.add_scatter(x=fs["R_tgrid"] / 1000, y=fs["R_t"], line=dict(color=ui.PALETTE[1], width=2), showlegend=False, row=2, col=1)
    fig.update_yaxes(title_text="cell (node)", row=1, col=1)
    fig.update_yaxes(title_text="R(t)", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    ui.show(ui.style(fig, 430, legend=False))
with c2:
    st.markdown("##### Single-cell Ca²⁺ transients (fluo-4-like ΔF/F₀)")
    ii, jj = np.nonzero(tis["excitable"])
    fig = go.Figure()
    if len(ii):
        sel = np.random.default_rng(1).choice(len(ii), min(6, len(ii)), replace=False)
        for n, s_ in enumerate(sel):
            i, j = ii[s_], jj[s_]
            fig.add_scatter(x=sim["t_frames"] / 1000, y=tis["amp"][i, j] * sim["ca"][:, i, j] + 0.0 * n,
                            line=dict(width=1.5, color=ui.PALETTE[n % 8]), name=f"cell node {i},{j}")
    fig.add_scatter(x=sim["t_ms"] / 1000, y=sim["ca_dff"], line=dict(color="black", width=2.5), name="field average")
    fig.update_layout(xaxis_title="time (s)", yaxis_title="ΔF/F₀")
    ui.show(ui.style(fig, 430))

ca, cadf = ep.single_cell_calcium(tis, sim, n=60, seed=day)
if ca:
    k = st.columns(5)
    ui.kpi(k[0], "Amplitude", f"{ca['amplitude_dff']:.2f}", "ΔF/F₀ (median cell)")
    ui.kpi(k[1], "Time to peak", f"{ca['time_to_peak_ms']:.0f} ms", "10 % → peak")
    ui.kpi(k[2], "CaD50", f"{ca['cad50_ms']:.0f} ms", "duration at 50 % decay")
    ui.kpi(k[3], "CaD90", f"{ca['cad90_ms']:.0f} ms", "duration at 90 % decay")
    ui.kpi(k[4], "Decay τ", f"{ca['tau_decay_ms']:.0f} ms", f"truth {out['true_ca_tau_decay_ms']:.0f} ms")

st.markdown("---")
st.markdown("##### Across the culture (6 wells × 8 days, mean ± 95 % CI)")
cols = st.columns(3)
for col, key in zip(cols * 2, ["kuramoto_R", "rate_dispersion", "activation_spread_ms", "ca_time_to_peak_ms", "ca_cad90_ms", "ca_tau_decay_ms"]):
    s = day_summary(study, key)
    fig = go.Figure()
    for w, g in study.groupby("well"):
        fig.add_scatter(x=g.day, y=g[key], mode="lines", line=dict(color=ui.PALETTE[7], width=1), opacity=0.35, showlegend=False)
    fig.add_scatter(x=s.day, y=s["mean"], mode="lines+markers", line=dict(color=ui.PALETTE[0], width=3), showlegend=False,
                    error_y=dict(type="data", symmetric=False, array=s.ci_hi - s["mean"], arrayminus=s["mean"] - s.ci_lo))
    fig.update_layout(xaxis_title="day")
    with col:
        ui.show(ui.style(fig, 260, legend=False, title=ui.flabel(key)))
ui.finding("Synchronisation emerges as gap-junction coupling matures and intrinsic-rate heterogeneity narrows "
           "(Kuramoto-type entrainment), then collapses when damage removes excitable cells and uncouples the network. "
           "Ca²⁺ handling accelerates with maturation (SERCA-like decay) and slows again with damage.")
st.caption("Model note: FitzHugh–Nagumo is phenomenological, so trends in synchrony, rate dispersion and wave spread "
           "are meaningful while absolute conduction velocities are in model units. A Paci-type hiPSC-CM ionic model would "
           "give quantitative action potentials (see Methods).")
