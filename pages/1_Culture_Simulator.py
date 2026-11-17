"""Culture simulator: one well on one day, end to end — and a user-configurable culture."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from cardiosim import ui
from cardiosim.model import CultureParams, latent, logistic, make_study, phenotypes

ui.setup("Culture simulator · Cardiomyocyte Lab", "🧫")
ui.hero("🧫 Culture simulator", "Look at any replicate well on any day: synthetic microscopy, the beating movie, "
        "and what the analysis pipeline measured versus the ground truth.")
study, cells, dose, morris, meta = ui.load_data()
P = CultureParams()

with st.sidebar:
    st.header("Well and day")
    well = st.selectbox("Replicate well", list(range(P.n_wells)), format_func=lambda w: f"Well {w + 1}")
    day = st.slider("Culture day", 1, 8, 5)
    overlay = st.toggle("Show segmentation outlines", value=False)

out, art = ui.well_day(well, day)
cs = art["cells"]
k = st.columns(6)
ui.kpi(k[0], "Maturation M", f"{out['true_M']:.2f}", "latent, 0–1")
ui.kpi(k[1], "Damage D", f"{out['true_D']:.2f}", "latent, 0–1")
ui.kpi(k[2], "Attached cells", f"{out['true_n_cells']}", f"{out['true_dying']} apoptotic")
ui.kpi(k[3], "Synchrony R", f"{out['kuramoto_R']:.2f}", "Kuramoto order parameter")
ui.kpi(k[4], "Striated cells", f"{out['img_striated_fraction']:.0%}", "periodic Z-disc signal")
ui.kpi(k[5], "Beat rate", f"{out.get('mot_beat_rate_bpm', np.nan):.0f} bpm", "from optical flow")

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### α-actinin / DAPI immunofluorescence (synthetic)")
    rgb = art["image"]["rgb"].copy()
    if overlay:
        from skimage.segmentation import find_boundaries
        b = find_boundaries(art["masks"][0], mode="outer")
        rgb[b] = [255, 255, 0]
    st.image(rgb, use_container_width=True, caption="276 × 276 µm, 0.36 µm/px. Yellow: watershed cell outlines." if overlay
             else "276 × 276 µm, 0.36 µm/px")
    with st.expander("Zoom on sarcomeres (centre 60 µm)"):
        z = art["image"]["rgb"][300:468, 300:468]
        st.image(Image.fromarray(z).resize((504, 504), Image.NEAREST), use_container_width=True)
with c2:
    st.markdown("##### Bright-field beating movie (synthetic)")
    from cardiosim import render

    @st.cache_data(show_spinner="Encoding movie…")
    def movie(w, d):
        o, a = ui.well_day(w, d)
        return render.to_gif(a["frames"][::2], fps=20, levels=32)
    st.image(movie(well, day), use_container_width=True, caption="5 s at 40 fps (shown at half rate). Cells shorten along their "
             "long axis in proportion to their computed active tension.")
    mo = art["motion"]
    fig = go.Figure()
    fig.add_scatter(x=mo["t_s"], y=mo["displacement_um"], name="optical-flow displacement (µm)", line=dict(color=ui.PALETTE[0], width=2))
    gt = art["gt_tension"]
    if np.ptp(gt) > 0:
        fig.add_scatter(x=mo["t_s"], y=gt / gt.max() * max(mo["displacement_um"].max(), 1e-6), name="ground-truth tension (scaled)",
                        line=dict(color=ui.PALETTE[1], dash="dot"))
    fig.add_scatter(x=mo["t_s"][art["peaks"]], y=mo["displacement_um"][art["peaks"]], mode="markers", name="detected beats",
                    marker=dict(symbol="triangle-down", size=11, color=ui.PALETTE[5]))
    fig.update_layout(xaxis_title="time (s)", yaxis_title="µm")
    ui.show(ui.style(fig, 300))

st.markdown("##### Measured by the pipeline vs ground truth (this well-day)")
rows = [
    ("Attached cells", out["true_n_cells"], out["img_n_cells"], "nuclei-seeded watershed"),
    ("Orientation order S", out["true_orientation_order"], out["img_orientation_order"], "nematic order of cell axes"),
    ("Median aspect ratio", out["true_med_aspect_ratio"], out["img_mean_aspect_ratio"], "moments of segmented cells"),
    ("Sarcomere length (µm)", out["true_med_sarcomere_length"], out["img_sarcomere_length_um"], "2-D FFT, striated cells only"),
    ("Ca²⁺ decay τ (ms)", out["true_ca_tau_decay_ms"], out.get("ca_tau_decay_ms", np.nan), "single-cell traces, exp. fit"),
    ("Beats in movie", out.get("true_n_beats", np.nan), out.get("mot_n_beats", np.nan), f"sensitivity {out.get('val_beat_sensitivity', np.nan):.0%}, "
     f"timing error {out.get('val_beat_timing_ms', np.nan):.0f} ms"),
]
df = pd.DataFrame(rows, columns=["Readout", "Ground truth", "Measured", "Method / note"])
df["Ground truth"] = df["Ground truth"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "—")
df["Measured"] = df["Measured"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "not detected")
st.dataframe(df, hide_index=True, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Per-cell morphometrics")
    ct = art["cell_table"].drop(columns=["bbox", "centroid_x", "centroid_y"], errors="ignore")
    st.dataframe(ct.round(3), use_container_width=True, height=300)
with c2:
    st.markdown("##### Latent maturation and damage of all wells")
    t = np.linspace(0, 8.5, 200)
    eff = make_study(P)
    fig = go.Figure()
    for w in eff:
        M, D = latent(t, P, w.u, w.v)
        fig.add_scatter(x=t, y=M, line=dict(color=ui.PALETTE[0], width=3 if w.index == well else 1), opacity=1 if w.index == well else 0.35,
                        name="maturation M" if w.index == well else None, showlegend=w.index == well)
        fig.add_scatter(x=t, y=D, line=dict(color=ui.PALETTE[5], width=3 if w.index == well else 1), opacity=1 if w.index == well else 0.35,
                        name="damage D" if w.index == well else None, showlegend=w.index == well)
    fig.add_vline(x=day, line_dash="dot", line_color=ui.INK2)
    fig.update_layout(xaxis_title="day", yaxis_title="latent state")
    ui.show(ui.style(fig, 300))

st.markdown("---")
st.markdown("##### Design your own culture")
st.caption("Change the biology and see how the ground-truth phenotype trajectories respond (instant), then run the full "
           "pipeline on one day of your culture (≈10 s).")
c = st.columns(4)
mid = c[0].slider("Maturation midpoint (day)", 2.0, 5.0, P.mat_midpoint, 0.1)
rate = c[1].slider("Maturation rate (1/day)", 0.6, 2.5, P.mat_rate, 0.1)
onset = c[2].slider("Damage onset (day)", 5.0, 9.0, P.dmg_onset, 0.1)
gmax = c[3].slider("Max gap-junction coupling", 0.3, 1.5, P.g_max, 0.05)
c = st.columns(4)
haz = c[0].slider("Damage-induced death hazard (1/day)", 0.1, 1.2, P.dmg_hazard, 0.05)
pace = c[1].slider("Pacemaker fraction", 0.0, 0.4, P.pacemaker_fraction, 0.02)
rcv = c[2].slider("Rate heterogeneity (immature CV)", 0.05, 0.6, P.rate_cv_immature, 0.05)
seed = c[3].number_input("Random seed", 1, 9999, 42)
mine = CultureParams(mat_midpoint=mid, mat_rate=rate, dmg_onset=onset, g_max=gmax, dmg_hazard=haz,
                     pacemaker_fraction=pace, rate_cv_immature=rcv, seed=int(seed))
t = np.linspace(0, 10, 200)
Md, Dd = latent(t, P)
Mm, Dm = latent(t, mine)
phd, phm = phenotypes(Md, Dd, P), phenotypes(Mm, Dm, mine)
show = [("coupling", "Gap-junction coupling"), ("sarcomere_order", "Sarcomere order"), ("aspect_ratio", "Aspect ratio"),
        ("ca_tau_decay_ms", "Ca²⁺ decay τ (ms)")]
cols = st.columns(4)
for col, (key, lab) in zip(cols, show):
    fig = go.Figure()
    fig.add_scatter(x=t, y=phd[key], name="default", line=dict(color=ui.PALETTE[7], dash="dot"))
    fig.add_scatter(x=t, y=phm[key], name="your culture", line=dict(color=ui.PALETTE[1], width=2.5))
    fig.update_layout(xaxis_title="day")
    with col:
        ui.show(ui.style(fig, 260, title=lab))
# survival
tg = np.linspace(0, 10, 200)
S_def = np.exp(-np.cumsum((P.base_hazard + P.dmg_hazard * Dd) * (tg[1] - tg[0])))
S_my = np.exp(-np.cumsum((mine.base_hazard + mine.dmg_hazard * Dm) * (tg[1] - tg[0])))
ui.finding(f"Expected survival of attached cells at day 8: default <b>{np.interp(8, tg, S_def):.0%}</b>, "
           f"your culture <b>{np.interp(8, tg, S_my):.0%}</b>.")

run_day = st.select_slider("Day to run through the full pipeline", options=list(range(1, 9)), value=5)
if st.button("▶️ Simulate, image and analyse this day of my culture", type="primary"):
    from cardiosim.pipeline import run_well_day
    with st.spinner("Running agents → electrophysiology → microscopy → analysis…"):
        w0 = make_study(CultureParams(**{**mine.to_dict(), "n_wells": 1}))[0]
        res, a = run_well_day(mine, w0, run_day, keep=True)
    c1, c2 = st.columns(2)
    c1.image(a["image"]["rgb"], caption=f"Your culture, day {run_day}", use_container_width=True)
    c2.image(render.to_gif(a["frames"][::2], fps=20, levels=32), caption="Beating movie", use_container_width=True)
    keys = ["kuramoto_R", "rate_dispersion", "img_striated_fraction", "img_sarcomere_length_um", "img_orientation_order",
            "ca_time_to_peak_ms", "ca_tau_decay_ms", "mot_beat_rate_bpm", "mot_amplitude_um"]
    ref = study[(study.day == run_day)][keys].mean()
    st.dataframe(pd.DataFrame({"Readout": [ui.flabel(k_) for k_ in keys],
                               "Your culture": [res.get(k_, np.nan) for k_ in keys],
                               "Default study (mean of 6 wells)": [ref[k_] for k_ in keys]}).round(3),
                 hide_index=True, use_container_width=True)
