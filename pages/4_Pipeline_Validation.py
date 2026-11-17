"""Validation of the measurement pipeline against the simulator's ground truth."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cardiosim import ui
from cardiosim.stats import bland_altman

ui.setup("Pipeline validation · Cardiomyocyte Lab", "✅")
ui.hero("✅ Pipeline validation", "Because the simulator knows the truth, every measurement can be checked: "
        "agreement, systematic bias and the conditions under which the analysis fails.")
study, *_ = ui.load_data()

PAIRS = [
    ("Attached cells", "true_n_cells", "img_n_cells", "cells"),
    ("Orientation order S", "true_orientation_order", "img_orientation_order", ""),
    ("Cell aspect ratio (median)", "true_med_aspect_ratio", "img_mean_aspect_ratio", ""),
    ("Cell area (median)", "true_med_area_um2", "img_mean_area_um2", "µm²"),
    ("Sarcomere length (striated cells)", "true_med_sarcomere_length", "img_sarcomere_length_um", "µm"),
    ("Ca²⁺ decay τ", "true_ca_tau_decay_ms", "ca_tau_decay_ms", "ms"),
    ("Sarcomere order → organisation score", "true_sarcomere_order", "img_sarcomere_score", "(different scales)"),
]
rows = []
for name, t, m, unit in PAIRS:
    ba = bland_altman(study[m], study[t])
    rows.append(dict(Readout=name, n=ba["n"], r=ba["r"], CCC=ba["ccc"], bias=ba["bias"], LoA=f"{ba['loa'][0]:.3g} to {ba['loa'][1]:.3g}",
                     slope=ba["slope"], unit=unit))
tab = pd.DataFrame(rows)
st.markdown("##### Agreement with ground truth over all 48 well-days")
st.dataframe(tab.style.format({"r": "{:.3f}", "CCC": "{:.3f}", "bias": "{:.3g}", "slope": "{:.2f}"}), hide_index=True,
             use_container_width=True)
st.caption("r: Pearson correlation; CCC: Lin's concordance correlation (agreement on the identity line); bias and 95 % limits of "
           "agreement (LoA) from Bland–Altman analysis; slope of measured on true (1 = no attenuation).")

name = st.selectbox("Inspect a readout", [p[0] for p in PAIRS])
_, t, m, unit = next(p for p in PAIRS if p[0] == name)
ba = bland_altman(study[m], study[t])
d = study[[t, m, "day", "well"]].dropna()
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_scatter(x=d[t], y=d[m], mode="markers", marker=dict(color=d.day, colorscale="Viridis", size=9, showscale=True,
                                                               colorbar=dict(title="day")), name="well-days")
    lo, hi = np.nanmin([d[t].min(), d[m].min()]), np.nanmax([d[t].max(), d[m].max()])
    if "different" not in unit:
        fig.add_scatter(x=[lo, hi], y=[lo, hi], line=dict(color=ui.INK2, dash="dot"), name="identity")
    xs = np.linspace(d[t].min(), d[t].max(), 20)
    fig.add_scatter(x=xs, y=ba["intercept"] + ba["slope"] * xs, line=dict(color=ui.PALETTE[1]), name="linear fit")
    fig.update_layout(xaxis_title=f"ground truth {unit}", yaxis_title=f"measured {unit}")
    ui.show(ui.style(fig, 400, title="Measured vs true"))
with c2:
    fig = go.Figure()
    fig.add_scatter(x=ba["mean"], y=ba["diff"], mode="markers", marker=dict(color=ui.PALETTE[0], size=8), showlegend=False)
    for y, dash in [(ba["bias"], "solid"), (ba["loa"][0], "dash"), (ba["loa"][1], "dash")]:
        fig.add_hline(y=y, line_dash=dash, line_color=ui.PALETTE[1])
    fig.update_layout(xaxis_title="mean of measured and true", yaxis_title="measured − true")
    ui.show(ui.style(fig, 400, legend=False, title="Bland–Altman"))

st.markdown("##### Beat detection from optical flow")
v = study.groupby("day")[["val_beat_sensitivity", "val_beat_precision", "val_trace_corr", "mot_amplitude_um"]].mean().reset_index()
fig = go.Figure()
fig.add_bar(x=v.day, y=v.val_beat_sensitivity, name="sensitivity", marker_color=ui.PALETTE[0])
fig.add_bar(x=v.day, y=v.val_beat_precision, name="precision", marker_color=ui.PALETTE[2])
fig.add_scatter(x=v.day, y=v.val_trace_corr, name="trace correlation with true tension", mode="lines+markers",
                line=dict(color=ui.PALETTE[1], width=3))
fig.update_layout(barmode="group", xaxis_title="day", yaxis_range=[-0.2, 1.05])
ui.show(ui.style(fig, 340))

st.markdown("##### What the validation shows")
T = tab.set_index("Readout")
good = T.loc[["Attached cells", "Orientation order S", "Cell area (median)"]]
ui.finding("<b>Counts, orientation order and cell area agree closely with the truth</b> (CCC "
           + ", ".join(f"{v:.2f}" for v in good.CCC) + ").", "ok")
sl = T.loc["Sarcomere length (striated cells)"]
ba_sl = bland_altman(study["img_sarcomere_length_um"], study["true_med_sarcomere_length"])
ui.finding(f"<b>Sarcomere length is accurate in absolute terms</b> (bias {ba_sl['bias']:+.3f} µm, limits of agreement "
           f"{ba_sl['loa'][0]:+.2f} to {ba_sl['loa'][1]:+.2f} µm), yet its CCC is only {sl.CCC:.2f}: among striated cultures the "
           "true length spans < 0.1 µm, and correlation-type indices collapse when the true range is narrow. Absolute agreement "
           "(Bland–Altman) is the right criterion here. Immature days return “not striated” instead of a spurious length.", "ok")
ar = T.loc["Cell aspect ratio (median)"]
ui.finding(f"<b>Aspect ratio is attenuated</b> (slope {ar.slope:.2f}): in a confluent monolayer, touching elongated cells are "
           f"split by the watershed, which shortens them. Trends survive (r = {ar.r:.2f}) but absolute elongation is "
           "underestimated — a known bias of intensity-based segmentation that instance-segmentation models (e.g. Cellpose) reduce.", "warn")
ca = T.loc["Ca²⁺ decay τ"]
ui.finding(f"<b>Ca²⁺ decay τ is underestimated</b> (bias {ca.bias:.0f} ms). F₀ is taken from the diastolic minimum, which sits above "
           "true resting Ca²⁺ when decay is incomplete between beats; transients measured against an elevated baseline look faster. "
           "The effect grows with beat rate and slow decay — a confound worth controlling with pacing or baseline correction.", "warn")
ui.finding("<b>Beat detection fails where contraction is weak and asynchronous</b> (days 1–3, 8): the field-averaged displacement "
           "is tiny and the true tension of unsynchronised cells has no clear peaks. Mature, synchronous days are detected "
           "almost perfectly. Report motion metrics only above an amplitude / synchrony threshold.", "warn")
