"""Shared Streamlit helpers: styling, cached data and model access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d63a7a", "#3bb3c4", "#7a7a72"]
INK, INK2, GRID = "#111111", "#55534e", "#e6e5e0"
DAY_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#6a51a3", "#e6550d", "#a63603"]

CSS = """
<style>
.hero{background:linear-gradient(135deg,#7f1d1d 0%,#9f1239 45%,#1e3a8a 100%);color:#fff;padding:1.5rem 1.8rem;
  border-radius:16px;margin:.2rem 0 1rem}
.hero h1{color:#fff;margin:0;font-size:1.9rem;line-height:1.2}
.hero p{margin:.45rem 0 0;opacity:.93;font-size:1rem}
.kpi{background:#f7f7f5;border-radius:12px;padding:.85rem 1rem;border-left:5px solid #9f1239;height:100%}
.kpi .lbl{font-size:.74rem;color:#55534e;text-transform:uppercase;letter-spacing:.04em}
.kpi .val{font-size:1.45rem;font-weight:700;color:#111;line-height:1.2;margin-top:.15rem}
.kpi .sub{font-size:.8rem;color:#55534e}
.finding{background:#f7f7f5;border-radius:10px;padding:.7rem 1rem;margin:.35rem 0;border-left:4px solid #2a78d6}
.finding.warn{border-left-color:#d63a3a;background:#fff6f6}
.finding.ok{border-left-color:#1baf7a;background:#f2fbf6}
.note{font-size:.84rem;color:#55534e}
.sim-badge{display:inline-block;background:#fff4e5;color:#8a4b00;border:1px solid #f3c98b;border-radius:999px;
  padding:.1rem .6rem;font-size:.78rem;margin-left:.3rem}
.step{background:#fff;border:1px solid #e6e5e0;border-radius:12px;padding:.8rem .9rem;height:100%}
.step b{color:#9f1239}
</style>
"""


def setup(title, icon="🫀"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def finding(text, kind=""):
    st.markdown(f'<div class="finding {kind}">{text}</div>', unsafe_allow_html=True)


def style(fig, height=380, legend=True, title=None):
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)))
    fig.update_layout(template="plotly_white", height=height, showlegend=legend,
                      margin=dict(l=10, r=10, t=50 if (title or fig.layout.title.text) else 20, b=10),
                      font=dict(color=INK, size=13),
                      legend=dict(orientation="h", yanchor="top", y=-0.26, x=0, title=None, font=dict(color=INK2)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


@st.cache_data(show_spinner=False)
def load_data():
    study = pd.read_csv(DATA / "study.csv")
    cells = pd.read_csv(DATA / "cells.csv")
    dose = pd.read_csv(DATA / "dose_response.csv")
    morris = pd.read_csv(DATA / "morris.csv")
    meta = json.load(open(DATA / "meta.json"))
    return study, cells, dose, morris, meta


@st.cache_resource(show_spinner=False)
def default_study():
    from .model import CultureParams, make_study
    p = CultureParams()
    return p, make_study(p)


@st.cache_resource(show_spinner="Simulating this well-day (electrophysiology, imaging, motion)…", max_entries=12)
def well_day(well: int, day: int, video=True):
    from .pipeline import run_well_day
    p, wells = default_study()
    return run_well_day(p, wells[well], day, video=video, keep=True)


FEATURES = {
    # column: (label, unit, group)
    "img_n_cells": ("Attached cells (segmented)", "cells / field", "Morphology"),
    "img_mean_area_um2": ("Cell area", "µm²", "Morphology"),
    "img_mean_aspect_ratio": ("Cell aspect ratio", "", "Morphology"),
    "img_orientation_order": ("Orientation order S", "0–1", "Morphology"),
    "img_nc_ratio": ("Nucleus : cell area", "", "Morphology"),
    "img_n_fragments": ("Nuclear fragments / debris", "objects", "Morphology"),
    "img_striated_fraction": ("Striated cells", "fraction", "Sarcomeres"),
    "img_sarcomere_length_um": ("Sarcomere length", "µm", "Sarcomeres"),
    "img_sarcomere_score": ("Sarcomere organisation score", "", "Sarcomeres"),
    "kuramoto_R": ("Synchrony (Kuramoto R)", "0–1", "Electrophysiology"),
    "rate_dispersion": ("Beat-rate dispersion (CV)", "", "Electrophysiology"),
    "beat_rate_bpm": ("Spontaneous rate (EP)", "bpm", "Electrophysiology"),
    "activation_spread_ms": ("Activation spread time", "ms (model)", "Electrophysiology"),
    "ca_amplitude_dff": ("Ca²⁺ transient amplitude", "ΔF/F₀", "Calcium"),
    "ca_time_to_peak_ms": ("Ca²⁺ time to peak", "ms", "Calcium"),
    "ca_cad90_ms": ("Ca²⁺ CaD90", "ms", "Calcium"),
    "ca_tau_decay_ms": ("Ca²⁺ decay τ", "ms", "Calcium"),
    "mot_amplitude_um": ("Contraction amplitude", "µm", "Contractility"),
    "mot_contraction_velocity_um_s": ("Contraction velocity", "µm/s", "Contractility"),
    "mot_cd50_ms": ("Contraction duration CD50", "ms", "Contractility"),
    "mot_sync_index": ("Regional contraction synchrony", "r", "Contractility"),
    "mot_rr_cv": ("Beat-interval variability (CV)", "", "Contractility"),
}


def flabel(col):
    lab, unit, _ = FEATURES.get(col, (col, "", ""))
    return f"{lab} ({unit})" if unit else lab


def sim_badge():
    return '<span class="sim-badge">in-silico data</span>'
