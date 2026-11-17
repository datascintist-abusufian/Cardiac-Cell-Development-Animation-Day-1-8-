"""Motion analysis of beating cardiomyocytes from video (upload or simulated)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cardiosim import analysis as A
from cardiosim import io, render, ui

ui.setup("Motion analysis · Cardiomyocyte Lab", "🎥")
ui.hero("🎥 Beating-motion analysis", "Dense optical flow (Farneback) against an automatically chosen relaxed reference frame, "
        "beat detection, contraction–relaxation kinetics, rhythm variability and regional synchrony.")

with st.sidebar:
    src = st.radio("Video source", ["Simulated well-day", "Upload a video"])
    if src == "Simulated well-day":
        well = st.selectbox("Replicate well", list(range(6)), format_func=lambda w: f"Well {w + 1}")
        day = st.slider("Culture day", 1, 8, 5)
    else:
        up = st.file_uploader("Video (MP4, AVI, MOV, multi-page TIFF or GIF)", type=["mp4", "avi", "mov", "tif", "tiff", "gif", "m4v"])
        fps_in = st.number_input("Frame rate (fps) — used if the file does not state it", 1.0, 1000.0, 30.0)
        um_px = st.number_input("Pixel size (µm / px) of the original video", 0.01, 50.0, 1.0, help="Displacements are reported in µm.")
        max_s = st.slider("Seconds to analyse", 2, 30, 10)

frames = fps = um_per_px = None
gt = None
if src == "Simulated well-day":
    out, art = ui.well_day(well, day)
    frames, fps, um_per_px, gt = art["frames"], 40.0, 276 / 256, art["gt_tension"]
elif up is not None:
    with st.spinner("Decoding video…"):
        try:
            frames, fps_file, scale = io.read_video(up, max_frames=2000, max_side=320)
        except Exception as e:
            st.error(str(e))
            st.stop()
    fps = fps_file if np.isfinite(fps_file) and fps_file > 0 else fps_in
    frames = frames[: int(max_s * fps)]
    um_per_px = um_px * scale
    st.caption(f"{len(frames)} frames at {fps:.1f} fps, {frames.shape[2]}×{frames.shape[1]} px after down-scaling (×{scale:.2f}).")
else:
    st.info("Upload a bright-field or phase-contrast movie of beating cardiomyocytes, or switch to a simulated well-day.")
    st.stop()


@st.cache_data(show_spinner="Computing dense optical flow…", max_entries=8)
def analyse(_frames, key, fps, um_per_px):
    return A.motion_analysis(_frames, fps, um_per_px=um_per_px)


key = f"{src}-{len(frames)}-{int(frames[::7].sum())}"
mo = analyse(frames, key, fps, um_per_px)
beats, summ, pk, smooth = A.beat_kinetics(mo["displacement_um"], fps)
rs = A.regional_synchrony(mo["tile_traces"], fps)

k = st.columns(6)
ui.kpi(k[0], "Beats", f"{summ['n_beats']}", f"{len(frames) / fps:.1f} s analysed")
ui.kpi(k[1], "Beat rate", f"{summ.get('beat_rate_bpm', np.nan):.0f} bpm", "median RR")
ui.kpi(k[2], "Amplitude", f"{summ.get('amplitude_um', np.nan):.2f} µm", "mean |displacement|")
ui.kpi(k[3], "Contraction velocity", f"{summ.get('contraction_velocity_um_s', np.nan):.1f} µm/s", "max d(disp)/dt")
ui.kpi(k[4], "RR variability", f"{summ.get('rr_cv', np.nan):.2f}", f"CV · RMSSD {summ.get('rmssd_ms', np.nan):.0f} ms")
ui.kpi(k[5], "Regional synchrony", f"{rs['sync_index']:.2f}", f"lag SD {rs['lag_sd_ms']:.0f} ms (4×4 tiles)")

c1, c2 = st.columns([1, 1.3])
with c1:
    @st.cache_data(show_spinner=False, max_entries=8)
    def gif(_frames, key, fps):
        step = max(1, int(round(fps / 20)))
        return render.to_gif(_frames[::step][:150], fps=fps / step, levels=48)
    st.image(gif(frames, key, fps), use_container_width=True, caption=f"Input movie · reference frame {mo['ref_frame']}"
             + (" (polarity corrected: first candidate was a contracted frame)" if mo["reference_flipped"] else ""))
with c2:
    fig = go.Figure()
    fig.add_scatter(x=mo["t_s"], y=mo["displacement_um"], name="displacement (µm)", line=dict(color=ui.PALETTE[0], width=2))
    fig.add_scatter(x=mo["t_s"], y=mo["speed_um_s"] / max(mo["speed_um_s"].max(), 1e-9) * max(mo["displacement_um"].max(), 1e-9),
                    name="speed (scaled)", line=dict(color=ui.PALETTE[2], width=1))
    if gt is not None and np.ptp(gt) > 0:
        fig.add_scatter(x=mo["t_s"], y=gt / gt.max() * mo["displacement_um"].max(), name="ground-truth tension (scaled)",
                        line=dict(color=ui.PALETTE[1], dash="dot"))
    fig.add_scatter(x=mo["t_s"][pk], y=mo["displacement_um"][pk], mode="markers", name="beats",
                    marker=dict(symbol="triangle-down", size=12, color=ui.PALETTE[5]))
    fig.update_layout(xaxis_title="time (s)", yaxis_title="µm")
    ui.show(ui.style(fig, 360, title="Contraction trace"))

c1, c2, c3 = st.columns(3)
with c1:
    fig = px.imshow(mo["max_disp_map"], color_continuous_scale="Inferno", labels=dict(color="µm"))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    ui.show(ui.style(fig, 330, legend=False, title="Peak displacement map"))
with c2:
    if rs["corr"] is not None:
        fig = px.imshow(rs["corr"], color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        ui.show(ui.style(fig, 330, legend=False, title="Tile–tile correlation of motion"))
with c3:
    rr = np.diff(pk) / fps * 1000
    fig = go.Figure()
    if len(rr) > 2:
        fig.add_scatter(x=rr[:-1], y=rr[1:], mode="markers", marker=dict(size=10, color=ui.PALETTE[4]), showlegend=False)
        lim = [rr.min() * 0.9, rr.max() * 1.1]
        fig.add_scatter(x=lim, y=lim, line=dict(dash="dot", color=ui.INK2), showlegend=False)
    fig.update_layout(xaxis_title="RRₙ (ms)", yaxis_title="RRₙ₊₁ (ms)")
    ui.show(ui.style(fig, 330, legend=False, title="Poincaré plot of beat intervals"))

if len(beats):
    st.markdown("##### Per-beat kinetics")
    st.dataframe(beats.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download per-beat table (CSV)", beats.to_csv(index=False).encode(), "beat_kinetics.csv", "text/csv")
    tr = pd.DataFrame(dict(t_s=mo["t_s"], displacement_um=mo["displacement_um"], speed_um_s=mo["speed_um_s"]))
    st.download_button("⬇️ Download traces (CSV)", tr.to_csv(index=False).encode(), "motion_traces.csv", "text/csv")
else:
    ui.finding("No beats were detected. Weak or asynchronous contraction produces a flat field-averaged trace; try a region "
               "with visible beating, a higher frame rate, or check the reference-frame choice.", "warn")
with st.expander("Method details"):
    st.markdown("""
* **Optical flow:** OpenCV Farneback (pyramid scale 0.5, 3 levels, window 15 px, 3 iterations), computed on texture-rich pixels.
* **Reference frame:** the frame with the lowest smoothed frame-to-frame speed. Contraction shortens cells, so the flow from a
  relaxed reference to a contracted frame is convergent (negative divergence); if the divergence at peak displacement is
  positive, the reference was contracted and the frame of maximal displacement is used instead.
* **Beats:** peaks of the displacement trace with prominence ≥ 30 % of its 2–98 % range and ≥ 300 ms apart.
* **Kinetics:** time to peak and relaxation time between 10 % levels; CD50 = width at half amplitude; velocities are the
  extremes of d(displacement)/dt during contraction and relaxation.
* **Regional synchrony:** mean pairwise Pearson correlation of 16 tile traces; lag SD from cross-correlation with the field trace.
* Inspired by MUSCLEMOTION (Sala et al., *Circ Res* 2018). Validated on simulated movies with known tension (see Pipeline validation).
""")
