"""Morphology and sarcomere analysis of fluorescence images (upload or simulated)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cardiosim import analysis as A
from cardiosim import io, ui

ui.setup("Morphology & sarcomeres · Cardiomyocyte Lab", "🔬")
ui.hero("🔬 Morphology &amp; sarcomere analysis", "Nuclei-seeded watershed segmentation, per-cell shape descriptors, "
        "myofibril alignment and Fourier-based sarcomere length and organisation.")

with st.sidebar:
    src = st.radio("Image source", ["Simulated well-day", "Upload an image"])
    if src == "Simulated well-day":
        well = st.selectbox("Replicate well", list(range(6)), format_func=lambda w: f"Well {w + 1}")
        day = st.slider("Culture day", 1, 8, 5)
    else:
        up = st.file_uploader("Fluorescence image (TIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
        um_px = st.number_input("Pixel size (µm / px)", 0.02, 5.0, 0.20, 0.01,
                                help="Needed for areas and sarcomere length. Sarcomeres need ≤ 0.4 µm/px to be resolved.")
        st.caption("Composite images: choose which colour channel holds α-actinin (or another Z-disc / myofibril stain) "
                   "and which holds the nuclear stain.")

if src == "Simulated well-day":
    out, art = ui.well_day(well, day)
    act, dapi, um_per_px = art["image"]["actinin"], art["image"]["dapi"], art["image"]["um_per_px"]
    rgb = art["image"]["rgb"]
else:
    if up is None:
        st.info("Upload an immunofluorescence image of cardiomyocytes (e.g. α-actinin + DAPI), or switch to a simulated well-day.")
        st.stop()
    img = io.read_image(up)
    um_per_px = um_px
    if img.ndim == 3:
        names = ["red", "green", "blue"][: img.shape[-1]]
        c1, c2 = st.sidebar.columns(2)
        ca = c1.selectbox("Myofibril channel", names, index=1 if len(names) > 1 else 0)
        cn = c2.selectbox("Nuclear channel", names, index=2 if len(names) > 2 else 0)
        act, dapi = img[..., names.index(ca)], img[..., names.index(cn)]
        rgb = (img[..., :3] * 255).astype(np.uint8) if img.shape[-1] >= 3 else None
    else:
        st.warning("Single-channel image: nuclei are detected as the brightest compact objects in the same channel.")
        act = dapi = img
        rgb = (np.stack([img] * 3, -1) * 255).astype(np.uint8)
    if max(act.shape) > 1600:
        st.warning("Large image — analysing the central 1600 × 1600 px.")
        h, w = act.shape
        sl = (slice(max(0, h // 2 - 800), h // 2 + 800), slice(max(0, w // 2 - 800), w // 2 + 800))
        act, dapi = act[sl], dapi[sl]
        rgb = rgb[sl] if rgb is not None else None


@st.cache_data(show_spinner="Segmenting cells and nuclei, computing sarcomere spectra…", max_entries=8)
def run(_act, _dapi, key, um_per_px):
    return A.morphometrics(_act, _dapi, um_per_px)


key = f"{src}-{act.shape}-{float(act[::17, ::17].sum()):.4f}"
cells_df, summ, (lab, nuc, frag) = run(act, dapi, key, um_per_px)

k = st.columns(6)
ui.kpi(k[0], "Cells", f"{summ['n_cells']}", f"{summ['n_fragments']} nuclear fragments")
ui.kpi(k[1], "Median area", f"{summ['mean_area_um2']:.0f} µm²", f"coverage {summ['coverage']:.0%}")
ui.kpi(k[2], "Aspect ratio", f"{summ['mean_aspect_ratio']:.2f}", f"circularity {summ['mean_circularity']:.2f}")
ui.kpi(k[3], "Orientation order", f"{summ['orientation_order']:.2f}", "0 isotropic · 1 aligned")
ui.kpi(k[4], "Striated cells", f"{summ.get('striated_fraction', np.nan):.0%}", f"score ≥ {A.SCORE_MIN}")
ui.kpi(k[5], "Sarcomere length", f"{summ['sarcomere_length_um']:.2f} µm" if np.isfinite(summ['sarcomere_length_um']) else "—",
       "median of striated cells")

c1, c2 = st.columns(2)
with c1:
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    base = rgb.copy() if rgb is not None else (np.stack([act] * 3, -1) * 255).astype(np.uint8)
    b = find_boundaries(lab, mode="outer")
    base[b] = [255, 255, 0]
    base[find_boundaries(nuc, mode="outer")] = [255, 0, 255]
    base[frag > 0] = [255, 80, 80]
    st.image(base, use_container_width=True, caption="Yellow: cells · magenta: nuclei · red: nuclear fragments / debris")
with c2:
    st.image((label2rgb(lab, bg_label=0, alpha=1) * 255).astype(np.uint8), use_container_width=True, caption="Cell instances")

if len(cells_df):
    c1, c2, c3 = st.columns(3)
    with c1:
        th = np.mod(cells_df.orientation_rad, np.pi)
        hist, edges = np.histogram(np.r_[th, th + np.pi], bins=36, range=(0, 2 * np.pi))
        fig = go.Figure(go.Barpolar(r=hist, theta=np.degrees((edges[:-1] + edges[1:]) / 2), marker_color=ui.PALETTE[0]))
        fig.update_layout(polar=dict(angularaxis=dict(direction="counterclockwise")))
        ui.show(ui.style(fig, 330, legend=False, title="Cell-axis orientations (axial rose)"))
    with c2:
        fig = px.scatter(cells_df, x="area_um2", y="aspect_ratio", color="sarcomere_score", color_continuous_scale="Viridis",
                         labels=dict(area_um2="area (µm²)", aspect_ratio="aspect ratio", sarcomere_score="sarcomere score"))
        ui.show(ui.style(fig, 330, legend=False, title="Shape vs sarcomere organisation"))
    with c3:
        fig = px.histogram(cells_df.dropna(subset=["sarcomere_length_um"]), x="sarcomere_length_um", nbins=20,
                           color_discrete_sequence=[ui.PALETTE[1]])
        fig.update_layout(xaxis_title="sarcomere length (µm)")
        ui.show(ui.style(fig, 330, legend=False, title="Sarcomere length of striated cells"))

    st.markdown("##### Sarcomere spectrum of a single cell")
    cid = st.selectbox("Cell", cells_df.cell.tolist(), format_func=lambda c: f"cell {c} · score "
                       f"{cells_df.set_index('cell').loc[c, 'sarcomere_score']:.2f}")
    sp = A.sarcomere_spectrum(act, lab == cid, um_per_px)
    r = cells_df.set_index("cell").loc[cid]
    y0, x0, y1, x1 = r.bbox
    c1, c2 = st.columns([1, 1.6])
    with c1:
        patch = (np.clip(act[y0:y1, x0:x1] * (lab[y0:y1, x0:x1] == cid), 0, 1) * 255).astype(np.uint8)
        st.image(patch, use_container_width=True, caption=f"cell {cid}: area {r.area_um2:.0f} µm², AR {r.aspect_ratio:.2f}")
    with c2:
        if sp["radial"] is not None:
            f, pw = sp["radial"]
            fig = go.Figure(go.Scatter(x=f, y=pw, line=dict(color=ui.PALETTE[0], width=2)))
            fig.add_vrect(x0=1 / 2.6, x1=1 / 1.3, fillcolor=ui.PALETTE[2], opacity=0.1, line_width=0,
                          annotation_text="sarcomere band", annotation_position="top left")
            if np.isfinite(sp["sarcomere_length_um"]):
                fig.add_vline(x=1 / sp["sarcomere_length_um"], line_dash="dot", line_color=ui.PALETTE[1],
                              annotation_text=f"SL = {sp['sarcomere_length_um']:.2f} µm", annotation_position="bottom right")
            fig.update_layout(xaxis_title="spatial frequency (cycles / µm)", yaxis_title="radial power", yaxis_type="log")
            ui.show(ui.style(fig, 320, legend=False))
    show_cols = ["cell", "area_um2", "aspect_ratio", "circularity", "solidity", "nc_ratio", "sarcomere_length_um", "sarcomere_score", "striated"]
    st.dataframe(cells_df[[c for c in show_cols if c in cells_df]].round(3), use_container_width=True, hide_index=True, height=280)
    st.download_button("⬇️ Download per-cell table (CSV)", cells_df.drop(columns=["bbox"]).to_csv(index=False).encode(),
                       "cell_morphometrics.csv", "text/csv")
with st.expander("Method details"):
    st.markdown(f"""
* **Nuclei:** Gaussian smoothing (σ = 1.2 px), Otsu threshold, distance-transform watershed to split touching nuclei.
  Objects < 25 µm² or < 30 % of the median nuclear area are classed as **fragments / apoptotic bodies**.
* **Cells:** Gaussian smoothing (σ = 3 px), 0.75 × Otsu foreground, marker-controlled watershed on the inverted
  myofibril signal seeded by the nuclei.
* **Shape:** area, major/minor axis from second moments, aspect ratio, circularity 4πA/P², solidity.
  **Orientation order** S = |⟨e^(2iθ)⟩| over cell axes (nematic order parameter).
* **Sarcomeres:** mean-subtracted myofibril signal inside each cell → zero-padded 2-D FFT → radial power profile.
  The peak inside 1.3–2.6 µm gives the sarcomere length (parabolic refinement); the **score** is the share of
  spectral power within ± 0.07 cycles/µm of that peak. Cells with score ≥ {A.SCORE_MIN} count as striated; edge peaks are rejected.
* Validated on simulated images with known sarcomere length and alignment (see Pipeline validation).
""")
