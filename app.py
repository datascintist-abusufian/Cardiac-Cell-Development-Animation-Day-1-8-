"""
Cardiomyocyte Maturation & Degeneration Lab — home page.

An in-silico human iPSC-cardiomyocyte culture followed from day 1 to day 8, measured with a
real image- and video-analysis pipeline, analysed with longitudinal statistics, and probed
with virtual experiments. The same analysis tools accept your own microscopy images and
beating-cell videos.
"""
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from cardiosim import ui
from cardiosim.stats import hill_fit, turning_point

ui.setup("Cardiomyocyte Maturation & Degeneration Lab")
ui.hero("🫀 Cardiomyocyte Maturation &amp; Degeneration Lab",
        "An in-silico hiPSC-cardiomyocyte culture (day 1–8) · biophysical electrophysiology · synthetic microscopy · "
        "validated image &amp; motion analysis · longitudinal statistics · virtual experiments")

study, cells, dose, morris, meta = ui.load_data()

st.markdown(f"""
Cardiomyocytes derived from human induced pluripotent stem cells (hiPSC-CMs) **mature** for several days after
plating — they elongate, align, build striated sarcomeres, couple through gap junctions and begin to beat in
synchrony — and then, in long or stressed cultures, **degenerate**. This app turns that narrative into a
quantitative, reproducible model {ui.sim_badge()} and analyses it the way a real imaging study would be analysed:

1. a **stochastic agent model** of {meta['params']['n_wells']} replicate wells (cell attachment, maturation, damage, death),
2. a **coupled-oscillator electrophysiology model** on the cell layout (FitzHugh–Nagumo lattice with gap-junction coupling,
   calcium release and active tension),
3. **synthetic α-actinin/DAPI images and bright-field beating movies** rendered from the model,
4. an **analysis pipeline** (segmentation, sarcomere spectra, optical-flow motion, calcium kinetics) whose output is
   **validated against the known ground truth**,
5. **longitudinal statistics** (mixed models, change-points, Gaussian-process forecasts, state-space trajectories,
   survival) and **in-silico experiments** (gap-junction blocker dose–response, culture stress, global sensitivity).

The analysis pages also accept **your own** beating-cell videos and fluorescence images.
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ headline numbers
by_day = study.groupby("day").mean(numeric_only=True)
tp = turning_point(study, "kuramoto_R", n_boot=200)
hf = hill_fit(dose.conc_uM.values, dose.kuramoto_R.values)
ok = study.dropna(subset=["img_sarcomere_length_um"])
r_sl = np.corrcoef(ok.img_sarcomere_length_um, ok.true_med_sarcomere_length)[0, 1]
mature = study[study.day.between(4, 7)]
k = st.columns(5)
ui.kpi(k[0], "Peak synchrony", f"R = {by_day.kuramoto_R.max():.2f}", f"day {int(by_day.kuramoto_R.idxmax())} (mean of {meta['params']['n_wells']} wells)")
ui.kpi(k[1], "Synchrony turning point", f"day {tp['tau']:.1f}", f"95% CI {tp['ci'][0]:.1f}–{tp['ci'][1]:.1f} (cluster bootstrap)")
ui.kpi(k[2], "Striated cells", f"{by_day.img_striated_fraction.min():.0%} → {by_day.img_striated_fraction.max():.0%}",
       "day 1 → peak, from image spectra")
ui.kpi(k[3], "Functional IC₅₀ (synchrony)", f"{hf['ic50']:.1f} µM", f"molecular IC₅₀ {meta['true_ic50_uM']} µM · CI {hf['ic50_ci'][0]:.1f}–{hf['ic50_ci'][1]:.1f}")
ui.kpi(k[4], "Pipeline validation", f"r = {r_sl:.2f}", f"sarcomere length vs truth · beat sensitivity {mature.val_beat_sensitivity.mean():.0%} (days 4–7)")

st.markdown("")
# ------------------------------------------------------------------ pipeline diagram
st.markdown("##### How the study is built")
steps = [("1 · Agent model", "Wells with random effects; cells attach, mature (logistic M), accumulate damage (logistic D) and die with hazard h₀ + h₁D."),
         ("2 · Electrophysiology", "FitzHugh–Nagumo oscillators on the cell layout, gap-junction coupling ∝ maturation; Ca²⁺ release gate; tension ∝ Ca³ × sarcomere order."),
         ("3 · Synthetic microscopy", "α-actinin striations at the true sarcomere length, DAPI nuclei (pyknosis, karyorrhexis), debris, blur and Poisson noise; contracting bright-field movies."),
         ("4 · Measurement", "Watershed segmentation, 2-D FFT sarcomere analysis, Farneback optical flow, beat kinetics, single-cell Ca²⁺ kinetics."),
         ("5 · Inference", "Mixed models, hinge change-points, GP forecasts, GMM states, Kaplan–Meier, Hill fits, Morris sensitivity — all checked against ground truth.")]
cols = st.columns(len(steps))
for c, (h, t) in zip(cols, steps):
    c.markdown(f'<div class="step"><b>{h}</b><br><span class="note">{t}</span></div>', unsafe_allow_html=True)

st.markdown("")
# ------------------------------------------------------------------ timelapse
c1, c2 = st.columns([1.05, 1])
with c1:
    st.markdown("##### Culture timelapse — α-actinin (green) and DAPI (blue)")
    well = st.selectbox("Replicate well", list(range(meta["params"]["n_wells"])), format_func=lambda w: f"Well {w + 1}")

    @st.cache_data(show_spinner="Rendering eight days of microscopy…")
    def timelapse(w):
        from cardiosim import render
        from cardiosim.model import cell_states
        p, wells = ui.default_study()
        frames = []
        for d in range(1, 9):
            cs = cell_states(wells[w], d, p)
            debris = int(((wells[w].cells.death_day <= d) & (wells[w].cells.attach_day <= d)).sum() * 6)
            im = Image.fromarray(render.fluorescence(cs, debris_n=debris, seed=d * 100 + w)["rgb"]).resize((420, 420))
            dr = ImageDraw.Draw(im)
            dr.rectangle([8, 8, 128, 40], fill=(0, 0, 0))
            dr.text((16, 14), f"Day {d}", fill=(255, 255, 255), font_size=22)
            dr.rectangle([420 - 20 - 76, 400, 420 - 20, 404], fill=(255, 255, 255))   # 50 um scale bar
            frames.append(np.asarray(im))
        return render.to_gif(np.stack(frames), fps=1.1)
    st.image(timelapse(well), caption="Field of view 276 µm; scale bar 50 µm. Rendered from the agent model of the selected well.")
with c2:
    st.markdown("##### Where to go next")
    pages = [("pages/1_Culture_Simulator.py", "🧫 Culture simulator", "any well, any day; design your own culture"),
             ("pages/2_Electrophysiology_and_Calcium.py", "⚡ Electrophysiology & calcium", "waves, synchrony, Ca²⁺ kinetics"),
             ("pages/3_Longitudinal_Phenotyping.py", "📈 Longitudinal phenotyping", "mixed models, turning points, states"),
             ("pages/4_Pipeline_Validation.py", "✅ Pipeline validation", "measured vs ground truth"),
             ("pages/5_Motion_Analysis.py", "🎥 Motion analysis", "upload a beating-cell video"),
             ("pages/6_Morphology_and_Sarcomeres.py", "🔬 Morphology & sarcomeres", "upload a fluorescence image"),
             ("pages/7_In_silico_Experiments.py", "🧪 In-silico experiments", "dose–response, stress, sensitivity"),
             ("pages/8_Methods.py", "📑 Methods & limitations", "equations, parameters, references")]
    for path, label, desc in pages:
        st.page_link(path, label=f"**{label}** — {desc}")
    st.markdown("")
    ui.finding("All culture data on these pages are <b>simulated</b> with parameters chosen inside ranges reported for "
               "hiPSC-CMs; they illustrate methods and hypotheses and are not experimental measurements. "
               "The motion and morphology pages analyse real uploads with the same validated code.", "warn")

st.markdown("---")
st.markdown(f"<div class='note' style='text-align:center'>cardiosim v3 · default study pre-computed with "
            f"<code>scripts/precompute.py</code> ({meta['created_utc'][:10]}) · Md Abu Sufian · "
            f"<a href='https://orcid.org/0009-0007-3503-6942'>ORCID 0009-0007-3503-6942</a></div>", unsafe_allow_html=True)
