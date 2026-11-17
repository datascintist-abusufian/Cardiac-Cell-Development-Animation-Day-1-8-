"""Methods, parameters, assumptions, limitations and references."""
import pandas as pd
import streamlit as st

from cardiosim import ui
from cardiosim.model import CultureParams

ui.setup("Methods · Cardiomyocyte Lab", "📑")
ui.hero("📑 Methods &amp; limitations", "Everything needed to understand, reproduce and criticise the results in this app.")
study, cells, dose, morris, meta = ui.load_data()

st.markdown("### 1 · Culture model")
st.markdown("Each replicate well *w* has random effects $u_w, v_w \\sim \\mathcal N(0, s^2)$ that shift its maturation and damage "
            "timing. Two latent processes drive all phenotypes:")
st.latex(r"M_w(t)=\frac{1}{1+e^{-k_m\,(t-t_m-u_w)}},\qquad D_w(t)=\frac{1}{1+e^{-k_d\,(t-t_d-v_w)}}")
st.markdown("Cells attach after an exponential delay, carry a cell-intrinsic maturation offset, and die with hazard "
            "$h(t)=h_0+h_1D(t)$ (death times by inverse-transform sampling of the cumulative hazard with log-normal frailty). "
            "A cell enters an apoptotic phase (condensed or fragmented nucleus, reduced signal, inexcitable) 0.9 days before "
            "detachment. Ground-truth phenotypes are smooth functions of $M$ and $D$, e.g.")
st.latex(r"g_{gap}=g_{max}(0.08+0.92M)(1-0.85D),\quad SL=1.62+0.30M-0.22D\ \mu m,\quad \tau_{Ca}=420-190M+260D\ ms")
p = CultureParams()
desc = {
    "n_wells": ("replicate wells", "experimental unit"), "n_days": ("days followed", ""),
    "cells_seeded": ("cells in the imaged field", "276 × 276 µm field"),
    "attach_mean_days": ("mean attachment delay", "days"),
    "mat_midpoint": ("day of half-maximal maturation", "early hiPSC-CM structural maturation over days–weeks"),
    "mat_rate": ("maturation slope", "1/day"), "dmg_onset": ("day of half-maximal damage", "culture stress / degeneration"),
    "dmg_rate": ("damage slope", "1/day"), "base_hazard": ("baseline death hazard", "1/day"),
    "dmg_hazard": ("extra hazard at full damage", "1/day"), "g_max": ("maximal relative gap-junction coupling", "Cx43-like"),
    "pacemaker_fraction": ("fraction of faster (pacemaker) cells", "+25 % intrinsic rate"),
    "rate_bpm": ("reference spontaneous rate", "hiPSC-CMs typically ~30–80 bpm"),
    "rate_cv_immature": ("cell-to-cell rate CV when immature", "narrows with maturation"),
    "well_sd_days": ("SD of well random effects", "days"), "seed": ("random seed", "reproducibility"),
}
st.dataframe(pd.DataFrame([dict(Parameter=k, Value=v, Meaning=desc.get(k, ("", ""))[0], Note=desc.get(k, ("", ""))[1])
                           for k, v in p.to_dict().items()]), hide_index=True, use_container_width=True)

st.markdown("### 2 · Electrophysiology, calcium and tension")
st.markdown("A 32 × 32 lattice covers the field; each node belongs to the nearest living cell (Voronoi assignment within the "
            "cell's reach) or is empty. Every excitable node is a FitzHugh–Nagumo oscillator with diffusive gap-junction coupling:")
st.latex(r"\dot v_i=v_i-\tfrac{v_i^3}{3}-w_i+I+\sum_{j\in\mathcal N(i)} g_{ij}(v_j-v_i),\qquad \dot w_i=\varepsilon_i\,(v_i+a-bw_i)")
st.markdown("with $a=0.7$, $b=0.8$, $I=0.5$ (oscillatory regime) and $\\varepsilon_i=\\varepsilon_0 f_i$; because the period scales "
            "≈ 1/ε, $f_i$ sets each cell's intrinsic rate ($f_i \\sim 1+\\text{CV}\\,z_i$, pacemakers × 1.25). Edge conductances are "
            "log-normal around the day's coupling (more heterogeneous when immature), 3.0 within a cell, 0 to empty nodes. Model "
            "time is calibrated so the reference oscillator beats at the reference rate. Calcium uses a release gate $r$ "
            "(set to 1 at each upstroke, inactivating with τ = 70 ms):")
st.latex(r"\dot{Ca}=r\,\frac{1-Ca}{\tau_{on}}-\frac{Ca}{\tau_{decay}},\qquad \frac{F}{F_0}=1+A\,Ca,\qquad T=SO\cdot\frac{Ca^3}{Ca^3+K^3}")
st.markdown("Synchrony is the time-averaged Kuramoto order parameter $R=|\\langle e^{i\\varphi_j(t)}\\rangle_j|$ with phases "
            "interpolated between each node's activation times.")

st.markdown("### 3 · Synthetic microscopy")
st.markdown("""
* **Fluorescence (768 × 768 px, 0.36 µm/px):** each cell is a soft ellipse with orientation drawn from a von Mises
  distribution whose concentration grows with maturation. α-actinin Z-discs are a sharpened cosine along the cell axis at
  the cell's sarcomere length; phase noise, myofibril bending and punctate Z-bodies increase as sarcomere order falls.
  Nuclei are ellipses (pyknotic: small and bright; karyorrhectic: 3–5 fragments). Debris scales with cumulative deaths.
  Optics: Gaussian PSF (σ = 1 px), Poisson photon noise and Gaussian read noise.
* **Bright-field movies (256 × 256 px, 40 fps, 5 s):** cells shorten along their long axis by up to 13 % × their mean
  nodal tension and widen slightly; internal texture moves with the material, so optical flow sees real deformation.
""")
st.markdown("### 4 · Analysis pipeline")
st.markdown("""
Segmentation, sarcomere spectra, optical flow, beat kinetics and calcium kinetics are described on the analysis pages
(🎥 Motion analysis and 🔬 Morphology & sarcomeres, *Method details*). The same functions process simulated and uploaded data.
""")
st.markdown("### 5 · Statistics")
st.markdown("""
* **Experimental unit:** the well (n = 6 per day); images and cells are sub-samples.
* **Day effect:** linear mixed model `y ~ C(day) + (1 | well)` fitted by ML, likelihood-ratio test vs `y ~ 1 + (1 | well)`;
  Benjamini–Hochberg FDR across phenotypes; intraclass correlation of wells.
* **Effect sizes:** Hedges' g (small-sample corrected) vs a reference day.
* **Turning points:** continuous hinge regression, grid search over the breakpoint, cluster (well) bootstrap 95 % CI.
* **Forecasts:** Gaussian-process regression, kernel C·RBF + white noise, hyper-parameters by marginal likelihood.
* **States:** PCA of standardised phenotypes; Gaussian mixture on PC1–PC2 with k chosen by BIC.
* **Survival:** Kaplan–Meier with Greenwood intervals; log-rank test; censoring at day 8.
* **Dose–response:** four-parameter Hill model by non-linear least squares; IC₅₀ CI by the delta method on log IC₅₀.
* **Sensitivity:** Morris elementary effects (μ*, σ), 4 levels.
* **Validation:** Pearson r, Lin's concordance correlation, Bland–Altman bias and limits of agreement.
""")

st.markdown("### 6 · Limitations")
ui.finding("""<ul style="margin:0">
<li><b>Simulated data.</b> Parameters are plausible, not fitted to a specific experiment; conclusions are about methods and
mechanisms, not about a particular cell line.</li>
<li><b>Phenomenological electrophysiology.</b> FitzHugh–Nagumo reproduces excitability, oscillation, entrainment and wave
propagation but not ionic currents; conduction velocities and action-potential shapes are in model units. Replacing it with a
Paci-type hiPSC-CM ionic model would make drug and APD read-outs quantitative.</li>
<li><b>2-D, single field.</b> No 3-D tissue, no extracellular matrix stiffness, no fibroblasts, no mechanical coupling between cells.</li>
<li><b>Segmentation bias.</b> Intensity watershed under-estimates elongation in confluent layers (see Pipeline validation);
deep instance segmentation (e.g. Cellpose) would reduce it.</li>
<li><b>Small n.</b> Six wells give wide confidence intervals for effect sizes and turning points, as in many real studies.</li>
</ul>""", "warn")

st.markdown("### 7 · Reproducibility")
st.markdown(f"Default study generated by `python scripts/precompute.py` on {meta['created_utc'][:10]} in {meta['runtime_s']:.0f} s "
            f"(seed {meta['params']['seed']}). Software: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + ".")

st.markdown("### 8 · References")
st.markdown("""
1. FitzHugh R. Impulses and physiological states in theoretical models of nerve membrane. *Biophys J* 1961;1:445–466.
2. Nagumo J, Arimoto S, Yoshizawa S. An active pulse transmission line simulating nerve axon. *Proc IRE* 1962;50:2061–2070.
3. Kuramoto Y. *Chemical Oscillations, Waves, and Turbulence.* Springer, 1984.
4. Paci M, Hyttinen J, Aalto-Setälä K, Severi S. Computational models of ventricular- and atrial-like human induced pluripotent stem cell derived cardiomyocytes. *Ann Biomed Eng* 2013;41:2334–2348.
5. Lundy SD, Zhu W-Z, Regnier M, Laflamme MA. Structural and functional maturation of cardiomyocytes derived from human pluripotent stem cells. *Stem Cells Dev* 2013;22:1991–2002.
6. Karbassi E, Fenix A, Marchiano S, et al. Cardiomyocyte maturation: advances in knowledge and implications for regenerative medicine. *Nat Rev Cardiol* 2020;17:341–359.
7. Pasqualini FS, Sheehy SP, Agarwal A, Aratyn-Schaus Y, Parker KK. Structural phenotyping of stem cell-derived cardiomyocytes. *Stem Cell Reports* 2015;4:340–347.
8. Sala L, van Meer BJ, Tertoolen LGJ, et al. MUSCLEMOTION: a versatile open software tool to quantify cardiomyocyte and cardiac muscle contraction in vitro and in vivo. *Circ Res* 2018;122:e5–e16.
9. Farnebäck G. Two-frame motion estimation based on polynomial expansion. *Scandinavian Conference on Image Analysis* 2003:363–370.
10. Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. *Nat Methods* 2021;18:100–106.
11. Morris MD. Factorial sampling plans for preliminary computational experiments. *Technometrics* 1991;33:161–174.
12. Campolongo F, Cariboni J, Saltelli A. An effective screening design for sensitivity analysis of large models. *Environ Model Softw* 2007;22:1509–1518.
13. Rasmussen CE, Williams CKI. *Gaussian Processes for Machine Learning.* MIT Press, 2006.
14. Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. *J Am Stat Assoc* 1958;53:457–481.
15. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet* 1986;1:307–310.
16. Lin LI. A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 1989;45:255–268.
""")
