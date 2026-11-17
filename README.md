# Cardiomyocyte Maturation & Degeneration Lab (Day 1–8)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abusufian-cardiac-morphology-development.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red)

An in-silico human iPSC-cardiomyocyte (hiPSC-CM) culture followed from day 1 to day 8, imaged with synthetic
microscopy, measured with a real image- and video-analysis pipeline that is **validated against the model's ground
truth**, analysed with longitudinal statistics, and probed with virtual experiments. The analysis pages also accept
**your own** beating-cell videos and fluorescence images.

**Live app:** https://abusufian-cardiac-morphology-development.streamlit.app/

> All culture data are simulated with parameters chosen inside ranges reported for hiPSC-CMs. They illustrate
> methods and mechanisms; they are not experimental measurements.

## What the app does

| Page | Content |
|---|---|
| 🫀 **Home** | Study overview, headline results, day-by-day fluorescence timelapse |
| 🧫 **Culture simulator** | Any well on any day: α-actinin/DAPI image, beating bright-field movie, measured vs true readouts, per-cell table; design your own culture and run the full pipeline on it |
| ⚡ **Electrophysiology & calcium** | Voltage-wave movie, activation map, activation raster, Kuramoto synchrony R(t), single-cell Ca²⁺ transients (amplitude, time to peak, CaD50/90, decay τ), trajectories across days |
| 📈 **Longitudinal phenotyping** | 22 phenotypes × 6 wells × 8 days: mixed-model tests with FDR and ICC, Hedges' g heatmap, hinge-regression turning points with cluster-bootstrap CIs, Gaussian-process forecasts, PCA + Gaussian-mixture cell states (BIC), Kaplan–Meier survival, phenotype correlations |
| ✅ **Pipeline validation** | Measured vs ground truth: Pearson r, Lin's CCC, Bland–Altman bias and limits of agreement, beat-detection sensitivity/precision by day |
| 🎥 **Motion analysis** | Upload a video (MP4/AVI/MOV/TIFF/GIF) or use a simulated one: Farneback optical flow with automatic relaxed-reference selection, beat detection, contraction/relaxation kinetics, RR variability (Poincaré), displacement map, regional synchrony |
| 🔬 **Morphology & sarcomeres** | Upload an image or use a simulated one: nuclei-seeded watershed segmentation, shape descriptors, nematic orientation order, 2-D FFT sarcomere length and organisation score, per-cell spectra |
| 🧪 **In-silico experiments** | Gap-junction blocker dose–response with Hill fit, culture stress with Kaplan–Meier/log-rank, Morris global sensitivity analysis |
| 📑 **Methods** | Equations, parameter table, statistics, limitations, references |

## Model in brief

* **Agents:** replicate wells with random effects; cells attach, mature (logistic *M*), accumulate damage (logistic *D*)
  and die with hazard *h₀ + h₁D*; apoptotic cells show condensed or fragmented nuclei.
* **Electrophysiology:** FitzHugh–Nagumo oscillators on the Voronoi map of the living cells, gap-junction coupling that
  grows with maturation and falls with damage, maturation-dependent rate heterogeneity → Kuramoto-type synchronisation.
  Calcium release gate, SERCA-like decay, active tension ∝ Ca³ × sarcomere order.
* **Synthetic microscopy:** α-actinin Z-disc striations at the true sarcomere length with order-dependent disorder,
  DAPI nuclei, debris, PSF blur, Poisson–Gaussian noise; bright-field movies in which each cell shortens with its tension.

## Selected results (default study, 6 wells)

* Synchrony rises from R ≈ 0.3 (day 1) to a peak around day 5–6, then collapses as damage removes excitable cells.
* Striated cells rise from ≈ 1 % to > 75 %; measured sarcomere length agrees closely with the ground truth.
* Intensity-watershed segmentation **under-estimates cell elongation** in confluent layers (attenuation slope < 1), and
  optical-flow beat detection fails on weak, asynchronous days — both quantified on the validation page.
* The functional IC₅₀ of a gap-junction blocker measured from synchrony (~8 µM) is ~5× higher and steeper than the
  molecular IC₅₀ (1.5 µM), because entrainment needs coupling above a critical value.

## Repository structure

```
app.py                     # home page
pages/                     # 8 Streamlit pages
cardiosim/
  model.py                 # latent states, agents, survival, ground-truth phenotypes
  ep.py                    # FitzHugh–Nagumo lattice, calcium, tension, synchrony, activation maps
  render.py                # synthetic fluorescence images and bright-field movies
  analysis.py              # segmentation, sarcomere FFT, optical flow, beat and calcium kinetics
  stats.py                 # mixed models, change-points, GP, GMM states, survival, Morris, Hill, Bland–Altman
  pipeline.py              # one well-day end to end (simulate → image → measure)
  io.py, ui.py             # upload readers, shared UI helpers
scripts/precompute.py      # regenerates data/ (≈ 8 min on 2 cores)
data/                      # pre-computed default study
```

## Running locally

```bash
pip install -r requirements.txt        # Python 3.10–3.12
streamlit run app.py
python scripts/precompute.py           # optional: regenerate data/
```

## Limitations

Phenomenological electrophysiology (conduction velocities in model units), a single 2-D field, no fibroblasts or
mechanics between cells, and simulated rather than measured data. See the Methods page for details.

## Author

Md Abu Sufian · ORCID [0009-0007-3503-6942](https://orcid.org/0009-0007-3503-6942)
