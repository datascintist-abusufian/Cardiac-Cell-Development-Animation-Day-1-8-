"""
Image and signal analysis. Every function here works on simulated data and on uploads.

Morphology    nuclei-seeded watershed segmentation of an alpha-actinin + DAPI image;
              per-cell area, aspect ratio, circularity, solidity, orientation, N:C ratio;
              nematic orientation order; nuclear condensation and fragment counts.
Sarcomeres    2-D Fourier analysis of the alpha-actinin signal inside each cell: sarcomere
              length from the dominant spatial frequency (1.3-2.6 um band) and a sarcomere
              organisation score (share of spectral power in the peak ring).
Motion        dense optical flow (Farneback) against a quiescent reference frame
              (MUSCLEMOTION-style, Sala et al. 2018) and between consecutive frames; beat
              detection and per-beat contraction kinetics; regional synchrony on a tile grid.
Calcium       transient amplitude (dF/F0), time to peak, CaD50/CaD90 and decay constant.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import signal

try:  # numpy >= 2
    _trapz = np.trapezoid
except AttributeError:  # pragma: no cover
    _trapz = np.trapz


# =========================================================================== morphology
SCORE_MIN = 0.22   # minimum spectral-ring share for a cell to count as striated


def segment(actinin, dapi, um_per_px):
    """Return (cell_labels, nuclei_labels, fragment_mask)."""
    from skimage import feature, filters, measure, morphology, segmentation
    a = filters.gaussian(actinin, 3)
    fg = a > filters.threshold_otsu(a) * 0.75
    fg = morphology.remove_small_objects(fg, int(80 / um_per_px ** 2))
    fg = ndi.binary_fill_holes(fg)
    n = filters.gaussian(dapi, 1.2)
    nmask = n > filters.threshold_otsu(n)
    nmask = morphology.remove_small_objects(nmask, int(2 / um_per_px ** 2))
    dist = ndi.distance_transform_edt(nmask)
    lab0 = measure.label(nmask)
    pk = feature.peak_local_max(dist, min_distance=max(2, int(3.0 / um_per_px)), labels=lab0, exclude_border=False)
    mk = np.zeros_like(lab0)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    nlab = segmentation.watershed(-dist, mk, mask=nmask)
    # small nuclear objects = fragments / debris (karyorrhexis, apoptotic bodies)
    areas = np.bincount(nlab.ravel())[1:] * um_per_px ** 2
    if len(areas):
        big = areas[areas >= 25]
        med = np.median(big) if len(big) else 25.0
        small = np.isin(nlab, np.nonzero((areas < 25) | (areas < 0.3 * med))[0] + 1)
    else:
        small = np.zeros_like(nmask)
    nuclei = np.where(small, 0, nlab)
    nuclei = measure.label(nuclei > 0) if nuclei.max() else nuclei
    markers = np.where(fg, nuclei, 0)
    cells = segmentation.watershed(-a, markers, mask=fg)
    return cells, nuclei, small


def morphometrics(actinin, dapi, um_per_px, cells=None, nuclei=None, fragments=None):
    """Per-cell table and field-level summary."""
    import pandas as pd
    from skimage import measure
    if cells is None:
        cells, nuclei, fragments = segment(actinin, dapi, um_per_px)
    props = measure.regionprops(cells, intensity_image=actinin)
    nprops = {r.label: r for r in measure.regionprops(nuclei, intensity_image=dapi)}
    rows = []
    for r in props:
        if r.area * um_per_px ** 2 < 60:
            continue
        nuc = nprops.get(r.label)
        maj, mnr = r.axis_major_length, max(r.axis_minor_length, 1e-6)
        rows.append(dict(
            cell=r.label, area_um2=r.area * um_per_px ** 2, major_um=maj * um_per_px, minor_um=mnr * um_per_px,
            aspect_ratio=maj / mnr, orientation_rad=-r.orientation + np.pi / 2,
            circularity=4 * np.pi * r.area / max(r.perimeter, 1) ** 2, solidity=r.solidity,
            actinin_mean=r.intensity_mean,
            nuclear_area_um2=(nuc.area * um_per_px ** 2) if nuc else np.nan,
            nuclear_intensity=nuc.intensity_mean if nuc else np.nan,
            centroid_y=r.centroid[0], centroid_x=r.centroid[1], bbox=r.bbox,
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df["nc_ratio"] = df.nuclear_area_um2 / df.area_um2
        sl = [sarcomere_spectrum(actinin, cells == c, um_per_px) for c in df.cell]
        df["sarcomere_score"] = [s["score"] for s in sl]
        # a sarcomere length is reported only where a periodic Z-disc pattern is detected
        df["striated"] = df.sarcomere_score >= SCORE_MIN
        df["sarcomere_length_um"] = [s["sarcomere_length_um"] if ok else np.nan
                                     for s, ok in zip(sl, df.striated)]
    n_frag = int(ndi.label(fragments)[1]) if fragments is not None else 0
    summary = dict(
        n_cells=len(df), n_nuclei=int(nuclei.max()), n_fragments=n_frag,
        coverage=float((cells > 0).mean()),
        mean_area_um2=float(df.area_um2.mean()) if len(df) else np.nan,
        mean_aspect_ratio=float(df.aspect_ratio.median()) if len(df) else np.nan,
        mean_circularity=float(df.circularity.median()) if len(df) else np.nan,
        orientation_order=float(np.abs(np.mean(np.exp(2j * df.orientation_rad)))) if len(df) else np.nan,
        nc_ratio=float(df.nc_ratio.median()) if len(df) else np.nan,
        # median over striated cells, reported only when at least 3 cells are striated
        sarcomere_length_um=float(np.nanmedian(df.sarcomere_length_um)) if len(df) and df.striated.sum() >= 3 else np.nan,
        sarcomere_score=float(np.nanmedian(df.sarcomere_score)) if len(df) else np.nan,
        striated_fraction=float(df.striated.mean()) if len(df) else np.nan,
        nuclear_condensation=float(np.nanmedian(df.nuclear_intensity / np.sqrt(df.nuclear_area_um2))) if len(df) else np.nan,
    )
    return df, summary, (cells, nuclei, fragments)


def sarcomere_spectrum(img, mask, um_per_px, band=(1.3, 2.6)):
    """Dominant sarcomere period inside `mask` from the 2-D power spectrum."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return dict(sarcomere_length_um=np.nan, score=np.nan, radial=None)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = img[y0:y1, x0:x1] * mask[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    patch = np.where(m, patch - patch[m].mean(), 0)
    n = int(2 ** np.ceil(np.log2(max(patch.shape) * 2)))
    P = np.abs(np.fft.fftshift(np.fft.fft2(patch, (n, n)))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, d=um_per_px))
    FX, FY = np.meshgrid(f, f)
    R = np.hypot(FX, FY)
    lo, hi = 1 / band[1], 1 / band[0]
    bins = np.linspace(0.15, 1 / (2 * um_per_px), 160)
    idx = np.digitize(R.ravel(), bins)
    radial = np.bincount(idx, P.ravel(), len(bins) + 1)[1:len(bins)] / np.maximum(np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)], 1)
    centers = (bins[:-1] + bins[1:]) / 2
    inband = (centers >= lo) & (centers <= hi)
    if not inband.any() or radial[inband].max() <= 0:
        return dict(sarcomere_length_um=np.nan, score=np.nan, radial=(centers, radial))
    ib = np.nonzero(inband)[0]
    k = ib[np.argmax(radial[inband])]
    if k in (ib[0], ib[-1]):          # maximum on the band edge = no periodic peak
        return dict(sarcomere_length_um=np.nan, score=0.0, radial=(centers, radial))
    # parabolic refinement
    if 0 < k < len(radial) - 1:
        y_1, y0_, y1_ = np.log(radial[k - 1] + 1e-12), np.log(radial[k] + 1e-12), np.log(radial[k + 1] + 1e-12)
        d = 0.5 * (y_1 - y1_) / (y_1 - 2 * y0_ + y1_ + 1e-12)
        fpk = centers[k] + d * (centers[1] - centers[0])
    else:
        fpk = centers[k]
    ring = np.abs(R - fpk) < 0.07
    total = P[(R > 0.15) & (R < 1 / (2 * um_per_px))].sum()
    score = float(P[ring].sum() / total) if total > 0 else np.nan
    return dict(sarcomere_length_um=float(1 / fpk), score=score, radial=(centers, radial))


# =========================================================================== motion
def _flow(a, b):
    import cv2
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def motion_analysis(frames, fps, um_per_px=1.0, tiles=4, max_frames=400):
    """Optical-flow motion analysis of a beating-cell movie (T, H, W) uint8.

    1. consecutive-frame flow (half resolution) gives a speed trace;
    2. the quiescent frame (lowest smoothed speed) is the first reference candidate;
    3. displacement = |flow(reference -> frame)|; contraction shortens cells, so the flow from a
       relaxed reference to a contracted frame converges (negative divergence). If the
       divergence at peak displacement is positive the reference was contracted, and the
       frame of maximal displacement (the relaxed state) is used as reference instead.
    """
    import cv2
    frames = np.asarray(frames)[:max_frames]
    if frames.ndim == 4:
        frames = frames.mean(-1).astype(np.uint8)
    T, H, W = frames.shape
    g = ndi.gaussian_gradient_magnitude(frames[0].astype(float), 2)
    mask = g > np.percentile(g, 40)
    small = [cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for f in frames]
    msmall = cv2.resize(mask.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
    speed = np.zeros(T)
    for t in range(1, T):
        fl = _flow(small[t - 1], small[t])
        speed[t] = np.hypot(fl[..., 0], fl[..., 1])[msmall].mean() * 2 * fps * um_per_px
    speed[0] = speed[1]
    sm = ndi.uniform_filter1d(speed, max(3, int(0.1 * fps)))

    def run(ref):
        disp = np.zeros(T)
        flows = []
        for t in range(T):
            fl = _flow(frames[ref], frames[t])
            flows.append(fl)
            disp[t] = np.hypot(fl[..., 0], fl[..., 1])[mask].mean() * um_per_px
        return disp, flows

    ref = int(np.argmin(sm))
    disp, flows = run(ref)
    tp = int(np.argmax(disp))
    div = (np.gradient(flows[tp][..., 0], axis=1) + np.gradient(flows[tp][..., 1], axis=0))[mask].mean()
    flipped = False
    if div > 0:                          # reference was the contracted state
        ref, flipped = tp, True
        disp, flows = run(ref)
    th, tw = H // tiles, W // tiles
    tile_tr = np.zeros((tiles * tiles, T))
    dmap = np.zeros((H, W))
    for t, fl in enumerate(flows):
        mag = np.hypot(fl[..., 0], fl[..., 1]) * um_per_px
        dmap = np.maximum(dmap, ndi.uniform_filter(mag, 5))
        for i in range(tiles):
            for j in range(tiles):
                m = mask[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
                tile_tr[i * tiles + j, t] = mag[i * th:(i + 1) * th, j * tw:(j + 1) * tw][m].mean() if m.any() else 0
    return dict(t_s=np.arange(T) / fps, displacement_um=disp, speed_um_s=speed, ref_frame=ref,
                reference_flipped=flipped, max_disp_map=dmap, tile_traces=tile_tr, mask=mask, fps=fps)


def detect_beats(trace, fps, min_interval_s=0.3, rel_prom=0.3):
    x = ndi.uniform_filter1d(np.asarray(trace, float), max(1, int(0.03 * fps)))
    rng = np.percentile(x, 98) - np.percentile(x, 2)
    if rng <= 0:
        return np.array([], int), x
    pk, _ = signal.find_peaks(x, prominence=rel_prom * rng, distance=max(1, int(min_interval_s * fps)))
    return pk, x


def beat_kinetics(trace, fps, speed=None):
    """Per-beat contraction metrics from a displacement trace (um)."""
    import pandas as pd
    pk, x = detect_beats(trace, fps)
    base = np.percentile(x, 5)
    rows = []
    for p in pk:
        amp = x[p] - base
        if amp <= 0:
            continue
        lo = p
        while lo > 0 and x[lo] > base + 0.1 * amp:
            lo -= 1
        hi = p
        while hi < len(x) - 1 and x[hi] > base + 0.1 * amp:
            hi += 1
        w50 = signal.peak_widths(x, [p], rel_height=0.5)[0][0] / fps
        dx = np.gradient(x) * fps
        rows.append(dict(t_peak_s=p / fps, amplitude_um=amp, time_to_peak_ms=(p - lo) / fps * 1000,
                         relaxation_ms=(hi - p) / fps * 1000, cd50_ms=w50 * 1000,
                         contraction_velocity_um_s=float(dx[lo:p + 1].max()) if p > lo else np.nan,
                         relaxation_velocity_um_s=float(-dx[p:hi + 1].min()) if hi > p else np.nan))
    df = pd.DataFrame(rows)
    rr = np.diff(pk) / fps
    summary = dict(n_beats=len(pk), beat_rate_bpm=60 / np.median(rr) if len(rr) else np.nan,
                   rr_cv=float(np.std(rr) / np.mean(rr)) if len(rr) > 1 else np.nan,
                   rmssd_ms=float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000) if len(rr) > 2 else np.nan,
                   **({k: float(df[k].median()) for k in df.columns if k != "t_peak_s"} if len(df) else {}))
    return df, summary, pk, x


def regional_synchrony(tile_traces, fps):
    """Mean pairwise correlation and contraction-lag dispersion across tiles."""
    tr = np.asarray(tile_traces, float)
    ok = tr.std(1) > 1e-6
    tr = tr[ok]
    if len(tr) < 2:
        return dict(sync_index=np.nan, lag_sd_ms=np.nan, corr=None)
    C = np.corrcoef(tr)
    iu = np.triu_indices(len(tr), 1)
    ref = tr.mean(0) - tr.mean()
    lags = []
    for x in tr:
        cc = signal.correlate(x - x.mean(), ref, mode="full")
        lag = (np.argmax(cc) - (len(ref) - 1)) / fps * 1000
        if abs(lag) < 500:
            lags.append(lag)
    return dict(sync_index=float(np.nanmean(C[iu])), lag_sd_ms=float(np.std(lags)) if lags else np.nan, corr=C)


def match_beats(det_s, true_s, tol_s=0.12):
    """Sensitivity / precision / timing error of detected vs ground-truth beat times."""
    det_s, true_s = np.asarray(det_s), np.asarray(true_s)
    used, errs = set(), []
    for t in true_s:
        if len(det_s) == 0:
            break
        k = int(np.argmin(np.abs(det_s - t)))
        if abs(det_s[k] - t) <= tol_s and k not in used:
            used.add(k)
            errs.append((det_s[k] - t) * 1000)
    tp = len(used)
    return dict(sensitivity=tp / max(len(true_s), 1), precision=tp / max(len(det_s), 1),
                timing_error_ms=float(np.mean(np.abs(errs))) if errs else np.nan)


# =========================================================================== calcium
def calcium_metrics(t_ms, F, min_interval_ms=300):
    """Transient metrics from a fluorescence trace (F or dF/F0 + 1)."""
    t_ms, F = np.asarray(t_ms, float), np.asarray(F, float)
    if len(F) < 10 or np.ptp(F) <= 0:
        return dict(n_transients=0), None
    F0 = np.percentile(F, 8)
    dff = (F - F0) / (abs(F0) if abs(F0) > 1e-6 else 1)
    dt = np.median(np.diff(t_ms))
    pk, _ = signal.find_peaks(dff, prominence=0.3 * np.ptp(dff), distance=max(1, int(min_interval_ms / dt)))
    rows = []
    for p in pk:
        amp = dff[p]
        lo = p
        while lo > 0 and dff[lo] > 0.1 * amp:
            lo -= 1
        def cross(level):
            k = p
            while k < len(dff) - 1 and dff[k] > level * amp:
                k += 1
            return k
        k50, k90 = cross(0.5), cross(0.1)
        seg = slice(cross(0.8), cross(0.2) + 1)
        y = dff[seg]
        tau = np.nan
        if len(y) > 3 and (y > 0).all():
            slope = np.polyfit(t_ms[seg], np.log(y), 1)[0]
            tau = -1 / slope if slope < 0 else np.nan
        rows.append(dict(t_peak_ms=t_ms[p], amplitude_dff=amp, time_to_peak_ms=t_ms[p] - t_ms[lo],
                         cad50_ms=t_ms[k50] - t_ms[lo], cad90_ms=t_ms[k90] - t_ms[lo], tau_decay_ms=tau))
    import pandas as pd
    df = pd.DataFrame(rows)
    if not len(df):
        return dict(n_transients=0), df
    ibi = np.diff(df.t_peak_ms.values)
    return dict(n_transients=len(df), rate_bpm=60000 / np.median(ibi) if len(ibi) else np.nan,
                **{k: float(df[k].median()) for k in df.columns if k != "t_peak_ms"}), df
