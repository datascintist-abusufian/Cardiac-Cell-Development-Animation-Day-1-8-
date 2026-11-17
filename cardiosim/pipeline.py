"""
End-to-end processing of one simulated well on one day:

    agents (model) -> lattice electrophysiology (ep) -> fluorescence image + beating movie (render)
                   -> segmentation, sarcomere spectra, optical flow, calcium kinetics (analysis)

`run_well_day` returns the measured readouts next to the ground truth so the pipeline can be
validated, plus the raw artefacts for display.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import analysis as A
from . import ep, render
from .model import CultureParams, cell_states, latent, orientation_order, phenotypes

FPS = 40
VIDEO_S = 5.0
VIDEO_PX = 256
FOV_UM = 276.0


def simulate_function(p: CultureParams, well, day, coupling_scale=1.0, N=32, duration_s=7.0, seed=None):
    cs = cell_states(well, day, p)
    M, D = latent(day, p, well.u, well.v)
    ph = phenotypes(M, D, p)
    seed = int(p.seed * 31 + well.index * 7 + day) if seed is None else seed
    tis = ep.build_tissue(cs, p, float(ph["coupling"]) * coupling_scale, N=N, seed=seed, rate_cv=float(ph["rate_cv"]))
    sim = ep.simulate(tis, p.rate_bpm, duration_s=duration_s, seed=seed)
    return cs, tis, sim, dict(M=float(M), D=float(D), **{k: float(v) for k, v in ph.items()})


def run_well_day(p: CultureParams, well, day, video=True, image=True, keep=False):
    cs, tis, sim, truth = simulate_function(p, well, day)
    out = dict(well=well.index, day=day, **{f"true_{k}": v for k, v in truth.items()},
               true_n_cells=len(cs), true_orientation_order=orientation_order(cs.theta.values),
               true_med_aspect_ratio=float(cs.aspect_ratio.median()) if len(cs) else np.nan,
               true_med_area_um2=float(cs.area_um2.median()) if len(cs) else np.nan,
               true_med_sarcomere_length=float(cs.sarcomere_length_um.median()) if len(cs) else np.nan,
               true_dying=int(cs.dying.sum()))
    art = dict(cells=cs, tissue=tis, sim=sim)
    # electrophysiology readouts
    fs = ep.functional_summary(tis, sim)
    out.update({k: v for k, v in fs.items() if np.isscalar(v)})
    ca, _ = ep.single_cell_calcium(tis, sim, seed=day)
    out.update({f"ca_{k}": v for k, v in ca.items()})
    art["functional"] = fs
    # image readouts
    if image:
        debris = int(((well.cells.death_day <= day) & (well.cells.attach_day <= day)).sum() * 6)
        img = render.fluorescence(cs, debris_n=debris, seed=int(day * 100 + well.index))
        cells_df, sm, masks = A.morphometrics(img["actinin"], img["dapi"], img["um_per_px"])
        out.update({f"img_{k}": v for k, v in sm.items()})
        art.update(image=img, cell_table=cells_df, masks=masks)
    # motion readouts
    if video:
        frames, t, gt = render.brightfield_video(cs, tis, sim, size=VIDEO_PX, fov_um=FOV_UM, fps=FPS,
                                                 seconds=VIDEO_S, seed=day)
        mo = A.motion_analysis(frames, FPS, um_per_px=FOV_UM / VIDEO_PX)
        beats, bsum, pk, _ = A.beat_kinetics(mo["displacement_um"], FPS)
        rs = A.regional_synchrony(mo["tile_traces"], FPS)
        gpk, _ = signal.find_peaks(gt, prominence=0.3 * max(np.ptp(gt), 1e-9), distance=int(0.3 * FPS))
        val = A.match_beats(pk / FPS, gpk / FPS)
        out.update({f"mot_{k}": v for k, v in bsum.items()})
        out.update(mot_sync_index=rs["sync_index"], mot_lag_sd_ms=rs["lag_sd_ms"],
                   val_beat_sensitivity=val["sensitivity"], val_beat_precision=val["precision"],
                   val_beat_timing_ms=val["timing_error_ms"], val_trace_corr=float(np.corrcoef(mo["displacement_um"], gt)[0, 1]) if np.ptp(gt) > 0 else np.nan,
                   true_n_beats=len(gpk))
        if keep:
            art.update(frames=frames, gt_tension=gt, motion=mo, beats=beats, peaks=pk)
    return out, (art if keep else None)
