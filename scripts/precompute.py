"""
Pre-compute the default in-silico study shown in the app (about 5-8 minutes on 2 CPU cores).

    python scripts/precompute.py            # writes data/*.csv and data/*.json

Outputs
    data/study.csv          6 wells x 8 days: ground truth + pipeline-measured readouts
    data/cells.csv          every cell agent (attachment and death day) for survival analysis
    data/dose_response.csv  gap-junction blocker series on day 5 (3 wells x 7 concentrations)
    data/morris.csv         Morris elementary effects of 6 model parameters on 5 outputs
    data/meta.json          parameters, true IC50, software versions, run time
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cardiosim import ep  # noqa: E402
from cardiosim.model import CultureParams, cell_states, make_study  # noqa: E402
from cardiosim.pipeline import run_well_day, simulate_function  # noqa: E402
from cardiosim.stats import morris  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
P = CultureParams()
WELLS = make_study(P)
TRUE_IC50, TRUE_HILL = 1.5, 1.3
CONC = [0, 0.1, 0.3, 1, 3, 10, 30]


def _study(job):
    w, d = job
    out, _ = run_well_day(P, WELLS[w], d)
    print(f"  well {w} day {d} done", flush=True)
    return out


def _dose(job):
    w, c = job
    scale = 1 / (1 + (c / TRUE_IC50) ** TRUE_HILL) if c > 0 else 1.0
    cs, tis, sim, truth = simulate_function(P, WELLS[w], 5, coupling_scale=scale)
    fs = ep.functional_summary(tis, sim)
    return dict(well=w, conc_uM=c, coupling_scale=scale, **{k: v for k, v in fs.items() if np.isscalar(v)})


NAMES = ["mat_midpoint", "mat_rate", "dmg_onset", "dmg_hazard", "g_max", "rate_cv_immature"]
BOUNDS = [(2.5, 4.5), (0.8, 2.0), (5.5, 7.5), (0.2, 0.9), (0.5, 1.5), (0.10, 0.45)]


def model_outputs(x):
    p = CultureParams(**{**P.to_dict(), **x, "n_wells": 1, "seed": 5})
    well = make_study(p)[0]
    Rs, alive = [], []
    for d in range(1, 9):
        cs, tis, sim, _ = simulate_function(p, well, d, N=24, duration_s=5.0)
        Rs.append(ep.functional_summary(tis, sim)["kuramoto_R"])
        alive.append(len(cs) / p.cells_seeded)
    Rs, alive = np.array(Rs), np.array(alive)
    days = np.arange(1, 9)
    return dict(peak_synchrony=float(Rs.max()), day_of_peak_synchrony=float(days[np.argmax(Rs)]),
                functional_window_days=float((Rs >= 0.7).sum()), mean_viability=float(alive.mean()),
                day8_viability=float(alive[-1]))


def _morris_chunk(seed):
    df, _ = morris(model_outputs, BOUNDS, NAMES, r=4, levels=4, seed=seed)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(w, d) for w in range(P.n_wells) for d in range(1, P.n_days + 1)]
    with Pool(2) as pool:
        rows = pool.map(_study, jobs)
    pd.DataFrame(rows).sort_values(["well", "day"]).to_csv(os.path.join(OUT, "study.csv"), index=False)
    print(f"study done in {time.time() - t0:.0f}s", flush=True)
    cells = pd.concat([w.cells.assign(well=w.index, u=w.u, v=w.v) for w in WELLS])
    cells.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    with Pool(2) as pool:
        dose = pool.map(_dose, [(w, c) for w in range(3) for c in CONC])
    pd.DataFrame(dose).to_csv(os.path.join(OUT, "dose_response.csv"), index=False)
    print(f"dose-response done in {time.time() - t0:.0f}s", flush=True)
    with Pool(2) as pool:
        parts = pool.map(_morris_chunk, [1, 2])
    # merge chunks: average mu* / sigma over the two independent sets of trajectories
    m = pd.concat(parts).groupby(["output", "parameter"]).mean(numeric_only=True).reset_index()
    m.to_csv(os.path.join(OUT, "morris.csv"), index=False)
    import cv2
    import scipy
    import skimage
    import sklearn
    meta = dict(params=P.to_dict(), true_ic50_uM=TRUE_IC50, true_hill=TRUE_HILL, concentrations_uM=CONC,
                morris=dict(names=NAMES, bounds=BOUNDS, trajectories=8, levels=4),
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                              skimage=skimage.__version__, opencv=cv2.__version__),
                runtime_s=round(time.time() - t0, 1),
                created_utc=pd.Timestamp.utcnow().isoformat(timespec="seconds"))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
