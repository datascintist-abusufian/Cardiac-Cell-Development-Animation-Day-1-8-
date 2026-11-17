"""
Synthetic microscopy of the in-silico culture.

fluorescence(...)  two-channel immunofluorescence-like image (alpha-actinin, DAPI) with
                   Z-disc striations at the cell's sarcomere length, disorder that grows as
                   sarcomere order falls (punctate Z-bodies in immature cells), condensed or
                   fragmented nuclei in dying cells, debris, optical blur and Poisson-Gaussian
                   noise. Also returns the ground-truth instance masks.
brightfield_video  phase-contrast-like movie in which every cell shortens along its long axis
                   in proportion to the active tension of its lattice nodes (ep.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _smooth_noise(shape, sigma, rng):
    z = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return z / (z.std() + 1e-9)


def _local_coords(cx, cy, theta, half_len, shape):
    """Bounding box and rotated local coordinates (u along the long axis, v across)."""
    H, W = shape
    r = int(np.ceil(half_len)) + 3
    x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    c, s = np.cos(theta), np.sin(theta)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return (slice(y0, y1), slice(x0, x1)), u, v


def fluorescence(cells, debris_n=0, size=768, um_per_px=0.36, seed=0, noise=0.035):
    """Render alpha-actinin (ch0) and DAPI (ch1). Returns dict(actinin, dapi, rgb, labels, nuclei_labels)."""
    rng = np.random.default_rng(seed)
    shape = (size, size)
    act = np.zeros(shape, np.float32)
    dapi = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int32)
    nlabels = np.zeros(shape, np.int32)
    best = np.zeros(shape, np.float32)
    for k, c in enumerate(cells.itertuples(), start=1):
        cx, cy = c.x_um / um_per_px, c.y_um / um_per_px
        a = c.major_um / 2 / um_per_px
        b = c.minor_um / 2 / um_per_px
        loc = _local_coords(cx, cy, c.theta, a, shape)
        if loc is None:
            continue
        sl, u, v = loc
        # slightly irregular outline
        wob = 1 + 0.08 * np.sin(3 * np.arctan2(v, u) + c.phase * 6.28)
        r = (u / a) ** 2 + (v / b) ** 2
        body = 1 / (1 + np.exp(np.clip((r - wob) * 9, -50, 50)))
        if c.dying:
            body *= 0.55 + 0.3 * rng.random()
        # sarcomeres: Z-discs along u with period SL, disorder grows as order falls
        so = float(c.sarcomere_order)
        sl_px = c.sarcomere_length_um / um_per_px
        disorder = (1 - so) ** 1.5 * 2.8 * _smooth_noise(u.shape, max(2, sl_px * 0.8), rng)
        bend = 0.18 * (1 - so) * _smooth_noise(u.shape, 10, rng)        # local myofibril bending
        uu = u + bend * v
        zline = ((1 + np.cos(2 * np.pi * uu / sl_px + disorder + 6.28 * c.phase)) / 2) ** 6
        # myofibril bundles: banded across v
        bundles = 0.55 + 0.45 * ((1 + np.cos(2 * np.pi * v / (sl_px * 1.3) + 3 * c.phase)) / 2)
        puncta = np.clip(_smooth_noise(u.shape, 1.2, rng) - 1.2, 0, None)  # Z-bodies (immature)
        pattern = so * zline * bundles + (1 - so) * 0.9 * puncta + 0.22
        sig = body * pattern * (0.75 + 0.25 * rng.random())
        act[sl] = np.maximum(act[sl], sig)
        upd = body > best[sl]
        labels[sl][upd & (body > 0.5)] = k
        best[sl] = np.maximum(best[sl], body)
        # nucleus
        na = c.nuclear_area_um2 / um_per_px ** 2
        nb = np.sqrt(na / (np.pi * 1.35))
        nA = nb * 1.35
        off_u, off_v = rng.normal(0, a * 0.1), rng.normal(0, b * 0.1)
        if c.dying and rng.random() < 0.5:     # karyorrhexis: fragmented nucleus
            for _ in range(rng.integers(3, 6)):
                fu, fv = off_u + rng.normal(0, nA * 0.9), off_v + rng.normal(0, nb * 0.9)
                fr = nb * rng.uniform(0.25, 0.45)
                blob = np.exp(-(((u - fu) ** 2 + (v - fv) ** 2) / (2 * fr ** 2)))
                dapi[sl] = np.maximum(dapi[sl], 1.6 * blob)
        else:
            rn = ((u - off_u) / nA) ** 2 + ((v - off_v) / nb) ** 2
            nuc = 1 / (1 + np.exp(np.clip((rn - 1) * 12, -50, 50)))
            chrom = 0.75 + 0.25 * np.clip(_smooth_noise(u.shape, 1.5, rng), -2, 2) / 2
            bright = 1.7 if c.dying else 1.0          # pyknosis: small and bright
            dapi[sl] = np.maximum(dapi[sl], nuc * chrom * bright)
            nlabels[sl][nuc > 0.5] = k
    # debris: small bright DAPI fragments and actinin specks
    for _ in range(int(debris_n)):
        x, y = rng.uniform(0, size, 2)
        r = rng.uniform(1.2, 3.2)
        loc = _local_coords(x, y, 0, r * 3, shape)
        if loc is None:
            continue
        sl, u, v = loc
        blob = np.exp(-(u ** 2 + v ** 2) / (2 * r ** 2))
        dapi[sl] = np.maximum(dapi[sl], blob * rng.uniform(0.8, 1.8))
        act[sl] = np.maximum(act[sl], blob * rng.uniform(0.2, 0.7))
    # optics + noise (Poisson-Gaussian)
    def acquire(img, photons):
        img = ndi.gaussian_filter(img, 1.0) + 0.04
        img = rng.poisson(np.clip(img, 0, None) * photons) / photons
        return np.clip(img + rng.normal(0, noise, img.shape), 0, None).astype(np.float32)
    act, dapi = acquire(act, 60), acquire(dapi, 80)
    rgb = np.stack([0.25 * act, act, np.clip(dapi * 1.1, 0, None)], -1)
    rgb = np.clip(rgb / np.percentile(rgb, 99.7), 0, 1)
    return dict(actinin=act, dapi=dapi, rgb=(rgb * 255).astype(np.uint8), labels=labels,
                nuclei_labels=nlabels, um_per_px=um_per_px)


def brightfield_video(cells, tissue, sim, size=256, fov_um=276.0, fps=50, seconds=4.0,
                      start_s=2.0, contraction=0.13, seed=0):
    """Phase-contrast-like movie; each cell shortens along its axis with its nodes' tension."""
    rng = np.random.default_rng(seed)
    um_px = fov_um / size
    t_frames = sim["t_frames"]
    times = start_s * 1000 + np.arange(int(seconds * fps)) * 1000 / fps
    idx = np.clip(np.searchsorted(t_frames, times), 0, len(t_frames) - 1)
    # per-cell tension time course = mean tension of its lattice nodes
    cid = tissue["cell_id"]
    tens = {}
    for c in cells.itertuples():
        m = cid == c.cell
        tens[c.cell] = sim["tension"][idx][:, m].mean(1) if m.any() else np.zeros(len(idx))
    bg = 0.55 + 0.03 * _smooth_noise((size, size), 20, rng)
    textures = {c.cell: rng.random() for c in cells.itertuples()}
    frames = np.empty((len(idx), size, size), np.uint8)
    for f in range(len(idx)):
        img = bg.copy()
        for c in cells.itertuples():
            T = float(tens[c.cell][f])
            a = c.major_um / 2 / um_px * (1 - contraction * T)
            b = c.minor_um / 2 / um_px * (1 + 0.35 * contraction * T)
            loc = _local_coords(c.x_um / um_px, c.y_um / um_px, c.theta, a, (size, size))
            if loc is None:
                continue
            sl, u, v = loc
            r = (u / a) ** 2 + (v / b) ** 2
            body = 1 / (1 + np.exp(np.clip((r - 1) * 7, -50, 50)))
            halo = np.exp(-((np.sqrt(r) - 1.05) ** 2) / 0.006)
            # texture that moves with the material (coordinates scaled by the deformation)
            un, vn = u / (1 - contraction * T), v / (1 + 0.35 * contraction * T)
            ph = textures[c.cell] * 50
            tex = 0.5 * np.sin(un * 0.9 + ph) * np.sin(vn * 1.3 + 2 * ph) + 0.5 * np.sin(0.45 * un + 0.7 * vn + ph)
            img[sl] = img[sl] - 0.22 * body + 0.08 * body * tex + 0.35 * halo
        img = img + rng.normal(0, 0.012, img.shape)
        frames[f] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    gt_tension = np.mean([tens[c] for c in tens], axis=0) if tens else np.zeros(len(idx))
    return frames, times - times[0], gt_tension


def to_gif(frames, fps=25, scale=1.0, max_frames=200, levels=None):
    """Encode a (T,H,W) or (T,H,W,3) uint8 array as an animated GIF (bytes)."""
    from io import BytesIO

    from PIL import Image
    frames = frames[:max_frames]
    if levels and frames.ndim == 3:   # quantise grey movies for much smaller files
        step = 256 // levels
        frames = (ndi.gaussian_filter(frames.astype(np.float32), (0, 0.8, 0.8)) // step * step).astype(np.uint8)
    ims = []
    for fr in frames:
        im = Image.fromarray(fr)
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
        ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=128) if im.mode == "RGB" else im)
    buf = BytesIO()
    ims[0].save(buf, format="GIF", save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    return buf.getvalue()
