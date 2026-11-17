"""Reading uploaded movies and images."""
from __future__ import annotations

import os
import tempfile

import numpy as np


def read_video(uploaded, max_frames=400, max_side=320):
    """Return (frames uint8 (T,H,W), fps). Supports mp4/avi/mov via OpenCV and multi-page TIFF/GIF."""
    import cv2
    name = getattr(uploaded, "name", "upload").lower()
    data = uploaded.read() if hasattr(uploaded, "read") else uploaded
    if name.endswith((".tif", ".tiff", ".gif")):
        from io import BytesIO

        from PIL import Image, ImageSequence
        im = Image.open(BytesIO(data))
        frames = [np.asarray(f.convert("L")) for f in ImageSequence.Iterator(im)][:max_frames]
        fps = 1000.0 / im.info["duration"] if im.info.get("duration") else np.nan
        arr = np.stack(frames)
    else:
        suffix = os.path.splitext(name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or np.nan
        frames = []
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        os.unlink(path)
        if not frames:
            raise ValueError("Could not decode the video. Try MP4 (H.264), AVI or a TIFF stack.")
        arr = np.stack(frames)
    T, H, W = arr.shape
    s = max(H, W) / max_side
    if s > 1:
        import cv2
        arr = np.stack([cv2.resize(f, (int(W / s), int(H / s)), interpolation=cv2.INTER_AREA) for f in arr])
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / max(np.ptp(arr), 1)).astype(np.uint8)
    return arr, float(fps), float(s if s > 1 else 1.0)


def read_image(uploaded):
    """Return a float32 array (H, W) or (H, W, C) scaled to [0, 1]."""
    from io import BytesIO

    from PIL import Image
    data = uploaded.read() if hasattr(uploaded, "read") else uploaded
    im = Image.open(BytesIO(data))
    arr = np.asarray(im)
    if arr.dtype == bool:
        arr = arr.astype(np.float32)
    arr = arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    lo, hi = np.percentile(arr, [0.5, 99.8])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
