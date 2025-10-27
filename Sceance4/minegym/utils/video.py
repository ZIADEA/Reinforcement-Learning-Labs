import os, time, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
from PIL import Image
# ----------------- outils vidéo -----------------
def downscale_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return rgb
    h, w = rgb.shape[:2]
    nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
    img = Image.fromarray(rgb)
    img = img.resize((nw, nh), resample=Image.BILINEAR)
    return np.asarray(img)

def canvas_to_rgb(fig):
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
        rgba = np.asarray(renderer.buffer_rgba())
        return rgba[:, :, :3]
    except Exception:
        pass
    if hasattr(fig.canvas, "tostring_rgb"):
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)
    if hasattr(fig.canvas, "tostring_argb"):
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, [1, 2, 3]]
    raise RuntimeError("Backend Matplotlib non supporté.")