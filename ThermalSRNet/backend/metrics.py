"""PSNR + SSIM metrics on single-channel float images in [0, 1]."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _to_np(x):
    """Accept a torch tensor of shape (1,1,H,W)/(1,H,W)/(H,W) or an ndarray."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    x = np.squeeze(x)
    return np.clip(x, 0.0, 1.0)


def compute_mae(pred, target):
    p, t = _to_np(pred), _to_np(target)
    return float(np.abs(p - t).mean())


def compute_rmse(pred, target):
    p, t = _to_np(pred), _to_np(target)
    return float(np.sqrt(((p - t) ** 2).mean()))


def compute_psnr(pred, target):
    p, t = _to_np(pred), _to_np(target)
    # data_range is 1.0 because images are normalized to [0, 1]
    if np.allclose(p, t):
        return 100.0
    return float(sk_psnr(t, p, data_range=1.0))


def compute_ssim(pred, target):
    p, t = _to_np(pred), _to_np(target)
    h, w = t.shape[-2:]
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    return float(sk_ssim(t, p, data_range=1.0, win_size=win))
