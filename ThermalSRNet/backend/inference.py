"""Model singleton + full inference pipeline (Part B)."""
import os, io, base64, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from model import ThermalSRNet
from metrics import compute_psnr, compute_ssim, compute_mae, compute_rmse
import temperature
import physics

DEVICE = "cpu"
_HERE = os.path.dirname(__file__)
_CKPT_DIR = os.path.join(_HERE, "checkpoints")


def _pick_checkpoint():
    """THERMAL_CKPT env var (file name in checkpoints/ or a full path) wins; otherwise prefer
    the temperature-fixed weights, then fall back to the original best checkpoint."""
    override = os.environ.get("THERMAL_CKPT")
    if override:
        return override if os.path.isabs(override) else os.path.join(_CKPT_DIR, override)
    for name in ("thermalsrnet_best_tempfix.pt", "thermalsrnet_best.pt"):
        path = os.path.join(_CKPT_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(_CKPT_DIR, "thermalsrnet_best.pt")


_CKPT = _pick_checkpoint()
CKPT_NAME = os.path.basename(_CKPT)

# ---------------------------------------------------------------------------
# Singleton: build the model once and load weights once at import time.
# ---------------------------------------------------------------------------
def _extract_state_dict(obj):
    """Accept a raw state_dict, common wrappers, or a full pickled model/nn.Module."""
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj  # assume it already IS the state_dict
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _strip_prefixes(sd):
    """Remove 'module.' (DataParallel) or '_orig_mod.' (torch.compile) prefixes."""
    out = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "_orig_mod."):
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


_model = ThermalSRNet()
MODEL_LOADED = False
LOAD_ERROR = None

if not os.path.exists(_CKPT):
    LOAD_ERROR = (
        f"Checkpoint not found at {_CKPT}. Put thermalsrnet_best_tempfix.pt (or "
        f"thermalsrnet_best.pt) in backend/checkpoints/, then restart."
    )
    print(f"[MODEL] {LOAD_ERROR}")
else:
    try:
        raw = torch.load(_CKPT, map_location=DEVICE)
        sd = _strip_prefixes(_extract_state_dict(raw))
        missing, unexpected = _model.load_state_dict(sd, strict=False)
        _model.eval()
        # One warm-up pass: PyTorch initialises its CPU kernels on the first call, which
        # would otherwise inflate the first measured inference time ~10x.
        with torch.no_grad():
            _model(torch.zeros(1, 1, 32, 32), torch.zeros(1, 3, 128, 128))
        MODEL_LOADED = True
        print(f"[MODEL] Loaded {_CKPT}")
        if missing:
            print(f"[MODEL] WARNING missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[MODEL] WARNING unexpected keys ({len(unexpected)}): {unexpected[:8]}")
        if missing or unexpected:
            print("[MODEL] Key mismatch means backend model.py != the architecture that was "
                  "trained. Ensure model.py matches your Colab definition exactly.")
    except Exception as e:
        LOAD_ERROR = f"Failed to load {_CKPT}: {type(e).__name__}: {e}"
        MODEL_LOADED = False
        print(f"[MODEL] {LOAD_ERROR}")


# ---------------------------------------------------------------------------
# Colormap + base64 helpers (Part B4)
# ---------------------------------------------------------------------------
def apply_inferno(t):
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    a = np.squeeze(a).astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    rgb = (cm.inferno(a)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(data_url):
    raw = data_url.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(raw)))


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
def _minmax(t):
    """Per-image (per-channel) min-max to [0,1] over spatial dims — matches the training
    normalize(): (t - amin) / (amax - amin + 1e-6)."""
    mn = t.amin(dim=(-2, -1), keepdim=True)
    mx = t.amax(dim=(-2, -1), keepdim=True)
    return (t - mn) / (mx - mn + 1e-6)


def _optical_tensor(pil_img):
    img = pil_img.convert("RGB").resize((128, 128), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)                  # (128,128,3), 0..255
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,128,128)
    t = _minmax(t)                                           # per-channel min-max, as in training
    return t, img


def _looks_colormapped(pil_img):
    """A raw thermal is single-channel (or R==G==B). If the channels diverge, the PNG
    was almost certainly saved with a colormap (e.g. cmap='inferno'), which destroys the
    thermal signal when converted to grayscale. Return True to warn the caller."""
    if pil_img.mode not in ("RGB", "RGBA"):
        return False
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.int16)
    # Max per-pixel channel spread; ~0 means a true grayscale stored as RGB.
    spread = (arr.max(axis=2) - arr.min(axis=2)).mean()
    return spread > 6.0


HR_SIZE = 128            # model output / ground-truth size
LR_SIZE = HR_SIZE // 4   # model input size (x4)


def _thermal_tensors(pil_img):
    """Fair evaluation protocol.

    * Thermal >= 128x128  -> it is a REAL high-res image: use it as ground truth (GT) and make the
      low-res input from it with the SAME bicubic x4 degradation used in training.
    * Thermal  < 128x128  -> there is no real high-res data, so the image itself is the low-res
      input and NO ground truth is invented. (Enlarging a small image to make a fake GT gives a
      bicubic image, which bicubic then "matches" perfectly - an unfair test.)

    Returns (lr, gt_or_None, lo, hi). lo/hi = raw 8-bit range, used to map back to temperature.
    NOTE: thermal must be a RAW grayscale export, not a colormapped PNG.
    """
    gray = pil_img.convert("L")
    w, h = gray.size
    if w >= HR_SIZE and h >= HR_SIZE:
        hr_arr = np.asarray(gray.resize((HR_SIZE, HR_SIZE), Image.BICUBIC), dtype=np.float32)
        # Normalize the HR patch ONCE (per-image min-max) - exactly as in training.
        lo, hi = float(hr_arr.min()), float(hr_arr.max())
        gt_t = torch.from_numpy((hr_arr - lo) / (hi - lo + 1e-6))[None, None]   # (1,1,128,128)
        # LR = normalized HR, bicubic-downsampled x4 (same operator as training; do NOT re-normalize).
        lr_t = F.interpolate(gt_t, size=(LR_SIZE, LR_SIZE), mode="bicubic", align_corners=False).clamp(0, 1)
        return lr_t, gt_t, lo, hi

    lr_arr = np.asarray(gray.resize((LR_SIZE, LR_SIZE), Image.BICUBIC), dtype=np.float32)
    lo, hi = float(lr_arr.min()), float(lr_arr.max())
    lr_t = torch.from_numpy((lr_arr - lo) / (hi - lo + 1e-6))[None, None]          # (1,1,32,32)
    return lr_t, None, lo, hi


def _predict(optical_pil, thermal_pil, alpha=1.0):
    """Preprocess, run the model and the bicubic baseline, timing each separately.

    alpha = residual strength. The model outputs SR = bicubic + residual, so
    out = bicubic + alpha * residual (0 = pure bicubic, 1 = the model as trained)."""
    if _looks_colormapped(thermal_pil):
        print("[WARN] Thermal image looks COLORMAPPED (non-grayscale channels). Re-export "
              "demo_thermal.png as a RAW single-channel grayscale (cmap='gray'); a colormapped "
              "PNG scrambles the thermal signal and the fusion output will be wrong.")
    optical_t, optical_img = _optical_tensor(optical_pil)
    lr_t, gt_t, lo, hi = _thermal_tensors(thermal_pil)

    with torch.no_grad():
        t0 = time.perf_counter()
        sr_full = _model(lr_t, optical_t)                                  # (1,1,128,128)
        base = F.interpolate(lr_t, size=(HR_SIZE, HR_SIZE), mode="bicubic", align_corners=False)
        sr = (base + alpha * (sr_full - base)).clamp(0, 1)                 # residual blend
        t1 = time.perf_counter()
        bicubic = F.interpolate(lr_t, size=(HR_SIZE, HR_SIZE), mode="bicubic",
                                align_corners=False).clamp(0, 1)            # (1,1,128,128)
        t2 = time.perf_counter()

    w, h = thermal_pil.size
    fair = gt_t is not None
    return {
        "optical_img": optical_img, "lr": lr_t, "gt": gt_t, "lo": lo, "hi": hi,
        "sr": sr, "bicubic": bicubic, "alpha": float(alpha),
        "model_ms": (t1 - t0) * 1000.0, "bicubic_ms": (t2 - t1) * 1000.0,
        "thermal_size": [w, h],
        "fair": fair,
        "mode": "hr_reference" if fair else "lr_input",
    }


_GT_KEYS = ("mae", "rmse", "psnr", "ssim", "mae_c", "rmse_c", "max_error_c", "bias_c", "within_tol_pct")


def _score(pred, p, t_min, t_max, tolerance):
    """Metrics for one prediction (model or bicubic). Ground-truth metrics only exist when a real
    high-res ground truth exists (fair mode); physics consistency needs no ground truth."""
    gt, lr = p["gt"], p["lr"]
    phys = physics.consistency(pred, lr)
    out = {
        "physics": round(phys, 5),
        "physics_c": round(temperature.norm_to_celsius_delta(phys, p["lo"], p["hi"], t_min, t_max), 3),
    }
    if gt is None:
        out.update({k: None for k in _GT_KEYS})
        return out
    out.update({
        "mae": round(compute_mae(pred, gt), 5),
        "rmse": round(compute_rmse(pred, gt), 5),
        "psnr": round(compute_psnr(pred, gt), 2),
        "ssim": round(compute_ssim(pred, gt), 4),
    })
    out.update(temperature.celsius_errors(pred, gt, p["lo"], p["hi"], t_min, t_max, tolerance))
    return out


def compare(p, t_min=temperature.DEFAULT_T_MIN_C, t_max=temperature.DEFAULT_T_MAX_C,
            tolerance=temperature.DEFAULT_TOLERANCE_C):
    """Bicubic vs proposed model on the same image, plus timings."""
    bic = _score(p["bicubic"], p, t_min, t_max, tolerance)
    ours = _score(p["sr"], p, t_min, t_max, tolerance)
    bic["time_ms"] = round(p["bicubic_ms"], 2)
    ours["time_ms"] = round(p["model_ms"], 2)
    return {"bicubic": bic, "ours": ours, "alpha": p["alpha"], "fair": p["fair"],
            "mode": p["mode"], "thermal_size": p["thermal_size"], "min_hr_size": HR_SIZE}


def evaluate_pair(optical_pil, thermal_pil, t_min=temperature.DEFAULT_T_MIN_C,
                  t_max=temperature.DEFAULT_T_MAX_C, tolerance=temperature.DEFAULT_TOLERANCE_C,
                  alpha=1.0):
    """Metrics only (no images) - used by the test-set evaluation."""
    if not MODEL_LOADED:
        raise RuntimeError("model_not_loaded")
    return compare(_predict(optical_pil, thermal_pil, alpha), t_min, t_max, tolerance)


def run_inference(optical_pil, thermal_pil, t_min=temperature.DEFAULT_T_MIN_C,
                  t_max=temperature.DEFAULT_T_MAX_C, tolerance=temperature.DEFAULT_TOLERANCE_C,
                  alpha=1.0):
    if not MODEL_LOADED:
        raise RuntimeError("model_not_loaded")

    t0 = time.time()
    p = _predict(optical_pil, thermal_pil, alpha)
    optical_img, lr_t, gt_t, lo, hi = p["optical_img"], p["lr"], p["gt"], p["lo"], p["hi"]
    sr, bicubic = p["sr"], p["bicubic"]
    elapsed_ms = round((time.time() - t0) * 1000.0, 1)
    cmp_ = compare(p, t_min, t_max, tolerance)

    if p["fair"]:
        # Temperature preservation: real HR ground truth vs predicted HR.
        temp_stats, temp_imgs = temperature.analyze(sr, gt_t, lo, hi, t_min, t_max, tolerance,
                                                    reference="hr", baseline=bicubic)
    else:
        # No HR ground truth: check every low-res pixel keeps its temperature, by shrinking the
        # prediction back to the input size (energy conservation) and comparing with the input.
        temp_stats, temp_imgs = temperature.analyze(physics.degrade(sr), lr_t, lo, hi, t_min, t_max,
                                                    tolerance, reference="input")

    b, o = cmp_["bicubic"], cmp_["ours"]

    def diff(a, c, d):
        return round(a - c, d) if a is not None and c is not None else None

    return {
        "comparison": cmp_,
        "optical_b64": to_b64(optical_img),
        "lr_thermal_b64": to_b64(apply_inferno(lr_t)),
        "bicubic_b64": to_b64(apply_inferno(bicubic)),
        "sr_thermal_b64": to_b64(apply_inferno(sr)),
        "gt_thermal_b64": to_b64(apply_inferno(gt_t)) if gt_t is not None else None,
        "metrics": {
            "psnr_bicubic": b["psnr"],
            "psnr_ours": o["psnr"],
            "ssim_bicubic": b["ssim"],
            "ssim_ours": o["ssim"],
            "improvement_psnr": diff(o["psnr"], b["psnr"], 2),
            "improvement_ssim": diff(o["ssim"], b["ssim"], 4),
        },
        "temperature": temp_stats,
        "temp_hr_b64": to_b64(temp_imgs["hr"]),
        "temp_pred_b64": to_b64(temp_imgs["pred"]),
        "temp_error_b64": to_b64(temp_imgs["error"]),
        "processing_time_ms": elapsed_ms,
    }
