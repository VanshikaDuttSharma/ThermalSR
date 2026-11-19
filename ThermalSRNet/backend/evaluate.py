"""Test-set evaluation: bicubic vs the proposed model on every image pair available.

Test pairs come from:
  * backend/test_data/<name>_optical.<png|jpg|tif>  +  <name>_thermal.<png|jpg|tif>
  * backend/demo/demo_optical.png + demo_thermal.png (always included if present)

Add more Landsat pairs to backend/test_data/ (never ones used for training) to make the test
results meaningful. Thermal images should be RAW grayscale and at least 128x128: only those have
a real high-res ground truth and count towards the averages. Smaller thermals are used as the
low-res input only (no ground-truth metrics), so they cannot distort the comparison.
The test always uses alpha = 1.0, i.e. the model exactly as trained.

Run from the command line to also save the results:   python evaluate.py
The API serves the same results at GET /api/evaluation (cached until files change).
"""
import os, json, glob, time
import numpy as np
from PIL import Image

import inference as inf

_HERE = os.path.dirname(__file__)
TEST_DIR = os.path.join(_HERE, "test_data")
DEMO_DIR = os.path.join(_HERE, "demo")
OUT_PATH = os.path.join(_HERE, "evaluation.json")
_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

METRIC_KEYS = ("mae", "rmse", "psnr", "ssim", "mae_c", "rmse_c", "max_error_c",
               "within_tol_pct", "physics", "physics_c", "time_ms")

_cache = {"key": None, "result": None}


def find_pairs():
    """Return [(name, optical_path, thermal_path), ...] sorted by name."""
    pairs = []
    demo_o = os.path.join(DEMO_DIR, "demo_optical.png")
    demo_t = os.path.join(DEMO_DIR, "demo_thermal.png")
    if os.path.exists(demo_o) and os.path.exists(demo_t):
        pairs.append(("demo", demo_o, demo_t))
    if os.path.isdir(TEST_DIR):
        for opt in sorted(glob.glob(os.path.join(TEST_DIR, "*_optical.*"))):
            if not opt.lower().endswith(_EXTS):
                continue
            name = os.path.basename(opt).rsplit("_optical.", 1)[0]
            thermal = next((os.path.join(TEST_DIR, f"{name}_thermal{e}") for e in _EXTS
                            if os.path.exists(os.path.join(TEST_DIR, f"{name}_thermal{e}"))), None)
            if thermal:
                pairs.append((name, opt, thermal))
    return pairs


def _mean_std(rows, model, key):
    vals = [r[model][key] for r in rows if r[model].get(key) is not None]
    if not vals:
        return None, None
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


def evaluate():
    pairs = find_pairs()
    if not inf.MODEL_LOADED:
        raise RuntimeError(inf.LOAD_ERROR or "model not loaded")
    rows = []
    for name, o, t in pairs:
        r = inf.evaluate_pair(Image.open(o), Image.open(t))
        r["name"] = name
        rows.append(r)

    # Fair protocol: only images with a real >=128px ground truth count towards the averages.
    fair_rows = [r for r in rows if r["fair"]]
    summary = {}
    for model in ("bicubic", "ours"):
        summary[model] = {}
        for k in METRIC_KEYS:
            m, s = _mean_std(fair_rows, model, k)
            summary[model][k] = m
            summary[model][f"{k}_std"] = s

    return {
        "checkpoint": inf.CKPT_NAME,
        "n_images": len(rows),
        "n_fair": len(fair_rows),
        "n_lr_only": len(rows) - len(fair_rows),
        "alpha": 1.0,
        "test_dir": "backend/test_data",
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_image": rows,
        "summary": summary,
    }


def _files_key():
    # Checkpoint mtime too: a retrained model saved under the same name must be re-evaluated.
    ckpt_mtime = os.path.getmtime(inf._CKPT) if os.path.exists(inf._CKPT) else None
    return (inf.CKPT_NAME, ckpt_mtime) + tuple((n, os.path.getmtime(o), os.path.getmtime(t)) for n, o, t in find_pairs())


def cached_evaluation():
    """Re-evaluate only when the checkpoint or test files change."""
    key = _files_key()
    if _cache["key"] != key:
        _cache["result"] = evaluate()
        _cache["key"] = key
    return _cache["result"]


if __name__ == "__main__":
    res = evaluate()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Evaluated {res['n_images']} pair(s) with {res['checkpoint']} -> {OUT_PATH}")
    for model in ("bicubic", "ours"):
        s = res["summary"][model]
        print(f"  {model:8s} PSNR {s['psnr']}  SSIM {s['ssim']}  MAE {s['mae_c']} C  "
              f"RMSE {s['rmse_c']} C  physics {s['physics']}  time {s['time_ms']} ms")
