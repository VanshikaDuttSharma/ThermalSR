"""FastAPI app: ThermalSRNet inference API (Part B)."""
import os, io, json, base64, math
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image

import inference as inf
import temperature
import physics
import evaluate

app = FastAPI(title="ThermalSRNet API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Any localhost port: Vite moves to 5174, 5175… when 5173 is already taken.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE = os.path.dirname(__file__)
_DEMO_DIR = os.path.join(_HERE, "demo")


def _history_path():
    """Training history that belongs to the loaded checkpoint:
    thermalsrnet_best_tempfix.pt -> training_history_tempfix.json, else training_history.json."""
    stem = os.path.splitext(inf.CKPT_NAME)[0]                    # thermalsrnet_best_tempfix
    suffix = stem.split("thermalsrnet_best", 1)[-1]              # _tempfix
    for name in (f"training_history{suffix}.json", "training_history.json"):
        path = os.path.join(_HERE, name)
        if os.path.exists(path):
            return path
    return None


def _load_history():
    path = _history_path()
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), os.path.basename(path)


# History keys that hold the physics loss. The Colab run logs it as "temp".
_PHYSICS_KEYS = ("temp", "temp_loss", "val_temp")


def _physics_status(history):
    """Physics loss counts as 'used in training' only if the history actually logged it."""
    keys = [k for k in (history or {}) if "physics" in k.lower() or k in _PHYSICS_KEYS]
    return {
        "used_in_training": bool(keys),
        "history_keys": keys,
        "computed_at_inference": True,
        "formula": physics.PHYSICS_FORMULA,
    }


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": inf.MODEL_LOADED,
        "device": inf.DEVICE,
        "reason": None if inf.MODEL_LOADED else inf.LOAD_ERROR,
    }


@app.get("/api/model-info")
def model_info():
    # Surface the learned fusion gate so a soft-but-serving-correct output is diagnosable:
    # gamma ~0 => optical stream was pulled out during training (not a serving bug).
    fusion_gamma = None
    n_params = sum(p.numel() for p in inf._model.parameters())
    history, history_file = _load_history()
    phys = _physics_status(history)
    loss = "L1 (x1.0) + Edge/Sobel (x0.5) + SSIM (x0.5) + Total Variation (x0.01)"
    if phys["used_in_training"]:
        loss += " + Physics consistency"
    if inf.MODEL_LOADED:
        try:
            fusion_gamma = round(float(inf._model.fusion.gamma.item()), 4)
        except Exception:
            fusion_gamma = None
    return {
        "architecture": "ThermalSRNet V2",
        "checkpoint": inf.CKPT_NAME,
        "fusion_gamma": fusion_gamma,
        "parameters": f"{n_params / 1e6:.2f}M",
        "parameter_count": n_params,
        "scale_factor": 4,
        "dataset": "NASA/USGS Landsat 8/9 OLI-TIRS (SSL4EO-L)",
        "thermal_band": "B10 (idx 9)",
        "rgb_bands": "B4,B3,B2 (3,2,1)",
        "loss": loss,
        "physics_loss": phys,
        "history_file": history_file,
        "design": "residual-over-bicubic + RCAB + cross-modal fusion",
        "temperature_defaults": {
            "t_min_c": temperature.DEFAULT_T_MIN_C,
            "t_max_c": temperature.DEFAULT_T_MAX_C,
            "tolerance_c": temperature.DEFAULT_TOLERANCE_C,
        },
    }


# ---------------------------------------------------------------------------
# Demo pair
# ---------------------------------------------------------------------------
def _synthesize_demo():
    """Generate a matched optical/thermal demo pair when the real files are
    absent, so the frontend 'Load demo pair' button always works."""
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # A synthetic scene: warm circular "urban" hotspot + a cool diagonal "river".
    cx, cy = w * 0.62, h * 0.42
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    heat = np.exp(-(dist ** 2) / (2 * 55.0 ** 2))
    river = np.exp(-((xx - yy) ** 2) / (2 * 22.0 ** 2))
    thermal = np.clip(0.35 + 0.9 * heat - 0.45 * river, 0, 1)
    thermal += 0.04 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    thermal = np.clip(thermal, 0, 1)
    thermal_img = Image.fromarray((thermal * 255).astype(np.uint8), mode="L")

    # Optical shares the same structures but as RGB texture.
    r = np.clip(0.30 + 0.7 * heat + 0.05 * np.sin(xx / 6.0), 0, 1)
    g = np.clip(0.45 + 0.4 * river + 0.05 * np.cos(yy / 6.0), 0, 1)
    b = np.clip(0.55 + 0.6 * river - 0.2 * heat, 0, 1)
    optical = np.stack([r, g, b], axis=-1)
    optical_img = Image.fromarray((optical * 255).astype(np.uint8), mode="RGB")
    return optical_img, thermal_img


@app.get("/api/demo-pair")
def demo_pair():
    opt_path = os.path.join(_DEMO_DIR, "demo_optical.png")
    thm_path = os.path.join(_DEMO_DIR, "demo_thermal.png")
    if os.path.exists(opt_path) and os.path.exists(thm_path):
        optical_img = Image.open(opt_path)
        thermal_img = Image.open(thm_path)
    else:
        optical_img, thermal_img = _synthesize_demo()
    # IMPORTANT: return the thermal as RAW single-channel grayscale. The frontend turns this
    # data URL into the File it POSTs back to /api/infer, so it must NOT be colormapped here —
    # colormapping + later .convert("L") would scramble the thermal signal. The inferno look is
    # applied by the backend on the OUTPUT panels, not on the model input.
    return {
        "optical_b64": inf.to_b64(optical_img.convert("RGB")),
        "thermal_b64": inf.to_b64(thermal_img.convert("L")),
    }


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------
def _json_safe(v):
    """Colab exports can contain Infinity/NaN (e.g. PSNR of identical images), which are not
    valid JSON and make the response fail. Replace them with null."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return v


@app.get("/api/training-history")
def training_history():
    history, history_file = _load_history()
    if history is None:
        return JSONResponse(status_code=404, content={"error": "training_history.json not found in backend/"})
    content = _json_safe(history)
    if isinstance(content, dict):
        content["_source_file"] = history_file
        content["_physics"] = _physics_status(history)
    return JSONResponse(content=content)


# ---------------------------------------------------------------------------
# Test-set evaluation (bicubic vs proposed model on backend/test_data + demo pair)
# ---------------------------------------------------------------------------
@app.get("/api/evaluation")
def evaluation():
    if not inf.MODEL_LOADED:
        return JSONResponse(status_code=503, content={"error": inf.LOAD_ERROR or "Model not loaded."})
    try:
        return JSONResponse(content=_json_safe(evaluate.cached_evaluation()))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Evaluation failed: {e}"})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@app.post("/api/infer")
async def infer(
    optical_image: UploadFile = File(...),
    thermal_image: UploadFile = File(...),
    t_min: float = Form(temperature.DEFAULT_T_MIN_C),
    t_max: float = Form(temperature.DEFAULT_T_MAX_C),
    tolerance: float = Form(temperature.DEFAULT_TOLERANCE_C),
    alpha: float = Form(1.0),
):
    if not inf.MODEL_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": inf.LOAD_ERROR or "Model not loaded. Put thermalsrnet_best.pt in backend/checkpoints/ and restart."},
        )
    if not t_max > t_min:
        return JSONResponse(status_code=400, content={"error": "t_max must be greater than t_min."})
    if not tolerance > 0:
        return JSONResponse(status_code=400, content={"error": "tolerance must be positive."})
    if not 0.0 <= alpha <= 1.0:
        return JSONResponse(status_code=400, content={"error": "alpha (detail strength) must be between 0 and 1."})
    try:
        optical_pil = Image.open(io.BytesIO(await optical_image.read()))
        thermal_pil = Image.open(io.BytesIO(await thermal_image.read()))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read images: {e}"})

    try:
        result = inf.run_inference(optical_pil, thermal_pil, t_min, t_max, tolerance, alpha)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Inference failed: {e}"})
    return result


# ---------------------------------------------------------------------------
# Download result
# ---------------------------------------------------------------------------
class DownloadBody(BaseModel):
    sr_thermal_b64: str


@app.post("/api/download-result")
def download_result(body: DownloadBody):
    raw = body.sr_thermal_b64.split(",", 1)[-1]
    data = base64.b64decode(raw)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="thermal_fusion_result.png"'},
    )
