# ThermalSRNet — Optical-Guided Thermal Fusion

A web app that demonstrates **ThermalSRNet**, a deep-learning model that fuses a
low-resolution **thermal** image with a high-resolution **optical** image of the
same location and outputs a single high-resolution, temperature-accurate **fused**
thermal map. Dark, mission-control "inferno" UI with a 5-panel comparison,
live PSNR/SSIM, a metrics bar chart, and the training loss curve.

```
PID 15-comp/
├── backend/            FastAPI + PyTorch (CPU) inference API
│   ├── main.py         routes + CORS
│   ├── model.py        ThermalSRNet architecture (verbatim — matches the checkpoint)
│   ├── inference.py    model singleton + inference pipeline + inferno colormap
│   ├── metrics.py      PSNR + SSIM
│   ├── requirements.txt
│   ├── training_history.json     (sample — replace with your Colab export)
│   ├── checkpoints/    <- put thermalsrnet_best.pt here
│   └── demo/           <- optional demo_optical.png / demo_thermal.png
└── frontend/           React 18 + Vite + Tailwind
```

## Colab → App handoff

After training in Colab, download and place the four exports:

| Colab file                | Destination                           |
| ------------------------- | ------------------------------------- |
| `thermalsrnet_best_tempfix.pt` (preferred) or `thermalsrnet_best.pt` | `backend/checkpoints/` |
| `training_history.json`   | `backend/` (overwrites the sample)    |
| `demo_optical.png`        | `backend/demo/`                       |
| `demo_thermal.png`        | `backend/demo/`                       |

> The app still runs without these: `/api/infer` returns **503** until the
> checkpoint is present, "Load demo pair" falls back to a **synthetic** pair, and
> the bundled sample `training_history.json` feeds the loss chart until you replace it.

## 1. Run the backend

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify: open http://localhost:8000/api/health → `{"status":"ok","model_loaded":true,"device":"cpu"}`
(`model_loaded` is `false` until `thermalsrnet_best.pt` is in `backend/checkpoints/`).

## 2. Run the frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

## 3. Use it

Open http://localhost:5173, click **Load demo pair** (or upload your own matched
optical + thermal images), then **Run Fusion**. You'll see the 5-panel comparison
(Optical · LR Thermal · Bicubic · **Ours** · Ground Truth), the inferno colormap
legend, live PSNR/SSIM cards + bar charts, and the training loss curve.

## API

| Method | Route                     | Purpose                                      |
| ------ | ------------------------- | -------------------------------------------- |
| GET    | `/api/health`             | status + model_loaded + device               |
| GET    | `/api/model-info`         | architecture / params / dataset / bands / loss |
| GET    | `/api/demo-pair`          | base64 optical + thermal demo images         |
| GET    | `/api/training-history`   | contents of `training_history.json`          |
| POST   | `/api/infer`              | multipart `optical_image`,`thermal_image` → fused + metrics |
| POST   | `/api/download-result`    | `{sr_thermal_b64}` → PNG download            |
| GET    | `/api/evaluation`         | test-set results: bicubic vs model, per image + mean ± std |

## Checkpoint selection

The backend loads, in order: the file named by the `THERMAL_CKPT` env var (a file name in
`backend/checkpoints/` or a full path), then `thermalsrnet_best_tempfix.pt`, then
`thermalsrnet_best.pt`. `/api/model-info` reports which one is in use (`checkpoint`).

## Temperature preservation

Compares the **high-resolution thermal (actual)** with the **predicted high-resolution thermal**.
`/api/infer` returns a `temperature` block plus three images (`temp_hr_b64`, `temp_pred_b64`,
`temp_error_b64`).

Temperature comes from the raw 8-bit pixel value on a fixed scale:
`T = t_min + raw/255 · (t_max − t_min)` (optional form fields `t_min`, `t_max`, `tolerance`;
defaults 10 / 50 / 1.0 °C). The model's normalized output is mapped back to raw values with the
input thermal's own min/max first, so the numbers differ for every image pair. It reports:

- **Probe points** — centre of the hottest and coldest 9×9 area (kept off the border) and a 3×3
  grid: actual vs predicted (3×3 window mean), difference, and whether it is within ±tolerance.
- **Charts** (only with a real ≥128 px ground truth, proposed vs bicubic, all pixels) — actual-vs-
  predicted scatter with R² against y = x and slope, an error histogram, and a temperature line
  profile along the row with the most edges (drawn as a blue dashed line on the images).
- **Summary** — average / hottest / coldest temperature of both images, mean and max absolute
  error, bias, % of pixels within tolerance.
- **Images** — actual and predicted on a shared colour scale with the probe markers, and the
  |difference| map.

The UI shows this in the **Temperature Preservation** section, where the range can be changed and re-run.

## Training, validation & testing

- **Training & Validation** reads the history file that matches the loaded checkpoint
  (`thermalsrnet_best_tempfix.pt` → `training_history_tempfix.json`). It plots every logged loss
  component and the final validation PSNR/SSIM. Train-vs-validation curves appear automatically
  when the history contains `val_total` (and `val_psnr`, `val_physics`, …) per epoch.
- **Testing** runs bicubic and the model on every pair in `backend/test_data/`
  (`name_optical.png` + `name_thermal.png`) plus the demo pair, and reports per-image results and
  mean ± std. Run `python evaluate.py` to also save them to `evaluation.json`.
- **Comparison table** (per image and test set): MAE, RMSE, PSNR, SSIM, temperature MAE/RMSE/max
  error, % within tolerance, physics-consistency error, inference time.

## Fair evaluation & detail strength (α)

- **Thermal ≥ 128×128:** treated as the real high-res ground truth. The low-res input is made from it
  with the same bicubic ×4 degradation used in training, and PSNR / SSIM / temperature error are
  measured against it.
- **Thermal < 128×128:** used directly as the low-res input. No ground truth is invented (enlarging
  it would create a bicubic "ground truth" that unfairly favours bicubic), so ground-truth scores are
  shown as "—". Temperature preservation is checked at the input resolution instead. Test-set averages
  only include images with a real ground truth.
- **α (detail strength):** output = bicubic + α × learned detail. 1.0 (default) = the model as
  trained, 0 = plain bicubic. Sent as the optional `alpha` form field (0–1); results are labelled
  "Proposed (α=…)" when α ≠ 1. The test set always uses α = 1.0. Choose α on validation data, never
  per test image.

## Physics loss

`backend/physics.py` implements an energy-conservation loss: `mean |Degrade(SR) − LR|`, where
`Degrade` is the same bicubic ×4 downsampling that creates the low-res input. It is computed on
every result (no ground truth needed). To use it **in training**, add `PhysicsConsistencyLoss` to
the Colab loss (start with weight 0.1), log `physics` / `val_physics` per epoch in the history
JSON, and keep it only if validation temperature error improves. The app reports
"used in training" only when the history file contains those keys.

## Notes

- **CPU-only** inference (`map_location='cpu'`) — runs on any laptop.
- The model is loaded **once at startup** (singleton), reused per request.
- The inferno colormap is applied to **all** thermal panels.
- CORS allows `http://localhost:5173` (Vite dev). Set `VITE_API_URL` in
  `frontend/.env` to point at a different backend.
- `backend/model.py` matches the trained architecture character-for-character;
  do not edit it or `torch.load` on the checkpoint will fail.

**Dataset:** NASA/USGS Landsat 8/9 OLI-TIRS · SSL4EO-L Benchmark (TorchGeo).
