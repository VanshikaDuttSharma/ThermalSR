"""
Parity test: does the BACKEND produce the same output as COLAB for the SAME input?

This isolates the bug:
  - If outputs MATCH  -> your training/model is the limit (retrain / tune). Serving is correct.
  - If outputs DIFFER -> the bug is in the backend preprocessing (normalization, resize,
    channel order). Fix run_inference() to mirror your Colab transform exactly.

--------------------------------------------------------------------------------------
STEP 1 — In COLAB, right after you load thermalsrnet_best.pt and pick ONE validation
sample, run this cell to export the EXACT tensors and the model's output:

    import numpy as np, torch
    model.eval()
    with torch.no_grad():
        sr = model(lr_thermal, optical)        # the tensors you fed during training/val
    np.savez("reference.npz",
             lr_thermal=lr_thermal.cpu().numpy(),   # (1,1,H_lr,W_lr)
             optical=optical.cpu().numpy(),         # (1,3,H,W)
             sr=sr.cpu().numpy())                   # (1,1,H,W)  <- the "expected" output
    print("lr", lr_thermal.shape, "optical", optical.shape, "sr", sr.shape)
    # also print the ranges — you will need these to match preprocessing:
    print("lr range", float(lr_thermal.min()), float(lr_thermal.max()))
    print("optical range", float(optical.min()), float(optical.max()))

Download reference.npz and place it next to this file (backend/reference.npz).

STEP 2 — Run:  python parity_test.py
--------------------------------------------------------------------------------------
"""
import os
import numpy as np
import torch

from inference import _model, MODEL_LOADED

REF = os.path.join(os.path.dirname(__file__), "reference.npz")


def main():
    if not MODEL_LOADED:
        print("[FAIL] Model not loaded. Put thermalsrnet_best.pt in checkpoints/ first.")
        return
    if not os.path.exists(REF):
        print(f"[SKIP] {REF} not found. Export it from Colab (see the header of this file).")
        return

    z = np.load(REF)
    lr = torch.from_numpy(z["lr_thermal"]).float()
    opt = torch.from_numpy(z["optical"]).float()
    ref_sr = z["sr"]

    print("Input ranges from Colab reference:")
    print(f"  lr_thermal: [{lr.min():.4f}, {lr.max():.4f}]  shape {tuple(lr.shape)}")
    print(f"  optical:    [{opt.min():.4f}, {opt.max():.4f}]  shape {tuple(opt.shape)}")

    with torch.no_grad():
        sr = _model(lr, opt).cpu().numpy()

    diff = np.abs(sr - ref_sr)
    mae = diff.mean()
    mx = diff.max()
    print(f"\nBackend vs Colab output:  MAE={mae:.6e}   max|diff|={mx:.6e}")

    if mx < 1e-4:
        print("[PASS] Serving matches Colab. The model IS the limit -> retrain / tune, "
              "or check gamma below. The app is NOT the problem.")
    else:
        print("[FAIL] Serving DIVERGES from Colab for identical tensors -> the checkpoint "
              "or model.py loaded here differs from Colab. Verify same architecture & weights.")

    # Bonus: is the fusion actually using the optical stream?
    try:
        g = float(_model.fusion.gamma.item())
        print(f"\nfusion.gamma = {g:.5f}  "
              + ("(~0 -> optical barely fused; model under-trained)" if abs(g) < 0.02
                 else "(non-trivial -> optical is being fused)"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
