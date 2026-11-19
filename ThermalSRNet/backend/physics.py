"""Physics-consistency (energy-conservation) loss.

Physical idea
-------------
A low-resolution thermal pixel records the combined emission of the ground it covers. So a
correct high-resolution prediction, degraded back to low resolution with the SAME operator
that produced the low-res data, must reproduce the low-res input:

    L_phys = mean | Degrade(SR) - LR |

`Degrade` here is bicubic x4 downsampling, because that is exactly how the low-res input is
built from the high-res thermal (see inference._thermal_tensors). Using any other operator
(e.g. average pooling) would penalise correct predictions.

Assumption: values are normalized brightness, so energy is treated linearly. Strictly,
emitted energy scales with T^4 (Stefan-Boltzmann); over the narrow temperature range of one
scene the linear form is a close approximation.

Two uses
--------
1. Inference (always on): `consistency()` measures how well a prediction conserves the
   input's energy. It needs no ground truth, so it works on any uploaded image.
2. Training: `PhysicsConsistencyLoss` is a drop-in nn.Module. Add it to the training loss in
   Colab and log it per epoch as `physics` / `val_physics` in the history JSON; the app then
   shows it as "used in training" and plots its curve:

       phys = PhysicsConsistencyLoss(scale=4)
       loss = l1 + 0.5 * edge + 0.5 * ssim + 0.01 * tv + PHYS_WEIGHT * phys(sr, lr)

   Start with PHYS_WEIGHT = 0.1 and keep it only if validation temperature error improves
   without lowering PSNR/SSIM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PHYSICS_FORMULA = "mean |Degrade(SR) - LR|, Degrade = bicubic x4 downsampling"


def degrade(hr, scale=4):
    """Same degradation used to create the low-res input from the high-res thermal."""
    h, w = hr.shape[-2:]
    return F.interpolate(hr, size=(h // scale, w // scale), mode="bicubic",
                         align_corners=False).clamp(0, 1)


def consistency(pred, lr, scale=4):
    """Mean absolute physics-consistency error in normalized [0,1] units (float)."""
    with torch.no_grad():
        return float((degrade(pred, scale) - lr).abs().mean())


class PhysicsConsistencyLoss(nn.Module):
    """Training loss: L1 between the degraded prediction and the low-res input."""

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, sr, lr):
        return F.l1_loss(degrade(sr, self.scale), lr)
