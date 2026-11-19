import math
import torch, torch.nn as nn, torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, ch, red=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(ch, ch//red, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(ch//red, ch, 1), nn.Sigmoid())
    def forward(self, x): return x * self.fc(self.pool(x))

class RCAB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(ch,ch,3,padding=1), CALayer(ch))
    def forward(self, x): return x + self.body(x)

def make_upsampler(scale, ch):
    layers = []
    for _ in range(int(math.log2(scale))):
        layers += [nn.Conv2d(ch, ch*4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class ThermalEncoderV2(nn.Module):
    def __init__(self, ch=64, n=3):
        super().__init__()
        self.head = nn.Conv2d(1, ch, 3, padding=1)
        self.body = nn.Sequential(*[RCAB(ch) for _ in range(n)])
    def forward(self, x): return self.body(self.head(x))

class OpticalEncoderV2(nn.Module):
    def __init__(self, ch=64, n=3):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[RCAB(ch) for _ in range(n)])
    def forward(self, x): return self.body(self.head(x))

class CrossModalFusion(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(ch*2, ch, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.ones(1))   # init 1.0
    def forward(self, t_feat, o_feat):
        g = self.gate(torch.cat([t_feat, o_feat], 1))
        return t_feat + self.gamma * g * o_feat

class ThermalSRNetV2(nn.Module):
    def __init__(self, ch=64, scale=4, n_body=4):
        super().__init__()
        self.scale = scale
        self.thermal_enc = ThermalEncoderV2(ch)
        self.optical_enc = OpticalEncoderV2(ch)
        self.up          = make_upsampler(scale, ch)
        self.fusion      = CrossModalFusion(ch)
        self.body        = nn.Sequential(*[RCAB(ch) for _ in range(n_body)])
        self.recon       = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(ch, 1, 3, padding=1))
    def forward(self, lr_thermal, optical):
        bic   = F.interpolate(lr_thermal, size=optical.shape[-2:], mode='bicubic', align_corners=False)
        tf    = self.up(self.thermal_enc(lr_thermal))
        of    = self.optical_enc(optical)
        fused = self.body(self.fusion(tf, of))
        res   = self.recon(fused)
        return bic + res


# Backwards-compatible alias so the rest of the backend keeps importing `ThermalSRNet`.
ThermalSRNet = ThermalSRNetV2
