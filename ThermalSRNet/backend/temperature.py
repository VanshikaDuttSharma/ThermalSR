"""Temperature preservation: HR ground-truth thermal vs predicted HR thermal.

How a pixel becomes a temperature
---------------------------------
The thermal PNG is an 8-bit image, so we treat its pixel value as a fixed sensor scale:

    T(deg C) = t_min + (raw / 255) * (t_max - t_min)       raw in [0, 255]

This scale is the SAME for every image, so a brighter image really is hotter and the
numbers change from one image pair to the next.

The model works on per-image min-max normalized values in [0, 1]. Before converting to
temperature, the prediction is mapped back to the raw scale with the input thermal's own
range (lo, hi):  raw = lo + v * (hi - lo).  The ground truth goes through the same mapping,
so both images sit on the same physical scale and their difference is a real deg C error.

What is reported
----------------
  * probe points - hottest spot, coldest spot and a 3x3 grid; HR vs predicted at each
                   (3x3 window mean so one noisy pixel doesn't dominate)
  * summary      - mean / max / min temperature of both images, mean absolute error,
                   max error, bias, % of pixels within +-tolerance
  * images       - HR and predicted thermal on a shared colour scale with the probe markers,
                   and an |error| map
  * charts       - (only with a real HR ground truth) every pixel, for the model AND bicubic:
                   actual-vs-predicted scatter (R2, slope), error histogram, and a temperature
                   line profile along the row with the most edges
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont

DEFAULT_T_MIN_C = 10.0     # temperature of raw pixel value 0
DEFAULT_T_MAX_C = 50.0     # temperature of raw pixel value 255
DEFAULT_TOLERANCE_C = 1.0

_PROBE_RADIUS = 1          # 3x3 window around each probe
_SPOT_WINDOW = 9           # hottest/coldest = centre of the hottest/coldest 9x9 area
_SCATTER_POINTS = 800      # pixels drawn in the scatter plot (stats always use every pixel)
_ERROR_BINS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
_VIEW_SIZE = 384           # px; images are upscaled so markers/labels are readable


def _to_np(t):
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return np.clip(np.squeeze(a).astype(np.float32), 0.0, 1.0)


def norm_to_celsius(norm, lo, hi, t_min, t_max):
    """Normalized [0,1] -> raw 8-bit [lo, hi] -> deg C on the fixed sensor scale."""
    raw = lo + norm * (hi - lo)
    return t_min + (raw / 255.0) * (t_max - t_min)


def _window_mean(img, y, x, r=_PROBE_RADIUS):
    h, w = img.shape
    return float(img[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].mean())


PROBE_METHOD = (
    "Not random. Point 1 is the centre of the hottest area and point 2 the centre of the coldest "
    "area of the actual high-res image (9x9 average, so ties and single pixels don't decide it); "
    "points 3-11 are the centres of a fixed 3x3 grid for even coverage. Each value is the mean "
    "of the 3x3 pixels around the point."
)


def _spot(img, hottest):
    """Centre of the hottest (or coldest) _SPOT_WINDOW x _SPOT_WINDOW area. Averaging avoids
    picking the first of many tied pixels (often a corner) and keeps the point off the border."""
    k = min(_SPOT_WINDOW, *img.shape)
    means = np.lib.stride_tricks.sliding_window_view(img, (k, k)).mean(axis=(2, 3))
    m = k // 2 if min(means.shape) > k else 0   # keep centres >= k px from the border
    inner = means[m:means.shape[0] - m, m:means.shape[1] - m]
    y, x = np.unravel_index(int(np.argmax(inner) if hottest else np.argmin(inner)), inner.shape)
    return int(y + m + k // 2), int(x + m + k // 2)


def _probe_locations(gt):
    """Hottest + coldest area centres, then a 3x3 grid of cell centres.
    Returns (label, y, x, reason) tuples."""
    h, w = gt.shape
    hy, hx = _spot(gt, hottest=True)
    cy, cx = _spot(gt, hottest=False)
    probes = [("Hottest", hy, hx, "centre of the hottest area in actual image"),
              ("Coldest", cy, cx, "centre of the coldest area in actual image")]
    rows, cols = ("top", "middle", "bottom"), ("left", "centre", "right")
    for i in range(3):
        for j in range(3):
            probes.append((f"P{i * 3 + j + 1}", int((i + 0.5) * h / 3), int((j + 0.5) * w / 3),
                           f"grid cell {rows[i]}-{cols[j]}"))
    return probes


def celsius_errors(pred, gt, lo, hi, t_min=DEFAULT_T_MIN_C, t_max=DEFAULT_T_MAX_C,
                   tolerance=DEFAULT_TOLERANCE_C):
    """Temperature error summary (deg C) of any prediction vs the ground truth. Used to score
    bicubic and the model on exactly the same scale."""
    gt_c = norm_to_celsius(_to_np(gt), lo, hi, t_min, t_max)
    pr_c = norm_to_celsius(_to_np(pred), lo, hi, t_min, t_max)
    diff = pr_c - gt_c
    err = np.abs(diff)
    return {
        "mae_c": round(float(err.mean()), 3),
        "rmse_c": round(float(np.sqrt((diff ** 2).mean())), 3),
        "max_error_c": round(float(err.max()), 3),
        "bias_c": round(float(diff.mean()), 3),
        "within_tol_pct": round(float((err <= tolerance).mean() * 100.0), 1),
    }


def norm_to_celsius_delta(delta_norm, lo, hi, t_min=DEFAULT_T_MIN_C, t_max=DEFAULT_T_MAX_C):
    """Convert a difference in normalized units to deg C (no offset)."""
    return delta_norm * (hi - lo) / 255.0 * (t_max - t_min)


def _font(size):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _profile_row(gt_c):
    """Row with the most temperature change along it (sum of |horizontal gradient|): the row
    where keeping edges sharp matters most. Chosen from the actual image only."""
    return int(np.abs(np.diff(gt_c, axis=1)).sum(axis=1).argmax())


def _charts(gt_c, preds_c, tolerance):
    """Pixel-level preservation charts for each method in preds_c ({name: deg C array})."""
    g = gt_c.ravel()
    ss_tot = float(((g - g.mean()) ** 2).sum()) + 1e-12
    idx = np.random.default_rng(0).choice(g.size, min(_SCATTER_POINTS, g.size), replace=False)
    labels = [f"{a:g}-{b:g}" if np.isfinite(b) else f">{a:g}" for a, b in zip(_ERROR_BINS, _ERROR_BINS[1:])]
    row = _profile_row(gt_c)

    scatter, hist = {}, []
    for name, pr_c in preds_c.items():
        p = pr_c.ravel()
        slope, intercept = np.polyfit(g, p, 1)
        err = np.abs(p - g)
        scatter[name] = {
            # R2 against the perfect line y = x (not a fitted line): 1 = every pixel exact.
            "r2": round(1.0 - float(((p - g) ** 2).sum()) / ss_tot, 4),
            "slope": round(float(slope), 3),
            "intercept": round(float(intercept), 2),
            "within_tol_pct": round(float((err <= tolerance).mean() * 100.0), 1),
            "points": [[round(float(g[i]), 2), round(float(p[i]), 2)] for i in idx],
        }
        counts, _ = np.histogram(err, bins=_ERROR_BINS)
        hist.append((name, counts / err.size * 100.0))

    lo = min(float(g.min()), *(float(v.min()) for v in preds_c.values()))
    hi = max(float(g.max()), *(float(v.max()) for v in preds_c.values()))
    return {
        "scatter": scatter,
        "scatter_range": [round(lo, 2), round(hi, 2)],
        "histogram": [{"bin": lab, **{n: round(float(c[i]), 1) for n, c in hist}} for i, lab in enumerate(labels)],
        "profile_row": row,
        "profile": [{"x": x, "actual": round(float(gt_c[row, x]), 2),
                     **{n: round(float(v[row, x]), 2) for n, v in preds_c.items()}}
                    for x in range(gt_c.shape[1])],
    }


def _view(temp_c, vmin, vmax, points, cmap=cm.inferno, profile_row=None):
    """Colour-mapped, upscaled image with numbered probe markers (and the profile line)."""
    h, w = temp_c.shape
    norm = np.clip((temp_c - vmin) / (vmax - vmin + 1e-6), 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((_VIEW_SIZE, _VIEW_SIZE), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    sx, sy = _VIEW_SIZE / w, _VIEW_SIZE / h
    if profile_row is not None:
        ly = (profile_row + 0.5) * sy
        for x0 in range(0, _VIEW_SIZE, 12):  # dashed line
            draw.line([(x0, ly), (x0 + 7, ly)], fill=(0, 0, 0), width=4)
            draw.line([(x0, ly), (x0 + 7, ly)], fill=(56, 189, 248), width=2)
    font = _font(13)
    r = 7
    for idx, p in enumerate(points, start=1):
        px, py = (p["x"] + 0.5) * sx, (p["y"] + 0.5) * sy
        ring = (53, 208, 165) if p["preserved"] else (255, 92, 108)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 0), width=4)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=ring, width=2)
        tx = min(px + r + 2, _VIEW_SIZE - 18)
        ty = max(py - r - 2, 0)
        draw.text((tx, ty), str(idx), fill=(255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return img


def analyze(pred, gt, lo, hi, t_min=DEFAULT_T_MIN_C, t_max=DEFAULT_T_MAX_C,
            tolerance=DEFAULT_TOLERANCE_C, reference="hr", baseline=None):
    # reference="hr":    gt = real high-res ground truth, pred = predicted high-res image.
    # reference="input": no high-res ground truth; gt = low-res input, pred = prediction shrunk
    #                    back to the input size (checks each input pixel keeps its temperature).
    """pred, gt: normalized [0,1] HR thermal (tensor or array). lo, hi: raw 8-bit range of
    the input thermal. baseline: optional bicubic prediction; when given, stats["charts"] holds
    the pixel-level charts for both. Returns (stats_dict, images_dict of PIL images)."""
    if not t_max > t_min:
        raise ValueError(f"t_max ({t_max}) must be greater than t_min ({t_min})")
    if not tolerance > 0:
        raise ValueError(f"tolerance ({tolerance}) must be positive")

    gt_c = norm_to_celsius(_to_np(gt), lo, hi, t_min, t_max)
    pr_c = norm_to_celsius(_to_np(pred), lo, hi, t_min, t_max)
    diff = pr_c - gt_c
    err = np.abs(diff)

    points = []
    for label, y, x, reason in _probe_locations(gt_c):
        g = _window_mean(gt_c, y, x)
        p = _window_mean(pr_c, y, x)
        points.append({
            "label": label, "x": x, "y": y, "reason": reason,
            "hr_c": round(g, 2),
            "pred_c": round(p, 2),
            "diff_c": round(p - g, 2),
            "preserved": bool(abs(p - g) <= tolerance),
        })

    def _r(v, d=2):
        return round(float(v), d)

    stats = {
        "t_min_c": _r(t_min), "t_max_c": _r(t_max), "tolerance_c": _r(tolerance),
        "raw_lo": _r(lo, 1), "raw_hi": _r(hi, 1),
        "hr_mean_c": _r(gt_c.mean()), "pred_mean_c": _r(pr_c.mean()),
        "hr_max_c": _r(gt_c.max()), "pred_max_c": _r(pr_c.max()),
        "hr_min_c": _r(gt_c.min()), "pred_min_c": _r(pr_c.min()),
        "mae_c": _r(err.mean(), 3),
        "rmse_c": _r(np.sqrt((diff ** 2).mean()), 3),
        "max_error_c": _r(err.max(), 3),
        "bias_c": _r(diff.mean(), 3),
        "within_tol_pct": _r((err <= tolerance).mean() * 100.0, 1),
        "points_preserved": sum(p["preserved"] for p in points),
        "points_total": len(points),
        "probe_method": PROBE_METHOD,
        "reference": reference,
        "size": list(gt_c.shape),
        "points": points,
    }

    # HR and predicted share one colour scale (the HR's own range) so they compare directly.
    vmin, vmax = float(gt_c.min()), float(gt_c.max())
    if vmax - vmin < 1e-3:
        vmax = vmin + 1e-3
    err_max = float(max(np.percentile(err, 99), 1e-3))
    stats["view_min_c"], stats["view_max_c"] = _r(vmin), _r(vmax)
    stats["error_view_max_c"] = _r(err_max)

    row = None
    if baseline is not None:
        bic_c = norm_to_celsius(_to_np(baseline), lo, hi, t_min, t_max)
        stats["charts"] = _charts(gt_c, {"ours": pr_c, "bicubic": bic_c}, tolerance)
        row = stats["charts"]["profile_row"]

    images = {
        "hr": _view(gt_c, vmin, vmax, points, profile_row=row),
        "pred": _view(pr_c, vmin, vmax, points, profile_row=row),
        "error": _view(err, 0.0, err_max, points, cmap=cm.magma),
    }
    return stats, images
