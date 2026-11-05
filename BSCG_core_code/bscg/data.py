"""Generic, user-supplied arrays; no dataset or private sample list is included."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, Path): return str(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, default=convert, indent=2,
                              ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_data(path, cfg, require_target=True):
    with np.load(path, allow_pickle=False) as f:
        d = {key: f[key] for key in f.files}
    required = {"x", "static", "history", "cows", "input_dates"}
    if require_target: required.add("y")
    if required - d.keys():
        raise ValueError(f"Missing input arrays: {sorted(required - d.keys())}")
    x = np.asarray(d["x"], dtype=np.float32)
    n = len(x)
    if not n: raise ValueError("Input archive has no samples")
    if x.shape != (n, cfg["points"], cfg["channels"]):
        raise ValueError("x shape differs from config points/channels")
    mask = np.asarray(d.get("mask", np.isfinite(x)), dtype=np.float32)
    if mask.shape != x.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must be binary and match x")
    if np.any((mask == 1) & ~np.isfinite(x)):
        raise ValueError("Observed entries must be finite")
    x = np.where(mask == 1, x, np.nan)
    static = np.asarray(d["static"], float)
    hist = np.asarray(d["history"], float)
    if static.shape != (n, cfg["static_features"]) or hist.shape != (n, cfg["history_days"]):
        raise ValueError("static/history width differs from the declared configuration")
    if not np.isfinite(hist).all() or (hist <= 0).any():
        raise ValueError("Complete positive milk history is required")
    if not np.allclose(static[:, :hist.shape[1]], hist, equal_nan=False):
        raise ValueError("The initial static columns must be the same raw milk history")
    cows = np.asarray(d["cows"]).astype(str)
    dates = np.asarray(d["input_dates"]).astype("datetime64[D]")
    if cows.shape != (n,) or dates.shape != (n,) or np.isnat(dates).any():
        raise ValueError("Invalid cow/date keys")
    dates = dates.astype(str)
    targets = (dates.astype("datetime64[D]") + np.timedelta64(1, "D")).astype(str)
    if "target_dates" in d and not np.array_equal(targets, d["target_dates"].astype(str)):
        raise ValueError("Targets must be dated input day + 1")
    ids = np.asarray(d.get("sample_ids", [f"{c}::{t}" for c, t in zip(cows, dates)])).astype(str)
    if len(set(ids)) != n or len(set(zip(cows, dates))) != n:
        raise ValueError("Duplicate sample IDs or cow/date keys")
    y = np.asarray(d["y"], float) if "y" in d else None
    if y is not None and (y.shape != (n,) or not np.isfinite(y).all() or (y <= 0).any()):
        raise ValueError("y must contain finite positive recorded yields")
    # Check every overlap present in the supplied cohort, including histories.
    record_values = {}
    for i, (cow, day) in enumerate(zip(cows, dates)):
        pairs = [(str(np.datetime64(day) - np.timedelta64(lag, "D")), v)
                 for lag, v in enumerate(hist[i])]
        if y is not None: pairs.append((targets[i], y[i]))
        for record_date, value in pairs:
            key = (cow, record_date)
            if key in record_values and not np.isclose(record_values[key], value,
                                                       rtol=1e-6, atol=1e-5):
                raise ValueError(f"Inconsistent overlapping milk record: {key}")
            record_values[key] = float(value)
    if cfg["dataset"] == "cowbase" and not (mask == 1).all():
        raise ValueError("CowBase eligibility requires twelve observed activity values")
    if cfg["dataset"] == "mmcows":
        for sl in [slice(0,3), slice(3,9), slice(9,10)]:
            if not (mask[:, :, sl].sum(axis=(1,2)) > 0).all():
                raise ValueError("Each MMCOWS window needs observations in each modality")
    return dict(x=x, mask=mask, static=static, history=hist, base=hist[:, 0], y=y,
                cows=cows, input_dates=dates, target_dates=targets, sample_ids=ids,
                source_sha256=sha256(path))
