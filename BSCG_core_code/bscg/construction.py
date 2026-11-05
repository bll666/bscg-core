"""Cow-day construction from caller-normalised tables.

Raw-device decoding and source clock conversion belong in the user's adapter.
No records, labels, example datasets or private paths are distributed.
"""
import numpy as np
import pandas as pd


def construct(sensor, milk, state, cfg):
    """Return canonical arrays from tidy frames (schemas in docs/INPUT.md)."""
    sensor, milk, state = sensor.copy(), milk.copy(), state.copy()
    for frame in [sensor, milk, state]: frame["cow"] = frame.cow.astype(str)
    sensor["time"] = pd.to_datetime(sensor["time"], errors="raise")
    for frame in [milk, state]: frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if milk.duplicated(["cow", "date"]).any() or state.duplicated(["cow", "date"]).any():
        raise ValueError("Milk/state keys must be unique cow/date pairs")
    channels = [f"x{i}" for i in range(cfg["channels"])]
    for col in channels: sensor[col] = pd.to_numeric(sensor[col], errors="coerce")
    continuous = set(cfg["continuous_channels"])
    reducers = {col: "mean" if i in continuous else "last" for i, col in enumerate(channels)}
    sensor = sensor.sort_values(["cow", "time"], kind="stable")
    sensor = sensor.groupby(["cow", "time"], as_index=False).agg(reducers)
    if cfg["dataset"] == "mmcows":
        sensor["time"] = sensor.time.dt.floor("min")
        sensor = sensor.groupby(["cow", "time"], as_index=False).agg(reducers)
    milk_map = {(str(r.cow), r.date): float(r.milk) for r in milk.itertuples()}
    groups = {str(c): g.set_index("time")[channels] for c, g in sensor.groupby("cow")}
    rows = []
    for r in state.sort_values(["cow", "date"]).itertuples():
        cow, day = str(r.cow), r.date
        if cow not in groups: continue
        h = np.array([milk_map.get((cow, day-pd.Timedelta(days=i)), np.nan)
                      for i in range(cfg["history_days"])])
        y = milk_map.get((cow, day+pd.Timedelta(days=1)), np.nan)
        if not np.isfinite(np.r_[h, y]).all() or (np.r_[h, y] <= 0).any(): continue
        if cfg["dataset"] == "mmcows":
            clock = pd.date_range(day, periods=1440, freq="min")
            x = groups[cow].reindex(clock).to_numpy(float)
            if not all(np.isfinite(x[:, sl]).any() for sl in [slice(0,3), slice(3,9), slice(9,10)]):
                continue
            s = np.r_[h, float(r.age), float(r.DIM)]
        else:
            clock = pd.date_range(day-pd.Timedelta(hours=2), periods=12, freq="2h")
            x = groups[cow].reindex(clock).to_numpy(float)
            if not np.isfinite(x).all(): continue
            s = np.r_[h, h[:3].mean(), h.mean(), h.std(), h[0]-h[1],
                      float(r.positive_sessions), float(r.hours_since_last_milking),
                      float(r.DIM), float(r.parity)]
        rows.append((cow, day.strftime("%Y-%m-%d"), x, s, h, y))
    if not rows: raise ValueError("No eligible cow-days; check clock and table schemas")
    return dict(cows=np.array([r[0] for r in rows]), input_dates=np.array([r[1] for r in rows]),
                x=np.array([r[2] for r in rows], dtype=np.float32),
                mask=np.isfinite(np.array([r[2] for r in rows])).astype(np.float32),
                static=np.array([r[3] for r in rows], float),
                history=np.array([r[4] for r in rows], float), y=np.array([r[5] for r in rows], float))
