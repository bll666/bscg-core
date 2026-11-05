"""Sample-local gap handling and training-partition-only fitted transforms."""
import numpy as np


def interpolate_short_gaps(x, continuous_channels, max_gap=2):
    result = np.array(x, dtype=float, copy=True)
    for sample in result:
        for c in continuous_channels:
            a = sample[:, c]
            edges = np.diff(np.r_[0, (~np.isfinite(a)).astype(int), 0])
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if 0 < start < end < len(a) and end - start <= max_gap:
                    a[start:end] = np.linspace(a[start-1], a[end], end-start+2)[1:-1]
    return result


class Standardizer:
    def fit(self, a):
        a = np.asarray(a, float)
        a = np.where(np.isfinite(a), a, np.nan)
        self.median = np.nanmedian(a, axis=0)
        if not np.isfinite(self.median).all():
            raise ValueError("An entire training feature is missing")
        filled = np.where(np.isfinite(a), a, self.median)
        self.mean = filled.mean(0)
        sd = filled.std(0, ddof=0)
        self.scale = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, a):
        a = np.where(np.isfinite(a), a, self.median)
        return np.clip((a-self.mean)/self.scale, -5, 5).astype(np.float32)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ["median", "mean", "scale"]}

    @classmethod
    def restore(cls, state):
        result = cls()
        for key, value in state.items(): setattr(result, key, np.asarray(value))
        return result


class Preprocessor:
    def __init__(self, cfg): self.cfg = cfg

    def fit(self, data, train):
        train = np.asarray(train, int)
        if not len(train): raise ValueError("Empty training partition")
        c = self.cfg["continuous_channels"]
        raw = interpolate_short_gaps(data["x"][train], c, self.cfg["max_gap"])
        self.sensor = Standardizer().fit(raw[:, :, c].reshape(-1, len(c)))
        self.static = Standardizer().fit(data["static"][train])
        residual = data["y"][train] - data["base"][train]
        self.residual_mean = float(residual.mean())
        self.residual_scale = float(residual.std()) or 1.0
        self.train_count = len(train)
        return self

    def transform(self, data, indices):
        indices = np.asarray(indices, int)
        c = self.cfg["continuous_channels"]
        raw = interpolate_short_gaps(data["x"][indices], c, self.cfg["max_gap"])
        raw[:, :, c] = self.sensor.transform(raw[:, :, c])
        binary = sorted(set(range(raw.shape[-1])) - set(c))
        if binary:
            a = raw[:, :, binary]
            if np.any(np.isfinite(a) & ~np.isin(a, [0, 1])):
                raise ValueError("Binary posture must be 0/1")
            # Median of binary training posture, recorded in fit below.
            raw[:, :, binary] = np.where(np.isfinite(a), a, self.binary_median)
        static_raw = data["static"][indices]
        static = np.column_stack([self.static.transform(static_raw),
                                  (~np.isfinite(static_raw)).astype(np.float32)])
        return dict(x=raw.astype(np.float32), mask=data["mask"][indices].astype(np.float32),
                    static=static.astype(np.float32))

    def fit_all(self, data, train):
        self.fit(data, train)
        binary = sorted(set(range(data["x"].shape[-1])) - set(self.cfg["continuous_channels"]))
        self.binary_median = []
        if binary:
            a = data["x"][train][:, :, binary].reshape(-1, len(binary))
            med = np.nanmedian(a, 0)
            if not np.isfinite(med).all(): raise ValueError("Entire binary training channel missing")
            self.binary_median = (med >= .5).astype(float).tolist()
        return self

    def state(self):
        return dict(cfg=self.cfg, sensor=self.sensor.state(), static=self.static.state(),
                    residual_mean=self.residual_mean, residual_scale=self.residual_scale,
                    binary_median=self.binary_median, train_count=self.train_count)

    @classmethod
    def restore(cls, state):
        obj = cls(state["cfg"])
        obj.sensor = Standardizer.restore(state["sensor"])
        obj.static = Standardizer.restore(state["static"])
        for key in ["residual_mean", "residual_scale", "binary_median", "train_count"]:
            setattr(obj, key, state[key])
        return obj
