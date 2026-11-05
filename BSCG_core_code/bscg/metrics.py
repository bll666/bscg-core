"""Equal-cow metrics and paired percentile bootstrap, calculated from predictions."""
import numpy as np


def per_cow(prediction, target, cows):
    p, y, c = np.asarray(prediction), np.asarray(target), np.asarray(cows)
    if p.shape != y.shape or p.shape != c.shape or not np.isfinite(p-y).all():
        raise ValueError("Invalid predictions/targets/cow keys")
    ids, inv = np.unique(c, return_inverse=True)
    counts = np.bincount(inv)
    e = p-y
    mae = np.bincount(inv, weights=np.abs(e)) / counts
    rmse = np.sqrt(np.bincount(inv, weights=e*e) / counts)
    return ids, mae, rmse


def interval(values, draws, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    # Batches bound memory for cohorts with many animals.
    means = []
    for start in range(0, draws, 250):
        ix = rng.integers(0, len(values), (min(250, draws-start), len(values)))
        means.append(values[ix].mean(1))
    return np.quantile(np.concatenate(means), [.025, .975]).tolist()


def scores(prediction, target, cows, draws=2000, seed=4473):
    ids, mae, rmse = per_cow(prediction, target, cows)
    return dict(cows=len(ids), samples=len(target), mae=float(mae.mean()),
                rmse=float(rmse.mean()), mae_ci=interval(mae, draws, seed),
                rmse_ci=interval(rmse, draws, seed))


def paired(control, target_prediction, truth, cows, draws=2000, seed=4473):
    ids, a, _ = per_cow(control, truth, cows)
    _, b, _ = per_cow(target_prediction, truth, cows)
    delta = a-b
    return dict(gain=float(delta.mean()), ci=interval(delta, draws, seed),
                improved=int((delta > 0).sum()), total=len(ids))
