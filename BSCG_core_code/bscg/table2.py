"""Regenerate Table 2 from local data and actual prediction CSV files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import load_data, write_json
from .splits import make_splits, check_splits
from .metrics import scores, paired, per_cow


def regenerate(plan_path, out, require_complete=False):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    resolve = lambda p: (plan_path.parent / p).resolve()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(resolve(p), dtype={"sample_id": str, "cow": str}) for p in plan["prediction_files"]]
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(predictions):
        required = {"cohort", "model", "seed", "sample_id", "prediction"}
        if not required <= set(predictions): raise ValueError(f"CSV requires {sorted(required)}")
        if predictions.duplicated(["cohort", "model", "seed", "sample_id"]).any():
            raise ValueError("Duplicate prediction keys across input files")
        if not np.isfinite(predictions.prediction).all(): raise ValueError("Nonfinite predictions")
    expected_cases = {}; settings = {}
    for entry in plan["datasets"]:
        cfg = json.loads(resolve(entry["config"]).read_text(encoding="utf-8"))
        data = load_data(resolve(entry["data"]), cfg)
        conditions = ["mm_loco", "mm_time", "mm_cow_time"] if cfg["dataset"] == "mmcows" else ["cowbase"]
        for condition in conditions:
            splits = make_splits(data, cfg, condition); check_splits(data, splits)
            for cohort in {k for f in splits for k in f["tests"]}:
                ix = np.concatenate([f["tests"][cohort] for f in splits if cohort in f["tests"]])
                if len(set(ix)) != len(ix): raise ValueError("Duplicate outer test rows")
                ix = ix[np.argsort(data["sample_ids"][ix])]
                expected_cases[cohort] = (data, ix)
                settings[cohort] = cfg
    metric_rows = []; pair_rows = []; cow_rows = []; seed_rows = []; missing = []
    if len(predictions) and set(predictions.cohort) - set(expected_cases):
        raise ValueError("Prediction file contains cohorts absent from the plan")
    case_order = [c for c in ["MM_LOCO", "MM_TIME", "MM_BOTH", "CB_SEEN", "CB_UNSEEN"] if c in expected_cases]
    for cohort in case_order:
        data, ix = expected_cases[cohort]
        ids = data["sample_ids"][ix]; truth = data["y"][ix]; cows = data["cows"][ix]
        outputs = {"Persistence": data["base"][ix], "RecentMean": data["history"][ix].mean(1)}
        sub = predictions[predictions.cohort == cohort] if len(predictions) else pd.DataFrame()
        if len(sub):
            for model, group in sub.groupby("model"):
                if model in outputs: raise ValueError("Fixed baselines are computed directly, not imported")
                vectors = []
                for seed, g in group.groupby("seed"):
                    if set(g.sample_id) != set(ids): raise ValueError(f"Unmatched cohort: {cohort}/{model}/{seed}")
                    g = g.set_index("sample_id").loc[ids]
                    if "target" in g and not np.allclose(g.target, truth, rtol=1e-6, atol=1e-5):
                        raise ValueError("Imported prediction targets disagree with supplied data")
                    if "cow" in g and not np.array_equal(g.cow.astype(str), cows):
                        raise ValueError("Imported cow IDs disagree")
                    vector = g.prediction.to_numpy(float); vectors.append(vector)
                    seed_rows.append(dict(cohort=cohort, model=model, seed=seed,
                        mae=float(per_cow(vector, truth, cows)[1].mean())))
                if model.startswith("BSCG") and set(group.seed.astype(int)) != set(settings[cohort]["seeds"]):
                    raise ValueError("BSCG must include every configured refit seed")
                outputs[model] = np.mean(vectors, axis=0)
        required = ["Persistence", "RecentMean", "HGB_H", "HGB_HS", "Transformer", "BSCG"]
        if cohort == "MM_LOCO": required += ["LSTM", "TCN"]
        missing += [dict(cohort=cohort, model=m) for m in required if m not in outputs]
        draws = 5000 if cohort.startswith("MM") else 2000
        base_seed = 4470 + ["MM_LOCO", "MM_TIME", "MM_BOTH", "CB_SEEN", "CB_UNSEEN"].index(cohort)
        for model, vector in outputs.items():
            seed = 4471 if cohort in {"MM_TIME", "MM_BOTH"} and model in {"Persistence", "RecentMean"} else base_seed
            metric_rows.append(dict(cohort=cohort, model=model, **scores(vector, truth, cows, draws, seed)))
            names, a, b = per_cow(vector, truth, cows)
            cow_rows += [dict(cohort=cohort, model=model, cow=str(c), mae=float(mae), rmse=float(rmse))
                         for c, mae, rmse in zip(names, a, b)]
            if "BSCG" in outputs and model != "BSCG":
                pair_rows.append(dict(cohort=cohort, control=model, target_model="BSCG",
                                      **paired(vector, outputs["BSCG"], truth, cows, draws, base_seed)))
    if require_complete and missing: raise ValueError(f"Table 2 is incomplete: {missing}")
    pd.DataFrame(metric_rows).to_csv(out/"table2_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out/"table2_paired.csv", index=False)
    pd.DataFrame(cow_rows).to_csv(out/"per_cow_metrics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out/"per_seed_metrics.csv", index=False)
    write_json(out/"table2_completeness.json", dict(complete=not missing, missing_models=missing,
                note="Comparators are imported; no scores or manuscript values are hard-coded."))
    rows = ["# Table 2", "", "Complete model coverage: " + str(not missing), "",
            "| Cohort | Model | Macro-cow MAE [95% CI] | Macro-cow RMSE [95% CI] |",
            "|---|---|---:|---:|"]
    for r in metric_rows:
        a, b = r["mae_ci"], r["rmse_ci"]
        rows.append(f'| {r["cohort"]} | {r["model"]} | {r["mae"]:.2f} [{a[0]:.2f}, {a[1]:.2f}] | '
                    f'{r["rmse"]:.2f} [{b[0]:.2f}, {b[1]:.2f}] |')
    (out/"Table2.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    return metric_rows
