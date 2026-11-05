"""Generate exact local fold assignments from supplied cow/date keys."""
import hashlib
import numpy as np


def grouped_inner(cows, train, folds=3):
    ids = sorted(set(cows[train]))
    if len(ids) < folds: raise ValueError("Not enough training cows for inner folds")
    groups = [ids[i::folds] for i in range(folds)]
    return [dict(train=train[~np.isin(cows[train], g)],
                 validation=train[np.isin(cows[train], g)]) for g in groups]


def make_splits(data, cfg, condition):
    cows, dates, targets = data["cows"], data["input_dates"], data["target_dates"]
    all_idx = np.arange(len(cows))
    if cfg["dataset"] == "mmcows":
        keep = (dates >= cfg["input_start"]) & (dates <= cfg["input_end"])
        ids = sorted(set(cows[keep]))
        if condition == "mm_loco":
            return [dict(name=f"LOCO_{c}", train=all_idx[keep & (cows != c)],
                         inner=grouped_inner(cows, all_idx[keep & (cows != c)]),
                         tests={"MM_LOCO": all_idx[keep & (cows == c)]}) for c in ids]
        early = keep & (dates <= cfg["fit_end"])
        late = keep & (dates >= cfg["test_start"])
        if condition == "mm_time":
            tr = all_idx[early]
            return [dict(name="time", train=tr, inner=grouped_inner(cows, tr),
                         tests={"MM_TIME": all_idx[late]})]
        if condition == "mm_cow_time":
            out = []
            for cow in ids:
                tr = all_idx[early & (cows != cow)]
                out.append(dict(name=f"COW_TIME_{cow}", train=tr,
                                inner=grouped_inner(cows, tr),
                                tests={"MM_BOTH": all_idx[late & (cows == cow)]}))
            return out
        raise ValueError("Unknown MMCOWS condition")
    if condition != "cowbase": raise ValueError("Use condition cowbase")
    def held(cow):
        # Canonical integer IDs match the manuscript's hash split.
        text = f"CowBase_external_v1|{int(cow)}"
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 5 == 0
    holdout = np.array([held(c) for c in cows])
    eligible = dates >= cfg["input_start"]
    fit = all_idx[eligible & ~holdout & (dates <= cfg["fit_end"])]
    validation = all_idx[eligible & ~holdout & (dates >= cfg["validation_start"])
                         & (dates <= cfg["validation_end"])]
    refit = all_idx[eligible & ~holdout & (dates <= cfg["refit_end"])]
    future = (targets >= cfg["target_start"]) & (targets <= cfg["target_end"])
    seen = np.isin(cows, cows[refit])
    return [dict(name="temporal", train=refit,
                 inner=[dict(train=fit, validation=validation)],
                 tests={"CB_SEEN": all_idx[future & seen],
                        "CB_UNSEEN": all_idx[future & ~seen]})]


def check_splits(data, splits):
    for fold in splits:
        if not len(fold["train"]): raise ValueError("Empty refit partition")
        for cohort, te in fold["tests"].items():
            if not len(te): raise ValueError(f"Empty test cohort: {cohort}")
            if set(te) & set(fold["train"]): raise ValueError("Train/test row overlap")
            if cohort in {"MM_LOCO", "MM_BOTH", "CB_UNSEEN"}:
                if set(data["cows"][te]) & set(data["cows"][fold["train"]]):
                    raise ValueError("Animal leakage")
            if cohort != "MM_LOCO":
                if max(data["target_dates"][fold["train"]]) >= min(data["input_dates"][te]):
                    raise ValueError("Training target reaches the future input window")
        for inner in fold["inner"]:
            tr, va = inner["train"], inner["validation"]
            if not len(tr) or not len(va) or set(tr) & set(va):
                raise ValueError("Invalid inner train/validation assignment")
            if not set(tr).union(va) <= set(fold["train"]):
                raise ValueError("Inner samples leave the outer development set")
            if fold["name"] == "temporal":
                if max(data["target_dates"][tr]) >= min(data["input_dates"][va]):
                    raise ValueError("Inner temporal boundary leaks training labels")
            elif set(data["cows"][tr]) & set(data["cows"][va]):
                raise ValueError("Inner animal leakage")


def assignment_manifest(data, splits):
    result = []
    for fold in splits:
        def ids(indices): return data["sample_ids"][indices].tolist()
        result.append(dict(name=fold["name"], train=ids(fold["train"]),
                           inner=[{k: ids(v) for k, v in inner.items()} for inner in fold["inner"]],
                           tests={k: ids(v) for k, v in fold["tests"].items()}))
    return dict(input_sha256=data["source_sha256"], folds=result)
