"""Real optimisation, validation selection, final refits and checkpoint inference."""
import copy
import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
from .model import BSCG, shuffled_view, reorder_blocks, parameter_count
from .preprocess import Preprocessor
from .metrics import per_cow
from .data import write_json
from .splits import assignment_manifest


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def model_config(cfg, variant):
    out = dict(channels=cfg["channels"], static_width=2*cfg["static_features"],
               segment_points=cfg["segment_points"], width=32, heads=4,
               feedforward=64, dropout=.2, local_layers=1, global_layers=2)
    if variant == "no_local": out["local_layers"] = 0
    if variant == "no_global": out["global_layers"] = 0
    return out


def weights(cows):
    _, inv, counts = np.unique(cows, return_inverse=True, return_counts=True)
    return len(cows)/(len(counts)*counts[inv])


def tensors(pr, data, indices, device):
    return {k: torch.as_tensor(v, device=device) for k, v in pr.transform(data, indices).items()}


def fixed_orders(ids, segments, seed, draw):
    result = []
    for key in ids:
        h = hashlib.sha256(f"{seed}|{draw}|{key}".encode()).digest()[:8]
        result.append(np.random.default_rng(int.from_bytes(h, "big")).permutation(segments))
    return np.asarray(result)


@torch.no_grad()
def predict(model, pr, data, indices, device, permutations=1, shuffled=False, seed=17):
    model.eval(); indices = np.asarray(indices, int)
    output = np.zeros(len(indices), float)
    batch = pr.cfg.get("inference_batch_size", pr.cfg["batch_size"])
    for start in range(0, len(indices), batch):
        ix = indices[start:start+batch]
        t = tensors(pr, data, ix, device)
        value = np.zeros(len(ix), float)
        for draw in range(permutations if shuffled else 1):
            x, m = t["x"], t["mask"]
            if shuffled:
                order = fixed_orders(data["sample_ids"][ix], x.shape[1]//model.segment_points, seed, draw)
                x, m = reorder_blocks(x, m, model.segment_points, torch.as_tensor(order, device=device))
            p, _ = model(x, m, t["static"])
            value += p.cpu().numpy() / (permutations if shuffled else 1)
        output[start:start+len(ix)] = data["base"][ix] + pr.residual_mean + pr.residual_scale*value
    return output


def fit(data, cfg, train, validation, variant, lr, seed, device, epochs):
    seed_all(seed)
    pr = Preprocessor(cfg).fit_all(data, train)
    model = BSCG(**model_config(cfg, variant)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    w = weights(data["cows"][train])
    residual = (data["y"][train]-data["base"][train]-pr.residual_mean)/pr.residual_scale
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed+202)
    best, best_epoch, best_state, wait = float("inf"), 0, None, 0
    shuffled = variant.endswith("_P")
    lam = 0.0 if variant.startswith("BSCG0") else cfg["auxiliary_weight"]
    trace = []
    for epoch in range(1, epochs+1):
        model.train(); order = rng.permutation(len(train)); total, count = 0.0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            local = order[start:start+cfg["batch_size"]]; ix = train[local]
            t = tensors(pr, data, ix, device)
            x, m = t["x"], t["mask"]
            if shuffled: x, m = shuffled_view(x, m, model.segment_points, gen)
            opt.zero_grad(set_to_none=True)
            p, h = model(x, m, t["static"])
            target = torch.as_tensor(residual[local], dtype=p.dtype, device=device)
            loss_rows = (p-target).square()
            if lam:
                loss_rows = loss_rows + lam*model.auxiliary(h, x, m, cfg["continuous_channels"],
                                     cfg["noise_sd"], cfg["temperature"], gen)
            loss = (loss_rows*torch.as_tensor(w[local], dtype=p.dtype, device=device)).mean()
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); total += float(loss.detach())*len(ix); count += len(ix)
        row = dict(epoch=epoch, loss=total/count)
        if validation is not None:
            pred = predict(model, pr, data, validation, device, cfg["permutations"], shuffled, seed)
            score = float(per_cow(pred, data["y"][validation], data["cows"][validation])[1].mean())
            row["validation_macro_mae"] = score
            if score < best-cfg["min_improvement"]:
                best, best_epoch, wait = score, epoch, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else: wait += 1
        else: best_epoch = epoch
        trace.append(row)
        if validation is not None and wait >= cfg["patience"]: break
    if validation is not None:
        if best_state is None: raise RuntimeError("No finite validation epoch")
        model.load_state_dict(best_state)
    return model, pr, dict(best_epoch=best_epoch, best_validation_mae=None if validation is None else best,
                           trace=trace, parameters=parameter_count(model))


def run(data, cfg, splits, out, variant="BSCG", device="cpu", only_fold=None,
        reference_run=None):
    allowed = {"BSCG", "BSCG_P", "BSCG0", "BSCG0_P", "no_local", "no_global"}
    if variant not in allowed: raise ValueError("Unsupported BSCG variant")
    reference = None
    if variant in {"no_local", "no_global"} and reference_run is None:
        raise ValueError("Encoder ablations inherit the full BSCG schedule: supply --reference-run")
    if reference_run is not None:
        root = Path(reference_run)
        prior = json.loads((root/"run_config.json").read_text(encoding="utf-8"))
        if prior["variant"] != "BSCG" or prior["input_sha256"] != data["source_sha256"]:
            raise ValueError("Reference must be an ordered full BSCG run on the same input archive")
        reference = json.loads((root/"selected_configurations.json").read_text(encoding="utf-8"))
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"fold_assignments.json", assignment_manifest(data, splits))
    write_json(out/"environment.json", dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                device=str(device), cuda_available=torch.cuda.is_available()))
    write_json(out/"run_config.json", dict(config=cfg, variant=variant, input_sha256=data["source_sha256"],
                                         only_fold=only_fold, reference_run=str(reference_run) if reference_run else None))
    rows = []; selected_all = {}
    for fold in splits:
        if only_fold and fold["name"] != only_fold: continue
        fd = out/fold["name"]; fd.mkdir()
        candidates = []
        if reference is None:
            for lr in cfg["learning_rates"]:
                evaluations = []
                for j, inner in enumerate(fold["inner"]):
                    model, pr, info = fit(data, cfg, inner["train"], inner["validation"], variant,
                                          lr, cfg["selection_seed"], device, cfg["max_epochs"])
                    evaluations.append(info)
                    write_json(fd/f"candidate_{lr:g}_inner_{j}.json", info)
                    del model
                candidates.append(dict(lr=lr, score=float(np.mean([i["best_validation_mae"] for i in evaluations])),
                                       epochs=[i["best_epoch"] for i in evaluations]))
            chosen = min(candidates, key=lambda c: c["score"])
            epochs = int(np.median(chosen["epochs"]))
        else:
            chosen = reference[fold["name"]]["selected"]
            epochs = reference[fold["name"]]["refit_epochs"]
        selected_all[fold["name"]] = dict(selected=chosen, refit_epochs=epochs, candidates=candidates)
        write_json(out/"selected_configurations.json", selected_all)
        for seed in cfg["seeds"]:
            model, pr, info = fit(data, cfg, fold["train"], None, variant, chosen["lr"], seed, device, epochs)
            checkpoint = dict(model_config=model_config(cfg, variant), state_dict=model.state_dict(),
                              preprocessing=pr.state(), config=cfg, variant=variant, seed=seed,
                              input_sha256=data["source_sha256"], epochs=epochs)
            # Python streams also support non-ASCII Windows paths on older torch.
            with (fd/f"seed_{seed}.pt").open("wb") as handle:
                torch.save(checkpoint, handle)
            write_json(fd/f"refit_seed_{seed}.json", info)
            for cohort, ix in fold["tests"].items():
                p = predict(model, pr, data, ix, device, cfg["permutations"], variant.endswith("_P"), seed)
                for i, value in zip(ix, p):
                    rows.append(dict(cohort=cohort, model=variant, seed=seed, sample_id=data["sample_ids"][i],
                                     prediction=float(value), cow=data["cows"][i], target=float(data["y"][i]),
                                     input_date=data["input_dates"][i], fold=fold["name"]))
            del model
    if not rows: raise ValueError("No selected fold produced predictions")
    with (out/"predictions.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out/"completion.json", dict(complete=True, prediction_rows=len(rows),
                                         executed_folds=list(selected_all), full_fold_run=only_fold is None))
    return out/"predictions.csv"


def load_checkpoint(path, device="cpu"):
    with Path(path).open("rb") as handle:
        ck = torch.load(handle, map_location=device, weights_only=True)
    model = BSCG(**ck["model_config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, Preprocessor.restore(ck["preprocessing"]), ck
