"""python -m bscg --help"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import load_data, write_json
from .splits import make_splits, check_splits, assignment_manifest


def main():
    p = argparse.ArgumentParser(description="BSCG training and reproducibility tools")
    subs = p.add_subparsers(dest="command", required=True)
    for name in ["validate", "splits", "train", "inspect-model"]:
        q = subs.add_parser(name); q.add_argument("--config", required=True)
        if name != "inspect-model": q.add_argument("--data", required=True)
        if name in {"splits", "train"}:
            q.add_argument("--condition", required=True, choices=["mm_loco","mm_time","mm_cow_time","cowbase"])
            q.add_argument("--out", required=True)
        if name == "train":
            q.add_argument("--variant", default="BSCG", choices=["BSCG","BSCG_P","BSCG0","BSCG0_P","no_local","no_global"])
            q.add_argument("--device", default="cpu"); q.add_argument("--only-fold")
            q.add_argument("--reference-run", help="Full BSCG run supplying the fixed ablation/scale schedule")
    q = subs.add_parser("construct")
    for a in ["config", "sensor", "milk", "state", "out"]: q.add_argument("--"+a, required=True)
    q = subs.add_parser("predict")
    for a in ["checkpoint", "data", "out"]: q.add_argument("--"+a, required=True)
    q.add_argument("--device", default="cpu")
    q = subs.add_parser("table2"); q.add_argument("--plan", required=True); q.add_argument("--out", required=True)
    q.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    if a.command == "table2":
        from .table2 import regenerate
        regenerate(a.plan, a.out, a.require_complete); return
    if a.command == "predict":
        import pandas as pd
        from .engine import load_checkpoint, predict
        model, pr, ck = load_checkpoint(a.checkpoint, a.device)
        d = load_data(a.data, ck["config"], require_target=False)
        values = predict(model, pr, d, np.arange(len(d["x"])), a.device,
                         ck["config"]["permutations"], ck["variant"].endswith("_P"), ck["seed"])
        pd.DataFrame(dict(sample_id=d["sample_ids"], prediction=values)).to_csv(a.out, index=False)
        return
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.command == "construct":
        import pandas as pd
        from .construction import construct
        arrays = construct(*[pd.read_csv(getattr(a,k)) for k in ["sensor","milk","state"]], cfg)
        path = Path(a.out)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays); load_data(path, cfg)
        return
    if a.command == "inspect-model":
        from .model import BSCG, parameter_count
        from .engine import model_config
        print(json.dumps({v: parameter_count(BSCG(**model_config(cfg,v)))
                          for v in ["BSCG","BSCG0","no_local","no_global"]}, indent=2)); return
    d = load_data(a.data, cfg)
    if a.command == "validate":
        print(json.dumps(dict(samples=len(d["x"]), cows=len(set(d["cows"])), input_sha256=d["source_sha256"]))); return
    splits = make_splits(d, cfg, a.condition); check_splits(d, splits)
    if a.command == "splits": write_json(a.out, assignment_manifest(d, splits)); return
    from .engine import run
    print(run(d, cfg, splits, a.out, a.variant, a.device, a.only_fold, a.reference_run))


if __name__ == "__main__": main()
