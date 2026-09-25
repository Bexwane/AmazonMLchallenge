"""End-to-end entry point.

  python -m ber.run cv      --data DIR --work DIR --exp ID          # OOF CV + stress CV on train
  python -m ber.run fit     --data DIR --work DIR --exp ID          # final model on all train pairs
  python -m ber.run predict --data DIR --work DIR --exp ID --out output/
  python -m ber.run block   --data DIR --work DIR --exp ID --split test   # blocking only (cache)
  python -m ber.run holdout --data DIR --work DIR --exp ID                # one cheap validation split

DIR is the official `dataset/` folder (holding train/ and test/) or a subset built by
scripts/make_subset.py. Every expensive stage is cached under WORK/<split>/.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .blocking import generate_candidates
from .channels import retrieve
from .config import SEED
from .features import context_features, pair_features, string_features
from .io import read_ground_truth, write_id_lists
from .labels import blocking_report, label_pairs
from .metric import breakdown
from .postprocess import select, tune_threshold
from .prepare import load_prepared

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=63,
              seed=SEED, verbose=-1, num_threads=0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stage_candidates(args, split):
    """Prepared frames and blocking candidates for a split (cached)."""
    s1, s23 = load_prepared(args.data, split, args.work)
    log(f"{split}: {len(s1)} S1, {len(s23)} S2/S3")
    d = Path(args.work) / split
    d.mkdir(parents=True, exist_ok=True)
    pc = d / f"cand_{args.block_tag}.parquet"
    if pc.exists():
        cand = pd.read_parquet(pc)
    else:
        if args.channels:
            cand = retrieve(s1, s23, args.channels, m=args.rrf_m, df_cap=args.df_cap, log=log)
        else:
            cand = generate_candidates(s1, s23, k_comb=args.k_comb, k_name=args.k_name, df_cap=args.df_cap,
                                       chunk=args.block_chunk, log=log)
        cand.to_parquet(pc, index=False)
    log(f"{split}: {len(cand)} candidate pairs ({len(cand) / len(s1):.1f}/S1)")
    return s1, s23, cand


def stage_train(args):
    """Train split with features for the pairs of a random `cv_frac` sample of Source-1 records.
    Context features are computed on all pairs first, so competition between S1s is realistic."""
    s1, s23, cand = stage_candidates(args, "train")
    rng = np.random.default_rng(SEED)
    s1_keep = rng.random(len(s1)) < args.cv_frac
    rows = s1_keep[cand["s1_idx"].to_numpy()]
    pf = Path(args.work) / "train" / f"feat2_{args.block_tag}_f{args.cv_frac:g}.parquet"
    if pf.exists():
        F = pd.read_parquet(pf)
    else:
        t = time.time()
        F = pair_features(cand, s1, s23, rows=rows).reset_index(drop=True)
        F.to_parquet(pf, index=False)
        log(f"train: features built in {(time.time() - t) / 60:.1f} min")
    cand = cand.loc[rows, ["s1_idx", "r_idx", "name_cos", "addr_cos"]].reset_index(drop=True)
    gc.collect()
    log(f"train: features {F.shape} for {int(s1_keep.sum())} sampled S1")
    return s1, s23, cand, F, s1["entity_id"].to_numpy()[s1_keep]


def train_lgb(X, y, Xv=None, yv=None, rounds=1500):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is None:
        return lgb.train(PARAMS, dtr, num_boost_round=rounds)
    dv = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def cmd_cv(args):
    s1, s23, cand, F, eval_ids = stage_train(args)
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    truth = {k: truth[k] for k in eval_ids}
    y = label_pairs(cand, s1, s23, truth)
    brep = blocking_report(cand, y, s1, s23, truth)
    log("blocking", json.dumps(brep))
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    groups = cand["s1_idx"].to_numpy()
    oof = np.zeros(len(cand), np.float32)
    iters = []
    for k, (tr, va) in enumerate(GroupKFold(args.folds).split(F, y, groups)):
        m = train_lgb(F.iloc[tr], y[tr], F.iloc[va], y[va])
        oof[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"fold {k}: best_iter={m.best_iteration}")
    all_ids = eval_ids
    thr, grid = tune_threshold(cand, oof, s1_ids, r_ids, truth, all_ids)
    pred = select(cand, oof, s1_ids, r_ids, thr)
    res = {"cv": breakdown(pred, truth, all_ids), "threshold": thr, "grid": grid, "best_iters": iters,
           "no_exclusive": breakdown(select(cand, oof, s1_ids, r_ids, thr, exclusive=False), truth, all_ids)}
    s1c = s1["country"].to_numpy()
    ec = s1.set_index("entity_id").loc[all_ids, "country"].to_numpy()
    for c in sorted(set(ec)):
        res[f"cv_{c}"] = breakdown(pred, truth, all_ids[ec == c])
    # stress: hold out one whole country (proxy for the unseen test country)
    cc = s1c[groups]
    for c in sorted(set(ec)):
        tr, va = np.flatnonzero(cc != c), np.flatnonzero(cc == c)
        if len(tr) == 0 or len(va) == 0:
            continue
        m = train_lgb(F.iloc[tr], y[tr], rounds=int(np.mean(iters)))
        p = np.zeros(len(cand), np.float32)
        p[va] = m.predict(F.iloc[va])
        ids_c = all_ids[ec == c]
        pred_c = select(cand.iloc[va], p[va], s1_ids, r_ids, thr)
        res[f"stress_holdout_{c}"] = breakdown(pred_c, truth, ids_c)
    imp = pd.Series(m.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    res["blocking"] = brep
    res["feature_gain_top"] = (imp / imp.sum()).round(4).head(20).to_dict()
    out = Path(args.work) / "experiments" / args.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv_metrics.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({"s1_idx": cand["s1_idx"], "r_idx": cand["r_idx"], "oof": oof, "y": y}).to_parquet(out / "oof.parquet")
    log("CV", json.dumps({k: v for k, v in res.items() if k.startswith(("cv", "stress", "threshold", "no_excl"))}, indent=1))


def cmd_holdout(args):
    """One cheap validation split (spec: no repeated k-fold). Sampled S1s are split 70/15/15 by entity:
    train / val-A (early stopping + threshold) / val-B (reported, untouched by any tuning)."""
    s1, s23, cand, F, eval_ids = stage_train(args)
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    truth = {k: truth[k] for k in eval_ids}
    y = label_pairs(cand, s1, s23, truth)
    brep = blocking_report(cand, y, s1, s23, truth)
    log("blocking (sampled S1)", json.dumps(brep))
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    u = np.random.default_rng(SEED + 1).random(len(s1))[cand["s1_idx"].to_numpy()]
    tr, va, vb = np.flatnonzero(u < 0.70), np.flatnonzero((u >= 0.70) & (u < 0.85)), np.flatnonzero(u >= 0.85)
    t = time.time()
    m = train_lgb(F.iloc[tr], y[tr], F.iloc[va], y[va])
    log(f"trained on {len(tr)} pairs in {(time.time() - t) / 60:.1f} min, best_iter={m.best_iteration}")
    p = np.zeros(len(cand), np.float32)
    p[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
    p[vb] = m.predict(F.iloc[vb], num_iteration=m.best_iteration)
    # evaluation S1s of each half: every sampled S1 falls in exactly one half (singletons without candidates too)
    u_s1 = pd.Series(np.random.default_rng(SEED + 1).random(len(s1)), index=s1_ids)
    ev = pd.Series(eval_ids)
    ids_a = ev[((u_s1[eval_ids] >= 0.70) & (u_s1[eval_ids] < 0.85)).to_numpy()].to_numpy()
    ids_b = ev[(u_s1[eval_ids] >= 0.85).to_numpy()].to_numpy()
    thr, grid = tune_threshold(cand.iloc[va], p[va], s1_ids, r_ids, truth, ids_a)
    pred_b = select(cand.iloc[vb], p[vb], s1_ids, r_ids, thr)
    res = {"val_B": breakdown(pred_b, truth, ids_b), "threshold": thr, "grid_val_A": grid,
           "best_iter": m.best_iteration, "n_train_pairs": int(len(tr)), "blocking_sampled": brep,
           "val_B_no_exclusive": breakdown(select(cand.iloc[vb], p[vb], s1_ids, r_ids, thr, exclusive=False), truth, ids_b)}
    ec = s1.set_index("entity_id").loc[ids_b, "country"].to_numpy()
    for c in sorted(set(ec)):
        res[f"val_B_{c}"] = breakdown(pred_b, truth, ids_b[ec == c])
    imp = pd.Series(m.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    res["feature_gain_top"] = (imp / imp.sum()).round(4).head(25).to_dict()
    out = Path(args.work) / "experiments" / args.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_metrics.json").write_text(json.dumps(res, indent=1))
    log("HOLDOUT", json.dumps({k: v for k, v in res.items() if k.startswith(("val_B", "threshold", "best_iter"))}, indent=1))


def cmd_block(args):
    stage_candidates(args, args.split)


def cmd_fit(args):
    s1, s23, cand, F, _ = stage_train(args)
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    y = label_pairs(cand, s1, s23, truth)
    out = Path(args.work) / "experiments" / args.exp
    if (out / "cv_metrics.json").exists():
        cv = json.loads((out / "cv_metrics.json").read_text())
        rounds, thr, src = int(np.mean(cv["best_iters"]) * 1.1), cv["threshold"], "cv"
    else:
        ho = json.loads((out / "holdout_metrics.json").read_text())
        rounds, thr, src = int(ho["best_iter"] * 1.15), ho["threshold"], "holdout"
    del cand, s1, s23
    gc.collect()
    m = train_lgb(F, y, rounds=rounds)
    m.save_model(str(out / "model.txt"))
    (out / "fit.json").write_text(json.dumps({"rounds": rounds, "threshold": thr, "n_pairs": int(len(y)),
                                              "settings_from": src, "block_tag": args.block_tag}))
    log(f"saved model ({rounds} rounds, {len(y)} pairs, threshold {thr} from {src})")


def cmd_predict(args):
    out = Path(args.work) / "experiments" / args.exp
    fit = json.loads((out / "fit.json").read_text())
    m = lgb.Booster(model_file=str(out / "model.txt"))
    s1, s23, cand = stage_candidates(args, "test")
    C = context_features(cand)
    cand = cand[["s1_idx", "r_idx"]]  # the extra blocking columns now live in C
    gc.collect()
    p = np.empty(len(cand), np.float32)
    step = 4_000_000  # stream string features so the full test matrix never sits in memory
    for a in range(0, len(cand), step):
        sub = cand.iloc[a:a + step]
        Fc = pd.concat([string_features(sub, s1, s23), C.iloc[a:a + step]], axis=1)
        p[a:a + step] = m.predict(Fc[m.feature_name()])
        log(f"predicted {min(a + step, len(cand))}/{len(cand)}")
    pd.DataFrame({"s1_idx": cand["s1_idx"], "r_idx": cand["r_idx"], "p": p}).to_parquet(out / "test_pairs_proba.parquet")
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    pred = select(cand, p, s1_ids, r_ids, fit["threshold"])
    cands = {}
    for a, b in zip(s1_ids[cand["s1_idx"].to_numpy()], r_ids[cand["r_idx"].to_numpy()]):
        cands.setdefault(a, []).append(b)
    od = Path(args.out)
    write_id_lists(od / "matching_results.tsv", s1_ids, pred, "matched_entity_ids")
    write_id_lists(od / "candidate_pairs.tsv", s1_ids, cands, "candidate_entity_ids")
    n = sum(len(v) for v in pred.values())
    log(f"wrote {od}: {n} matches for {len(pred)}/{len(s1)} S1 ({len(s1) - len(pred)} predicted singletons)")


def block_tag(args):
    """Cache tag of a blocker configuration (shared with scripts/eval_blocking_full.py)."""
    if args.channels:
        return f"ch-{args.channels.replace(',', '+').replace(':', '')}_m{args.rrf_m}_df{args.df_cap}"
    return f"k{args.k_comb}_n{args.k_name}_df{args.df_cap}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "fit", "predict", "block", "holdout"])
    ap.add_argument("--split", default="train", choices=["train", "test"], help="for `block`")
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k-comb", type=int, default=40)
    ap.add_argument("--k-name", type=int, default=10)
    ap.add_argument("--df-cap", type=int, default=2000)
    ap.add_argument("--cv-frac", type=float, default=1.0, help="fraction of train S1 used for CV/fit features")
    ap.add_argument("--block-chunk", type=int, default=2000)
    ap.add_argument("--channels", default="", help="multi-channel blocker, e.g. base_wide,conj,bm25,name_noaddr,rescue "
                                                    "(empty = legacy E003 blocker)")
    ap.add_argument("--rrf-m", type=int, default=0, help="optional cap per S1 after RRF (0 = keep the whole union; "
                                                         "rescue pairs are always kept)")
    ap.add_argument("--lr", type=float, default=0.1, help="LightGBM learning rate (0.05 in E001-E003)")
    args = ap.parse_args()
    PARAMS["learning_rate"] = args.lr
    args.block_tag = block_tag(args)
    {"cv": cmd_cv, "fit": cmd_fit, "predict": cmd_predict, "block": cmd_block, "holdout": cmd_holdout}[args.cmd](args)


if __name__ == "__main__":
    main()
