"""End-to-end entry point.

  python -m ber.run cv      --data DIR --work DIR --exp ID          # OOF CV + stress CV on train
  python -m ber.run fit     --data DIR --work DIR --exp ID          # final model on all train pairs
  python -m ber.run predict --data DIR --work DIR --exp ID --out output/
  python -m ber.run block   --data DIR --work DIR --exp ID --split test   # blocking only (cache)
  python -m ber.run holdout --data DIR --work DIR --exp ID                # one cheap validation split
  python -m ber.run stack   --data DIR --work DIR --exp ID                # stage 2 on the cv out-of-fold p
  python -m ber.run select  --data DIR --work DIR --exp ID --out DIR [--count-match unseen]  # re-select saved p
  python -m ber.run dense   --data DIR --work DIR --exp ID --split test --dense-model e5s   # dense cosines (GPU, cache)

DIR is the official `dataset/` folder (holding train/ and test/) or a subset built by
scripts/make_subset.py. Every expensive stage is cached under WORK/<split>/.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .blocking import generate_candidates
from .buckets import assign_modes, eval_slices, name_frequency, pair_table
from .channels import pair_conj_cos, retrieve
from .config import SEED
from .dense import DENSE_INFO, dense_context_arrays, load_encoder, pair_dense_cos
from .features import (context2_arrays, context_arrays, context_columns, context_features, crowd_arrays,
                       crowd_features, extra_features, name_amb_arrays, name_amb_features, pair_features,
                       string_features)
from .io import read_ground_truth, write_grouped
from .labels import blocking_report, label_pairs
from .metric import breakdown
from .postprocess import (apply_calibrator, count_match_delta, fit_calibrator, rule_mask, select, select_rule,
                          selection_study, shift_logit, tune_threshold)
from .prepare import load_prepared
from .profile import profile_arrays, profile_features
from .stack import stack_features

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
    args.ctx_cols = context_columns(cand)
    X4 = None
    if args.feat in ("v4", "v5", "v6"):  # name/address competition + shared-address counts (see features.context2_arrays)
        p4 = Path(args.work) / "train" / f"featx4_{args.block_tag}_f{args.cv_frac:g}.parquet"
        if p4.exists():
            X4 = pd.read_parquet(p4)
        else:
            t = time.time()
            si, ri = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
            C2 = context2_arrays(si, ri, cand["name_cos"].to_numpy(), cand["addr_cos"].to_numpy(), rows)
            X4 = pd.DataFrame({k: v.astype(np.float32) for k, v in C2.items()})
            del C2
            X4 = pd.concat([X4, crowd_features(si[rows], ri[rows], crowd_arrays(s1, s23))], axis=1)
            X4.to_parquet(p4, index=False)
            log(f"train: v4 features built in {(time.time() - t) / 60:.1f} min")
    if args.feat in ("v5", "v6"):  # global name ambiguity (+ dense multilingual cosines unless --dense-model none)
        pa = Path(args.work) / "train" / f"featx5a_{args.block_tag}_f{args.cv_frac:g}.parquet"
        if pa.exists():
            XA = pd.read_parquet(pa)
        else:
            si, ri = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
            XA = name_amb_features(si[rows], ri[rows], name_amb_arrays(s1, s23)).reset_index(drop=True)
            XA.to_parquet(pa, index=False)
        X4 = pd.concat([X4, XA], axis=1)
    if args.feat in ("v5", "v6") and args.dense_model != "none":
        p5 = Path(args.work) / "train" / f"featx5_{args.dense_model}_{args.block_tag}_f{args.cv_frac:g}.parquet"
        if p5.exists():
            X5 = pd.read_parquet(p5)
        else:
            si, ri = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
            D = dense_arrays(args, "train", s1, s23, si, ri)
            X5 = pd.DataFrame({k: v.astype(np.float32) for k, v in dense_context_arrays(si, ri, D, rows).items()})
            del D
            X5.to_parquet(p5, index=False)
        X4 = pd.concat([X4, X5], axis=1)
    cand = cand.loc[rows, ["s1_idx", "r_idx", "name_cos", "addr_cos"]].reset_index(drop=True)
    gc.collect()
    if args.feat == "v6":  # legal-form / business-word / house-number profiles (see profile.py) at stage 1 too
        p6 = Path(args.work) / "train" / f"featx6_{args.block_tag}_f{args.cv_frac:g}.parquet"
        if p6.exists():
            X6 = pd.read_parquet(p6)
        else:
            t = time.time()
            si, ri = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
            X6 = profile_features(si, ri, profile_arrays(s1, s23), amb=False).reset_index(drop=True)  # cand = sampled rows here
            X6.to_parquet(p6, index=False)
            log(f"train: v6 profile features built in {(time.time() - t) / 60:.1f} min")
        X4 = pd.concat([X4, X6], axis=1)
    if args.feat in ("v3", "v4", "v5", "v6"):  # extra pair features, cached separately so the v2 matrix is reused
        px = Path(args.work) / "train" / f"featx3_{args.block_tag}_f{args.cv_frac:g}.parquet"
        if px.exists():
            X = pd.read_parquet(px)
        else:
            t = time.time()
            X = extra_features(cand, s1, s23).reset_index(drop=True)
            X["conj_cos"] = pair_conj_cos(cand, s1, s23, log=log)
            X.to_parquet(px, index=False)
            log(f"train: v3 features built in {(time.time() - t) / 60:.1f} min")
        F = pd.concat([F, X], axis=1)
    if X4 is not None:
        F = pd.concat([F, X4], axis=1)
    log(f"train: features {F.shape} for {int(s1_keep.sum())} sampled S1")
    return s1, s23, cand, F, s1["entity_id"].to_numpy()[s1_keep]


def dense_arrays(args, split, s1, s23, si, ri):
    """Per-pair dense cosines for the full candidate table of a split, cached as float16 .npy."""
    d = Path(args.work) / split
    paths = {f: d / f"dense_{args.dense_model}_{f}_{args.block_tag}.npy" for f in ("name", "full")}
    if all(p.exists() for p in paths.values()):
        D = {f: np.load(p) for f, p in paths.items()}
        assert all(len(v) == len(si) for v in D.values()), "dense cache does not match the candidate table"
        return D
    t = time.time()
    enc = load_encoder(args.dense_model)
    D = pair_dense_cos(s1, s23, si, ri, enc, prefix=getattr(enc, "prefix", ""), log=log)
    del enc
    gc.collect()
    for f, p in paths.items():
        np.save(p, D[f])
    DENSE_INFO.update({"split": split, "pairs": int(len(si)), "minutes": round((time.time() - t) / 60, 1)})
    (d / f"dense_{args.dense_model}_info.json").write_text(json.dumps(DENSE_INFO, indent=1))
    log(f"{split}: dense cosines for {len(si)} pairs in {(time.time() - t) / 60:.1f} min: {json.dumps(DENSE_INFO)}")
    return D


def cmd_dense(args):
    """Dense cosines for a split's cached candidates (GPU), so cv/stack/predict find them cached."""
    s1, s23 = load_prepared(args.data, args.split, args.work)
    pc = Path(args.work) / args.split / f"cand_{args.block_tag}.parquet"
    if not pc.exists():
        stage_candidates(args, args.split)
    cols = _read_columns(pc)
    dense_arrays(args, args.split, s1, s23, cols["s1_idx"], cols["r_idx"])


def train_lgb(X, y, Xv=None, yv=None, rounds=1500, w=None, wv=None):
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    if Xv is None:
        return lgb.train(PARAMS, dtr, num_boost_round=rounds)
    dv = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
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
    fold = np.zeros(len(cand), np.int8)
    iters = []
    out = Path(args.work) / "experiments" / args.exp
    out.mkdir(parents=True, exist_ok=True)
    for k, (tr, va) in enumerate(GroupKFold(args.folds).split(F, y, groups)):
        m = train_lgb(F.iloc[tr], y[tr], F.iloc[va], y[va])
        oof[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
        fold[va] = k
        iters.append(m.best_iteration)
        m.save_model(str(out / f"model_f{k}.txt"), num_iteration=m.best_iteration)
        log(f"fold {k}: best_iter={m.best_iteration}")
    all_ids = eval_ids
    thr, grid = tune_threshold(cand, oof, s1_ids, r_ids, truth, all_ids)
    pred = select(cand, oof, s1_ids, r_ids, thr)
    res = {"cv": breakdown(pred, truth, all_ids), "threshold": thr, "grid": grid, "best_iters": iters,
           "no_exclusive": breakdown(select(cand, oof, s1_ids, r_ids, thr, exclusive=False), truth, all_ids)}
    rule, f_rule, table = selection_study(cand, oof, y, s1_ids, r_ids, truth, all_ids, fold, log=log)
    res["selection"] = {"best": {k: v for k, v in rule.items() if k != "cal"}, "macro_f05": f_rule, "table": table[:15]}
    res["rule"] = rule
    s1c = s1["country"].to_numpy()
    ec = s1.set_index("entity_id").loc[all_ids, "country"].to_numpy()
    for c in sorted(set(ec)):
        res[f"cv_{c}"] = breakdown(pred, truth, all_ids[ec == c])
    for b, ids_b in eval_slices(s1, s23, truth, all_ids).items():  # script / crowd / cluster-size diagnostics
        res[f"slice_{b}"] = breakdown(pred, truth, ids_b)
    # stress: hold out one whole country (proxy for the unseen test country)
    cc = s1c[groups]
    rule_nc = {k: v for k, v in rule.items() if k != "cal"}
    target = rule_mask(cand, oof if rule["method"] == "threshold" else apply_calibrator(rule["cal"], oof),
                       {k: v for k, v in rule.items() if k != "cal"}).sum() / len(all_ids)
    res["selected_per_s1_oof"] = float(target)
    for c in sorted(set(ec)) if args.stress else []:
        tr, va = np.flatnonzero(cc != c), np.flatnonzero(cc == c)
        if len(tr) == 0 or len(va) == 0:
            continue
        m = train_lgb(F.iloc[tr], y[tr], rounds=int(np.mean(iters)))
        p = np.zeros(len(cand), np.float32)
        p[va] = m.predict(F.iloc[va])
        ids_c = all_ids[ec == c]
        cv_, pv = cand.iloc[va], p[va]
        # calibrator and matches-per-S1 target from the training countries only (no peeking at country c)
        rule_c = {**rule_nc, "cal": fit_calibrator(oof[tr], y[tr])} if rule["method"] == "expf" else rule_nc
        q_tr = apply_calibrator(rule_c["cal"], oof[tr]) if rule["method"] == "expf" else oof[tr]
        target_c = rule_mask(cand.iloc[tr], q_tr, {k: v for k, v in rule_c.items() if k != "cal"}).sum() / int((ec != c).sum())
        res[f"stress_holdout_{c}"] = breakdown(select(cv_, pv, s1_ids, r_ids, thr), truth, ids_c)
        res[f"stress_rule_{c}"] = breakdown(select_rule(cv_, pv, s1_ids, r_ids, rule_c), truth, ids_c)
        d = count_match_delta(cv_, pv, rule_c, target_c, len(ids_c))
        res[f"stress_rule_countmatch_{c}"] = {**breakdown(select_rule(cv_, shift_logit(pv, d), s1_ids, r_ids, rule_c),
                                                          truth, ids_c), "delta": d, "target": float(target_c)}
        log(f"stress {c}: thr {res[f'stress_holdout_{c}']['macro_f05']:.4f} | rule {res[f'stress_rule_{c}']['macro_f05']:.4f}"
            f" | rule+count-match {res[f'stress_rule_countmatch_{c}']['macro_f05']:.4f} (delta {d:+.3f})")
    imp = pd.Series(m.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    res["blocking"] = brep
    res["feature_gain_top"] = (imp / imp.sum()).round(4).head(20).to_dict()
    (out / "cv_metrics.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({"s1_idx": cand["s1_idx"], "r_idx": cand["r_idx"], "oof": oof, "y": y, "fold": fold}).to_parquet(
        out / "oof.parquet")
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


def _augment_decoys(cand, decoy, weight, n_r):
    """Test-like out-of-fold set: test has ~1.9x the decoy records of train (records owned by no S1: twins,
    noise). Each decoy pair gets weight-1 extra clones (a fractional part is sampled) as a new, unique
    record, so selection, calibration and the metric see test's decoy density. Returns the row index into
    `cand` of every augmented row and the augmented (s1_idx, r_idx) table."""
    rng = np.random.default_rng(SEED + 7)
    extra = np.flatnonzero(decoy)
    reps = int(np.floor(weight - 1))
    clones = np.concatenate([np.repeat(extra, reps), extra[rng.random(len(extra)) < weight - 1 - reps]])
    idx = np.concatenate([np.arange(len(cand)), clones])
    c = pd.DataFrame({"s1_idx": cand["s1_idx"].to_numpy()[idx],
                      "r_idx": np.concatenate([cand["r_idx"].to_numpy(), n_r + np.arange(len(clones))]).astype(np.int64)})
    return idx, c


def _crossfit(p, y, folds):
    q = np.empty(len(p), np.float32)
    for k in np.unique(folds):
        q[folds == k] = apply_calibrator(fit_calibrator(p[folds != k], y[folds != k]), p[folds == k])
    return q


def cmd_stack(args):
    """Stage 2 on the stage-1 out-of-fold probabilities (same folds), then choose the selection rule."""
    out = Path(args.work) / "experiments" / args.exp
    out.mkdir(parents=True, exist_ok=True)
    O = pd.read_parquet(Path(args.work) / "experiments" / (args.p1_from or args.exp) / "oof.parquet")
    if args.no_ctx:  # pairs come from the stage-1 source; no candidate table or stage-1 feature cache needed
        s1, s23 = load_prepared(args.data, "train", args.work)
        cand = O[["s1_idx", "r_idx"]].copy()
        cand["name_cos"] = cand["addr_cos"] = np.float32(np.nan)  # only used by the error-sample report
        s1_keep = np.random.default_rng(SEED).random(len(s1)) < args.cv_frac  # the stage_train sample
        eval_ids = s1["entity_id"].to_numpy()[s1_keep]
        F = pd.DataFrame(index=cand.index)
        args.ctx_cols = []
        log(f"stack: {len(cand)} pairs from {args.p1_from} (no candidate-table context)")
    else:
        s1, s23, cand, F, eval_ids = stage_train(args)
    assert np.array_equal(O["s1_idx"].to_numpy(), cand["s1_idx"].to_numpy()), "oof rows do not match the features"
    p1, y = O["oof"].to_numpy(np.float32), O["y"].to_numpy().astype(np.int8)
    if "fold" in O:
        folds = O["fold"].to_numpy()
    else:  # older cv runs: GroupKFold is deterministic for the same groups
        folds = np.zeros(len(O), np.int8)
        for k, (_, va) in enumerate(GroupKFold(args.folds).split(F, y, cand["s1_idx"].to_numpy())):
            folds[va] = k
    truth = read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv")
    truth = {k: truth[k] for k in eval_ids}
    prune = None
    if args.prune_loss > 0:  # learned blocking: drop pairs stage 1 is sure about, keeping candidate recall
        n_true = sum(len(truth[k]) for k in eval_ids)
        rec_block = float(y.sum() / n_true)
        loss = min(args.prune_loss, 1 - args.min_cand_recall / rec_block)
        if loss > 0:
            t_min = float(np.quantile(p1[y == 1], loss))
            keep = np.flatnonzero(p1 >= t_min)
            prune = {"p1_min": t_min, "blocking_recall": rec_block, "cand_recall": float(y[keep].sum() / n_true),
                     "pairs_before": int(len(cand)), "pairs_after": int(len(keep)),
                     "pairs_per_s1_before": len(cand) / len(eval_ids), "pairs_per_s1_after": len(keep) / len(eval_ids)}
            cand, F = cand.iloc[keep].reset_index(drop=True), F.iloc[keep].reset_index(drop=True)
            p1, y, folds = p1[keep], y[keep], folds[keep]
            gc.collect()
            log("prune", json.dumps(prune))
        else:
            log(f"prune skipped: blocking recall {rec_block:.4f} leaves no room above {args.min_cand_recall}")
    t = time.time()
    X2 = stack_features(cand, p1, s23, F[args.ctx_cols], s1=s1 if args.stack_feat in ("v2", "v3") else None,
                        pair=args.stack_feat == "v3")
    log(f"stack: features {X2.shape} in {(time.time() - t) / 60:.1f} min")
    del F
    gc.collect()
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    decoy = None
    if args.decoy_weight > 1:  # records owned by no S1 (twins, noise): ~1.9x denser in test than in train
        owned = set().union(*read_ground_truth(Path(args.data) / "train" / "train_ground_truth.tsv").values())
        decoy = (y == 0) & ~pd.Series(r_ids[cand["r_idx"].to_numpy()]).isin(owned).to_numpy()
        log(f"decoys: {int(decoy.sum())} of {int((y == 0).sum())} negative pairs; test-like weight {args.decoy_weight}")

    def fit_stage2(wt, tag):
        pr, its, mm = np.zeros(len(cand), np.float32), [], None
        for k in np.unique(folds):
            tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            mm = train_lgb(X2.iloc[tr], y[tr], X2.iloc[va], y[va],
                           w=None if wt is None else wt[tr], wv=None if wt is None else wt[va])
            pr[va] = mm.predict(X2.iloc[va], num_iteration=mm.best_iteration)
            mm.save_model(str(out / f"{tag}_f{k}.txt"), num_iteration=mm.best_iteration)
            its.append(mm.best_iteration)
            log(f"{tag} fold {k}: best_iter={mm.best_iteration}")
        return pr, its, mm

    p2, iters, m = fit_stage2(None, "stack")
    imp = pd.Series(m.feature_importance("gain"), index=X2.columns).sort_values(ascending=False)
    rule1, f1, tab1 = selection_study(cand, p1, y, s1_ids, r_ids, truth, eval_ids, folds, log=log)
    rule2, f2, tab2 = selection_study(cand, p2, y, s1_ids, r_ids, truth, eval_ids, folds, log=log)
    use_stack = f2 > f1 + 0.0005
    rule, p = (rule2, p2) if use_stack else (rule1, p1)
    testlike = None
    if decoy is not None:  # choose model + rule on the test-like (decoy-augmented) out-of-fold set
        p2w, iters_w, mw = fit_stage2(np.where(decoy, args.decoy_weight, 1.0).astype(np.float32), "stackw")
        ia, cand_a = _augment_decoys(cand, decoy, args.decoy_weight, len(r_ids))
        r_ids_a = np.concatenate([r_ids.astype(object), np.array([f"DECOY-{i}" for i in range(len(ia) - len(cand))], object)])
        y_a, folds_a = y[ia], folds[ia]
        testlike = {"decoy_weight": args.decoy_weight, "decoy_pairs": int(decoy.sum()), "clones": int(len(ia) - len(cand))}
        best = None
        for name, pv in (("stage1", p1), ("stage2", p2), ("stage2w", p2w)):
            r_, f_, t_ = selection_study(cand_a, pv[ia], y_a, s1_ids, r_ids_a, truth, eval_ids, folds_a, log=log)
            testlike[name] = {"best": {k: v for k, v in r_.items() if k != "cal"}, "macro_f05": f_, "table": t_[:5]}
            if best is None or f_ > best[1] + (0.0005 if name != "stage2w" else 0.0):
                best = (name, f_, r_, pv)
        name, _, rule, p = best
        testlike["chosen_variant"] = name
        use_stack = name != "stage1"
        if name == "stage2w":  # the weighted models become the stage-2 models used at test time
            for k in np.unique(folds):
                os.replace(out / f"stackw_f{k}.txt", out / f"stack_f{k}.txt")
            iters, m = iters_w, mw
            imp = pd.Series(m.feature_importance("gain"), index=X2.columns).sort_values(ascending=False)
        qa = _crossfit(p[ia], y_a, folds_a) if rule["method"] == "expf" else p[ia]
        keep_a = rule_mask(cand_a, qa, {k: v for k, v in rule.items() if k != "cal"})
        pred_a = {}
        for a, b in zip(s1_ids[cand_a["s1_idx"].to_numpy()[keep_a]], r_ids_a[cand_a["r_idx"].to_numpy()[keep_a]]):
            pred_a.setdefault(a, []).append(b)
        testlike["chosen"] = breakdown(pred_a, truth, eval_ids)
        testlike["selected_per_s1"] = float(keep_a.sum() / len(eval_ids))
        log("test-like", json.dumps({k: (v["macro_f05"] if isinstance(v, dict) and "macro_f05" in v else v)
                                     for k, v in testlike.items() if k != "chosen"}))
    q = _crossfit(p, y, folds) if rule["method"] == "expf" else p
    keep = rule_mask(cand, q, {k: v for k, v in rule.items() if k != "cal"})
    sel = cand[keep]
    pred = {}
    for a, b in zip(s1_ids[sel["s1_idx"].to_numpy()], r_ids[sel["r_idx"].to_numpy()]):
        pred.setdefault(a, []).append(b)
    res = {"stage1": {"best": {k: v for k, v in rule1.items() if k != "cal"}, "macro_f05": f1, "table": tab1[:10]},
           "stage2": {"best": {k: v for k, v in rule2.items() if k != "cal"}, "macro_f05": f2, "table": tab2[:10],
                      "best_iters": iters},
           "use_stack": bool(use_stack), "rule": rule, "chosen": breakdown(pred, truth, eval_ids),
           "stack_feat": args.stack_feat, "prune": prune, "p1_from": args.p1_from,
           "selected_per_s1_oof": testlike["selected_per_s1"] if testlike else float(keep.sum() / len(eval_ids)),
           "testlike": testlike,
           "stack_feature_gain_top": (imp / imp.sum()).round(4).head(25).to_dict()}
    ec = s1.set_index("entity_id").loc[eval_ids, "country"].to_numpy()
    for c in sorted(set(ec)):
        res[f"chosen_{c}"] = breakdown(pred, truth, eval_ids[ec == c])
    for b, ids_b in eval_slices(s1, s23, truth, eval_ids).items():
        res[f"slice_{b}"] = breakdown(pred, truth, ids_b)
    (out / "stack_metrics.json").write_text(json.dumps(res, indent=1))
    # error sample for offline analysis: false negatives, false positives, true positives
    rng = np.random.default_rng(SEED)
    idx = {"FN": np.flatnonzero((y == 1) & ~keep), "FP": np.flatnonzero((y == 0) & keep),
           "TP": np.flatnonzero((y == 1) & keep)}
    picks = {k: rng.choice(v, min(20000, len(v)), replace=False) for k, v in idx.items()}
    pick = np.concatenate(list(picks.values()))
    kind = np.concatenate([[k] * len(v) for k, v in picks.items()])
    E = cand.iloc[pick].reset_index(drop=True)
    E["comb"] = E["name_cos"] + E["addr_cos"]
    T = pair_table(E, s1, s23, name_frequency(s23))
    T["kind"], T["p1"], T["p_final"] = kind, p1[pick], p[pick]
    T["mode"] = assign_modes(T)
    T.to_parquet(out / "oof_errors.parquet", index=False)
    log("STACK", json.dumps({"stage1": f1, "stage2": f2, "use_stack": bool(use_stack), "chosen": res["chosen"]}, indent=1))


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


def _load_models(out, prefix):
    return [lgb.Booster(model_file=str(f)) for f in sorted(out.glob(f"{prefix}_f*.txt"))]


def _s1_chunks(s1_idx, rows_per_chunk=4_000_000):
    """Row index arrays covering whole Source-1 groups (stage-2 features need all pairs of an entity)."""
    order = np.argsort(s1_idx, kind="stable")
    g = s1_idx[order]
    cuts = [0]
    while cuts[-1] < len(order):
        c = min(cuts[-1] + rows_per_chunk, len(order))
        while c < len(order) and g[c] == g[c - 1]:
            c += 1
        cuts.append(c)
    return [order[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def _read_columns(path):
    """Parquet columns as numpy arrays, one column at a time (no pandas block consolidation copy)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    return {c: pf.read(columns=[c]).column(0).to_numpy() for c in pf.schema_arrow.names}


def _predict_from_p1(args, out):
    """Stage 2 on another experiment's test p1 without the candidate table (--no-ctx)."""
    s1, s23 = load_prepared(args.data, "test", args.work)
    T = pd.read_parquet(Path(args.work) / "experiments" / args.p1_from / "test_pairs_proba.parquet",
                        columns=["s1_idx", "r_idx", "p1"])
    cand, p1 = T[["s1_idx", "r_idx"]], T["p1"].to_numpy(np.float32)
    del T
    log(f"test: {len(cand)} pairs with p1 from {args.p1_from}")
    sm = json.loads((out / "stack_metrics.json").read_text())
    if sm.get("prune") and sm["prune"].get("p1_min") is not None:
        keep = np.flatnonzero(p1 >= sm["prune"]["p1_min"])
        n0 = len(cand)
        cand, p1 = cand.iloc[keep].reset_index(drop=True), p1[keep]
        gc.collect()
        log(f"prune: p1 >= {sm['prune']['p1_min']:.2e} keeps {len(cand)}/{n0} pairs ({len(cand) / len(s1):.1f}/S1)")
    p = p1
    stack = _load_models(out, "stack") if sm["use_stack"] else []
    if stack:
        names2 = stack[0].feature_name()
        prof = s1 if sm.get("stack_feat") in ("v2", "v3") else None
        p = np.empty(len(cand), np.float32)
        for rows in _s1_chunks(cand["s1_idx"].to_numpy()):
            X2 = stack_features(cand.iloc[rows], p1[rows], s23, pd.DataFrame(index=range(len(rows))), s1=prof,
                                pair=sm.get("stack_feat") == "v3")[names2]
            p[rows] = np.mean([m.predict(X2) for m in stack], axis=0)
            log(f"stage 2: {len(rows)} pairs")
    pd.DataFrame({"s1_idx": cand["s1_idx"], "r_idx": cand["r_idx"], "p1": p1, "p": p}).to_parquet(
        out / "test_pairs_proba.parquet")
    write_selection(args, out, cand, p, s1, s23)


def cmd_predict(args):
    """Test prediction, memory-lean: the 1e8-row candidate table is kept as separate numpy columns and
    feature matrices are only built 4M rows at a time."""
    out = Path(args.work) / "experiments" / args.exp
    if args.no_ctx:
        return _predict_from_p1(args, out)
    if args.p1_from:  # E011: reuse another experiment's stage-1 test probabilities
        models, names = [], []
    else:
        models = _load_models(out, "model") or [lgb.Booster(model_file=str(out / "model.txt"))]
        names = models[0].feature_name()
        log(f"stage 1: {len(models)} model(s), {len(names)} features")
    pc = Path(args.work) / "test" / f"cand_{args.block_tag}.parquet"
    if not pc.exists():  # blocking in this process; prefer `ber.run block --split test` beforehand
        stage_candidates(args, "test")
        gc.collect()
    s1, s23 = load_prepared(args.data, "test", args.work)
    cols = _read_columns(pc)
    cand = pd.DataFrame({"s1_idx": cols["s1_idx"], "r_idx": cols["r_idx"]})
    log(f"test: {len(cand)} candidate pairs ({len(cand) / len(s1):.1f}/S1)")
    C = context_arrays(cols)
    del cols
    gc.collect()
    ctx_cols = list(C)
    A = None
    if "name_rank_r" in names:  # v4 (float16 until each chunk is cast, as in training)
        C.update(context2_arrays(cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy(), C["name_cos"], C["addr_cos"]))
        A = crowd_arrays(s1, s23)
        gc.collect()
    AN = name_amb_arrays(s1, s23) if "r_name_s1cnt" in names else None
    PA = profile_arrays(s1, s23) if "lg_add" in names else None
    if "d_full" in names:  # v5 dense cosines (cached by `ber.run dense --split test`)
        si_, ri_ = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
        D = dense_arrays(args, "test", s1, s23, si_, ri_)
        C.update(dense_context_arrays(si_, ri_, D))
        del D, si_, ri_
        gc.collect()
    if "conj_cos" in names:
        C["conj_cos"] = pair_conj_cos(cand, s1, s23, log=log)
    p1 = np.empty(len(cand), np.float32)
    step = 4_000_000  # features are built per chunk; the full test matrix never exists
    if args.p1_from:
        T = pd.read_parquet(Path(args.work) / "experiments" / args.p1_from / "test_pairs_proba.parquet",
                            columns=["s1_idx", "r_idx", "p1"])
        assert np.array_equal(T["s1_idx"].to_numpy(), cand["s1_idx"].to_numpy()) and             np.array_equal(T["r_idx"].to_numpy(), cand["r_idx"].to_numpy()), "stage-1 probabilities do not match the candidates"
        p1[:] = T["p1"].to_numpy(np.float32)
        del T
        log(f"stage 1: reused p1 from {args.p1_from}")
    for a in range(0, len(cand), step) if not args.p1_from else ():
        sub = cand.iloc[a:a + step]
        parts = [string_features(sub, s1, s23)]
        if "a_num_tset" in names:
            parts.append(extra_features(sub, s1, s23))
        parts.append(pd.DataFrame({k: np.asarray(v[a:a + step], np.float32) for k, v in C.items()}, index=sub.index))
        if A is not None:
            parts.append(crowd_features(sub["s1_idx"].to_numpy(), sub["r_idx"].to_numpy(), A).set_axis(sub.index))
        if AN is not None:
            parts.append(name_amb_features(sub["s1_idx"].to_numpy(), sub["r_idx"].to_numpy(), AN).set_axis(sub.index))
        if PA is not None:
            parts.append(profile_features(sub["s1_idx"].to_numpy(), sub["r_idx"].to_numpy(), PA, amb=False).set_axis(sub.index))
        Fc = pd.concat(parts, axis=1)[names]
        p1[a:a + step] = np.mean([m.predict(Fc) for m in models], axis=0)
        del Fc, parts
        log(f"stage 1: predicted {min(a + step, len(cand))}/{len(cand)}")
    sm = json.loads((out / "stack_metrics.json").read_text()) if (out / "stack_metrics.json").exists() else None
    if sm and sm.get("prune") and sm["prune"].get("p1_min") is not None:  # learned blocking (stage 1 as filter)
        keep = np.flatnonzero(p1 >= sm["prune"]["p1_min"])
        n0 = len(cand)
        cand, p1 = cand.iloc[keep].reset_index(drop=True), p1[keep]
        C = {k: np.asarray(C[k][keep]) for k in ctx_cols}
        gc.collect()
        log(f"prune: p1 >= {sm['prune']['p1_min']:.2e} keeps {len(cand)}/{n0} pairs ({len(cand) / len(s1):.1f}/S1)")
    p = p1
    stack = _load_models(out, "stack") if sm and sm["use_stack"] else []
    if stack:
        names2 = stack[0].feature_name()
        prof = s1 if sm.get("stack_feat") in ("v2", "v3") else None
        p = np.empty(len(cand), np.float32)
        for rows in _s1_chunks(cand["s1_idx"].to_numpy()):
            Cc = pd.DataFrame({k: C[k][rows] for k in ctx_cols})
            X2 = stack_features(cand.iloc[rows], p1[rows], s23, Cc, s1=prof, pair=sm.get("stack_feat") == "v3")[names2]
            p[rows] = np.mean([m.predict(X2) for m in stack], axis=0)
            log(f"stage 2: {len(rows)} pairs")
    del C
    gc.collect()
    pd.DataFrame({"s1_idx": cand["s1_idx"], "r_idx": cand["r_idx"], "p1": p1, "p": p}).to_parquet(
        out / "test_pairs_proba.parquet")
    write_selection(args, out, cand, p, s1, s23)


def write_selection(args, out, cand, p, s1, s23):
    """Apply the chosen selection rule (optionally with per-country count matching) and write both TSVs."""
    sm = json.loads((out / "stack_metrics.json").read_text()) if (out / "stack_metrics.json").exists() else None
    cvm = json.loads((out / "cv_metrics.json").read_text()) if (out / "cv_metrics.json").exists() else {}
    if sm:
        rule = sm["rule"]
    elif "rule" in cvm:
        rule = cvm["rule"]
    else:
        rule = {"method": "threshold", "thr": json.loads((out / "fit.json").read_text())["threshold"]}
    log("selection rule:", json.dumps({k: v for k, v in rule.items() if k != "cal"}))
    s1c = s1["country"].to_numpy()
    info = {"rule": {k: v for k, v in rule.items() if k != "cal"}, "count_match": args.count_match, "delta": {}}
    if args.count_match != "none":
        p = p.copy()
        target = (sm or {}).get("selected_per_s1_oof") or cvm.get("selected_per_s1_oof")
        train_c = set(pd.read_parquet(Path(args.work) / "prepared" / "train" / "s1.parquet", columns=["country"])["country"])
        scale = {k: float(v) for k, v in (kv.split("=") for kv in args.count_scale.split(",") if kv)}
        info["count_scale"] = scale
        pc = s1c[cand["s1_idx"].to_numpy()]
        for c in sorted(set(s1c)):
            if args.count_match == "unseen" and c in train_c:
                continue
            rows = np.flatnonzero(pc == c)
            t_c = target * scale.get(c, 1.0)
            d = count_match_delta(cand.iloc[rows], p[rows], rule, t_c, int((s1c == c).sum()))
            p[rows] = shift_logit(p[rows], d)
            info["delta"][c] = d
            log(f"count match {c}: delta {d:+.3f} -> {t_c:.3f} selected per S1")
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    keep = rule_mask(cand, p, rule)
    si, ri = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    od = Path(args.out)
    write_grouped(od / "matching_results.tsv", s1_ids, si[keep], ri[keep], r_ids, "matched_entity_ids")
    linked = False
    if args.cand_from:  # identical for every selection variant: link instead of another ~1.3 GB copy
        (od / "candidate_pairs.tsv").unlink(missing_ok=True)
        try:
            (od / "candidate_pairs.tsv").symlink_to(Path(args.cand_from).resolve() / "candidate_pairs.tsv")
            linked = True
        except OSError:  # no symlink privilege (Windows): write the file
            pass
    if not linked:
        write_grouped(od / "candidate_pairs.tsv", s1_ids, si, ri, r_ids, "candidate_entity_ids")
    n = np.bincount(si[keep], minlength=len(s1_ids))  # matches per S1 (exclusivity makes pairs unique)
    info["per_country"] = {c: {"mean_matches": float(n[s1c == c].mean()), "empty_rate": float((n[s1c == c] == 0).mean()),
                               "n_s1": int((s1c == c).sum())} for c in sorted(set(s1c))}
    (od / "selection_info.json").write_text(json.dumps(info, indent=1))
    log(f"wrote {od}: {int(n.sum())} matches; per country {json.dumps(info['per_country'])}")


def cmd_select(args):
    """Re-apply selection to saved test probabilities (no feature or model work)."""
    out = Path(args.work) / "experiments" / args.exp
    s1, s23 = load_prepared(args.data, "test", args.work)
    T = pd.read_parquet(out / "test_pairs_proba.parquet")
    write_selection(args, out, T[["s1_idx", "r_idx"]], T["p"].to_numpy(np.float32), s1, s23)


def block_tag(args):
    """Cache tag of a blocker configuration (shared with scripts/eval_blocking_full.py)."""
    if args.channels:
        return f"ch-{args.channels.replace(',', '+').replace(':', '')}_m{args.rrf_m}_df{args.df_cap}"
    return f"k{args.k_comb}_n{args.k_name}_df{args.df_cap}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "fit", "predict", "block", "holdout", "stack", "select", "dense"])
    ap.add_argument("--split", default="train", choices=["train", "test"], help="for `block` and `dense`")
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
    ap.add_argument("--feat", default="v2", choices=["v2", "v3", "v4", "v5", "v6"],
                    help="v3 adds house-number and conj_cos features; v4 adds name/address competition and shared-address "
                         "counts; v5 adds global name ambiguity and, with --dense-model, dense multilingual cosines; "
                         "v6 adds legal-form / business-word / house-number profiles")
    ap.add_argument("--dense-model", default="none", choices=["none", "e5s", "e5b", "bge"],
                    help="dense encoder for --feat v5 (none = v5 without dense cosines)")
    ap.add_argument("--no-stress", dest="stress", action="store_false", help="skip the country-holdout stress models")
    ap.add_argument("--count-scale", default="", help="per-country factor on the count-matching target, e.g. "
                                                        "France=0.97 (label-free probes for the unseen country)")
    ap.add_argument("--count-match", default="none", choices=["none", "unseen", "all"],
                    help="shift each test country's logits to the out-of-fold matches-per-S1 (unseen = new countries only)")
    ap.add_argument("--p1-from", default="", help="stack/predict: reuse this experiment's stage-1 OOF and test p1")
    ap.add_argument("--no-ctx", action="store_true",
                    help="stack/predict with --p1-from: pairs from that experiment's oof/test probabilities, "
                         "no candidate table (stage-2 features without candidate-table context)")
    ap.add_argument("--decoy-weight", type=float, default=0.0,
                    help=">1: stage 2 also trained with this weight on negatives from records owned by no S1, and the model "
                         "and selection rule are chosen on a test-like out-of-fold set with that decoy density (~1.9)")
    ap.add_argument("--stack-feat", default="v1", choices=["v1", "v2", "v3"],
                    help="v2 adds legal-form/word/house-number profiles; v3 also the S1-vs-record name/address similarities "
                         "(stage 1's string features: with --no-ctx stage 2 otherwise sees them only through p1)")
    ap.add_argument("--prune-loss", type=float, default=0.0,
                    help="stack: drop pairs below the p1 quantile losing this share of true candidate pairs")
    ap.add_argument("--min-cand-recall", type=float, default=0.98, help="pruning never takes candidate recall below this")
    ap.add_argument("--cand-from", default="", help="select: symlink candidate_pairs.tsv from this output dir")
    args = ap.parse_args()
    PARAMS["learning_rate"] = args.lr
    args.block_tag = block_tag(args)
    if os.environ.get("BER_DENSE_FAKE") and args.dense_model != "none":  # smoke tests: never mix stand-in cosines with real model caches
        args.dense_model = "fake"
    {"cv": cmd_cv, "fit": cmd_fit, "predict": cmd_predict, "block": cmd_block, "holdout": cmd_holdout,
     "stack": cmd_stack, "select": cmd_select, "dense": cmd_dense}[args.cmd](args)


if __name__ == "__main__":
    main()
