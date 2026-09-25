"""End-to-end entry point.

  python -m ber.run cv      --data DIR --work DIR --exp ID          # OOF CV + stress CV on train
  python -m ber.run fit     --data DIR --work DIR --exp ID          # final model on all train pairs
  python -m ber.run predict --data DIR --work DIR --exp ID --out output/
  python -m ber.run block   --data DIR --work DIR --exp ID --split test   # blocking only (cache)
  python -m ber.run holdout --data DIR --work DIR --exp ID                # one cheap validation split
  python -m ber.run stack   --data DIR --work DIR --exp ID                # stage 2 on the cv out-of-fold p
  python -m ber.run select  --data DIR --work DIR --exp ID --out DIR [--count-match unseen]  # re-select saved p

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
from .buckets import assign_modes, name_frequency, pair_table
from .channels import pair_conj_cos, retrieve
from .config import SEED
from .features import context_columns, context_features, extra_features, pair_features, string_features
from .io import read_ground_truth, write_id_lists
from .labels import blocking_report, label_pairs
from .metric import breakdown
from .postprocess import (apply_calibrator, count_match_delta, fit_calibrator, rule_mask, select, select_rule,
                          selection_study, shift_logit, tune_threshold)
from .prepare import load_prepared
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
    cand = cand.loc[rows, ["s1_idx", "r_idx", "name_cos", "addr_cos"]].reset_index(drop=True)
    gc.collect()
    if args.feat == "v3":  # extra pair features, cached separately so the v2 matrix is reused
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


def _crossfit(p, y, folds):
    q = np.empty(len(p), np.float32)
    for k in np.unique(folds):
        q[folds == k] = apply_calibrator(fit_calibrator(p[folds != k], y[folds != k]), p[folds == k])
    return q


def cmd_stack(args):
    """Stage 2 on the stage-1 out-of-fold probabilities (same folds), then choose the selection rule."""
    s1, s23, cand, F, eval_ids = stage_train(args)
    out = Path(args.work) / "experiments" / args.exp
    O = pd.read_parquet(out / "oof.parquet")
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
    t = time.time()
    X2 = stack_features(cand, p1, s23, F[args.ctx_cols])
    log(f"stack: features {X2.shape} in {(time.time() - t) / 60:.1f} min")
    del F
    gc.collect()
    p2 = np.zeros(len(cand), np.float32)
    iters = []
    for k in np.unique(folds):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        m = train_lgb(X2.iloc[tr], y[tr], X2.iloc[va], y[va])
        p2[va] = m.predict(X2.iloc[va], num_iteration=m.best_iteration)
        m.save_model(str(out / f"stack_f{k}.txt"), num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"stack fold {k}: best_iter={m.best_iteration}")
    imp = pd.Series(m.feature_importance("gain"), index=X2.columns).sort_values(ascending=False)
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    rule1, f1, tab1 = selection_study(cand, p1, y, s1_ids, r_ids, truth, eval_ids, folds, log=log)
    rule2, f2, tab2 = selection_study(cand, p2, y, s1_ids, r_ids, truth, eval_ids, folds, log=log)
    use_stack = f2 > f1 + 0.0005
    rule, p = (rule2, p2) if use_stack else (rule1, p1)
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
           "selected_per_s1_oof": float(keep.sum() / len(eval_ids)),
           "stack_feature_gain_top": (imp / imp.sum()).round(4).head(25).to_dict()}
    ec = s1.set_index("entity_id").loc[eval_ids, "country"].to_numpy()
    for c in sorted(set(ec)):
        res[f"chosen_{c}"] = breakdown(pred, truth, eval_ids[ec == c])
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


def cmd_predict(args):
    out = Path(args.work) / "experiments" / args.exp
    models = _load_models(out, "model") or [lgb.Booster(model_file=str(out / "model.txt"))]
    names = models[0].feature_name()
    log(f"stage 1: {len(models)} model(s), {len(names)} features")
    s1, s23, cand = stage_candidates(args, "test")
    C = context_features(cand)
    ctx_cols = list(C.columns)
    cand = cand[["s1_idx", "r_idx"]]  # the extra blocking columns now live in C
    gc.collect()
    if "conj_cos" in names:
        C["conj_cos"] = pair_conj_cos(cand, s1, s23, log=log)
    p1 = np.empty(len(cand), np.float32)
    step = 4_000_000  # stream string features so the full test matrix never sits in memory
    for a in range(0, len(cand), step):
        sub = cand.iloc[a:a + step]
        parts = [string_features(sub, s1, s23)]
        if "a_num_tset" in names:
            parts.append(extra_features(sub, s1, s23))
        Fc = pd.concat(parts + [C.iloc[a:a + step]], axis=1)[names]
        p1[a:a + step] = np.mean([m.predict(Fc) for m in models], axis=0)
        log(f"stage 1: predicted {min(a + step, len(cand))}/{len(cand)}")
    p = p1
    sm = json.loads((out / "stack_metrics.json").read_text()) if (out / "stack_metrics.json").exists() else None
    stack = _load_models(out, "stack") if sm and sm["use_stack"] else []
    if stack:
        names2 = stack[0].feature_name()
        p = np.empty(len(cand), np.float32)
        for rows in _s1_chunks(cand["s1_idx"].to_numpy()):
            X2 = stack_features(cand.iloc[rows], p1[rows], s23, C.iloc[rows][ctx_cols])[names2]
            p[rows] = np.mean([m.predict(X2) for m in stack], axis=0)
            log(f"stage 2: {len(rows)} pairs")
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
        pc = s1c[cand["s1_idx"].to_numpy()]
        for c in sorted(set(s1c)):
            if args.count_match == "unseen" and c in train_c:
                continue
            rows = np.flatnonzero(pc == c)
            d = count_match_delta(cand.iloc[rows], p[rows], rule, target, int((s1c == c).sum()))
            p[rows] = shift_logit(p[rows], d)
            info["delta"][c] = d
            log(f"count match {c}: delta {d:+.3f} -> {target:.3f} selected per S1")
    s1_ids, r_ids = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    pred = select_rule(cand, p, s1_ids, r_ids, rule)
    cands = {}
    for a, b in zip(s1_ids[cand["s1_idx"].to_numpy()], r_ids[cand["r_idx"].to_numpy()]):
        cands.setdefault(a, []).append(b)
    od = Path(args.out)
    write_id_lists(od / "matching_results.tsv", s1_ids, pred, "matched_entity_ids")
    write_id_lists(od / "candidate_pairs.tsv", s1_ids, cands, "candidate_entity_ids")
    n = pd.Series({e: len(pred.get(e, ())) for e in s1_ids})
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
    ap.add_argument("cmd", choices=["cv", "fit", "predict", "block", "holdout", "stack", "select"])
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
    ap.add_argument("--feat", default="v2", choices=["v2", "v3"], help="v3 adds house-number and conj_cos features")
    ap.add_argument("--no-stress", dest="stress", action="store_false", help="skip the country-holdout stress models")
    ap.add_argument("--count-match", default="none", choices=["none", "unseen", "all"],
                    help="shift each test country's logits to the out-of-fold matches-per-S1 (unseen = new countries only)")
    args = ap.parse_args()
    PARAMS["learning_rate"] = args.lr
    args.block_tag = block_tag(args)
    {"cv": cmd_cv, "fit": cmd_fit, "predict": cmd_predict, "block": cmd_block, "holdout": cmd_holdout,
     "stack": cmd_stack, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
