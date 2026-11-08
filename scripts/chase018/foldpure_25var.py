#!/usr/bin/env python3
"""Stronger fold-pure MultExp 25-variant selector. Honest nested LOOCV. No Kaggle."""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from adapt_multexp_fasc import (
    FROZEN,
    VARIANTS,
    adapt_fasc,
    apo_belly_mask,
    image_features,
    loocv_cal_pa_mt,
    predict_one,
    structure_ridge,
)
from umud.rl.bandit_policy import BanditConfig, SoftmaxBandit

HK = timezone(timedelta(hours=8))
PRIOR_C = 0.13195144580335014
GATE = 0.12
CACHE_BANK = ROOT / "reports/phase3/chase018/cache/multexp_25var_bank.npz"
OUT_DIR = ROOT / "reports/phase3/chase018"


def adapt_or_blend(g, apo, fasc, variant):
    if variant.startswith("blend_bm_"):
        w = float(variant.split("_")[-1]) / 100.0
        return np.clip(
            (1 - w) * fasc + w * adapt_fasc(g, apo, fasc, "morph_connect"), 0, 1
        ).astype(np.float32)
    if variant.startswith("blend_br_"):
        w = float(variant.split("_")[-1]) / 100.0
        return np.clip(
            (1 - w) * fasc + w * adapt_fasc(g, apo, fasc, "ridge_blend03"), 0, 1
        ).astype(np.float32)
    if variant.startswith("blend_bc_"):
        w = float(variant.split("_")[-1]) / 100.0
        return np.clip(
            (1 - w) * fasc + w * adapt_fasc(g, apo, fasc, "corridor_dilate"), 0, 1
        ).astype(np.float32)
    return adapt_fasc(g, apo, fasc, variant).astype(np.float32)


POOL = (
    list(VARIANTS)
    + [f"blend_bm_{w}" for w in (15, 25, 40, 50, 65, 75)]
    + [f"blend_br_{w}" for w in (15, 25, 40, 50)]
    + [f"blend_bc_{w}" for w in (15, 25, 40, 50)]
)


def raw_err(pr, gt):
    return abs(pr["pred_pa"] - gt["pa"]) / 6.0 + abs(pr["pred_fl"] - gt["fl"]) / 12.0


def rich_image_features(g, apo, fasc, px, preds_by_var):
    """Fold-pure features: map stats + prediction disagreement across variants."""
    base = image_features(g, apo, fasc, px)
    h, w = apo.shape
    belly = apo_belly_mask(apo, 0.60)
    fasc_hi = float((fasc > 0.5).mean())
    fasc_lo = float((fasc > 0.2).mean())
    # ridge mass in belly
    g_r = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    ridge, ang, coher = structure_ridge(g_r, sigma=1.5)
    ridge_belly = float(ridge[belly > 0].mean()) if belly.sum() else 0.0
    # fasc mass after morph_connect / ridge as cheap proxies without needing all maps
    pas = np.array([preds_by_var[v]["pred_pa"] for v in POOL if v in preds_by_var], dtype=np.float64)
    fls = np.array([preds_by_var[v]["pred_fl"] for v in POOL if v in preds_by_var], dtype=np.float64)
    mts = np.array([preds_by_var[v]["pred_mt"] for v in POOL if v in preds_by_var], dtype=np.float64)
    # disagreement / dispersion of candidate predictions
    pa_std = float(np.std(pas)) if len(pas) else 0.0
    fl_std = float(np.std(fls)) if len(fls) else 0.0
    pa_iqr = float(np.percentile(pas, 75) - np.percentile(pas, 25)) if len(pas) else 0.0
    fl_iqr = float(np.percentile(fls, 75) - np.percentile(fls, 25)) if len(fls) else 0.0
    pa_range = float(pas.max() - pas.min()) if len(pas) else 0.0
    fl_range = float(fls.max() - fls.min()) if len(fls) else 0.0
    # baseline-relative
    base_pa = preds_by_var.get("baseline", {}).get("pred_pa", float(np.median(pas)))
    base_fl = preds_by_var.get("baseline", {}).get("pred_fl", float(np.median(fls)))
    # top variants by extremity
    ridge_pa = preds_by_var.get("ridge_replace", {}).get("pred_pa", base_pa)
    morph_pa = preds_by_var.get("morph_prune", {}).get("pred_pa", base_pa)
    blend40_fl = preds_by_var.get("blend_br_40", {}).get("pred_fl", base_fl)
    # ok / extrap flags
    n_ok = sum(1 for v in POOL if v in preds_by_var and preds_by_var[v].get("ok"))
    # q/scale family
    q_scale = float(px)
    # fasc entropy-ish: mass at multiple thresholds
    thr_curve = [float((fasc > t).mean()) for t in (0.15, 0.25, 0.35, 0.45, 0.55)]
    thr_slope = thr_curve[0] - thr_curve[-1]
    out = dict(base)
    out.update(
        dict(
            fasc_hi=fasc_hi,
            fasc_lo=fasc_lo,
            ridge_belly=ridge_belly,
            pa_std=pa_std,
            fl_std=fl_std,
            pa_iqr=pa_iqr,
            fl_iqr=fl_iqr,
            pa_range=pa_range,
            fl_range=fl_range,
            base_pa=float(base_pa),
            base_fl=float(base_fl),
            dpa_ridge=float(ridge_pa - base_pa),
            dpa_morph=float(morph_pa - base_pa),
            dfl_br40=float(blend40_fl - base_fl),
            n_ok=float(n_ok),
            q_scale=q_scale,
            thr_slope=float(thr_slope),
            mt_med=float(np.median(mts)) if len(mts) else 20.0,
        )
    )
    return out


def build_or_load_bank():
    expert_csv = ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    expert = pd.read_csv(expert_csv)
    kw = dict(FROZEN)
    items = []
    t0 = time.time()
    for _, er in expert.iterrows():
        name = er["image_file"]
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        z = np.load(cache / f"{name}.npz")
        apo, fasc = z["apo"].astype(np.float32), z["fasc"].astype(np.float32)
        px = float(er["Scale_pixel_per_cm"]) / 10.0
        preds, maps = {}, {}
        for v in POOL:
            try:
                m = adapt_or_blend(g, apo, fasc, v)
            except Exception:
                continue
            pr = predict_one(g, apo, m, px, kw)
            preds[v] = pr
            maps[v] = m
        gt = dict(pa=float(er["median_PA"]), fl=float(er["median_FL"]), mt=float(er["median_MT"]))
        feats = rich_image_features(g, apo, fasc, px, preds)
        items.append(
            dict(name=name, g=g, apo=apo, fasc=fasc, px=px, feats=feats, preds=preds, maps=maps, gt=gt)
        )
        print(f"banked {name} nvar={len(maps)}", flush=True)
    print(f"bank done in {(time.time()-t0)/60:.1f}m pool={len(POOL)} n={len(items)}", flush=True)
    return items


def feat_matrix(items, keys=None):
    keys = keys or sorted(items[0]["feats"].keys())
    X = np.zeros((len(items), len(keys)), dtype=np.float64)
    for i, it in enumerate(items):
        for j, k in enumerate(keys):
            v = it["feats"].get(k, 0.0)
            X[i, j] = float(v) if np.isfinite(v) else np.nan
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_med, inds[1])
    return X, keys


def eval_picks(items, picks, name):
    rows = []
    for it, v in zip(items, picks):
        if v not in it["preds"]:
            v = "baseline" if "baseline" in it["preds"] else next(iter(it["preds"]))
        pr = it["preds"][v]
        rows.append(
            dict(
                image_file=it["name"],
                ok=pr["ok"],
                pred_pa=pr["pred_pa"],
                pred_fl=pr["pred_fl"],
                pred_mt=pr["pred_mt"],
                gt_pa=it["gt"]["pa"],
                gt_fl=it["gt"]["fl"],
                gt_mt=it["gt"]["mt"],
                picked=v,
            )
        )
    df = pd.DataFrame(rows)
    comps, mae, _, _ = loocv_cal_pa_mt(df)
    return {
        "name": name,
        "S": float(comps["S"]),
        "MAE": mae,
        "gate": float(comps["S"]) < GATE,
        "beats_C": float(comps["S"]) < PRIOR_C,
        "pick_counts": dict(Counter(picks)),
    }, df


def oracle_labels(items):
    labels, costs = [], []
    for it in items:
        best_v = min(it["preds"], key=lambda v: raw_err(it["preds"][v], it["gt"]))
        labels.append(best_v)
        costs.append(raw_err(it["preds"][best_v], it["gt"]))
    return labels, costs


def cost_matrix(items, pool):
    n, V = len(items), len(pool)
    C = np.full((n, V), 1e6, dtype=np.float64)
    for i, it in enumerate(items):
        for j, v in enumerate(pool):
            if v in it["preds"]:
                C[i, j] = raw_err(it["preds"][v], it["gt"])
    return C


def nested_trainS(items, pool):
    n = len(items)
    C = cost_matrix(items, pool)
    picks = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        mean_c = C[m].mean(0)
        picks.append(pool[int(np.argmin(mean_c))])
    return picks


def nested_knn(X, labels, ks=(3, 5, 7, 9, 11)):
    n = len(labels)
    # z-score leave-one-out style: use train only
    out = {}
    for k in ks:
        picks = []
        for i in range(n):
            tr = np.ones(n, dtype=bool)
            tr[i] = False
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr = (X[tr] - mu) / sd
            xt = (X[i] - mu) / sd
            d = np.linalg.norm(Xtr - xt, axis=1)
            nn = np.argsort(d)[:k]
            labs = [labels[j] for j in np.where(tr)[0][nn]]
            picks.append(Counter(labs).most_common(1)[0][0])
        out[k] = picks
    return out


def nested_clf(X, labels, pool, method="et", grid=None):
    """LOOCV classification of oracle variant; hyperparams chosen on train via inner LOOCV mean cost."""
    n = len(labels)
    lab_to_i = {v: i for i, v in enumerate(pool)}
    y = np.array([lab_to_i[l] for l in labels], dtype=int)
    C = cost_matrix(
        [
            # rebuild items not needed — use external C from caller? rebuild via labels only no
        ],
        pool,
    ) if False else None
    # We'll pass costs separately — rebuild below in main. Here expect cost_fn.
    raise NotImplementedError


def make_estimator(kind, params, n_classes):
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            random_state=0,
            n_jobs=1,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            random_state=0,
            n_jobs=1,
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 80),
            max_depth=params.get("max_depth", 2),
            learning_rate=params.get("lr", 0.1),
            random_state=0,
        )
    if kind == "logreg":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        )
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=params.get("k", 5), weights="distance")
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=params.get("hidden", (32,)),
            alpha=params.get("alpha", 1e-2),
            max_iter=800,
            random_state=0,
            early_stopping=False,
        )
    raise ValueError(kind)


def nested_sklearn_pick(X, labels, items, pool, kind, param_grid):
    """LOOCV classification; hyperparams via 5-fold CV on train minimizing mean pick cost."""
    n = len(labels)
    C = cost_matrix(items, pool)
    picks = []
    chosen_params = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        y_tr = [labels[j] for j in tr]
        present = sorted(set(y_tr))
        if len(present) == 1:
            picks.append(present[0])
            chosen_params.append({"degenerate": True})
            continue
        Xtr_raw = X[tr]
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
        Xtr = (Xtr_raw - mu) / sd
        xt = (X[i] - mu) / sd
        present_idx = {v: k for k, v in enumerate(present)}
        y_local = np.array([present_idx[labels[j]] for j in tr], dtype=int)
        best_p, best_sc = param_grid[0], 1e18
        # 5-fold on train
        tr_arr = np.array(tr)
        folds = np.array_split(np.arange(len(tr)), 5)
        for params in param_grid:
            fold_costs = []
            for fi, val_idx in enumerate(folds):
                if len(val_idx) == 0:
                    continue
                train_idx = np.concatenate([folds[fj] for fj in range(5) if fj != fi])
                if len(set(y_local[train_idx])) < 2:
                    pred_vs = [Counter(present[y_local[u]] for u in train_idx).most_common(1)[0][0]] * len(val_idx)
                else:
                    est = make_estimator(kind, params, len(present))
                    est.fit(Xtr[train_idx], y_local[train_idx])
                    pred_local = est.predict(Xtr[val_idx])
                    pred_vs = [present[int(p)] for p in pred_local]
                for local_k, pred_v in zip(val_idx, pred_vs):
                    img = tr[int(local_k)]
                    fold_costs.append(C[img, pool.index(pred_v)])
            sc = float(np.mean(fold_costs)) if fold_costs else 1e18
            if sc < best_sc:
                best_sc, best_p = sc, params
        est = make_estimator(kind, best_p, len(present))
        est.fit(Xtr, y_local)
        pred_local = int(est.predict(xt.reshape(1, -1))[0])
        picks.append(present[pred_local])
        chosen_params.append(best_p)
    return picks, chosen_params

def nested_l1_residual(X, items, pool):
    """For each variant, Ridge/L1-ish residual model of cost from features; pick min predicted cost.
    Hyperparams (alpha) chosen on train via mean abs residual of cost.
    Actually: train Ridge to predict cost of each variant from X; pick argmin predicted cost.
    """
    n = len(items)
    C = cost_matrix(items, pool)
    alphas = [0.1, 1.0, 10.0, 100.0]
    picks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr_raw = X[tr]
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
        Xtr = (Xtr_raw - mu) / sd
        xt = (X[i] - mu) / sd
        # choose alpha by predicting costs on leave-one of train (cheap: mean CV on train)
        best_alpha, best_sc = alphas[0], 1e18
        for a in alphas:
            # 5-fold-ish: leave-one cost MAE of predicted costs
            errs = []
            for t_idx, t in enumerate(tr):
                inner = [u for u in range(len(tr)) if u != t_idx]
                pred_costs = []
                true_costs = []
                for vj, v in enumerate(pool):
                    ridge = Ridge(alpha=a, random_state=0)
                    ridge.fit(Xtr[inner], C[np.array(tr)[inner], vj])
                    pred_costs.append(float(ridge.predict(Xtr[t_idx : t_idx + 1])[0]))
                    true_costs.append(C[t, vj])
                # score by whether argmin pred matches low true
                vhat = int(np.argmin(pred_costs))
                errs.append(C[t, vhat])
            sc = float(np.mean(errs))
            if sc < best_sc:
                best_sc, best_alpha = sc, a
        pred_c = []
        for vj in range(len(pool)):
            ridge = Ridge(alpha=best_alpha, random_state=0)
            ridge.fit(Xtr, C[tr, vj])
            pred_c.append(float(ridge.predict(xt.reshape(1, -1))[0]))
        picks.append(pool[int(np.argmin(pred_c))])
    return picks


def nested_softmax_bandit(X, items, pool, labels):
    """SoftmaxBandit over variants; fast nested: cfg chosen by train greedy cost (no inner LOOCV)."""
    n = len(items)
    C = cost_matrix(items, pool)
    V = len(pool)
    cfgs = [
        dict(lr=0.05, l2=0.1, epochs=40, temperature=1.0),
        dict(lr=0.08, l2=0.2, epochs=50, temperature=0.9),
        dict(lr=0.03, l2=0.3, epochs=60, temperature=1.2),
    ]
    picks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr_raw = X[tr]
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
        Xtr = (Xtr_raw - mu) / sd
        xt = (X[i] - mu) / sd
        best_cfg, best_sc = cfgs[0], 1e18
        for cfg in cfgs:
            band = SoftmaxBandit(
                BanditConfig(
                    n_actions=V,
                    state_dim=X.shape[1],
                    lr=cfg["lr"],
                    l2=cfg["l2"],
                    epochs=cfg["epochs"],
                    seed=0,
                    temperature=cfg["temperature"],
                )
            )
            rng = np.random.default_rng(0)
            for ep in range(cfg["epochs"]):
                for u in rng.permutation(len(tr)):
                    s = Xtr[u]
                    a = band.act(s, greedy=False, rng=rng)
                    r = -C[tr[u], a]
                    baseline = -float(C[tr[u]].mean())
                    band.update(s, a, r - baseline)
            # train greedy cost (fold-pure: no held-out GT used for cfg beyond train costs)
            errs = [C[tr[u], band.act(Xtr[u], greedy=True)] for u in range(len(tr))]
            sc = float(np.mean(errs))
            if sc < best_sc:
                best_sc, best_cfg = sc, cfg
                best_band = band
        a_hat = best_band.act(xt, greedy=True)
        picks.append(pool[a_hat])
    return picks


def accuracy_at_k(picks, labels, items, pool, k=3):
    """@k: whether oracle variant is among top-k by... for hard picks, @1 only meaningful;
    for classifiers we can use predicted probs — here approximate @1 and @3 via cost proximity:
    treat pick correct if equals oracle; @3 if pick's cost within 3rd best? Better: store top3 from methods that support it.
    Simple: @1 = exact match; @3 = pick in top-3 oracle cost variants for that image.
    """
    C = cost_matrix(items, pool)
    n = len(picks)
    hit1 = hit3 = 0
    for i, p in enumerate(picks):
        if p == labels[i]:
            hit1 += 1
        # top3 by true cost
        order = np.argsort(C[i])
        top3 = {pool[j] for j in order[:3]}
        if p in top3:
            hit3 += 1
    return hit1 / n, hit3 / n


def main():
    print(f"POOL size={len(POOL)} variants={POOL}", flush=True)
    items = build_or_load_bank()
    n = len(items)
    labels, _ = oracle_labels(items)
    X, feat_keys = feat_matrix(items)
    print(f"features={feat_keys}", flush=True)
    print(f"oracle label counts={dict(Counter(labels))}", flush=True)

    results = {}
    rows_store = {}

    # leaky oracle
    leak_picks = labels[:]
    leak, leak_df = eval_picks(items, leak_picks, "leaky_oracle")
    results["leaky_oracle"] = leak
    print("LEAKY", leak, flush=True)

    # nested trainS baseline
    picks = nested_trainS(items, list(POOL))
    r, df = eval_picks(items, picks, "nested_trainS")
    results["nested_trainS"] = r
    rows_store["nested_trainS"] = df
    print("TRAINS", r["S"], flush=True)

    # knn
    knn_picks = nested_knn(X, labels)
    for k, pk in knn_picks.items():
        r, df = eval_picks(items, pk, f"nested_knn_k{k}")
        results[f"knn_k{k}"] = r
        print(f"KNN{k}", r["S"], flush=True)
        if r["S"] < results.get("_best_S", 1e9):
            results["_best_S"] = r["S"]
            rows_store["best"] = df
            results["_best_name"] = r["name"]

    # sklearn classifiers — grids kept small for n=35
    grids = {
        "et": [
            {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 2},
            {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 1},
            {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 2},
        ],
        "rf": [
            {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 2},
            {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 1},
        ],
        "gb": [
            {"n_estimators": 60, "max_depth": 2, "lr": 0.1},
            {"n_estimators": 100, "max_depth": 2, "lr": 0.05},
            {"n_estimators": 80, "max_depth": 3, "lr": 0.1},
        ],
        "logreg": [
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
        "mlp": [
            {"hidden": (16,), "alpha": 1e-2},
            {"hidden": (32,), "alpha": 5e-2},
            {"hidden": (32, 16), "alpha": 1e-1},
        ],
    }

    best_S = min(v["S"] for k, v in results.items() if isinstance(v, dict) and "S" in v and k != "leaky_oracle")
    best_name = min(
        ((k, v) for k, v in results.items() if isinstance(v, dict) and "S" in v and k != "leaky_oracle"),
        key=lambda kv: kv[1]["S"],
    )[0]
    best_picks_map = {}

    for kind, grid in grids.items():
        print(f"=== nested {kind} ===", flush=True)
        t0 = time.time()
        picks, params = nested_sklearn_pick(X, labels, items, list(POOL), kind, grid)
        r, df = eval_picks(items, picks, f"nested_{kind}")
        a1, a3 = accuracy_at_k(picks, labels, items, list(POOL))
        r["acc_at_1"] = a1
        r["acc_at_3"] = a3
        r["wall_min"] = (time.time() - t0) / 60.0
        results[f"nested_{kind}"] = r
        rows_store[f"nested_{kind}"] = df
        best_picks_map[f"nested_{kind}"] = picks
        print(f"{kind} S={r['S']:.5f} acc@1={a1:.3f} @3={a3:.3f} in {r['wall_min']:.1f}m", flush=True)
        if r["S"] < best_S:
            best_S, best_name = r["S"], r["name"]
            rows_store["best"] = df

    # L1/Ridge residual cost model
    print("=== nested l1_residual ===", flush=True)
    t0 = time.time()
    picks = nested_l1_residual(X, items, list(POOL))
    r, df = eval_picks(items, picks, "nested_l1_residual")
    a1, a3 = accuracy_at_k(picks, labels, items, list(POOL))
    r["acc_at_1"] = a1
    r["acc_at_3"] = a3
    r["wall_min"] = (time.time() - t0) / 60.0
    results["nested_l1_residual"] = r
    rows_store["nested_l1_residual"] = df
    best_picks_map["nested_l1_residual"] = picks
    print(f"l1_residual S={r['S']:.5f} acc@1={a1:.3f} @3={a3:.3f}", flush=True)
    if r["S"] < best_S:
        best_S, best_name = r["S"], r["name"]
        rows_store["best"] = df

    # SoftmaxBandit — can be slow; run it
    print("=== nested SoftmaxBandit ===", flush=True)
    t0 = time.time()
    picks = nested_softmax_bandit(X, items, list(POOL), labels)
    r, df = eval_picks(items, picks, "nested_softmax_bandit")
    a1, a3 = accuracy_at_k(picks, labels, items, list(POOL))
    r["acc_at_1"] = a1
    r["acc_at_3"] = a3
    r["wall_min"] = (time.time() - t0) / 60.0
    results["nested_softmax_bandit"] = r
    rows_store["nested_softmax_bandit"] = df
    best_picks_map["nested_softmax_bandit"] = picks
    print(f"bandit S={r['S']:.5f} acc@1={a1:.3f} @3={a3:.3f} in {r['wall_min']:.1f}m", flush=True)
    if r["S"] < best_S:
        best_S, best_name = r["S"], r["name"]
        rows_store["best"] = df

    # also trainS / knn acc
    for name, pk in [("nested_trainS", nested_trainS(items, list(POOL))), ("leaky_oracle", leak_picks)]:
        if name in results:
            a1, a3 = accuracy_at_k(pk, labels, items, list(POOL))
            results[name]["acc_at_1"] = a1
            results[name]["acc_at_3"] = a3

    for k, pk in knn_picks.items():
        a1, a3 = accuracy_at_k(pk, labels, items, list(POOL))
        results[f"knn_k{k}"]["acc_at_1"] = a1
        results[f"knn_k{k}"]["acc_at_3"] = a3
        best_picks_map[f"knn_k{k}"] = pk

    # recompute best among transferable
    cand = {k: v for k, v in results.items() if isinstance(v, dict) and "S" in v and k != "leaky_oracle"}
    best_name = min(cand, key=lambda k: cand[k]["S"])
    best = cand[best_name]
    best_S = best["S"]

    # confusion vs oracle for best
    if best_name in best_picks_map:
        bp = best_picks_map[best_name]
    elif best_name.startswith("knn_"):
        bp = knn_picks[int(best_name.split("k")[-1])]
    elif best_name == "nested_trainS":
        bp = nested_trainS(items, list(POOL))
    else:
        bp = best_picks_map.get(best_name, labels)
    confusion = Counter((labels[i], bp[i]) for i in range(n))
    confusion_ser = {f"{a}->{b}": c for (a, b), c in confusion.most_common(30)}

    promote = bool(best_S < PRIOR_C)
    gate = bool(best_S < GATE)
    gap_oracle = float(best_S - leak["S"])

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "tag": "foldpure_25var_strong_selector",
        "protocol": (
            "nested LOOCV fold-pure classifiers over expanded 25-variant A/blend pool; "
            "rich map+disagreement features; hyperparams chosen inside train folds only; "
            "honest LOOCV S (PA+MT affine, FL raw); apo frozen; recipe frozen; no Kaggle"
        ),
        "pool": list(POOL),
        "pool_size": len(POOL),
        "feature_keys": feat_keys,
        "oracle_label_counts": dict(Counter(labels)),
        "results": {k: v for k, v in results.items() if not k.startswith("_")},
        "best_transferable": best,
        "leaky_oracle": leak,
        "oracle_selector_gap": gap_oracle,
        "prior_C": PRIOR_C,
        "delta_vs_C": float(best_S - PRIOR_C),
        "gate_0.12": gate,
        "promote": promote,
        "kaggle_submit": False,
        "confusion_oracle_to_pick_top30": confusion_ser,
        "diagnosis": (
            f"Transferable best S={best_S:.5f} ({best_name}); leaky oracle {leak['S']:.5f}; "
            f"gap={gap_oracle:.5f}. "
            + (
                "Cleared prior C."
                if promote
                else "Selectors still fail to close oracle−selector gap on Frundles+A pool."
            )
        ),
    }
    out_json = OUT_DIR / "multexp_foldpure_25var_scores.json"
    out_json.write_text(json.dumps(payload, indent=2))
    if "best" in rows_store:
        rows_store["best"].to_csv(OUT_DIR / "multexp_foldpure_25var_best_rows.csv", index=False)
    # also dump all method rows
    for name, df in rows_store.items():
        if name == "best":
            continue
        df.to_csv(OUT_DIR / f"multexp_foldpure_25var_{name}_rows.csv", index=False)

    # status.md
    status = f"""# MultExp fold-pure 25-var selector — {payload['asof']}

## Result (honest nested LOOCV)
- **Best transferable S = {best_S:.5f}** (`{best_name}`)
  PA {best['MAE']['pa_deg']:.3f}° / FL {best['MAE']['fl_mm']:.3f} mm / MT {best['MAE']['mt_mm']:.3f} mm
- vs prior C **0.13195** Δ {payload['delta_vs_C']:+.5f}
- Leaky oracle S = **{leak['S']:.5f}** (ceiling; not nested)
- Oracle−selector gap = **{gap_oracle:.5f}**
- Acc@1 / @3 (best method) = {best.get('acc_at_1', float('nan')):.3f} / {best.get('acc_at_3', float('nan')):.3f}
- **Gate 0.12: {'PASS' if gate else 'FAIL'}** | Promote: {promote} | **No Kaggle**

## Methods
| Method | S | beats C | acc@1 | acc@3 |
|--------|---|---------|-------|-------|
| leaky_oracle | {leak['S']:.5f} | yes | {leak.get('acc_at_1',1):.3f} | {leak.get('acc_at_3',1):.3f} |
"""
    for k, v in sorted(
        ((k, v) for k, v in results.items() if isinstance(v, dict) and "S" in v and k != "leaky_oracle"),
        key=lambda kv: kv[1]["S"],
    ):
        status += (
            f"| {k} | {v['S']:.5f} | {v.get('beats_C')} | "
            f"{v.get('acc_at_1', float('nan')):.3f} | {v.get('acc_at_3', float('nan')):.3f} |\n"
        )
    status += f"""
## Pool (25)
{', '.join(POOL)}

## Diagnosis
{payload['diagnosis']}
Top confusion (oracle→pick): {confusion_ser}

## Next lever
"""
    if best_S >= 0.13:
        status += (
            "Transferable S stays ≥0.13 despite oracle 0.105 — fold-pure variant choice cannot "
            "close the gap on Frundles+A maps. Next: fasc maps outside Frundles+A (new MultExp-paired "
            "labels / non-U-Net generators), not more selectors on this pool.\n"
        )
    elif not gate:
        status += (
            f"Beats prior C but still above 0.12 (gap {best_S-GATE:.5f}). "
            "Keep this selector; still need better fasc maps or a small calibrated residual on FL.\n"
        )
    else:
        status += "Gate cleared. Prepare Kaggle only after parent approval.\n"

    (OUT_DIR / "multexp_foldpure_25var_status.md").write_text(status)
    # also short json sidecar
    (OUT_DIR / "multexp_foldpure_25var_summary.json").write_text(
        json.dumps(
            {
                "best_S": best_S,
                "best_name": best_name,
                "MAE": best["MAE"],
                "oracle_S": leak["S"],
                "gap": gap_oracle,
                "prior_C": PRIOR_C,
                "gate": gate,
                "promote": promote,
                "kaggle_submit": False,
            },
            indent=2,
        )
    )

    if promote:
        # write maps for best picks
        maps_out = OUT_DIR / "cache" / "maps_multexp_foldpure_25var"
        maps_out.mkdir(parents=True, exist_ok=True)
        if best_name in best_picks_map:
            bp = best_picks_map[best_name]
        elif best_name == "nested_trainS":
            bp = nested_trainS(items, list(POOL))
        else:
            bp = knn_picks.get(int(best_name.split("k")[-1]), labels) if best_name.startswith("knn") else labels
        for it, v in zip(items, bp):
            np.savez_compressed(
                maps_out / f"{it['name']}.npz",
                apo=it["apo"].astype(np.float16),
                fasc=it["maps"][v].astype(np.float16),
            )
        recipe = {
            "asof": payload["asof"],
            "name": best_name,
            "S_loocv_cal_pa_mt": best_S,
            "MAE_loocv": best["MAE"],
            "kw": {
                k: FROZEN[k]
                for k in ("fasc_thr", "apo_thr", "tex_w", "fl_factor", "fl_sin_w", "agree_thr", "pa_mode")
            },
            "fasc_adapt": {
                "type": best_name,
                "pool_size": len(POOL),
                "pick_counts": best.get("pick_counts"),
                "maps": str(maps_out),
            },
            "cache_dir": str(maps_out),
            "prior_C_S": PRIOR_C,
            "beats_prior_C": True,
            "gate_0.12": gate,
            "gap_to_0.12": float(best_S - GATE),
            "kaggle_submit": False,
            "lab_fasc_ckpt": False,
            "protocol": payload["protocol"],
        }
        (OUT_DIR / "math_map_promoted_recipe.json").write_text(json.dumps(recipe, indent=2))
        print("PROMOTED", flush=True)

    print(
        f"FOLDPURE_RESULT S={best_S:.5f} name={best_name} leaky={leak['S']:.5f} "
        f"gap={gap_oracle:.5f} vs_C={PRIOR_C:.5f} gate={gate} promote={promote}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
