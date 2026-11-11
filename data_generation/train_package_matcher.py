# -*- coding: utf-8 -*-
"""
Package-Matching & Propensity Model Trainer (Multi-Task v2)
===========================================================
Loads `data/synthetic/train_patient_snapshot.parquet` and `feature_groups.json`,
constructs a point-in-time feature matrix (zero target leakage), trains all 4 heads:
  1. Multi-class Primary Package Matcher (13 classes: L1_BASIC .. L4_COLON)
  2. Multi-label Package Eligibility Scorer (13 binary heads -> Top-K ranking)
  3. Multi-label Clinical Add-on Recommender (13 binary heads -> Add-on ranking)
  4. 90-day Purchase Propensity Model (binary classification)

Saves production weights & preprocessing statistics to:
  - models/package_matcher_v2_bundle.npz
  - models/package_matcher_v2_meta.json

Usage:
  python data_generation/train_package_matcher.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
MODEL_DIR = ROOT / "models"


def load_training_bundle(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, Dict]:
    snap = pd.read_parquet(data_dir / "train_patient_snapshot.parquet")
    meta = json.loads((data_dir / "feature_groups.json").read_text(encoding="utf-8"))
    return snap, meta


def build_design_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    fit_stats: Dict | None = None,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Converts mixed numeric + categorical clinical snapshot features into a clean
    float32 matrix with:
      - Missingness indicator flags (`<col>__isna`) for partially-ordered lab tests
      - Median imputation (fitted on train split only!)
      - Z-score standardization
      - One-hot encoding for categorical clinical results (EKG, CXR, FIT, payer, etc.)
    """
    fit_mode = fit_stats is None
    if fit_mode:
        fit_stats = {
            "feature_cols": list(feature_cols),
            "num_medians": {},
            "num_means": {},
            "num_stds": {},
            "cat_levels": {},
            "high_na_cols": [],
        }

    cols_out: List[np.ndarray] = []
    names_out: List[str] = []
    n_rows = len(df)

    for col in feature_cols:
        if col not in df.columns:
            # Graceful fallback when a single-patient dict/DataFrame omits optional columns
            if col in fit_stats.get("num_medians", {}):
                cols_out.append(np.zeros(n_rows, dtype=np.float32))
                names_out.append(col)
                if col in fit_stats.get("high_na_cols", []):
                    cols_out.append(np.ones(n_rows, dtype=np.float32))
                    names_out.append(f"{col}__isna")
            elif col in fit_stats.get("cat_levels", {}):
                for lv in fit_stats["cat_levels"][col]:
                    cols_out.append(np.full(n_rows, 1.0 if lv == "__MISSING__" else 0.0, dtype=np.float32))
                    names_out.append(f"{col}={lv}")
            continue

        s = df[col]
        is_cat = (
            (not fit_mode and col in fit_stats.get("cat_levels", {}))
            or (
                fit_mode
                and (
                    isinstance(s.dtype, pd.CategoricalDtype)
                    or s.dtype == object
                    or str(s.dtype).startswith(("str", "string"))
                )
            )
        )
        if not is_cat:
            arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            na_mask = np.isnan(arr)
            if fit_mode:
                na_rate = float(na_mask.mean())
                if 0.02 <= na_rate <= 0.98:
                    fit_stats["high_na_cols"].append(col)
                med = float(np.nanmedian(arr)) if not np.all(na_mask) else 0.0
                filled = np.where(na_mask, med, arr)
                mu = float(filled.mean())
                sd = float(filled.std())
                if sd < 1e-6:
                    sd = 1.0
                fit_stats["num_medians"][col] = med
                fit_stats["num_means"][col] = mu
                fit_stats["num_stds"][col] = sd
            else:
                med = fit_stats["num_medians"][col]
                mu = fit_stats["num_means"][col]
                sd = fit_stats["num_stds"][col]
                filled = np.where(na_mask, med, arr)

            cols_out.append(((filled - mu) / sd).astype(np.float32))
            names_out.append(col)

            if col in fit_stats["high_na_cols"]:
                cols_out.append(na_mask.astype(np.float32))
                names_out.append(f"{col}__isna")
        else:
            s_str = s.astype(object).where(s.notna(), "__MISSING__").astype(str)
            if fit_mode:
                levels = sorted(s_str.unique().tolist())
                if "__MISSING__" not in levels:
                    levels.append("__MISSING__")
                fit_stats["cat_levels"][col] = levels
            else:
                levels = fit_stats["cat_levels"][col]
            arr_str = s_str.to_numpy()
            for lv in levels:
                cols_out.append((arr_str == lv).astype(np.float32))
                names_out.append(f"{col}={lv}")

    X = np.column_stack(cols_out).astype(np.float32)
    return X, names_out, fit_stats


def fit_multinomial_l2(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-2) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    Y_oh = np.eye(n_classes, dtype=np.float64)[y]

    def loss_grad(w_flat):
        W = w_flat.reshape(d + 1, n_classes)
        logits = Xb @ W
        P = softmax(logits, axis=1)
        diff = (P - Y_oh) / n
        loss = -np.sum(Y_oh * np.log(np.clip(P, 1e-12, 1.0))) / n + 0.5 * l2 * np.sum(W[:-1] ** 2)
        grad = Xb.T @ diff
        grad[:-1] += l2 * W[:-1]
        return float(loss), grad.ravel()

    res = minimize(
        loss_grad,
        np.zeros((d + 1) * n_classes, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 180},
    )
    return res.x.reshape(d + 1, n_classes)


def predict_proba_multinomial(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    return softmax(Xb @ W, axis=1)


def fit_binary_l2(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    y_f = y.astype(np.float64)

    def loss_grad(w):
        z = Xb @ w
        p = expit(z)
        loss = -np.mean(y_f * np.log(np.clip(p, 1e-12, 1.0)) + (1 - y_f) * np.log(np.clip(1 - p, 1e-12, 1.0)))
        loss += 0.5 * l2 * np.sum(w[:-1] ** 2)
        grad = (Xb.T @ (p - y_f)) / n
        grad[:-1] += l2 * w[:-1]
        return float(loss), grad

    res = minimize(loss_grad, np.zeros(d + 1, dtype=np.float64), method="L-BFGS-B", jac=True, options={"maxiter": 120})
    return res.x


def roc_auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    snap, meta = load_training_bundle()
    feature_cols = meta["feature_columns"]
    leak_cols = set(meta["exclude_from_features"])
    assert not (set(feature_cols) & leak_cols), "Leakage columns detected in feature_cols!"

    # Temporal split (80% earlier index dates -> train, 20% latest index dates -> test)
    snap_sorted = snap.sort_values("index_date").reset_index(drop=True)
    split_idx = int(len(snap_sorted) * 0.80)
    df_tr = snap_sorted.iloc[:split_idx].reset_index(drop=True)
    df_te = snap_sorted.iloc[split_idx:].reset_index(drop=True)

    X_tr, feat_names, stats = build_design_matrix(df_tr, feature_cols, None)
    X_te, _, _ = build_design_matrix(df_te, feature_cols, stats)
    print(f"Temporal Split: Train={len(df_tr):,} (up to {df_tr['index_date'].max().date()}) | "
          f"Test={len(df_te):,} ({df_te['index_date'].min().date()} .. {df_te['index_date'].max().date()})")
    print(f"Design matrix: {X_tr.shape[1]} encoded features (from {len(feature_cols)} raw clinical columns + missingness flags)")

    # 1. Multi-class Primary Package Matching
    classes = sorted(df_tr["label_primary_package_code"].unique().tolist())
    cls_to_id = {c: i for i, c in enumerate(classes)}
    y_tr = df_tr["label_primary_package_code"].map(cls_to_id).to_numpy()
    y_te = df_te["label_primary_package_code"].map(cls_to_id).to_numpy()

    W_mc = fit_multinomial_l2(X_tr, y_tr, len(classes), l2=0.005)
    P_te = predict_proba_multinomial(X_te, W_mc)
    top1 = float((P_te.argmax(axis=1) == y_te).mean())
    top3_idx = np.argsort(-P_te, axis=1)[:, :3]
    top3 = float(np.any(top3_idx == y_te[:, None], axis=1).mean())
    elig_cols = [f"elig_{c}" for c in classes]
    elig_te = df_te[elig_cols].to_numpy()
    top1_eligible = float(elig_te[np.arange(len(df_te)), P_te.argmax(axis=1)].mean())
    print(f"\n[Task 1] Multi-Class Primary Package (13 classes):")
    print(f"  • Top-1 Exact Match Accuracy : {top1*100:.1f}%")
    print(f"  • Top-3 Match Accuracy       : {top3*100:.1f}%")
    print(f"  • Top-1 Clinically Eligible  : {top1_eligible*100:.1f}%")

    # 2. Multi-label Package Eligibility (13 binary models)
    W_ml_list = []
    aucs = []
    for c in classes:
        yb_tr = df_tr[f"elig_{c}"].to_numpy()
        yb_te = df_te[f"elig_{c}"].to_numpy()
        w_b = fit_binary_l2(X_tr, yb_tr, l2=0.005)
        W_ml_list.append(w_b)
        pb_te = expit(np.hstack([X_te, np.ones((len(X_te), 1), dtype=np.float32)]) @ w_b)
        aucs.append((c, roc_auc_fast(yb_te, pb_te)))
    W_ml = np.column_stack(W_ml_list)
    mean_auc = float(np.nanmean([a for _, a in aucs]))
    print(f"\n[Task 2] Multi-Label Eligibility across 13 Packages:")
    print(f"  • Macro ROC-AUC : {mean_auc:.4f}")
    print("  • Per-package ROC-AUC: " + ", ".join(f"{c}={a:.3f}" for c, a in aucs))

    # 3. Multi-label Clinical Add-ons (13 binary models)
    addon_cols = [c for c in meta["targets"]["multilabel_addons"] if c in df_tr.columns]
    W_addon_list = []
    addon_aucs = []
    for ac in addon_cols:
        ya_tr = df_tr[ac].to_numpy()
        ya_te = df_te[ac].to_numpy()
        w_a = fit_binary_l2(X_tr, ya_tr, l2=0.005)
        W_addon_list.append(w_a)
        pa_te = expit(np.hstack([X_te, np.ones((len(X_te), 1), dtype=np.float32)]) @ w_a)
        addon_aucs.append((ac.replace("addon_", ""), roc_auc_fast(ya_te, pa_te)))
    W_addons = np.column_stack(W_addon_list)
    mean_addon_auc = float(np.nanmean([a for _, a in addon_aucs]))
    print(f"\n[Task 3] Multi-Label Clinical Add-ons ({len(addon_cols)} add-ons):")
    print(f"  • Macro ROC-AUC : {mean_addon_auc:.4f}")

    # 4. 90-Day Purchase Propensity
    yp_tr = df_tr["label_purchased_90d"].to_numpy()
    yp_te = df_te["label_purchased_90d"].to_numpy()
    w_p = fit_binary_l2(X_tr, yp_tr, l2=0.01)
    pp_te = expit(np.hstack([X_te, np.ones((len(X_te), 1), dtype=np.float32)]) @ w_p)
    prop_auc = roc_auc_fast(yp_te, pp_te)
    print(f"\n[Task 4] 90-Day Package Purchase Propensity:")
    print(f"  • Test ROC-AUC : {prop_auc:.4f}")

    # 5. Refit on 100% of snapshot data and save production bundle
    print("\n[5/5] Refitting on 100% of dataset & saving production bundle to models/...")
    X_all, feat_names_all, stats_all = build_design_matrix(snap_sorted, feature_cols, None)
    y_all = snap_sorted["label_primary_package_code"].map(cls_to_id).to_numpy()
    W_mc_full = fit_multinomial_l2(X_all, y_all, len(classes), l2=0.005)
    W_ml_full = np.column_stack([fit_binary_l2(X_all, snap_sorted[f"elig_{c}"].to_numpy(), l2=0.005) for c in classes])
    W_addons_full = np.column_stack([fit_binary_l2(X_all, snap_sorted[ac].to_numpy(), l2=0.005) for ac in addon_cols])
    w_prop_full = fit_binary_l2(X_all, snap_sorted["label_purchased_90d"].to_numpy(), l2=0.01)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_DIR / "package_matcher_v2_bundle.npz",
        W_multiclass=W_mc_full.astype(np.float32),
        W_multilabel=W_ml_full.astype(np.float32),
        W_addons=W_addons_full.astype(np.float32),
        w_propensity=w_prop_full.astype(np.float32),
    )
    bundle_meta = {
        "version": "2.0",
        "classes": classes,
        "addon_cols": addon_cols,
        "encoded_feature_names": feat_names_all,
        "fit_stats": stats_all,
        "metrics_temporal_test": {
            "top1_accuracy": round(top1, 4),
            "top3_accuracy": round(top3, 4),
            "top1_clinically_eligible": round(top1_eligible, 4),
            "multilabel_macro_roc_auc": round(mean_auc, 4),
            "per_package_roc_auc": {c: round(a, 4) for c, a in aucs},
            "addons_macro_roc_auc": round(mean_addon_auc, 4),
            "propensity_90d_roc_auc": round(prop_auc, 4),
        },
    }
    (MODEL_DIR / "package_matcher_v2_meta.json").write_text(
        json.dumps(bundle_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Saved {MODEL_DIR / 'package_matcher_v2_bundle.npz'} and {MODEL_DIR / 'package_matcher_v2_meta.json'}")


if __name__ == "__main__":
    main()
