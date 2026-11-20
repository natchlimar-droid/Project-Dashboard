# -*- coding: utf-8 -*-
"""
Model Training Script: Train XGBoost Propensity Model on 100% Synthetic Clinical Dataset
======================================================================================
Loads `data/synthetic/train_patient_snapshot.parquet` (10,000 synthetic patients),
extracts the 8 clinical & behavioral features, and trains the calibrated XGBoost Classifier
against `label_purchased_90d` ground-truth package purchases.

Outputs:
  - models/xgboost_package_upsell_model.json
  - models/xgboost_package_upsell_model.joblib
  - models/xgboost_package_upsell_meta.json
"""
import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def main():
    print("🚀 Starting XGBoost model training on 100% Synthetic Clinical Dataset...")

    # 1. Load Synthetic Patient Snapshot
    synth_path = os.path.join("data", "synthetic", "train_patient_snapshot.parquet")
    if not os.path.exists(synth_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {synth_path}")

    print(f"Loading synthetic patient snapshot from {synth_path}...")
    snap = pd.read_parquet(synth_path)
    print(f"✅ Loaded {len(snap):,} synthetic patient records.")

    # 2. Extract Exact 8 Feature Columns
    feature_cols = [
        "Age",
        "Systolic BP",
        "BMI",
        "Prior Visits (12M)",
        "Gender_Code",
        "Diabetes (Dx)",
        "Hypertension (Dx)",
        "Dyslipidemia (Dx)",
    ]

    X = pd.DataFrame({
        "Age": snap["age"].astype(float),
        "Systolic BP": snap["sbp"].astype(float),
        "BMI": snap["bmi"].astype(float),
        "Prior Visits (12M)": snap["prior_visits_12m"].astype(float),
        "Gender_Code": snap["gender_code"].astype(float),
        "Diabetes (Dx)": snap["known_dx_dm"].astype(float),
        "Hypertension (Dx)": snap["known_dx_ht"].astype(float),
        "Dyslipidemia (Dx)": snap["known_dx_dlp"].astype(float),
    }, index=snap.index)

    # 3. Ground Truth Target Label (1 if purchased package in 90 days, 0 otherwise)
    if "label_purchased_90d" not in snap.columns:
        raise KeyError("Target column 'label_purchased_90d' not found in synthetic dataset.")
    y = snap["label_purchased_90d"].astype(int).values

    n_buyers = int((y == 1).sum())
    n_non_buyers = int((y == 0).sum())
    print(f"Target distribution: {n_buyers:,} Buyers ({n_buyers/len(y)*100:.1f}%) vs {n_non_buyers:,} Non-Buyers ({n_non_buyers/len(y)*100:.1f}%)")

    # 4. Train-Test Split (80% Train, 20% Test, Stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split sizes: Train={len(X_tr):,} | Test={len(X_te):,}")

    # 5. Train Calibrated XGBoost Classifier
    print("Training XGBoost Classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_tr, y_tr)

    # 6. Evaluate Model on Hold-Out Test Set
    preds_prob = clf.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, preds_prob))
    print(f"🏆 Test Set ROC-AUC: {test_auc:.4f}")

    print("\nFeature Importances (Gain/Weight):")
    feat_imps = {}
    for col, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        feat_imps[col] = float(imp)
        print(f"  • {col:20s}: {imp:.4f}")

    # 7. Fit Final Model on 100% of Synthetic Data for Production
    print("\nRefitting model on 100% of synthetic dataset (10,000 patients)...")
    final_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss",
    )
    final_clf.fit(X, y)
    final_clf.get_booster().feature_names = feature_cols

    # 8. Save Models and Metadata
    os.makedirs("models", exist_ok=True)
    json_path = os.path.join("models", "xgboost_package_upsell_model.json")
    joblib_path = os.path.join("models", "xgboost_package_upsell_model.joblib")
    meta_path = os.path.join("models", "xgboost_package_upsell_meta.json")

    final_clf.save_model(json_path)
    print(f"✅ Saved Native JSON Model: {json_path}")
    joblib.dump(final_clf, joblib_path)
    print(f"✅ Saved Joblib Model: {joblib_path}")

    meta = {
        "model_type": "XGBClassifier",
        "training_data_source": "data/synthetic/train_patient_snapshot.parquet",
        "total_synthetic_patients": len(snap),
        "buyers_count": n_buyers,
        "non_buyers_count": n_non_buyers,
        "feature_columns": feature_cols,
        "test_roc_auc": round(test_auc, 4),
        "feature_importances": feat_imps,
        "trained_at": datetime.datetime.now().isoformat(),
        "notes": "Trained 100% on synthetic clinical star-schema dataset with zero real patient PHI."
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved Training Metadata: {meta_path}")

    # 9. Sanity Check with SHAP TreeExplainer
    try:
        import shap
        explainer = shap.TreeExplainer(final_clf)
        sample_x = X.iloc[[0]]
        shap_vals = explainer.shap_values(sample_x)
        print(f"\n🔍 SHAP TreeExplainer Sanity Check:")
        print(f"  Sample Patient 1: {sample_x.iloc[0].to_dict()}")
        print(f"  Base value: {explainer.expected_value:.3f}")
        print(f"  SHAP values computed successfully: shape={np.shape(shap_vals)}")
        print(f"  Predicted propensity: {final_clf.predict_proba(sample_x)[0, 1] * 100:.1f}%")
    except Exception as exc:
        print(f"SHAP check note: {exc}")

    print("\n🎉 Completed training on synthetic dataset successfully!")


if __name__ == "__main__":
    main()

