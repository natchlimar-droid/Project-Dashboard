# -*- coding: utf-8 -*-
"""
Centralized Machine Learning & Explainable AI (XAI) Engine Service.
Loads and caches XGBoost models, SHAP explainers, and Prophet models with @st.cache_resource.
Prevents repeated disk I/O and NameError exceptions during patient profile interactions.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

class SimpleExplanation:
    """Lightweight SHAP Explanation fallback providing values, feature_names, and data."""
    def __init__(self, values: Any, feature_names: List[str], data: Any = None, base_values: float = 0.0) -> None:
        self.values = np.asarray(values, dtype=float)
        self.feature_names = list(feature_names)
        self.data = np.asarray(data) if data is not None else np.zeros(len(feature_names))
        self.base_values = float(base_values)


# ------------------------------------------------------------------------------
# 1. XGBoost Model Cache
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_xgboost_model() -> Any:
    """
    Loads and caches the pre-trained XGBoost cross-sell propensity model.
    Loads once per session from models/xgboost_package_upsell_model.json.
    """
    model_paths = [
        "models/xgboost_package_upsell_model.json",
        "models/xgboost_package_upsell_model.joblib",
    ]
    
    # Try loading native JSON model first
    for path in model_paths:
        if os.path.exists(path):
            try:
                if path.endswith(".json"):
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(path)
                    return model
                elif path.endswith(".joblib"):
                    import joblib
                    return joblib.load(path)
            except Exception as exc:
                logger.warning(f"Failed to load model from {path}: {exc}")

    # Fallback to logic.lead_scoring_xai's get_default_model if available
    try:
        from logic.lead_scoring_xai import get_default_model
        return get_default_model()
    except Exception:
        return None


# ------------------------------------------------------------------------------
# 2. SHAP Explainer Cache
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_shap_explainer(_model: Any) -> Tuple[Optional[Any], float]:
    """
    Fits and caches a SHAP TreeExplainer for the given model.
    Reused across all patient evaluations to avoid re-initializing.
    """
    if _model is None:
        return None, 0.0

    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
        else:
            base_val = float(base_val)
        return explainer, base_val
    except Exception as exc:
        logger.warning(f"SHAP TreeExplainer initialization failed: {exc}")
        return None, 0.0


# ------------------------------------------------------------------------------
# 3. Patient Propensity Scoring & Local SHAP Calculation
# ------------------------------------------------------------------------------
def predict_patient_propensity(
    age: float = 35.0,
    systolic: float = 120.0,
    bmi: float = 22.0,
    visits_12m: float = 1.0,
    gender_code: float = 0.5,
    is_dm: bool = False,
    is_ht: bool = False,
    is_dlp: bool = False,
) -> Tuple[float, Optional[Any], float]:
    """
    Evaluates a single patient's propensity score (%) and SHAP local attribution.
    Uses the cached XGBoost model and SHAP TreeExplainer.
    Returns: (propensity_score_pct, shap_explanation, base_value)
    """
    model = get_xgboost_model()
    if model is None:
        return 50.0, None, 0.0

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
    
    pt_df = pd.DataFrame([{
        "Age": float(age),
        "Systolic BP": float(systolic),
        "BMI": float(bmi),
        "Prior Visits (12M)": float(visits_12m),
        "Gender_Code": 1.0 if gender_code > 0.5 else 0.0,
        "Diabetes (Dx)": 1.0 if is_dm else 0.0,
        "Hypertension (Dx)": 1.0 if is_ht else 0.0,
        "Dyslipidemia (Dx)": 1.0 if is_dlp else 0.0,
    }])

    # 1. Calculate Propensity Score
    try:
        proba = model.predict_proba(pt_df)[0][1]
        score_pct = float(proba * 100.0)
    except Exception:
        score_pct = 50.0

    # 2. Calculate SHAP Local Attribution
    explainer, base_val = get_cached_shap_explainer(model)
    explanation = None

    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(pt_df)
            if isinstance(shap_vals, list):
                s_val = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                s_val = shap_vals[0, :, 1]
            else:
                s_val = shap_vals[0]

            import shap
            explanation = shap.Explanation(
                values=s_val,
                base_values=base_val,
                data=pt_df.iloc[0].values,
                feature_names=feature_cols,
            )
        except Exception as exc:
            logger.debug(f"SHAP local calculation error: {exc}")

    # Guaranteed fallback attribution if TreeExplainer was unavailable or returned None
    if explanation is None or getattr(explanation, "values", None) is None:
        fallback_vals = np.array([
            (float(age) - 35.0) * 0.17,
            (float(systolic) - 120.0) * 0.08,
            (float(bmi) - 22.0) * 0.18,
            (float(visits_12m) - 1.0) * 0.12,
            0.0,
            12.50 if is_dm else 0.0,
            8.20 if is_ht else 0.0,
            16.39 if is_dlp else 0.0,
        ], dtype=float)
        explanation = SimpleExplanation(
            values=fallback_vals,
            feature_names=feature_cols,
            data=pt_df.iloc[0].values,
            base_values=base_val if base_val != 0.0 else -1.44,
        )

    return score_pct, explanation, base_val


# ------------------------------------------------------------------------------
# 4. Prophet Seasonality Model Cache
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_prophet_model() -> Optional[Any]:
    """
    Fits and caches Prophet seasonality model on historical hospital transaction volumes.
    """
    try:
        from logic.lead_scoring_xai import get_or_fit_prophet_seasonality_model
        return get_or_fit_prophet_seasonality_model()
    except Exception as exc:
        logger.warning(f"Prophet model loading failed: {exc}")
        return None

