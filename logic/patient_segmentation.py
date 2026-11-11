# -*- coding: utf-8 -*-
"""
Patient Segmentation & Package Matching Engine
Unified clinical decision engine for patient risk stratification and package matching.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from logic.config import CLINICAL_THRESHOLDS


def get_age_band(age: float) -> str:
    """Returns clinical age band string: <30, 30-34, 35-49, 50-59, 60+"""
    for band_name, low, high in CLINICAL_THRESHOLDS["age_bands"]:
        if low <= age < high:
            return band_name
    return "60+" if age >= 60 else "<30"


def segment_patient(
    age: float = 35.0,
    bmi: float = 22.0,
    systolic: float = 120.0,
    diastolic: float = 80.0,
    diagnosis_text: str = "",
    gender: str = "ไม่ระบุ",
    lab_data: Dict[str, Any] | None = None,
    **lab_kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluates patient parameters across 3 dimensions (Age, Clinical Vitals, and Diagnosis History),
    and automatically delegates to the Lab-Aware Multi-Label Package Matcher v2 when lab biomarkers
    are supplied via `lab_data` or `**lab_kwargs`.

    Returns:
        {
            "age_band": str,          # e.g. "35-49"
            "risk_tier": int,          # 1 to 4
            "risk_reasons": List[str], # list of human-readable rationale strings
            "recommended_package": str # short_name of the best matched package
            ... plus optional v2 keys (top3_packages, eligible_packages, recommended_addons, etc.)
        }
    """
    age_val = float(age) if age is not None else 35.0
    bmi_val = float(bmi) if bmi is not None else 22.0
    sys_val = float(systolic) if systolic is not None else 120.0
    dia_val = float(diastolic) if diastolic is not None else 80.0
    diag_str = str(diagnosis_text).lower() if diagnosis_text else ""
    gender_str = str(gender).strip().lower()
    is_male = gender_str in ["m", "male", "ช", "ชาย"]

    age_band = get_age_band(age_val)

    # If rich lab / biomarker dictionary is provided, delegate to Package Matcher v2
    merged_labs: Dict[str, Any] = {}
    if isinstance(lab_data, dict):
        merged_labs.update(lab_data)
    if lab_kwargs:
        merged_labs.update(lab_kwargs)
    _v2_trigger_keys = {
        "fbs", "hba1c", "insulin", "homa_ir", "tc", "ldl", "hdl", "tg", "non_hdl",
        "apob", "lpa", "hscrp", "creatinine", "egfr", "uacr", "uric_acid", "alt",
        "ast", "ggt", "fib4", "hb", "mcv", "ferritin", "tsh", "cea", "afp", "psa",
        "ca125", "ca199", "fit_result", "ekg_result", "cac_score", "bmd_tscore",
    }
    if any(k in merged_labs and pd.notna(merged_labs[k]) for k in _v2_trigger_keys):
        from logic.package_matching_v2 import predict_patient_packages_v2

        row_dict = dict(merged_labs)
        row_dict.setdefault("age", age_val)
        row_dict.setdefault("bmi", bmi_val)
        row_dict.setdefault("sbp", sys_val)
        row_dict.setdefault("dbp", dia_val)
        row_dict.setdefault("gender", "ชาย" if is_male else "หญิง")
        v2_res = predict_patient_packages_v2(row_dict)
        return {
            "age_band": age_band,
            "risk_tier": v2_res["risk_tier"],
            "risk_reasons": v2_res["risk_reasons"],
            "recommended_package": v2_res["primary_short_name"],
            "primary_package_code": v2_res["primary_package_code"],
            "top3_packages": v2_res["top3_packages"],
            "eligible_packages": v2_res["eligible_packages"],
            "recommended_addons": v2_res["recommended_addons"],
            "personalized_package": v2_res.get("personalized_package"),
            "needs_custom_package": v2_res.get("needs_custom_package", False),
            "propensity_90d_pct": v2_res["propensity_90d_pct"],
            "top_feature_drivers": v2_res["top_feature_drivers"],
        }

    reasons: List[str] = []

    # Clinical flags
    kw = CLINICAL_THRESHOLDS["keywords"]
    has_chronic_crisis = any(k in diag_str for k in kw["chronic_crisis"])
    has_dm = any(k in diag_str for k in kw["diabetes"])
    has_ht = any(k in diag_str for k in kw["hypertension"])
    has_lipid = any(k in diag_str for k in kw["lipid"])
    has_gout = any(k in diag_str for k in kw.get("gout", []))

    # Waist / Central obesity estimation
    waist_cm = bmi_val * (3.65 if is_male else 3.35)
    is_central_obesity = (is_male and waist_cm > 90) or ((not is_male) and waist_cm > 80)

    # ─────────────────────────────────────────────────────────────
    # Tier 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม (Critical / Deep Symptoms)
    # ─────────────────────────────────────────────────────────────
    is_bp_crisis = (sys_val >= CLINICAL_THRESHOLDS["bp"]["crisis_sys"] or
                    dia_val >= CLINICAL_THRESHOLDS["bp"]["crisis_dia"])
    is_severe_obese = (bmi_val >= CLINICAL_THRESHOLDS["bmi"]["severe_obese"])
    is_senior = (age_val >= 60)

    if is_bp_crisis or is_severe_obese or is_senior or has_chronic_crisis:
        if is_bp_crisis:
            reasons.append(f"ความดันโลหิตระดับวิกฤต (BP {sys_val:.0f}/{dia_val:.0f} mmHg)")
        if is_severe_obese:
            reasons.append(f"โรคอ้วนรุนแรง (BMI {bmi_val:.1f})")
        if is_senior:
            reasons.append(f"กลุ่มผู้สูงวัยความเสี่ยงสูง (อายุ {age_val:.0f} ปี)")
        if has_chronic_crisis:
            reasons.append("ประวัติหรืออาการเสี่ยงโรคเรื้อรัง/มะเร็ง/หัวใจ")

        # Pick best package in Tier 4
        if is_bp_crisis or any(k in diag_str for k in ["หัวใจ", "heart", "แน่นหน้าอก", "chest pain"]):
            rec_pkg = "หัวใจ"
        elif any(k in diag_str for k in ["ลำไส้", "colon", "rectal"]):
            rec_pkg = "Colonoscopy"
        elif any(k in diag_str for k in ["มะเร็ง", "cancer", "tumor", "ก้อน"]):
            rec_pkg = "คัดกรองมะเร็ง"
        elif is_senior:
            rec_pkg = "ตรวจสุขภาพผู้สูงวัยเชิงลึก"
        else:
            rec_pkg = "คัดกรองมะเร็ง"

        return {
            "age_band": age_band,
            "risk_tier": 4,
            "risk_reasons": reasons,
            "recommended_package": rec_pkg
        }

    # ─────────────────────────────────────────────────────────────
    # Tier 3: เริ่มมีความผิดปกติ / โรคระยะเริ่มต้น (Early-stage Disease)
    # ─────────────────────────────────────────────────────────────
    is_bp_stage1 = (CLINICAL_THRESHOLDS["bp"]["stage1_sys"] <= sys_val < CLINICAL_THRESHOLDS["bp"]["crisis_sys"] or
                    CLINICAL_THRESHOLDS["bp"]["stage1_dia"] <= dia_val < CLINICAL_THRESHOLDS["bp"]["crisis_dia"])
    is_obese_class1 = (CLINICAL_THRESHOLDS["bmi"]["obese"] <= bmi_val < CLINICAL_THRESHOLDS["bmi"]["severe_obese"])
    is_age_50_59 = (50 <= age_val < 60)

    if is_bp_stage1 or has_dm or has_ht or has_lipid or is_obese_class1 or is_age_50_59:
        if has_dm:
            reasons.append("พบค่าน้ำตาลในเลือดสูง / ประวัติเบาหวาน")
        if is_bp_stage1 or has_ht:
            reasons.append(f"ความดันโลหิตสูงระดับ 1 (BP {sys_val:.0f}/{dia_val:.0f} mmHg)")
        if has_lipid:
            reasons.append("ประวัติภาวะไขมันในเลือดสูง")
        if is_obese_class1:
            reasons.append(f"ภาวะอ้วนระดับ 1 (BMI {bmi_val:.1f})")
        if is_age_50_59 and not reasons:
            reasons.append(f"กลุ่มอายุ 50-59 ปี เสี่ยงโรคเรื้อรัง (อายุ {age_val:.0f} ปี)")

        # Pick best package in Tier 3
        if has_dm:
            rec_pkg = "เบาหวาน"
        elif is_bp_stage1 or has_ht:
            rec_pkg = "ความดันโลหิตสูง"
        elif has_lipid:
            rec_pkg = "ไขมันในเลือดสูง"
        else:
            rec_pkg = "เบาหวาน"

        return {
            "age_band": age_band,
            "risk_tier": 3,
            "risk_reasons": reasons,
            "recommended_package": rec_pkg
        }

    # ─────────────────────────────────────────────────────────────
    # Tier 2: ยังไม่พบโรคแต่มีความเสี่ยง (At-Risk / Metabolic)
    # ─────────────────────────────────────────────────────────────
    is_bp_elevated = (CLINICAL_THRESHOLDS["bp"]["elevated_sys"] <= sys_val < CLINICAL_THRESHOLDS["bp"]["stage1_sys"])
    is_overweight = (CLINICAL_THRESHOLDS["bmi"]["overweight"] <= bmi_val < CLINICAL_THRESHOLDS["bmi"]["obese"])
    is_age_35_49 = (35 <= age_val < 50)

    if is_bp_elevated or is_overweight or is_age_35_49 or is_central_obesity or has_gout:
        if is_bp_elevated:
            reasons.append(f"ความดันโลหิตเริ่มสูงกว่าปกติ (BP {sys_val:.0f}/{dia_val:.0f} mmHg)")
        if is_overweight:
            reasons.append(f"น้ำหนักเกินเกณฑ์มาตรฐาน (BMI {bmi_val:.1f})")
        if is_central_obesity:
            reasons.append("ตรวจพบภาวะอ้วนลงพุง (รอบเอวเกินเกณฑ์)")
        if has_gout:
            reasons.append("ประวัติโรคเกาต์ / กรดยูริกสูง")
        if is_age_35_49 and not reasons:
            reasons.append(f"เข้าเกณฑ์ตรวจคัดกรองวัย 35+ (อายุ {age_val:.0f} ปี)")

        # Pick best package in Tier 2
        if bmi_val >= 25.0:
            rec_pkg = "คนอ้วน"
        elif is_age_35_49:
            rec_pkg = "วัย 35+"
        else:
            rec_pkg = "9.9 METABOLIC"

        return {
            "age_band": age_band,
            "risk_tier": 2,
            "risk_reasons": reasons,
            "recommended_package": rec_pkg
        }

    # ─────────────────────────────────────────────────────────────
    # Tier 1: ตรวจสุขภาพทั่วไป (General / Baseline Check)
    # ─────────────────────────────────────────────────────────────
    reasons.append("สุขภาพโดยรวมอยู่ในเกณฑ์ปกติ ไม่พบปัจจัยเสี่ยงเด่นชัด")
    if age_val < 30:
        rec_pkg = "BASIC"
    elif age_val < 35:
        rec_pkg = "STANDARD"
    else:
        rec_pkg = "PREMIUM"

    return {
        "age_band": age_band,
        "risk_tier": 1,
        "risk_reasons": reasons,
        "recommended_package": rec_pkg
    }


def segment_population_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized population risk stratification and age band assignment.
    Adds 'age_band' and 'risk_tier' columns efficiently (0.1s for 200k+ rows).
    """
    if df.empty:
        res = df.copy()
        res["age_band"] = pd.Series(dtype="object")
        res["risk_tier"] = pd.Series(dtype="int64")
        return res

    res = df.copy()

    # 1. Age band assignment
    if "age" in res.columns:
        age_col = pd.to_numeric(res["age"], errors="coerce").fillna(35.0)
    elif "age_at_visit" in res.columns:
        age_col = pd.to_numeric(res["age_at_visit"], errors="coerce").fillna(35.0)
    else:
        age_col = pd.Series(35.0, index=res.index)
    res["age"] = age_col

    res["age_band"] = pd.cut(
        age_col,
        bins=[-1, 29.99, 34.99, 49.99, 59.99, 200],
        labels=["<30", "30-34", "35-49", "50-59", "60+"]
    ).astype(str)

    # 2. Extract clinical vitals
    if "systolic" in res.columns:
        sys_bp = pd.to_numeric(res["systolic"], errors="coerce").fillna(120.0)
    elif "systolic_bp" in res.columns:
        sys_bp = pd.to_numeric(res["systolic_bp"], errors="coerce").fillna(120.0)
    else:
        sys_bp = pd.Series(120.0, index=res.index)

    if "diastolic" in res.columns:
        dia_bp = pd.to_numeric(res["diastolic"], errors="coerce").fillna(80.0)
    elif "diastolic_bp" in res.columns:
        dia_bp = pd.to_numeric(res["diastolic_bp"], errors="coerce").fillna(80.0)
    else:
        dia_bp = pd.Series(80.0, index=res.index)

    bmi_col = pd.to_numeric(res["bmi"], errors="coerce").fillna(22.0) if "bmi" in res.columns else pd.Series(22.0, index=res.index)

    # 3. Keyword diagnosis matching
    diag_col = res["diagnosis_text"].fillna("").astype(str).str.lower() if "diagnosis_text" in res.columns else pd.Series("", index=res.index)

    kw = CLINICAL_THRESHOLDS["keywords"]
    c_crisis_pat = "|".join(kw["chronic_crisis"])
    c_dm_pat = "|".join(kw["diabetes"])
    c_ht_pat = "|".join(kw["hypertension"])
    c_lipid_pat = "|".join(kw["lipid"])
    c_gout_pat = "|".join(kw.get("gout", []))

    has_crisis = diag_col.str.contains(c_crisis_pat, regex=True, na=False)
    has_dm = diag_col.str.contains(c_dm_pat, regex=True, na=False)
    has_ht = diag_col.str.contains(c_ht_pat, regex=True, na=False)
    has_lipid = diag_col.str.contains(c_lipid_pat, regex=True, na=False)
    has_gout = diag_col.str.contains(c_gout_pat, regex=True, na=False)

    bp_cfg = CLINICAL_THRESHOLDS["bp"]
    bmi_cfg = CLINICAL_THRESHOLDS["bmi"]

    # 4. Priority conditions (highest risk tier evaluated first)
    cond4 = (
        (age_col >= 60) |
        (sys_bp >= bp_cfg["crisis_sys"]) |
        (dia_bp >= bp_cfg["crisis_dia"]) |
        (bmi_col >= bmi_cfg["severe_obese"]) |
        has_crisis
    )

    cond3 = (
        (age_col >= 50) |
        (sys_bp >= bp_cfg["stage1_sys"]) |
        (dia_bp >= bp_cfg["stage1_dia"]) |
        (bmi_col >= bmi_cfg["obese"]) |
        has_dm | has_ht | has_lipid
    )

    cond2 = (
        (age_col >= 35) |
        (sys_bp >= bp_cfg["elevated_sys"]) |
        (dia_bp >= bp_cfg["elevated_dia"]) |
        (bmi_col >= bmi_cfg["overweight"]) |
        has_gout
    )

    res["risk_tier"] = np.select([cond4, cond3, cond2], [4, 3, 2], default=1)
    return res

