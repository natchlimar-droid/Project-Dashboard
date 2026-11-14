# -*- coding: utf-8 -*-
"""
Package Matching Engine v2 (Lab-aware, Multi-label)
====================================================
Clinical ground-truth labeler used to (1) generate training labels for ML
package-matching models and (2) act as a transparent rule-based fallback.

Differences vs. logic/patient_segmentation.py (v1):
  * Uses real lab values (HbA1c, FBS, HOMA-IR, LDL, TG, HDL, eGFR, UACR, Uric,
    ALT/US liver, tumor markers, FIT, EKG, CAC, Thai CV risk) instead of
    diagnosis keywords + BMI-estimated waist.
  * Returns ONE primary package (multi-class target) AND a multi-hot
    eligibility vector over all 13 packages (multi-label target) + add-ons.
  * label_tier == catalog level of the primary package (always consistent).
  * Vectorized: works on a whole DataFrame (100k rows < 1s).

Package names / prices are read from logic.config.PACKAGE_CATALOG
(single source of truth). Only stable codes are added here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.config import PACKAGE_CATALOG, CATEGORY_ORDER, CLINICAL_THRESHOLDS

# ─────────────────────────────────────────────────────────────
# Stable package codes (short_name in config -> code)
# ─────────────────────────────────────────────────────────────
CODE_BY_SHORT_NAME: Dict[str, str] = {
    "BASIC": "L1_BASIC",
    "STANDARD": "L1_STANDARD",
    "PREMIUM": "L1_PREMIUM",
    "9.9 METABOLIC": "L2_METABOLIC",
    "คนอ้วน": "L2_OBESITY",
    "วัย 35+": "L2_AGE35",
    "เบาหวาน": "L3_DM",
    "ความดันโลหิตสูง": "L3_HT",
    "ไขมันในเลือดสูง": "L3_LIPID",
    "คัดกรองมะเร็ง": "L4_CANCER",
    "หัวใจ": "L4_HEART",
    "ตรวจสุขภาพผู้สูงวัยเชิงลึก": "L4_SENIOR",
    "Colonoscopy": "L4_COLON",
}
PACKAGE_CODES: List[str] = list(CODE_BY_SHORT_NAME.values())

ADDON_CODES: List[str] = [
    "ADDON_PSA", "ADDON_MAMMOGRAM", "ADDON_PAP_HPV", "ADDON_FIT",
    "ADDON_LDCT_LUNG", "ADDON_HBV_HCV_SCREEN", "ADDON_LIVER_US_AFP",
    "ADDON_BMD", "ADDON_THYROID", "ADDON_FIBROSCAN", "ADDON_CAC",
    "ADDON_FUNDUS", "ADDON_ECHO",
]

# Asian obesity cut-off used for the "คนอ้วน" package (WHO Asia-Pacific BMI >= 25)
OBESITY_PKG_BMI = 25.0

EKG_ABNORMAL = ["LVH", "ST-T change", "Atrial fibrillation", "Old MI (Q wave)"]
EKG_ISCHEMIC = ["ST-T change", "Old MI (Q wave)"]


def build_package_dim() -> pd.DataFrame:
    """Flatten PACKAGE_CATALOG into a dimension table with stable codes."""
    rows = []
    for cat_key in CATEGORY_ORDER:
        cat = PACKAGE_CATALOG[cat_key]
        for p in cat["packages"]:
            code = CODE_BY_SHORT_NAME.get(p["short_name"])
            if code is None:
                continue
            rows.append({
                "package_code": code,
                "short_name": p["short_name"],
                "package_name": p["name"],
                "level": int(p["level"]),
                "category": cat_key,
                "price_thb": float(p["price"]),
                "items": " | ".join(b.replace("<br>", " ") for b in p.get("bullets", [])),
            })
    dim = pd.DataFrame(rows)
    dim["package_code"] = pd.Categorical(dim["package_code"], categories=PACKAGE_CODES, ordered=True)
    return dim.sort_values("package_code").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Clinical calculators
# ─────────────────────────────────────────────────────────────
def ckd_epi_2021(creatinine_mg_dl, age, is_male) -> np.ndarray:
    """eGFR (mL/min/1.73m2), race-free CKD-EPI 2021 creatinine equation."""
    scr = np.asarray(creatinine_mg_dl, dtype=float)
    age = np.asarray(age, dtype=float)
    male = np.asarray(is_male, dtype=bool)
    kappa = np.where(male, 0.9, 0.7)
    alpha = np.where(male, -0.302, -0.241)
    ratio = scr / kappa
    egfr = 142.0 * np.minimum(ratio, 1.0) ** alpha * np.maximum(ratio, 1.0) ** -1.200 * 0.9938 ** age
    return egfr * np.where(male, 1.0, 1.012)


def thai_cv_risk(age, is_male, sbp, has_dm, total_chol, is_smoker) -> np.ndarray:
    """10-year CVD risk, Thai CV Risk Score (Rama-EGAT, lipid version)."""
    score = (0.08183 * np.asarray(age, float)
             + 0.39499 * np.asarray(is_male, float)
             + 0.02084 * np.asarray(sbp, float)
             + 0.69974 * np.asarray(has_dm, float)
             + 0.00212 * np.asarray(total_chol, float)
             + 0.41916 * np.asarray(is_smoker, float))
    return 1.0 - 0.978296 ** np.exp(score - 7.04423)


def fib4(age, ast, alt, platelet_k) -> np.ndarray:
    """FIB-4 index (platelet in 10^3/uL)."""
    alt = np.asarray(alt, float)
    return np.asarray(age, float) * np.asarray(ast, float) / (np.asarray(platelet_k, float) * np.sqrt(np.where(alt > 0, alt, np.nan)))


# ─────────────────────────────────────────────────────────────
# Helpers (NaN-safe column access)
# ─────────────────────────────────────────────────────────────
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float) > 0


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].astype(object).where(df[col].notna(), "").astype(str)


def _fmt(s: pd.Series, nd: int = 0) -> pd.Series:
    return s.round(nd).map(lambda v: "" if pd.isna(v) else (f"{v:.{nd}f}"))


# ─────────────────────────────────────────────────────────────
# Main labeler
# ─────────────────────────────────────────────────────────────
def compute_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive boolean clinical flags used by both primary & eligibility logic."""
    idx = df.index
    age = _num(df, "age")
    male = _str(df, "gender").str.strip().str.lower().isin(["ชาย", "m", "male", "ช", "1"])
    bmi, waist = _num(df, "bmi"), _num(df, "waist_cm")
    sbp, dbp = _num(df, "sbp"), _num(df, "dbp")
    fbs, a1c, homa = _num(df, "fbs"), _num(df, "hba1c"), _num(df, "homa_ir")
    ldl, hdl, tg, tc = _num(df, "ldl"), _num(df, "hdl"), _num(df, "tg"), _num(df, "tc")
    egfr, uacr, uric = _num(df, "egfr"), _num(df, "uacr"), _num(df, "uric_acid")
    alt, fl_grade, cap = _num(df, "alt"), _num(df, "us_fatty_liver_grade"), _num(df, "fibroscan_cap")
    hb, mcv, ferritin = _num(df, "hb"), _num(df, "mcv"), _num(df, "ferritin")
    cea, afp, psa = _num(df, "cea"), _num(df, "afp"), _num(df, "psa")
    ca125, ca199 = _num(df, "ca125"), _num(df, "ca199")
    cac, cvr, bmd = _num(df, "cac_score"), _num(df, "thai_cv_risk_10y"), _num(df, "bmd_tscore")
    pack_years, n_meds, lpa = _num(df, "pack_years"), _num(df, "n_meds"), _num(df, "lpa")
    ekg, fit = _str(df, "ekg_result"), _str(df, "fit_result").str.lower()
    cxr, hbsag = _str(df, "cxr_result"), _str(df, "hbsag").str.lower()
    smoke = _str(df, "smoking_status").str.lower()
    ever_smoker = smoke.isin(["current", "former"])
    alcohol = _str(df, "alcohol_level").str.lower()

    known = {k: _flag(df, f"known_dx_{k}") for k in ["dm", "ht", "dlp", "ckd", "cad", "gout", "masld", "cancer_hx"]}
    sym = {k: _flag(df, f"sym_{k}") for k in [
        "chest_pain", "dyspnea", "palpitation", "rectal_bleeding", "bowel_change",
        "weight_loss", "memory", "falls"]}
    fhx_cad, fhx_cancer, fhx_crc = _flag(df, "fhx_cad_premature"), _flag(df, "fhx_cancer"), _flag(df, "fhx_crc")

    bp_cfg = CLINICAL_THRESHOLDS["bp"]
    f = pd.DataFrame(index=idx)
    f["male"] = male
    # Glycemic (ADA)
    f["dm"] = (a1c >= 6.5) | (fbs >= 126) | known["dm"]
    f["predm"] = ~f["dm"] & (a1c.between(5.7, 6.49) | fbs.between(100, 125.9) | (homa >= 2.5))
    # Blood pressure (Thai HT 2019 / config thresholds)
    f["ht"] = (sbp >= bp_cfg["stage1_sys"]) | (dbp >= bp_cfg["stage1_dia"]) | known["ht"]
    f["ht_severe"] = (sbp >= bp_cfg["crisis_sys"]) | (dbp >= bp_cfg["crisis_dia"])
    f["elev_bp"] = ~f["ht"] & ((sbp >= bp_cfg["elevated_sys"]) | (dbp >= bp_cfg["elevated_dia"]))
    # Lipids
    low_hdl = (male & (hdl < 40)) | (~male & (hdl < 50))
    f["dlp"] = (ldl >= 160) | (tc >= 240) | (tg >= 200) | known["dlp"] | ((ldl >= 130) & (f["dm"] | f["ht"]))
    f["borderline_lipid"] = ~f["dlp"] & ((ldl >= 130) | (tg >= 150) | low_hdl)
    # Adiposity (Asian cut-offs)
    f["central_obesity"] = (male & (waist >= 90)) | (~male & (waist >= 80))
    f["obese_pkg"] = (bmi >= OBESITY_PKG_BMI) | ((bmi >= CLINICAL_THRESHOLDS["bmi"]["overweight"]) & f["central_obesity"])
    f["severe_obese"] = bmi >= CLINICAL_THRESHOLDS["bmi"]["severe_obese"]
    # Metabolic syndrome (harmonized, Asian waist)
    f["mets_count"] = (f["central_obesity"].astype(int) + (tg >= 150).astype(int) + low_hdl.astype(int)
                       + ((sbp >= 130) | (dbp >= 85) | known["ht"]).astype(int)
                       + ((fbs >= 100) | known["dm"]).astype(int))
    # Organ damage
    f["ckd"] = (egfr < 60) | (uacr >= 30) | known["ckd"]
    f["masld"] = (fl_grade >= 1) | (cap >= 248) | ((alt > 40) & (alcohol != "heavy")) | known["masld"]
    f["hyperuricemia"] = (male & (uric > 7.0)) | (~male & (uric > 6.0)) | known["gout"]
    # Cardiac
    ekg_abn = ekg.isin(EKG_ABNORMAL)
    f["heart_hard"] = ((sym["chest_pain"] & ekg_abn) | ekg.isin(EKG_ISCHEMIC) | (cac >= 400)
                       | (sbp >= 180) | (dbp >= 110) | (known["cad"] & sym["chest_pain"]))
    f["heart_finding"] = (sym["chest_pain"] | (sym["dyspnea"] & (age >= 40)) | sym["palpitation"]
                          | ekg_abn | (cac >= 100) | known["cad"] | f["ht_severe"])
    f["heart_risk"] = (cvr >= 0.20) | (fhx_cad & (age >= 40) & (f["dlp"] | f["dm"] | ever_smoker)) | (lpa >= 50) & (age >= 40)
    # Colorectal
    anemia = (male & (hb < 13)) | (~male & (hb < 12))
    ida = anemia & ((ferritin < 30) | (ferritin.isna() & (mcv < 80)))
    f["colon_hard"] = fit.eq("positive") | sym["rectal_bleeding"]
    f["colon_alarm"] = (sym["bowel_change"] & (age >= 45)) | (ida & (age >= 45))
    f["colon_risk"] = fhx_crc & (age >= 40)
    # Cancer (markers / alarm / risk)
    f["cancer_marker"] = ((cea > 5) | (afp > 20) | (male & (psa > 4)) | (~male & (ca125 > 35)) | (ca199 > 37)
                          | sym["weight_loss"] | cxr.eq("Lung nodule"))
    f["cancer_risk"] = ((fhx_cancer & (age >= 40)) | (ever_smoker & (pack_years >= 20) & (age >= 50))
                        | known["cancer_hx"] | (hbsag.eq("positive") & (age >= 40)))
    # Geriatric
    f["geriatric_marker"] = (age >= 60) & (sym["memory"] | sym["falls"] | (bmd <= -2.5) | (n_meds >= 5) | (age >= 75))
    f["senior_age"] = age >= 60

    f["any_tier3"] = f["dm"] | f["ht"] | f["dlp"] | f["ckd"] | f["masld"]
    f["any_tier2"] = (f["predm"] | f["elev_bp"] | f["borderline_lipid"] | f["central_obesity"]
                      | (bmi >= CLINICAL_THRESHOLDS["bmi"]["overweight"]) | (f["mets_count"] >= 2) | f["hyperuricemia"])
    f["any_tier4"] = (f["heart_hard"] | f["heart_finding"] | f["heart_risk"] | f["colon_hard"] | f["colon_alarm"]
                      | f["colon_risk"] | f["cancer_marker"] | f["cancer_risk"] | f["geriatric_marker"])
    f["healthy"] = ~(f["any_tier2"] | f["any_tier3"] | f["any_tier4"])
    return f


def assign_package_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame (same index) with:
      label_primary_package_code, label_primary_short_name, label_tier, label_rule_id,
      label_reason, elig_<CODE> x13, n_eligible_packages, addon_<CODE> x13, label_addons
    """
    f = compute_clinical_flags(df)
    idx = df.index
    age, bmi = _num(df, "age"), _num(df, "bmi")
    sbp, a1c, fbs = _num(df, "sbp"), _num(df, "hba1c"), _num(df, "fbs")
    ldl, tg, lpa = _num(df, "ldl"), _num(df, "tg"), _num(df, "lpa")
    uacr = _num(df, "uacr")
    ekg = _str(df, "ekg_result")
    high_spender = _str(df, "spending_tier").str.lower().eq("high")

    # ── Tier-3 severity scores (argmax picks the most pressing disease) ──
    a1c_sev = np.where(a1c.notna(), (a1c - 6.5) / 1.0, (fbs - 126) / 30.0)
    dm_s = np.where(f["dm"], 1.0 + np.clip(np.nan_to_num(a1c_sev), 0, 4) + 0.5 * f["ckd"] + 0.3 * (uacr >= 30), -np.inf)
    ht_s = np.where(f["ht"] | (f["ckd"] & ~f["dm"]),
                    1.0 + np.clip(np.nan_to_num((sbp - 140) / 10.0), 0, 4) + 0.5 * f["ckd"] + 0.3 * ekg.eq("LVH"), -np.inf)
    lip_s = np.where(f["dlp"] | f["masld"],
                     0.8 + np.clip(np.nan_to_num((ldl - 160) / 30.0), 0, 4) + np.clip(np.nan_to_num((tg - 200) / 150.0), 0, 2)
                     + 0.4 * f["masld"] + 0.5 * (lpa >= 50), -np.inf)
    t3 = np.vstack([dm_s + 0.02, ht_s + 0.01, lip_s])  # tie-break DM > HT > LIPID
    t3_choice = np.array(["L3_DM", "L3_HT", "L3_LIPID"])[np.argmax(t3, axis=0)]

    # ── Tier-1/2 choices ──
    t2_choice = np.where(f["obese_pkg"] | f["severe_obese"], "L2_OBESITY",
                np.where(f["predm"] | (f["mets_count"] >= 2) | f["hyperuricemia"] | f["borderline_lipid"] | f["elev_bp"],
                         "L2_METABOLIC", "L2_AGE35"))
    t1_choice = np.where(high_spender & (age >= 30), "L1_PREMIUM", np.where(age < 30, "L1_BASIC", "L1_STANDARD"))

    # ── Priority cascade (first match wins) ──
    rules: List[Tuple[str, pd.Series, object]] = [
        ("R01_heart_hard", f["heart_hard"], "L4_HEART"),
        ("R02_colon_hard", f["colon_hard"], "L4_COLON"),
        ("R03_cancer_marker", f["cancer_marker"], "L4_CANCER"),
        ("R04_heart_finding", f["heart_finding"], "L4_HEART"),
        ("R05_colon_alarm", f["colon_alarm"], "L4_COLON"),
        ("R06_geriatric_marker", f["geriatric_marker"], "L4_SENIOR"),
        ("R07_tier3_disease", f["any_tier3"], t3_choice),
        ("R08_heart_risk", f["heart_risk"], "L4_HEART"),
        ("R09_colon_risk", f["colon_risk"], "L4_COLON"),
        ("R10_cancer_risk", f["cancer_risk"], "L4_CANCER"),
        ("R11_senior_age", f["senior_age"], "L4_SENIOR"),
        ("R12_tier2_risk", f["any_tier2"], t2_choice),
        ("R13_age35_healthy", (age >= 35) & ~high_spender, "L2_AGE35"),
    ]
    primary = pd.Series(t1_choice, index=idx, dtype=object)
    rule_id = pd.Series("R14_tier1_healthy", index=idx, dtype=object)
    for rid, mask, choice in reversed(rules):  # apply lowest priority first, highest last
        m = mask.fillna(False).to_numpy(dtype=bool)
        ch = np.asarray(choice, dtype=object) if not isinstance(choice, str) else choice
        primary = primary.mask(m, pd.Series(ch, index=idx) if not isinstance(ch, str) else ch)
        rule_id = rule_id.mask(m, rid)

    # ── Multi-label eligibility ──
    elig = pd.DataFrame(index=idx)
    healthy_or_t2 = ~(f["any_tier3"] | f["any_tier4"])
    elig["elig_L1_BASIC"] = f["healthy"] & (age < 30)
    elig["elig_L1_STANDARD"] = healthy_or_t2 & (age < 45)
    elig["elig_L1_PREMIUM"] = healthy_or_t2 & (age >= 30)
    elig["elig_L2_METABOLIC"] = (f["predm"] | (f["mets_count"] >= 2) | f["hyperuricemia"] | f["borderline_lipid"]
                                 | f["elev_bp"] | f["masld"])
    elig["elig_L2_OBESITY"] = f["obese_pkg"] | f["severe_obese"]
    elig["elig_L2_AGE35"] = (age >= 35) & (age < 60)
    elig["elig_L3_DM"] = f["dm"]
    elig["elig_L3_HT"] = f["ht"] | (f["ckd"] & ~f["dm"])
    elig["elig_L3_LIPID"] = f["dlp"] | f["masld"]
    elig["elig_L4_CANCER"] = f["cancer_marker"] | f["cancer_risk"]
    elig["elig_L4_HEART"] = f["heart_hard"] | f["heart_finding"] | f["heart_risk"]
    elig["elig_L4_SENIOR"] = f["senior_age"]
    elig["elig_L4_COLON"] = f["colon_hard"] | f["colon_alarm"] | f["colon_risk"] | age.between(50, 75)
    elig = elig.fillna(False).astype(bool)
    # Guarantee: primary is always eligible
    for code in PACKAGE_CODES:
        elig[f"elig_{code}"] = elig[f"elig_{code}"] | (primary == code)
    elig = elig.astype("int8")

    # ── Add-ons ──
    sex_m = f["male"]
    pack_years = _num(df, "pack_years")
    smoke = _str(df, "smoking_status").str.lower()
    hbsag = _str(df, "hbsag").str.lower()
    tsh, cac, cvr = _num(df, "tsh"), _num(df, "cac_score"), _num(df, "thai_cv_risk_10y")
    fib = _num(df, "fib4")
    fit_done = _str(df, "fit_result").ne("")
    add = pd.DataFrame(index=idx)
    add["addon_ADDON_PSA"] = sex_m & ((age >= 50) | ((age >= 45) & _flag(df, "fhx_cancer")))
    add["addon_ADDON_MAMMOGRAM"] = ~sex_m & (age >= 40)
    add["addon_ADDON_PAP_HPV"] = ~sex_m & age.between(25, 65)
    add["addon_ADDON_FIT"] = age.between(45, 75) & ~fit_done & (primary != "L4_COLON")
    add["addon_ADDON_LDCT_LUNG"] = age.between(50, 80) & smoke.isin(["current", "former"]) & (pack_years >= 20)
    add["addon_ADDON_HBV_HCV_SCREEN"] = (hbsag == "") & (age >= 30)
    add["addon_ADDON_LIVER_US_AFP"] = hbsag.eq("positive")
    add["addon_ADDON_BMD"] = (~sex_m & (age >= 65)) | (sex_m & (age >= 70)) | (~sex_m & (age >= 50) & (bmi < 19))
    add["addon_ADDON_THYROID"] = tsh.isna() & ((~sex_m & (age >= 35)) | _flag(df, "sym_fatigue"))
    add["addon_ADDON_FIBROSCAN"] = f["masld"] & (fib >= 1.3)
    add["addon_ADDON_CAC"] = age.between(40, 75) & cvr.between(0.05, 0.1999) & cac.isna()
    add["addon_ADDON_FUNDUS"] = f["dm"]
    add["addon_ADDON_ECHO"] = ekg.eq("LVH") | _flag(df, "sym_dyspnea")
    add = add.fillna(False).astype("int8")
    addon_cols = list(add.columns)
    addon_names = np.array([c.replace("addon_", "") for c in addon_cols], dtype=object)
    add_mat = add.to_numpy(dtype=bool)
    label_addons = ["|".join(addon_names[r]) for r in add_mat]

    # ── Human-readable reasons ──
    reason_parts: List[Tuple[pd.Series, pd.Series | str]] = [
        (f["heart_hard"], "สัญญาณหัวใจผิดปกติชัดเจน (EKG/CAC/BP≥180)"),
        (f["colon_hard"], "FIT บวก / ถ่ายเป็นเลือด"),
        (f["cancer_marker"], "Tumor marker สูง / น้ำหนักลด / พบก้อนในปอด"),
        (f["heart_finding"] & ~f["heart_hard"], "อาการ/ผลตรวจหัวใจ (เจ็บหน้าอก, ใจสั่น, EKG, BP≥160)"),
        (f["colon_alarm"], "อาการลำไส้/โลหิตจางจากขาดเหล็ก อายุ≥45"),
        (f["geriatric_marker"], "ผู้สูงวัยมีภาวะเปราะบาง (หกล้ม/ความจำ/กระดูกพรุน/ยา≥5)"),
        (f["dm"], "เบาหวาน: HbA1c " + _fmt(a1c, 1) + "% FBS " + _fmt(fbs) ),
        (f["ht"], "ความดันสูง: SBP " + _fmt(sbp)),
        (f["dlp"], "ไขมันสูง: LDL " + _fmt(ldl) + " TG " + _fmt(tg)),
        (f["ckd"], "ไตเสื่อม/อัลบูมินรั่ว (eGFR<60 หรือ UACR≥30)"),
        (f["masld"], "ไขมันพอกตับ"),
        (f["heart_risk"], "Thai CV risk ≥20% / ประวัติครอบครัวหัวใจ / Lp(a) สูง"),
        (f["colon_risk"], "ประวัติครอบครัวมะเร็งลำไส้"),
        (f["cancer_risk"], "ปัจจัยเสี่ยงมะเร็ง (ครอบครัว/สูบบุหรี่/HBV)"),
        (f["predm"], "ภาวะก่อนเบาหวาน/ดื้ออินซูลิน"),
        (f["obese_pkg"], "ภาวะอ้วน/อ้วนลงพุง: BMI " + _fmt(bmi, 1)),
        (f["mets_count"] >= 3, "Metabolic syndrome (≥3 เกณฑ์)"),
        (f["hyperuricemia"], "กรดยูริกสูง"),
        (f["elev_bp"], "ความดันเริ่มสูง"),
        (f["borderline_lipid"], "ไขมันก้ำกึ่ง"),
    ]
    reasons = pd.Series("", index=idx, dtype=object)
    for mask, txt in reason_parts:
        m = mask.fillna(False)
        piece = txt if isinstance(txt, str) else txt.astype(str)
        reasons = reasons.where(~m, reasons + piece + " | ")
    reasons = reasons.str.rstrip(" |")
    reasons = reasons.mask(reasons.eq("") & (age >= 35), "เข้าเกณฑ์ตรวจคัดกรองตามอายุ")
    reasons = reasons.mask(reasons.eq(""), "สุขภาพโดยรวมปกติ")

    level_by_code = {c: int(c[1]) for c in PACKAGE_CODES}
    short_by_code = {v: k for k, v in CODE_BY_SHORT_NAME.items()}
    out = pd.DataFrame(index=idx)
    out["label_primary_package_code"] = primary.astype(str)
    out["label_primary_short_name"] = out["label_primary_package_code"].map(short_by_code)
    out["label_tier"] = out["label_primary_package_code"].map(level_by_code).astype("int8")
    out["label_rule_id"] = rule_id.astype(str)
    out["label_reason"] = reasons
    out = pd.concat([out, elig], axis=1)
    out["n_eligible_packages"] = elig.sum(axis=1).astype("int8")
    out = pd.concat([out, add], axis=1)
    out["label_addons"] = label_addons
    out["mets_count"] = f["mets_count"].astype("int8")
    return out


# ─────────────────────────────────────────────────────────────
# Add-on Catalog & Feature Labels for Dashboard / XAI
# ─────────────────────────────────────────────────────────────
ADDON_CATALOG: Dict[str, Dict[str, object]] = {
    "ADDON_PSA": {
        "name": "ตรวจสารบ่งชี้มะเร็งต่อมลูกหมาก (PSA)",
        "price": 650,
        "indication": "เพศชายอายุ ≥ 50 ปี หรือ ≥ 45 ปีที่มีประวัติครอบครัวเป็นมะเร็ง",
    },
    "ADDON_MAMMOGRAM": {
        "name": "ดิจิทัลแมมโมแกรมและอัลตราซาวด์เต้านม (Digital Mammogram + US)",
        "price": 2500,
        "indication": "เพศหญิงอายุ ≥ 40 ปี คัดกรองมะเร็งเต้านมระยะเริ่มต้น",
    },
    "ADDON_PAP_HPV": {
        "name": "คัดกรองมะเร็งปากมดลูก (ThinPrep + HPV DNA Test)",
        "price": 1890,
        "indication": "เพศหญิงอายุ 25–65 ปี คัดกรองเชื้อ HPV สายพันธุ์เสี่ยงสูง",
    },
    "ADDON_FIT": {
        "name": "ตรวจเม็ดเลือดแดงแฝงในอุจจาระ (FIT Test)",
        "price": 450,
        "indication": "อายุ 45–75 ปี ที่ยังไม่ได้ส่องกล้องลำไส้ใหญ่หรือตรวจ FIT ในปีนี้",
    },
    "ADDON_LDCT_LUNG": {
        "name": "เอกซเรย์คอมพิวเตอร์ปอดรังสีต่ำ (Low-Dose CT Chest)",
        "price": 3900,
        "indication": "อายุ 50–80 ปี ที่มีประวัติสูบบุหรี่สะสม ≥ 20 Pack-Years",
    },
    "ADDON_HBV_HCV_SCREEN": {
        "name": "ตรวจคัดกรองไวรัสตับอักเสบ บี และ ซี (HBsAg, Anti-HCV)",
        "price": 890,
        "indication": "อายุ ≥ 30 ปี ที่ยังไม่เคยตรวจคัดกรองไวรัสตับอักเสบ",
    },
    "ADDON_LIVER_US_AFP": {
        "name": "เฝ้าระวังมะเร็งตับ (Ultrasound Upper Abdomen + AFP)",
        "price": 1990,
        "indication": "ผู้ติดเชื้อไวรัสตับอักเสบบี (HBsAg Positive) เฝ้าระวังทุก 6 เดือน",
    },
    "ADDON_BMD": {
        "name": "ตรวจความหนาแน่นมวลกระดูก 2 ส่วน (Bone Mineral Density DXA)",
        "price": 1800,
        "indication": "หญิง ≥ 65 ปี, ชาย ≥ 70 ปี หรือหญิงวัยหมดประจำเดือนที่มีน้ำหนักตัวน้อย",
    },
    "ADDON_THYROID": {
        "name": "ตรวจฮอร์โมนไทรอยด์ (TSH, Free T4)",
        "price": 950,
        "indication": "หญิงอายุ ≥ 35 ปี หรือมีอาการอ่อนเพลีย/ใจสั่นที่ยังไม่ได้ตรวจไทรอยด์",
    },
    "ADDON_FIBROSCAN": {
        "name": "ตรวจวัดไขมันและพังผืดตับ (FibroScan CAP & LSM)",
        "price": 2500,
        "indication": "ผู้มีภาวะไขมันพอกตับร่วมกับค่าดัชนีพังผืดตับ FIB-4 ≥ 1.3",
    },
    "ADDON_CAC": {
        "name": "ตรวจคราบหินปูนหลอดเลือดหัวใจ (CT Coronary Calcium Score)",
        "price": 3500,
        "indication": "อายุ 40–75 ปี ที่มีความเสี่ยงโรคหัวใจ 10 ปี (Thai CV Risk) ระดับปานกลาง 5–20%",
    },
    "ADDON_FUNDUS": {
        "name": "ถ่ายภาพจอประสาทตาดิจิทัล (Digital Fundus Photography)",
        "price": 750,
        "indication": "ผู้ที่มีภาวะเบาหวานหรือน้ำตาลสะสม HbA1c ≥ 6.5% คัดกรองเบาหวานขึ้นตา",
    },
    "ADDON_ECHO": {
        "name": "อัลตราซาวด์หัวใจ (Echocardiogram)",
        "price": 3200,
        "indication": "ตรวจพบหัวใจห้องล่างซ้ายโต (EKG LVH) หรือมีอาการเหนื่อยหอบง่าย",
    },
}

MODULAR_TEST_CATALOG: Dict[str, Dict[str, object]] = {
    "MOD_CORE": {
        "name": "ชุดตรวจพื้นฐานแกนหลัก (Core Physical & Hematology)",
        "organ": "พื้นฐานทั่วไป",
        "price": 790,
        "tests": ["ตรวจร่างกายโดยแพทย์ (PE)", "ความสมบูรณ์ของเม็ดเลือด (CBC)", "ตรวจปัสสาวะสมบูรณ์แบบ (UA)"],
        "default_reason": "ชุดตรวจพื้นฐานบังคับสำหรับประเมินสุขภาพองค์รวมเบื้องต้น",
    },
    "MOD_RENAL": {
        "name": "ชุดตรวจเจาะลึกการทำงานของไตและโปรตีนรั่ว (Renal & CKD Protection)",
        "organ": "ไตและสมดุลเกลือแร่",
        "price": 1290,
        "tests": ["Creatinine & eGFR (CKD-EPI)", "BUN", "ไมโครอัลบูมินในปัสสาวะ (UACR)", "สมดุลเกลือแร่ (Electrolytes Na/K/Cl/CO2)"],
        "default_reason": "เฝ้าระวังภาวะไตเสื่อมระยะเริ่มต้นและอัลบูมินรั่วในปัสสาวะ (ไม่อยู่ในแพ็กเกจมาตรฐาน)",
    },
    "MOD_THYROID": {
        "name": "ชุดตรวจสมดุลฮอร์โมนไทรอยด์ครบวงจร (Thyroid Endocrine Panel)",
        "organ": "ต่อมไร้ท่อและไทรอยด์",
        "price": 1350,
        "tests": ["TSH (Thyroid Stimulating Hormone)", "Free T3", "Free T4"],
        "default_reason": "คัดกรองภาวะไทรอยด์ทำงานผิดปกติ เหนื่อยง่าย น้ำหนักเปลี่ยน หรือใจสั่น (ไม่อยู่ในแพ็กเกจมาตรฐาน)",
    },
    "MOD_ANEMIA": {
        "name": "ชุดตรวจภาวะโลหิตจางและธาตุเหล็กสะสม (Anemia & Iron Study)",
        "organ": "ระบบโลหิตวิทยา",
        "price": 1150,
        "tests": ["Serum Ferritin", "Serum Iron & TIBC", "Reticulocyte Count", "Vitamin B12 / Folate"],
        "default_reason": "ค้นหาสาเหตุภาวะโลหิตจางและภาวะขาดธาตุเหล็กสะสม (ไม่อยู่ในแพ็กเกจมาตรฐาน)",
    },
    "MOD_GOUT": {
        "name": "ชุดตรวจกรดยูริกและสุขภาพข้อ (Hyperuricemia & Gout Care)",
        "organ": "ข้อและกรดยูริก",
        "price": 690,
        "tests": ["Uric Acid ในเลือด", "hs-CRP (ค่าการอักเสบ)", "ประเมินความเสี่ยงนิ่วในไตและข้ออักเสบเกาต์"],
        "default_reason": "ติดตามภาวะกรดยูริกในเลือดสูง ป้องกันโรคเกาต์และนิ่วในไต",
    },
    "MOD_HEPATIC": {
        "name": "ชุดตรวจสุขภาพตับ ไขมันพอกตับ และพังผืดตับ (Hepatology & FibroScan)",
        "organ": "ตับและทางเดินน้ำดี",
        "price": 2490,
        "tests": ["เอนไซม์ตับครบชุด (AST, ALT, ALP, GGT)", "ดัชนีพังผืดตับ (FIB-4)", "ตรวจวัดไขมันและพังผืดตับ (FibroScan)"],
        "default_reason": "ประเมินภาวะตับอักเสบ ไขมันพอกตับ และพังผืดตับระยะเริ่มต้น (ไม่อยู่ในแพ็กเกจมาตรฐาน)",
    },
    "MOD_GLYCEMIC": {
        "name": "ชุดตรวจคัดกรองเบาหวานและภาวะดื้ออินซูลิน (Glycemic & Insulin Resistance)",
        "organ": "น้ำตาลและเมตาบอลิก",
        "price": 1390,
        "tests": ["ระดับน้ำตาลหลังอดอาหาร (FBS)", "น้ำตาลสะสม (HbA1c)", "Fasting Insulin & HOMA-IR", "ตรวจจอประสาทตา (Digital Fundus)"],
        "default_reason": "ประเมินการควบคุมระดับน้ำตาล ภาวะดื้ออินซูลิน และภาวะแทรกซ้อนทางตา",
    },
    "MOD_LIPID_VASC": {
        "name": "ชุดตรวจไขมันเชิงลึกและการอักเสบหลอดเลือด (Advanced Lipid & Vascular)",
        "organ": "ไขมันและหลอดเลือด",
        "price": 1490,
        "tests": ["Lipid Profile (Chol, TG, HDL, Direct LDL)", "Non-HDL & ApoB", "hs-CRP (การอักเสบหลอดเลือด)"],
        "default_reason": "ประเมินความเสี่ยงหลอดเลือดแข็งตัวและไขมันอนุภาคเล็กก่อโรค",
    },
    "MOD_CARDIO": {
        "name": "ชุดตรวจสมรรถภาพหัวใจและหลอดเลือด (Cardiovascular Deep Check)",
        "organ": "หัวใจและหลอดเลือด",
        "price": 3490,
        "tests": ["คลื่นไฟฟ้าหัวใจ (EKG 12-Lead)", "อัลตราซาวด์หัวใจ (Echocardiogram) หรือ EST", "NT-proBNP / hs-Troponin I"],
        "default_reason": "ประเมินการทำงานของกล้ามเนื้อหัวใจและหลอดเลือดหัวใจในกลุ่มเสี่ยงสูง",
    },
    "MOD_GI_COLON": {
        "name": "ชุดตรวจคัดกรองระบบทางเดินอาหารและลำไส้ใหญ่ (GI & Colon Screening)",
        "organ": "ทางเดินอาหารและลำไส้",
        "price": 1590,
        "tests": ["ตรวจเลือดแฝงในอุจจาระ (FIT Test)", "สารบ่งชี้มะเร็งลำไส้ (CEA)", "อัลตราซาวด์ช่องท้องส่วนบน-ล่าง"],
        "default_reason": "คัดกรองความเสี่ยงติ่งเนื้อและมะเร็งลำไส้ใหญ่ รวมถึงอวัยวะในช่องท้อง",
    },
    "MOD_ONCO": {
        "name": "ชุดตรวจสารบ่งชี้มะเร็งเฉพาะบุคคล (Targeted Tumor Markers)",
        "organ": "คัดกรองมะเร็ง",
        "price": 2290,
        "tests": ["AFP (ตับ)", "CEA (ลำไส้)", "CA19-9 (ตับอ่อน)", "PSA (ชาย) หรือ CA-125 & Mammogram/HPV (หญิง)"],
        "default_reason": "คัดกรองสารบ่งชี้มะเร็งตามช่วงวัย เพศ และประวัติครอบครัว",
    },
    "MOD_BONE_SENIOR": {
        "name": "ชุดตรวจมวลกระดูก วิตามินดี และผู้สูงวัย (Bone Density & Longevity)",
        "organ": "กระดูกและผู้สูงวัย",
        "price": 2190,
        "tests": ["ตรวจความหนาแน่นมวลกระดูก (BMD DXA 2 ส่วน)", "ระดับวิตามินดี (25-OH Vitamin D)", "Calcium & Phosphate"],
        "default_reason": "คัดกรองภาวะกระดูกบาง/กระดูกพรุน และภาวะพร่องวิตามินดี",
    },
}


def build_personalized_package(
    df: pd.DataFrame,
    row_lbl: pd.Series,
    primary_code: str,
    primary_fit_pct: float,
    elig_codes: List[str],
) -> Dict[str, object]:
    """
    Constructs a Tailor-Made / Bespoke Personalized Health Package (`CUSTOM_BESPOKE`)
    for patients whose clinical/lab abnormalities are not fully covered by any single
    standard catalog package (e.g. CKD/renal decline, thyroid disorder, anemia, gout,
    fatty liver/fibrosis, or multi-organ comorbidities), and also supports interactive
    custom package assembly for any patient.
    """
    r = df.iloc[0]

    def _num(col: str, default: float = np.nan) -> float:
        if col in r and pd.notna(r[col]):
            try:
                return float(r[col])
            except Exception:
                return default
        return default

    age = _num("age", 35.0)
    bmi = _num("bmi", 22.0)
    sbp = _num("sbp", 120.0)
    dbp = _num("dbp", 80.0)
    fbs = _num("fbs")
    hba1c = _num("hba1c")
    homa_ir = _num("homa_ir")
    ldl = _num("ldl")
    tg = _num("tg")
    hdl = _num("hdl")
    hscrp = _num("hscrp")
    egfr = _num("egfr")
    uacr = _num("uacr")
    creatinine = _num("creatinine")
    uric_acid = _num("uric_acid")
    alt = _num("alt")
    ast = _num("ast")
    ggt = _num("ggt")
    fib4_val = _num("fib4")
    hb = _num("hb")
    ferritin = _num("ferritin")
    tsh = _num("tsh")
    cea = _num("cea")
    afp = _num("afp")
    psa = _num("psa")
    ca125 = _num("ca125")
    ca199 = _num("ca199")
    bmd = _num("bmd_tscore")
    cac = _num("cac_score")

    gender_str = str(r.get("gender", "")).strip().lower()
    is_male = gender_str in ("ช", "ชาย", "m", "male", "1")

    selected_modules: List[Dict[str, object]] = []
    uncovered_gaps: List[str] = []

    def _add_mod(mod_code: str, custom_reason: str, is_uncovered_by_std: bool = False):
        spec = MODULAR_TEST_CATALOG[mod_code]
        selected_modules.append({
            "module_code": mod_code,
            "name": spec["name"],
            "organ": spec["organ"],
            "price": int(spec["price"]),
            "tests": list(spec["tests"]),
            "reason": custom_reason,
            "is_core": mod_code == "MOD_CORE",
            "is_uncovered_gap": is_uncovered_by_std,
        })
        if is_uncovered_by_std:
            uncovered_gaps.append(f"{spec['organ']}: {custom_reason}")

    # Always include Core Base module
    _add_mod("MOD_CORE", "ชุดตรวจร่างกายและโลหิตวิทยาพื้นฐานสำหรับผู้รับบริการทุกราย", False)

    # 1. Renal / CKD gap (No standard package in the 13-package catalog is a dedicated Renal/CKD package!)
    if (not np.isnan(egfr) and egfr < 60.0) or (not np.isnan(uacr) and uacr >= 30.0) or (not np.isnan(creatinine) and creatinine > (1.3 if is_male else 1.1)):
        parts = []
        if not np.isnan(egfr) and egfr < 60.0:
            parts.append(f"eGFR {egfr:.0f} mL/min (<60)")
        if not np.isnan(uacr) and uacr >= 30.0:
            parts.append(f"UACR {uacr:.0f} mg/g (≥30)")
        if not parts and not np.isnan(creatinine):
            parts.append(f"Creatinine {creatinine:.2f} mg/dL")
        _add_mod("MOD_RENAL", f"พบสัญญาณการทำงานของไตลดลง ({', '.join(parts)}) ซึ่งไม่มีแพ็กเกจไตโดยเฉพาะในชุดมาตรฐาน", True)

    # 2. Thyroid gap (No standard package covers Thyroid TSH/FT3/FT4!)
    if not np.isnan(tsh) and (tsh < 0.4 or tsh > 4.0):
        _add_mod("MOD_THYROID", f"ผลฮอร์โมนไทรอยด์ผิดปกติ (TSH = {tsh:.2f} mIU/L) ซึ่งแพ็กเกจมาตรฐานไม่ครอบคลุม", True)

    # 3. Anemia / Iron gap (No standard package covers Ferritin/Iron/B12 anemia workup!)
    hb_low = (not np.isnan(hb)) and ((is_male and hb < 13.0) or ((not is_male) and hb < 12.0))
    ferr_low = (not np.isnan(ferritin)) and ferritin < 30.0
    if hb_low or ferr_low:
        p_txt = f"Hb = {hb:.1f} g/dL" if hb_low else f"Ferritin = {ferritin:.0f} ng/mL"
        _add_mod("MOD_ANEMIA", f"ตรวจพบภาวะโลหิตจาง/เหล็กสะสมต่ำ ({p_txt}) ต้องตรวจหาสาเหตุเชิงลึก", True)

    # 4. Gout / Hyperuricemia gap
    uric_high = (not np.isnan(uric_acid)) and ((is_male and uric_acid >= 7.0) or ((not is_male) and uric_acid >= 6.0))
    if uric_high:
        _add_mod("MOD_GOUT", f"กรดยูริกในเลือดสูง (Uric Acid = {uric_acid:.1f} mg/dL) เสี่ยงเกาต์และนิ่วในไต", True)

    # 5. Hepatic / Fatty Liver / Fibrosis gap
    liver_abn = (not np.isnan(alt) and alt >= 40.0) or (not np.isnan(ast) and ast >= 40.0) or (not np.isnan(ggt) and ggt >= 60.0) or (not np.isnan(fib4_val) and fib4_val >= 1.30)
    if liver_abn:
        l_parts = []
        if not np.isnan(alt) and alt >= 40.0:
            l_parts.append(f"ALT {alt:.0f} U/L")
        if not np.isnan(fib4_val) and fib4_val >= 1.30:
            l_parts.append(f"FIB-4 {fib4_val:.2f}")
        if not l_parts and not np.isnan(ggt):
            l_parts.append(f"GGT {ggt:.0f} U/L")
        _add_mod("MOD_HEPATIC", f"เอนไซม์ตับหรือดัชนีพังผืดตับสูง ({', '.join(l_parts)}) แนะนำตรวจตับและ FibroScan", True)

    # 6. Glycemic / Diabetes module
    dm_abn = (not np.isnan(fbs) and fbs >= 100.0) or (not np.isnan(hba1c) and hba1c >= 5.7) or (not np.isnan(homa_ir) and homa_ir >= 2.5)
    if dm_abn:
        g_parts = []
        if not np.isnan(fbs) and fbs >= 100.0:
            g_parts.append(f"FBS {fbs:.0f} mg/dL")
        if not np.isnan(hba1c) and hba1c >= 5.7:
            g_parts.append(f"HbA1c {hba1c:.1f}%")
        _add_mod("MOD_GLYCEMIC", f"ระดับน้ำตาล/ดื้ออินซูลินเกินเกณฑ์ ({', '.join(g_parts) or 'เสี่ยงเบาหวาน'})", primary_code not in ("L3_DM", "L2_METABOLIC"))

    # 7. Lipid & Vascular inflammation module
    lip_abn = (not np.isnan(ldl) and ldl >= 130.0) or (not np.isnan(tg) and tg >= 150.0) or (not np.isnan(hscrp) and hscrp >= 2.0)
    if lip_abn:
        lp_parts = []
        if not np.isnan(ldl) and ldl >= 130.0:
            lp_parts.append(f"LDL {ldl:.0f} mg/dL")
        if not np.isnan(tg) and tg >= 150.0:
            lp_parts.append(f"TG {tg:.0f} mg/dL")
        if not np.isnan(hscrp) and hscrp >= 2.0:
            lp_parts.append(f"hs-CRP {hscrp:.1f} mg/L")
        _add_mod("MOD_LIPID_VASC", f"ไขมันหรือค่าการอักเสบหลอดเลือดสูง ({', '.join(lp_parts)})", primary_code not in ("L3_LIPID", "L2_METABOLIC", "L4_HEART"))

    # 8. Cardiovascular module
    cv_abn = (sbp >= 140.0 or dbp >= 90.0) or (not np.isnan(cac) and cac >= 100.0) or ("L4_HEART" in elig_codes)
    if cv_abn:
        _add_mod("MOD_CARDIO", f"ความดันโลหิตสูง/ความเสี่ยงหัวใจ (BP {sbp:.0f}/{dbp:.0f} mmHg)", primary_code not in ("L4_HEART", "L3_HT"))

    # 9. GI & Colon module
    if ("L4_COLON" in elig_codes) or (not np.isnan(cea) and cea >= 5.0):
        _add_mod("MOD_GI_COLON", "พบความเสี่ยงระบบทางเดินอาหารและลำไส้ใหญ่", primary_code != "L4_COLON")

    # 10. Oncology module
    if ("L4_CANCER" in elig_codes) or (not np.isnan(afp) and afp >= 10.0) or (not np.isnan(psa) and psa >= 4.0) or (not np.isnan(ca125) and ca125 >= 35.0) or (not np.isnan(ca199) and ca199 >= 37.0):
        _add_mod("MOD_ONCO", "พบความเสี่ยงสารบ่งชี้มะเร็งหรือประวัติครอบครัวสายตรง", primary_code != "L4_CANCER")

    # 11. Bone & Senior module
    if age >= 60.0 or (not np.isnan(bmd) and bmd <= -1.5):
        _add_mod("MOD_BONE_SENIOR", f"ดูแลมวลกระดูกและสุขภาพผู้สูงวัย (อายุ {age:.0f} ปี)", primary_code != "L4_SENIOR")

    # If the patient is young/healthy and only MOD_CORE was selected, add preventive age/lifestyle modules
    if len(selected_modules) == 1:
        if bmi >= 23.0:
            _add_mod("MOD_LIPID_VASC", f"คัดกรองสมดุลไขมันและเมตาบอลิกตามค่า BMI {bmi:.1f}", False)
        if not is_male and age >= 30.0:
            _add_mod("MOD_THYROID", "คัดกรองสมดุลฮอร์โมนไทรอยด์เชิงป้องกันสำหรับเพศหญิงวัยทำงาน", False)
        else:
            _add_mod("MOD_HEPATIC", "ตรวจคัดกรองการทำงานของตับและช่องท้องเชิงป้องกัน", False)

    selected_codes = {m["module_code"] for m in selected_modules}
    optional_modules = [
        {
            "module_code": code,
            "name": spec["name"],
            "organ": spec["organ"],
            "price": int(spec["price"]),
            "tests": list(spec["tests"]),
            "reason": spec["default_reason"],
        }
        for code, spec in MODULAR_TEST_CATALOG.items()
        if code not in selected_codes
    ]

    organ_names = [m["organ"] for m in selected_modules if m["module_code"] != "MOD_CORE"][:3]
    organ_title = " + ".join(organ_names) if organ_names else "ตรวจสุขภาพเชิงป้องกันรายบุคคล"
    custom_pkg_name = f"แพ็กเกจเฉพาะบุคคล (Tailor-Made): {organ_title}"

    full_price = sum(int(m["price"]) for m in selected_modules)
    discount_pct = 18 if len(selected_modules) >= 4 else 15
    net_price = int(round((full_price * (100 - discount_pct) / 100.0) / 10.0) * 10)
    savings_baht = full_price - net_price

    n_active_organs = max(1, len(selected_modules) - 1)
    n_uncovered = len(uncovered_gaps)
    std_coverage_pct = int(round(max(25.0, min(95.0, ((n_active_organs - n_uncovered) / n_active_organs) * 100.0))))

    needs_custom = (
        primary_fit_pct < 72.0
        or n_uncovered >= 1
        or len(elig_codes) >= 3
    )

    if n_uncovered >= 1:
        gap_reason_th = (
            f"ตรวจพบความเสี่ยงเฉพาะทาง {n_uncovered} ระบบ ที่แพ็กเกจมาตรฐานอันดับ 1 ไม่ได้ครอบคลุมโดยตรง "
            f"({'; '.join(uncovered_gaps[:3])}) ระบบจึงประกอบแพ็กเกจเฉพาะบุคคลให้ครบ 100% ของปัญหาสุขภาพผู้ป่วยรายนี้"
        )
    elif len(elig_codes) >= 3:
        gap_reason_th = (
            f"ผู้ป่วยมีความเสี่ยงซ้อนทับหลายกลุ่มโรคพร้อมกัน ({len(elig_codes)} แพ็กเกจ) "
            f"การจัดแพ็กเกจเฉพาะบุคคลแบบ Modular ช่วยรวมรายการตรวจที่จำเป็นโดยไม่ต้องซื้อหลายแพ็กเกจซ้ำซ้อน"
        )
    else:
        gap_reason_th = (
            "สามารถเลือกปรับแต่งรายการตรวจเฉพาะบุคคล (Modular Custom Package) "
            "เพื่อเน้นตรวจเฉพาะอวัยวะเป้าหมายที่ต้องการเพิ่มเติมได้ทันที"
        )

    all_tests_flat = []
    for m in selected_modules:
        for t in m["tests"]:
            if t not in all_tests_flat:
                all_tests_flat.append(t)

    return {
        "package_code": "CUSTOM_BESPOKE",
        "name": custom_pkg_name,
        "short_name": f"เฉพาะบุคคล ({organ_title})",
        "needs_custom_package": bool(needs_custom),
        "uncovered_gaps": uncovered_gaps,
        "gap_reason_th": gap_reason_th,
        "std_coverage_pct": std_coverage_pct,
        "custom_coverage_pct": 100,
        "selected_modules": selected_modules,
        "optional_modules": optional_modules,
        "full_price": full_price,
        "discount_pct": discount_pct,
        "price": net_price,
        "savings_baht": savings_baht,
        "bullets": all_tests_flat[:8],
        "all_tests": all_tests_flat,
    }

FEATURE_LABELS_TH: Dict[str, str] = {
    "age": "อายุ (Age)",
    "bmi": "ดัชนีมวลกาย (BMI)",
    "waist_cm": "รอบเอว (Waist)",
    "waist_height_ratio": "สัดส่วนรอบเอวต่อส่วนสูง (WHtR)",
    "sbp": "ความดันตัวบน (Systolic BP)",
    "dbp": "ความดันตัวล่าง (Diastolic BP)",
    "fbs": "น้ำตาลหลังอดอาหาร (FBS)",
    "hba1c": "น้ำตาลสะสม (HbA1c)",
    "insulin": "อินซูลิน (Fasting Insulin)",
    "homa_ir": "ดัชนีดื้ออินซูลิน (HOMA-IR)",
    "tc": "คอเลสเตอรอลรวม (Total Chol)",
    "ldl": "ไขมันเลว (LDL-C)",
    "hdl": "ไขมันดี (HDL-C)",
    "tg": "ไตรกลีเซอไรด์ (Triglyceride)",
    "non_hdl": "ไขมันก่อโรคหลอดเลือด (Non-HDL)",
    "apob": "จำนวนอนุภาคไขมัน (ApoB)",
    "lpa": "ไขมันพันธุกรรม Lp(a)",
    "hscrp": "ค่าการอักเสบหลอดเลือด (hs-CRP)",
    "creatinine": "ครีเอตินิน (Creatinine)",
    "egfr": "อัตราการกรองของไต (eGFR)",
    "uacr": "อัลบูมินรั่วในปัสสาวะ (UACR)",
    "uric_acid": "กรดยูริก (Uric Acid)",
    "alt": "เอนไซม์ตับ (ALT/SGPT)",
    "ast": "เอนไซม์ตับ (AST/SGOT)",
    "ggt": "เอนไซม์ตับ (GGT)",
    "fib4": "ดัชนีพังผืดตับ (FIB-4)",
    "hb": "ความเข้มข้นเลือด (Hemoglobin)",
    "mcv": "ขนาดเม็ดเลือดแดง (MCV)",
    "ferritin": "เฟอร์ริตินสะสมเหล็ก (Ferritin)",
    "tsh": "ฮอร์โมนไทรอยด์ (TSH)",
    "cea": "สารบ่งชี้มะเร็งลำไส้ (CEA)",
    "afp": "สารบ่งชี้มะเร็งตับ (AFP)",
    "psa": "สารบ่งชี้มะเร็งต่อมลูกหมาก (PSA)",
    "ca125": "สารบ่งชี้มะเร็งรังไข่ (CA-125)",
    "ca199": "สารบ่งชี้มะเร็งตับอ่อน/ทางเดินน้ำดี (CA19-9)",
    "cac_score": "คราบหินปูนหลอดเลือดหัวใจ (CAC Score)",
    "bmd_tscore": "ความหนาแน่นมวลกระดูก (BMD T-Score)",
    "thai_cv_risk_10y": "ความเสี่ยงโรคหัวใจ 10 ปี (Thai CV Risk)",
    "mets_count": "จำนวนเกณฑ์ Metabolic Syndrome",
    "n_abnormal_labs": "จำนวนผล Lab ที่ผิดปกติ",
    "prior_visits_12m": "จำนวนครั้งรับบริการใน 12 เดือน",
    "prior_package_count": "ประวัติซื้อแพ็กเกจสะสม",
    "line_oa_engaged": "การติดตามผ่าน LINE OA",
}

_CACHED_V2_BUNDLE = None


def load_package_matcher_v2_bundle():
    """Loads `models/package_matcher_v2_bundle.npz` and `models/package_matcher_v2_meta.json` once."""
    global _CACHED_V2_BUNDLE
    if _CACHED_V2_BUNDLE is not None:
        return _CACHED_V2_BUNDLE
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    npz_path = root / "models" / "package_matcher_v2_bundle.npz"
    meta_path = root / "models" / "package_matcher_v2_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None
    weights = dict(np.load(npz_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _CACHED_V2_BUNDLE = (weights, meta)
    return _CACHED_V2_BUNDLE


def predict_patient_packages_v2(patient_row_or_df, top_k: int = 3, **kwargs) -> Dict[str, object]:
    """
    Unified Lab-Aware Clinical + ML Package Matcher (v2).
    Accepts a single patient dict, pd.Series, or 1-row pd.DataFrame.
    Returns:
      - risk_tier (1..4)
      - primary_package_code (e.g. 'L3_LIPID')
      - primary_short_name (e.g. 'ไขมันในเลือดสูง')
      - risk_reasons (List[str])
      - top3_packages (List[Dict] with package dict, ml_score_pct, is_eligible, package_code)
      - eligible_packages (List[str] of short_names)
      - recommended_addons (List[Dict] with code, name, price, indication)
      - propensity_90d_pct (float)
      - top_feature_drivers (List[Tuple[str, float, str]])
    """
    from scipy.special import expit, softmax

    if isinstance(patient_row_or_df, pd.Series):
        df = patient_row_or_df.to_frame().T.reset_index(drop=True)
    elif isinstance(patient_row_or_df, dict):
        df = pd.DataFrame([patient_row_or_df])
    else:
        df = patient_row_or_df.iloc[:1].copy().reset_index(drop=True)

    # Normalize common column aliases (systolic -> sbp, diastolic -> dbp, age_at_visit -> age)
    if "sbp" not in df.columns and "systolic" in df.columns:
        df["sbp"] = df["systolic"]
    if "sbp" not in df.columns and "systolic_bp" in df.columns:
        df["sbp"] = df["systolic_bp"]
    if "dbp" not in df.columns and "diastolic" in df.columns:
        df["dbp"] = df["diastolic"]
    if "dbp" not in df.columns and "diastolic_bp" in df.columns:
        df["dbp"] = df["diastolic_bp"]
    if "age" not in df.columns and "age_at_visit" in df.columns:
        df["age"] = df["age_at_visit"]
    if "fbs" not in df.columns and "fasting_glucose" in df.columns:
        df["fbs"] = df["fasting_glucose"]
    if "tc" not in df.columns and "total_cholesterol" in df.columns:
        df["tc"] = df["total_cholesterol"]

    # 1. Run Clinical Rule Engine v2
    labels = assign_package_labels(df)
    row_lbl = labels.iloc[0]
    primary_code = str(row_lbl["label_primary_package_code"])
    primary_short = str(row_lbl["label_primary_short_name"])
    risk_tier = int(row_lbl["label_tier"])
    reasons_str = str(row_lbl["label_reason"])
    risk_reasons = [r.strip() for r in reasons_str.split("|") if r.strip()]

    short_by_code = {v: k for k, v in CODE_BY_SHORT_NAME.items()}
    pkg_obj_by_code: Dict[str, Dict] = {}
    for cat_key in CATEGORY_ORDER:
        cat = PACKAGE_CATALOG[cat_key]
        for p in cat.get("packages", []):
            code = CODE_BY_SHORT_NAME.get(p["short_name"])
            if code:
                p_copy = dict(p)
                p_copy["category_label"] = cat["label"]
                p_copy["category_subtitle"] = cat["subtitle"]
                p_copy["package_code"] = code
                pkg_obj_by_code[code] = p_copy

    elig_codes = [c for c in PACKAGE_CODES if int(row_lbl.get(f"elig_{c}", 0)) == 1]
    elig_shorts = [short_by_code[c] for c in elig_codes if c in short_by_code]

    # 2. Run Trained ML Model Bundle if available
    bundle = load_package_matcher_v2_bundle()
    scores_by_code = {c: (92.0 if c == primary_code else (78.0 if c in elig_codes else 45.0)) for c in PACKAGE_CODES}
    propensity_pct = 55.0
    top_drivers: List[Tuple[str, float, str]] = []

    if bundle is not None:
        try:
            from data_generation.train_package_matcher import build_design_matrix

            weights, meta = bundle
            fit_stats = meta["fit_stats"]
            feature_cols = fit_stats["feature_cols"]
            classes = meta["classes"]
            X_enc, enc_names, _ = build_design_matrix(df, feature_cols, fit_stats)
            xb = np.hstack([X_enc, np.ones((1, 1), dtype=np.float32)])

            p_mc = softmax(xb @ weights["W_multiclass"], axis=1)[0]
            p_ml = expit(xb @ weights["W_multilabel"])[0]
            p_prop = float(expit(xb @ weights["w_propensity"])[0])
            propensity_pct = round(p_prop * 100.0, 1)

            for i, c in enumerate(classes):
                # Blend multi-label eligibility probability (70%) + multi-class primary probability (30%)
                # plus clinical rule bonus so the primary package is always #1
                raw_score = 0.70 * float(p_ml[i]) + 0.30 * float(p_mc[i])
                if c == primary_code:
                    raw_score = max(raw_score, 0.88) + 0.08
                elif c in elig_codes:
                    raw_score = max(raw_score, 0.65)
                scores_by_code[c] = round(float(np.clip(raw_score * 100.0, 15.0, 99.0)), 1)

            # Compute per-patient feature attributions for the primary package's eligibility head
            cls_idx = classes.index(primary_code) if primary_code in classes else 0
            w_cls = weights["W_multilabel"][:-1, cls_idx]
            contribs = X_enc[0] * w_cls
            driver_items = []
            for fname, cval in zip(enc_names, contribs):
                if fname.endswith("__isna") or "=__MISSING__" in fname:
                    continue
                base_col = fname.split("=")[0]
                if base_col not in df.columns or pd.isna(df[base_col].iloc[0]):
                    continue
                label_th = FEATURE_LABELS_TH.get(base_col, fname)
                raw_val = df[base_col].iloc[0]
                raw_str = f"{float(raw_val):.1f}" if isinstance(raw_val, (int, float, np.number)) else str(raw_val)
                driver_items.append((f"{label_th} [{raw_str}]", float(cval), raw_str))
            driver_items.sort(key=lambda t: abs(t[1]), reverse=True)
            top_drivers = driver_items[:8]
        except Exception:
            pass

    # 3. Rank Top-3 packages (primary first, then highest-scoring eligible packages)
    other_codes = [c for c in PACKAGE_CODES if c != primary_code]
    other_codes.sort(key=lambda c: (1 if c in elig_codes else 0, scores_by_code.get(c, 0.0)), reverse=True)
    ranked_codes = [primary_code] + other_codes[:2]

    top3_packages = []
    for rank_i, code in enumerate(ranked_codes):
        p_info = dict(pkg_obj_by_code.get(code, {}))
        p_info["ml_score_pct"] = scores_by_code.get(code, 80.0 - rank_i * 7.0)
        p_info["is_eligible"] = code in elig_codes
        p_info["is_primary"] = rank_i == 0
        top3_packages.append(p_info)

    # 4. Collect Recommended Add-ons
    rec_addons = []
    for acode in ADDON_CODES:
        if int(row_lbl.get(f"addon_{acode}", 0)) == 1 and acode in ADDON_CATALOG:
            item = dict(ADDON_CATALOG[acode])
            item["addon_code"] = acode
            rec_addons.append(item)

    # 5. Build Dynamic Personalized / Bespoke Package (for unmatched or multi-organ profiles)
    primary_fit_pct = float(scores_by_code.get(primary_code, 85.0))
    personalized_pkg = build_personalized_package(
        df=df,
        row_lbl=row_lbl,
        primary_code=primary_code,
        primary_fit_pct=primary_fit_pct,
        elig_codes=elig_codes,
    )

    return {
        "risk_tier": risk_tier,
        "primary_package_code": primary_code,
        "primary_short_name": primary_short,
        "risk_reasons": risk_reasons,
        "top3_packages": top3_packages,
        "eligible_packages": elig_shorts,
        "eligible_package_codes": elig_codes,
        "recommended_addons": rec_addons,
        "personalized_package": personalized_pkg,
        "needs_custom_package": bool(personalized_pkg["needs_custom_package"]),
        "propensity_90d_pct": propensity_pct,
        "top_feature_drivers": top_drivers,
        "mets_count": int(row_lbl.get("mets_count", 0)),
    }


