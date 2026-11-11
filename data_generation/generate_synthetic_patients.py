# -*- coding: utf-8 -*-
"""
Synthetic Patient Data Generator — Data4U Clinic
=================================================
Generates a realistic, physiologically-correlated, longitudinal synthetic EMR
for training package-matching / propensity models.

Design
------
* Latent risk factors per patient (insulin resistance, BP gene, lipid gene,
  kidney frailty, hidden cancers, CAD ...) drive ALL observed values, so lab
  results are mutually consistent (e.g. HbA1c <-> FBS via ADAG, TC via
  Friedewald, eGFR via CKD-EPI 2021, HOMA-IR = insulin*FBS/405).
* Longitudinal: 1-5 annual check-ups per patient (2022-2026) + OPD visits,
  values drift with age/BMI; known diagnoses start treatment (meds lower values).
* Realistic missingness: specialty tests ordered only when clinically likely.
* Labels from logic.package_matching_v2 (lab-aware clinical ground truth),
  plus a stochastic 90-day purchase outcome for propensity models.
* Point-in-time correct: snapshot features use only data <= index visit.

Outputs (default: data/synthetic/)
----------------------------------
  dim_patient.parquet          1 row / patient (static + lifestyle + family hx)
  dim_package.csv              13 packages with stable codes (from logic.config)
  dim_lab_test.csv             lab dictionary (LOINC, unit, reference range)
  fact_encounter.parquet       1 row / visit (check-up + OPD) with vitals, ICD-10
  fact_lab_result.parquet      long format 1 row / test / visit (sorted)
  fact_package_purchase.parquet historical + 90-day-after-index purchases
  train_patient_snapshot.parquet  ML-ready wide table, 1 row / patient (index visit)
  train_patient_snapshot_sample.csv  first 300 rows (UTF-8-SIG, open in Excel)
  feature_groups.json          column groups: keys / features / labels / leakage

Usage
-----
  python data_generation/generate_synthetic_patients.py --n-patients 10000 --seed 42
  python data_generation/generate_synthetic_patients.py --n-patients 50000 --complete-panel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.package_matching_v2 import (  # noqa: E402
    PACKAGE_CODES, assign_package_labels, build_package_dim, ckd_epi_2021, fib4, thai_cv_risk,
)

REF_DATE = pd.Timestamp("2026-10-31")        # data extraction date
CHECKUP_START = pd.Timestamp("2022-01-01")
CHECKUP_END = pd.Timestamp("2026-07-31")     # leaves >= 90 days to observe purchase label
PURCHASE_WINDOW_DAYS = 90

# ─────────────────────────────────────────────────────────────
# Lab dictionary (code, LOINC, name, unit, ref_low, ref_high, panel)
# ─────────────────────────────────────────────────────────────
LAB_TESTS = [
    ("FBS", "1558-6", "Fasting blood sugar", "mg/dL", 70, 99, "Glycemic"),
    ("HBA1C", "4548-4", "Hemoglobin A1c", "%", 4.0, 5.6, "Glycemic"),
    ("INSULIN", "27873-9", "Fasting insulin", "uIU/mL", 2.6, 24.9, "Glycemic"),
    ("HOMA_IR", "-", "HOMA-IR (calculated)", "index", 0.0, 2.5, "Glycemic"),
    ("TC", "2093-3", "Total cholesterol", "mg/dL", 0, 199, "Lipid"),
    ("LDL", "18262-6", "LDL-C (direct)", "mg/dL", 0, 129, "Lipid"),
    ("HDL", "2085-9", "HDL-C", "mg/dL", 40, 999, "Lipid"),
    ("TG", "2571-8", "Triglyceride", "mg/dL", 0, 149, "Lipid"),
    ("NON_HDL", "43396-1", "Non-HDL cholesterol (calculated)", "mg/dL", 0, 159, "Lipid"),
    ("APOB", "1884-6", "Apolipoprotein B", "mg/dL", 0, 99, "Lipid"),
    ("LPA", "10835-7", "Lipoprotein(a)", "mg/dL", 0, 49, "Lipid"),
    ("HSCRP", "30522-7", "hs-CRP", "mg/L", 0, 2.9, "Inflammation"),
    ("BUN", "3094-0", "Blood urea nitrogen", "mg/dL", 7, 20, "Kidney"),
    ("CREAT", "2160-0", "Creatinine", "mg/dL", 0.5, 1.2, "Kidney"),
    ("EGFR", "98979-8", "eGFR CKD-EPI 2021", "mL/min/1.73m2", 90, 999, "Kidney"),
    ("UACR", "9318-7", "Urine albumin/creatinine ratio", "mg/g", 0, 29, "Kidney"),
    ("URIC", "3084-1", "Uric acid", "mg/dL", 2.5, 7.0, "Kidney"),
    ("AST", "1920-8", "AST (SGOT)", "U/L", 0, 40, "Liver"),
    ("ALT", "1742-6", "ALT (SGPT)", "U/L", 0, 40, "Liver"),
    ("ALP", "6768-6", "Alkaline phosphatase", "U/L", 40, 129, "Liver"),
    ("GGT", "2324-2", "Gamma GT", "U/L", 0, 60, "Liver"),
    ("ALB", "1751-7", "Albumin", "g/dL", 3.5, 5.2, "Liver"),
    ("TBIL", "1975-2", "Total bilirubin", "mg/dL", 0.2, 1.2, "Liver"),
    ("HB", "718-7", "Hemoglobin", "g/dL", 12.0, 17.0, "CBC"),
    ("HCT", "4544-3", "Hematocrit", "%", 36, 50, "CBC"),
    ("WBC", "6690-2", "White blood cells", "10^3/uL", 4.0, 10.5, "CBC"),
    ("PLT", "777-3", "Platelets", "10^3/uL", 150, 450, "CBC"),
    ("MCV", "787-2", "Mean corpuscular volume", "fL", 80, 100, "CBC"),
    ("FERRITIN", "2276-4", "Ferritin", "ng/mL", 30, 400, "CBC"),
    ("TSH", "3016-3", "TSH", "uIU/mL", 0.4, 4.5, "Thyroid"),
    ("FT4", "3024-7", "Free T4", "ng/dL", 0.9, 1.7, "Thyroid"),
    ("VITD", "62292-8", "25-OH Vitamin D", "ng/mL", 30, 100, "Vitamin"),
    ("CEA", "2039-6", "CEA", "ng/mL", 0, 5.0, "Tumor marker"),
    ("AFP", "1834-1", "AFP", "ng/mL", 0, 10.0, "Tumor marker"),
    ("PSA", "2857-1", "PSA total", "ng/mL", 0, 4.0, "Tumor marker"),
    ("CA125", "10334-1", "CA-125", "U/mL", 0, 35, "Tumor marker"),
    ("CA199", "24108-3", "CA 19-9", "U/mL", 0, 37, "Tumor marker"),
]
LAB_COLS = {  # snapshot column name -> lab code
    "fbs": "FBS", "hba1c": "HBA1C", "insulin": "INSULIN", "homa_ir": "HOMA_IR", "tc": "TC", "ldl": "LDL",
    "hdl": "HDL", "tg": "TG", "non_hdl": "NON_HDL", "apob": "APOB", "lpa": "LPA", "hscrp": "HSCRP",
    "bun": "BUN", "creatinine": "CREAT", "egfr": "EGFR", "uacr": "UACR", "uric_acid": "URIC", "ast": "AST",
    "alt": "ALT", "alp": "ALP", "ggt": "GGT", "albumin": "ALB", "tbil": "TBIL", "hb": "HB", "hct": "HCT",
    "wbc": "WBC", "plt": "PLT", "mcv": "MCV", "ferritin": "FERRITIN", "tsh": "TSH", "ft4": "FT4", "vitd": "VITD",
    "cea": "CEA", "afp": "AFP", "psa": "PSA", "ca125": "CA125", "ca199": "CA199",
}
# Qualitative / procedure results kept in fact_encounter (and snapshot)
PROC_COLS = ["ua_protein", "ua_glucose", "hbsag", "anti_hcv", "fit_result", "ekg_result", "cxr_result",
             "us_fatty_liver_grade", "fibroscan_cap", "fibroscan_kpa", "cac_score", "abi", "bapwv",
             "echo_ef", "bmd_tscore"]
SYM_COLS = ["sym_chest_pain", "sym_dyspnea", "sym_palpitation", "sym_rectal_bleeding", "sym_bowel_change",
            "sym_weight_loss", "sym_fatigue", "sym_memory", "sym_falls", "sym_joint_pain", "sym_headache",
            "sym_epigastric_pain"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ═════════════════════════════════════════════════════════════
# 1) Patients
# ═════════════════════════════════════════════════════════════
def gen_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    comp = rng.choice(3, n, p=[0.25, 0.45, 0.30])
    age_ref = np.select([comp == 0, comp == 1], [rng.normal(29, 5, n), rng.normal(46, 8, n)], rng.normal(64, 9, n))
    age_ref = np.clip(age_ref, 20, 90)
    male = rng.random(n) < 0.44
    birth = REF_DATE - pd.to_timedelta((age_ref * 365.25).astype(int), unit="D")

    region = rng.choice(["กรุงเทพฯและปริมณฑล", "ภาคกลาง", "ภาคตะวันออกเฉียงเหนือ", "ภาคเหนือ", "ภาคใต้"], n,
                        p=[0.60, 0.15, 0.10, 0.08, 0.07])
    payer = rng.choice(["เงินสด", "ประกันสุขภาพเอกชน", "ประกันสังคม", "สวัสดิการบริษัท", "ข้าราชการ"], n,
                       p=[0.45, 0.20, 0.20, 0.12, 0.03])
    spend_p = {"เงินสด": [0.35, 0.45, 0.20], "ประกันสุขภาพเอกชน": [0.10, 0.45, 0.45],
               "ประกันสังคม": [0.55, 0.40, 0.05], "สวัสดิการบริษัท": [0.15, 0.55, 0.30], "ข้าราชการ": [0.30, 0.55, 0.15]}
    spending = np.array([rng.choice(["low", "mid", "high"], p=spend_p[p]) for p in payer])
    occupation = np.where(age_ref >= 62, "เกษียณ", rng.choice(
        ["พนักงานออฟฟิศ", "เจ้าของกิจการ", "รับราชการ", "ใช้แรงงาน", "แม่บ้าน/พ่อบ้าน", "อาชีพอิสระ"], n,
        p=[0.38, 0.15, 0.10, 0.15, 0.08, 0.14]))
    occupation = np.where(age_ref < 23, "นักศึกษา", occupation)

    smoke_u = rng.random(n)
    smoking = np.where(male, np.select([smoke_u < 0.28, smoke_u < 0.43], ["current", "former"], "never"),
                       np.select([smoke_u < 0.03, smoke_u < 0.05], ["current", "former"], "never"))
    pack_years = np.where(smoking == "never", 0.0,
                          np.clip(rng.gamma(2.0, 1.0, n) * np.clip(age_ref - 18, 1, None) * 0.25, 0.5, 90)).round(1)
    alc_levels = np.array(["none", "social", "regular", "heavy"])
    alcohol = np.where(male, rng.choice(alc_levels, n, p=[0.35, 0.35, 0.20, 0.10]),
                       rng.choice(alc_levels, n, p=[0.65, 0.28, 0.06, 0.01]))
    exercise = np.where(rng.random(n) < 0.28, 0, np.clip(rng.gamma(1.6, 70, n), 10, 900)).round(-1)

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1, dtype=np.int64),
        "hn": [f"HN{6800000 + i:07d}" for i in range(1, n + 1)],
        "gender": np.where(male, "ชาย", "หญิง"),
        "birth_date": birth.normalize(),
        "region": region,
        "payer_type": payer,
        "spending_tier": spending,
        "occupation": occupation,
        "smoking_status": smoking,
        "pack_years": pack_years,
        "alcohol_level": alcohol,
        "exercise_min_week": exercise,
        "sleep_hours": np.clip(rng.normal(6.6, 1.0, n), 3.5, 10).round(1),
        "stress_score_st5": np.clip(rng.poisson(5, n), 0, 15),
        "diet_score": np.clip(rng.normal(5.5, 1.8, n), 0, 10).round(1),
        "fhx_dm": (rng.random(n) < 0.25).astype("int8"),
        "fhx_ht": (rng.random(n) < 0.30).astype("int8"),
        "fhx_cad_premature": (rng.random(n) < 0.08).astype("int8"),
        "fhx_cancer": (rng.random(n) < 0.18).astype("int8"),
        "fhx_crc": (rng.random(n) < 0.05).astype("int8"),
        "line_oa_engaged": (rng.random(n) < np.where(age_ref < 55, 0.42, 0.22)).astype("int8"),
    })
    # ── hidden latent traits (NOT exported to snapshot; kept in _latent_ columns) ──
    df["_male"] = male
    df["_age_ref"] = age_ref
    df["_ir_gene"] = rng.normal(0, 1, n)
    df["_bp_gene"] = rng.normal(0, 1, n)
    df["_lipid_gene"] = rng.normal(0, 1, n)
    df["_fh"] = rng.random(n) < 0.004                      # familial hypercholesterolemia
    df["_hdl_gene"] = rng.normal(0, 1, n)
    df["_tg_gene"] = rng.normal(0, 1, n)
    df["_uric_gene"] = rng.normal(0, 1, n)
    df["_kidney_frailty"] = rng.normal(0, 1, n)
    df["_fat_gene"] = rng.normal(0, 1, n)
    df["_lpa"] = np.exp(rng.normal(np.log(15), 1.15, n)).clip(1, 300)
    df["_bmi0"] = np.clip((np.where(male, 24.0, 23.0) + 0.05 * np.clip(age_ref - 35, -15, 25)
                           + 0.9 * np.isin(alcohol, ["regular", "heavy"]) - 0.004 * exercise
                           + rng.normal(0, 3.8, n)), 16, 46)
    df["_bmi_slope"] = rng.normal(0.12, 0.45, n)            # BMI change per year
    df["_height"] = np.where(male, rng.normal(170, 6.5, n), rng.normal(158, 6.0, n))
    df["_hb0"] = rng.normal(0, 1.0, n)
    df["_thal_trait"] = rng.random(n) < 0.15                # thalassemia trait (common in Thailand)
    df["_ida"] = (~male) & (age_ref < 52) & (rng.random(n) < 0.10)
    df["_hypothyroid"] = rng.random(n) < np.where(male, 0.015, 0.06)
    born_before_1992 = birth.year < 1992
    df["_hbv"] = rng.random(n) < np.where(born_before_1992, 0.06, 0.012)
    df["_hcv"] = rng.random(n) < 0.008
    df["_af"] = rng.random(n) < np.clip(0.002 * np.exp(0.07 * (age_ref - 50)), 0, 0.12)
    df["_gilbert"] = rng.random(n) < 0.05
    df["_bph"] = male & (rng.random(n) < np.clip((age_ref - 45) / 60, 0, 0.6))
    df["_endometriosis"] = (~male) & (age_ref < 50) & (rng.random(n) < 0.06)

    # hidden cancers with onset dates (influence visits after onset)
    def onset(prob):
        has = rng.random(n) < np.clip(prob, 0, 0.5)
        days = rng.integers(0, (REF_DATE - CHECKUP_START).days, n)
        return np.where(has, (CHECKUP_START + pd.to_timedelta(days, unit="D")).to_numpy(), np.datetime64("NaT", "ns"))
    df["_crc_onset"] = onset(0.0012 * np.exp(0.06 * (age_ref - 45)) * np.where(df["fhx_crc"] == 1, 3.0, 1.0))
    df["_adenoma"] = rng.random(n) < np.clip(0.03 + 0.004 * np.clip(age_ref - 45, 0, None), 0, 0.25)
    df["_liver_ca_onset"] = onset(np.where(df["_hbv"], 0.02, 0.0004))
    df["_prostate_onset"] = onset(np.where(male, 0.002 * np.exp(0.08 * (age_ref - 50)), 0.0))
    df["_ovarian_onset"] = onset(np.where(~male, 0.0006 * np.exp(0.03 * (age_ref - 50)), 0.0))
    df["_lung_onset"] = onset(np.where(smoking != "never", 0.0008 * np.exp(0.05 * (age_ref - 50)) * (1 + pack_years / 20), 0.0002))
    return df


# ═════════════════════════════════════════════════════════════
# 2) Visit schedule
# ═════════════════════════════════════════════════════════════
def gen_checkup_schedule(pt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(pt)
    n_chk = np.clip(1 + rng.poisson(1.3, n), 1, 5)
    span = (CHECKUP_END - CHECKUP_START).days
    first_max = np.clip(span - (n_chk - 1) * 330, 30, span)
    first = rng.integers(0, 1_000_000_000, n) % first_max
    rows_pid, rows_date, rows_num = [], [], []
    for i in range(n):
        d = int(first[i])
        for k in range(n_chk[i]):
            if d > span:
                break
            rows_pid.append(pt["patient_id"].iat[i])
            rows_date.append(d)
            rows_num.append(k + 1)
            d += int(rng.normal(365, 50))
    v = pd.DataFrame({"patient_id": rows_pid, "day": rows_date, "checkup_no": rows_num})
    v["visit_date"] = CHECKUP_START + pd.to_timedelta(v["day"], unit="D")
    v = v.drop(columns="day")
    v["n_checkups_total"] = v.groupby("patient_id")["checkup_no"].transform("max")
    v["is_index_visit"] = (v["checkup_no"] == v["n_checkups_total"]).astype("int8")
    return v


# ═════════════════════════════════════════════════════════════
# 3) Physiology at a visit (vectorized)
# ═════════════════════════════════════════════════════════════
def simulate_checkups(v: pd.DataFrame, pt: pd.DataFrame, rng: np.random.Generator, complete_panel: bool) -> pd.DataFrame:
    d = v.merge(pt, on="patient_id", how="left")
    n = len(d)
    male = d["_male"].to_numpy()
    age = ((d["visit_date"] - d["birth_date"]).dt.days / 365.25).to_numpy()
    first_date = d.groupby("patient_id")["visit_date"].transform("min")
    t = ((d["visit_date"] - first_date).dt.days / 365.25).to_numpy()
    alc_num = d["alcohol_level"].map({"none": 0, "social": 1, "regular": 2, "heavy": 3}).to_numpy()
    smoker = (d["smoking_status"] == "current").to_numpy()
    exer = d["exercise_min_week"].to_numpy()
    N = lambda s: rng.normal(0, s, n)  # noqa: E731

    def after(col):
        o = d[col].to_numpy(dtype="datetime64[ns]")
        return (~np.isnat(o)) & (d["visit_date"].to_numpy(dtype="datetime64[ns]") >= o)
    crc, liver_ca, prost_ca = after("_crc_onset"), after("_liver_ca_onset"), after("_prostate_onset")
    ovar_ca, lung_ca = after("_ovarian_onset"), after("_lung_onset")
    any_ca = crc | liver_ca | prost_ca | ovar_ca | lung_ca

    # ── Anthropometrics ──
    bmi = np.clip(d["_bmi0"].to_numpy() + d["_bmi_slope"].to_numpy() * t + N(0.35) - 1.5 * any_ca, 15, 50)
    height = d["_height"].to_numpy() + N(0.4)
    weight = bmi * (height / 100) ** 2
    waist = np.where(male, 3.0 * bmi + 15, 2.9 * bmi + 8) + 0.10 * (age - 40) + N(3.5)

    # ── Glycemic ──
    z_bmi = (bmi - 23) / 4
    ir = (0.55 * z_bmi + 0.03 * (age - 40) + 0.5 * d["fhx_dm"].to_numpy() + 0.35 * (exer < 60)
          + 0.6 * d["_ir_gene"].to_numpy() + 0.2 * (d["diet_score"].to_numpy() < 4))
    a1c_u = 5.25 + 0.32 * ir + 1.4 * np.clip(ir - 1.3, 0, None) ** 1.2 + 0.04 * t + N(0.15)

    # ── BP ──
    sbp_u = (104 + 0.75 * (age - 30) + 1.3 * (bmi - 23) + 4 * smoker + np.select([alc_num == 3, alc_num == 2], [5, 2], 0)
             + 5 * d["fhx_ht"].to_numpy() + 8 * d["_bp_gene"].to_numpy() + 0.3 * d["stress_score_st5"].to_numpy() + N(8))

    # ── Lipids ──
    ldl_u = (95 + 0.9 * np.clip(age - 20, 0, 40) + 10 * ((~male) & (age > 50)) + 1.5 * (bmi - 23)
             + 30 * d["_lipid_gene"].to_numpy() + 110 * d["_fh"].to_numpy() + 4 * (d["diet_score"].to_numpy() < 4) + N(10))

    # ── Treatment status: known dx at visit (diagnosed before / at previous visit) ──
    d["_a1c_u"], d["_sbp_u"], d["_ldl_u"] = a1c_u, sbp_u, ldl_u
    d = d.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    for col, thr in [("_a1c_u", 6.5), ("_sbp_u", 140), ("_ldl_u", 160)]:
        d[col + "_prev"] = d.groupby("patient_id")[col].shift(1)
    # re-align arrays to sorted order
    idx_order = d.index.to_numpy()
    first = d["checkup_no"].to_numpy() == 1
    pre_dx_u = rng.random(len(d))
    known_dm = np.where(first, (d["_a1c_u"] >= 6.5) & (pre_dx_u < 0.60), d["_a1c_u_prev"].fillna(0) >= 6.5)
    known_ht = np.where(first, (d["_sbp_u"] >= 140) & (pre_dx_u < 0.50), d["_sbp_u_prev"].fillna(0) >= 140)
    known_dlp = np.where(first, (d["_ldl_u"] >= 160) & (pre_dx_u < 0.45), d["_ldl_u_prev"].fillna(0) >= 160)
    # once known always known
    for name, arr in [("known_dx_dm", known_dm), ("known_dx_ht", known_ht), ("known_dx_dlp", known_dlp)]:
        d[name] = arr.astype(int)
        d[name] = d.groupby("patient_id")[name].cummax().astype("int8")
    # re-extract arrays in sorted order
    male = d["_male"].to_numpy()
    age = ((d["visit_date"] - d["birth_date"]).dt.days / 365.25).to_numpy()
    n = len(d)
    alc_num = d["alcohol_level"].map({"none": 0, "social": 1, "regular": 2, "heavy": 3}).to_numpy()
    smoker = (d["smoking_status"] == "current").to_numpy()
    exer = d["exercise_min_week"].to_numpy()
    bmi, height, weight, waist, ir = (x[idx_order] for x in (bmi, height, weight, waist, ir))
    crc, liver_ca, prost_ca, ovar_ca, lung_ca, any_ca = (x[idx_order] for x in (crc, liver_ca, prost_ca, ovar_ca, lung_ca, any_ca))
    a1c_u, sbp_u, ldl_u = d["_a1c_u"].to_numpy(), d["_sbp_u"].to_numpy(), d["_ldl_u"].to_numpy()

    adherence = rng.uniform(0.4, 1.1, n)
    on_metformin = d["known_dx_dm"].to_numpy() == 1
    on_insulin = on_metformin & (a1c_u > 9.0) & (rng.random(n) < 0.5)
    on_antihtn = (d["known_dx_ht"].to_numpy() == 1) & (rng.random(n) < 0.88)
    on_statin = ((d["known_dx_dlp"].to_numpy() == 1) | on_metformin) & (rng.random(n) < 0.70)

    a1c = np.clip(a1c_u - on_metformin * 0.8 * adherence - on_insulin * 1.0 + N(0.1), 4.2, 14.5)
    fbs = np.clip(28.7 * a1c - 46.7 - 14 + N(9), 62, 450)
    insulin = np.clip(np.exp(np.log(7) + 0.38 * ir + N(0.3)) + on_insulin * 8, 1.5, 90)
    homa = insulin * fbs / 405
    sbp = np.clip(sbp_u - on_antihtn * rng.normal(14, 7, n) * adherence, 85, 230)
    dbp = np.clip(0.45 * sbp + 18 - 0.15 * np.clip(age - 55, 0, None) + 0.35 * (bmi - 23) + N(6.5), 45, 135)
    hr = np.clip(76 + 3 * smoker + N(10) + d["_af"].to_numpy() * rng.normal(15, 10, n), 45, 150)

    ldl = np.clip(ldl_u * (1 - on_statin * 0.40 * np.clip(adherence, 0, 1)), 35, 400)
    hdl = np.clip(np.where(male, 46, 56) - 0.9 * (bmi - 23) + 0.02 * np.clip(exer, 0, 300) - 4 * smoker
                  + 4 * np.isin(alc_num, [1, 2]) - 2.5 * np.clip(ir, 0, None) + 6 * d["_hdl_gene"].to_numpy() + N(4), 18, 110)
    tg = np.clip(np.exp(np.log(95) + 0.28 * ir + 0.12 * alc_num + 0.35 * d["_tg_gene"].to_numpy() + N(0.18)), 30, 1500)
    tc = ldl + hdl + tg / 5
    non_hdl = tc - hdl
    apob = np.clip(0.6 * non_hdl + 12 + N(7), 30, 250)
    lpa = d["_lpa"].to_numpy()
    hscrp = np.clip(np.exp(0.12 * (bmi - 23) + 0.4 * smoker + 0.3 * np.clip(ir, 0, None) + 1.2 * any_ca + N(0.7)), 0.1, 30)

    # ── Kidney ──
    dm_true = a1c_u >= 6.5
    kf = d["_kidney_frailty"].to_numpy()
    log_cr = (np.log(np.where(male, 0.95, 0.72)) + 0.008 * np.clip(age - 40, 0, None) + 0.10 * dm_true
              + 0.004 * np.clip(sbp_u - 130, 0, None) + 0.07 * kf + np.where(kf > 1.65, 0.35 * (kf - 1.65) + 0.3, 0) + N(0.06))
    creat = np.clip(np.exp(log_cr), 0.3, 12)
    egfr = np.clip(ckd_epi_2021(creat, age, male), 3, 150)
    uacr = np.clip(np.exp(np.log(7) + 0.9 * dm_true + 0.25 * np.clip(a1c_u - 6.5, 0, None) + 0.025 * np.clip(sbp_u - 125, 0, None)
                          + 0.4 * kf + N(0.6)), 1, 3000)
    bun = np.clip(7 + 7 * creat + N(2.5), 4, 120)
    uric = np.clip(np.where(male, 6.1, 4.7) + 0.13 * (bmi - 23) + 0.35 * alc_num + 0.6 * (egfr < 60)
                   + 0.9 * d["_uric_gene"].to_numpy() + N(0.4), 1.5, 14)
    known_gout = (uric > 8.5) & (rng.random(n) < 0.5)

    # ── Liver ──
    hbv, hcv = d["_hbv"].to_numpy(), d["_hcv"].to_numpy()
    fat = 0.6 * ir + 0.25 * alc_num + 0.5 * d["_fat_gene"].to_numpy() + N(0.15)
    alt = np.clip(np.exp(np.log(17) + 0.30 * fat + 0.5 * (hbv | hcv) + 0.8 * liver_ca + N(0.3)) + 5 * male, 5, 600)
    ast = np.clip(0.7 * alt + 9 + 4 * alc_num + N(3), 8, 600)
    ggt = np.clip(np.exp(np.log(22) + 0.2 * fat + 0.35 * alc_num + 0.4 * male + N(0.35)), 6, 900)
    alp = np.clip(rng.normal(75, 18, n) + 30 * liver_ca, 25, 600)
    alb = np.clip(4.4 - 0.012 * np.clip(age - 50, 0, None) - 0.5 * any_ca + N(0.25), 2.0, 5.5)
    tbil = np.clip(np.exp(np.log(0.65) + N(0.3)) + d["_gilbert"].to_numpy() * rng.uniform(0.6, 1.8, n), 0.1, 6)
    fl_grade = np.clip(np.floor(np.interp(fat + N(0.25), [-10, 0.55, 1.15, 1.75, 10], [0, 0.999, 1.999, 2.999, 3.0])), 0, 3)
    cap = np.clip(210 + 45 * fat + N(20), 100, 400)
    lsm = np.clip(np.exp(np.log(4.6) + 0.12 * fat + 0.5 * (alc_num == 3) + 0.6 * (hbv | hcv) + N(0.25)), 2.5, 75)

    # ── CBC / thyroid / vitamin ──
    thal, ida_pt = d["_thal_trait"].to_numpy(), d["_ida"].to_numpy()
    hb = np.clip(np.where(male, 14.9, 13.0) + d["_hb0"].to_numpy() + N(0.3) - 2.0 * ida_pt - 1.8 * crc - 0.6 * thal
                 - 0.8 * (egfr < 45) - 0.8 * any_ca, 6, 19)
    mcv = np.clip(88 + N(4) - 12 * ida_pt - 8 * crc - 14 * thal, 55, 110)
    ferritin = np.clip(np.exp(np.where(male, np.log(160), np.log(70)) + N(0.5) - 1.6 * ida_pt - 1.4 * crc), 3, 1500)
    hct = np.clip(3 * hb + N(1.0), 18, 60)
    wbc = np.clip(np.exp(np.log(6.8) + 0.12 * smoker + N(0.2)), 2.5, 25)
    plt_ = np.clip(rng.normal(265, 55, n) - 40 * (lsm > 10), 40, 700)
    hypo = d["_hypothyroid"].to_numpy()
    tsh = np.clip(np.exp(np.log(1.7) + N(0.45) + hypo * rng.normal(1.6, 0.4, n)), 0.05, 80)
    ft4 = np.clip(rng.normal(1.25, 0.15, n) - 0.35 * hypo, 0.3, 3)
    vitd = np.clip(rng.normal(24, 8, n) - 3 * (~male), 5, 80)

    # ── Tumor markers ──
    cea = np.clip(np.exp(np.log(1.6) + 0.5 * smoker + N(0.45) + crc * rng.normal(1.6, 0.8, n) + lung_ca * rng.normal(1.2, 0.8, n)), 0.2, 500)
    afp = np.clip(np.exp(np.log(3) + 0.4 * hbv + N(0.45) + liver_ca * rng.normal(3.0, 1.2, n)), 0.5, 5000)
    psa = np.where(male, np.clip(np.exp(np.log(0.7) + 0.035 * (age - 40) + 0.6 * d["_bph"].to_numpy() + N(0.5)
                                        + prost_ca * rng.normal(1.4, 0.6, n)), 0.05, 500), np.nan)
    ca125 = np.where(~male, np.clip(np.exp(np.log(12) + 0.7 * d["_endometriosis"].to_numpy() + N(0.5)
                                           + ovar_ca * rng.normal(2.5, 0.8, n)), 2, 5000), np.nan)
    ca199 = np.clip(np.exp(np.log(9) + N(0.6) + 0.15 * (a1c > 9) + crc * rng.normal(0.8, 0.6, n)), 1, 3000)

    # ── Cardio / procedures ──
    tcv = thai_cv_risk(age, male, sbp_u, dm_true, tc, smoker)
    cad = rng.random(n) < np.clip(tcv * 0.8, 0, 0.6)
    cad = pd.Series(cad).groupby(d["patient_id"].to_numpy()).cummax().to_numpy()
    known_cad = cad & (rng.random(n) < 0.35)
    p_cac0 = np.clip(1 - 0.02 * np.clip(age - 35, 0, None) * (1 + male) - 0.15 * dm_true - 0.1 * smoker - 0.5 * cad, 0.02, 1)
    cac = np.where(rng.random(n) < p_cac0, 0.0,
                   np.clip(np.exp(np.log(30) + 0.05 * (age - 45) + 1.5 * cad + N(1.2)), 1, 4000)).round(0)
    af = d["_af"].to_numpy()
    lvh = rng.random(n) < 0.5 * sigmoid((sbp_u - 165) / 10)
    stt = rng.random(n) < (0.02 + 0.30 * cad)
    old_mi = cad & (rng.random(n) < 0.08)
    brady = rng.random(n) < 0.03
    ekg = np.select([old_mi, af, stt, lvh, brady], ["Old MI (Q wave)", "Atrial fibrillation", "ST-T change", "LVH", "Sinus bradycardia"], "Normal")
    pad = cad & smoker & (rng.random(n) < 0.35)
    abi = np.clip(1.12 + N(0.07) - 0.30 * pad, 0.4, 1.4)
    bapwv = np.clip(1000 + 14 * (age - 30) + 6 * (sbp - 120) + 80 * dm_true + N(120), 700, 3000)
    echo_ef = np.clip(rng.normal(63, 5, n) - 15 * old_mi - 6 * (af & (rng.random(n) < 0.3)), 20, 80)
    bmd = np.where(age >= 45, -0.3 - 0.045 * np.clip(age - 50, 0, None) - 0.6 * ((~male) & (age > 50)) + 0.05 * (bmi - 23) + N(0.8), np.nan)
    cxr = np.select([lung_ca & (rng.random(n) < 0.7), (rng.random(n) < 0.012 * (d["smoking_status"] != "never")),
                     (echo_ef < 45) | (lvh & (rng.random(n) < 0.3)), rng.random(n) < 0.02],
                    ["Lung nodule", "Lung nodule", "Cardiomegaly", "Old TB scar"], "Normal")
    fit = np.where(rng.random(n) < np.where(crc, 0.75, np.where(d["_adenoma"].to_numpy() & (age >= 45), 0.25, 0.03)), "positive", "negative")
    ua_protein = np.select([uacr >= 300, uacr >= 30, rng.random(n) < 0.08], ["2+", "1+", "trace"], "negative")
    ua_glucose = np.where(fbs > 180, "positive", "negative")
    hbsag = np.where(hbv, "positive", "negative")
    anti_hcv = np.where(hcv, "positive", "negative")

    # ── Symptoms ──
    R = lambda p: (rng.random(n) < np.clip(p, 0, 1)).astype("int8")  # noqa: E731
    sym = {
        "sym_chest_pain": R(0.03 + 0.35 * cad),
        "sym_dyspnea": R(0.03 + 0.20 * cad + 0.05 * (bmi > 30) + 0.15 * (echo_ef < 50)),
        "sym_palpitation": R(0.03 + 0.5 * af + 0.15 * (tsh < 0.3)),
        "sym_rectal_bleeding": R(0.01 + 0.35 * crc),
        "sym_bowel_change": R(0.03 + 0.45 * crc),
        "sym_weight_loss": R(0.01 + 0.45 * any_ca + 0.15 * (a1c > 9.5)),
        "sym_fatigue": R(0.08 + 0.2 * (hb < 11) + 0.25 * hypo),
        "sym_memory": R(0.01 + 0.012 * np.clip(age - 60, 0, None)),
        "sym_falls": R(np.where(age >= 65, 0.06 + 0.01 * (age - 65), 0.01)),
        "sym_joint_pain": R(0.10 + 0.4 * known_gout + 0.004 * np.clip(age - 40, 0, None)),
        "sym_headache": R(0.05 + 0.2 * (sbp >= 160)),
        "sym_epigastric_pain": R(0.06),
    }

    n_meds = (on_metformin.astype(int) + on_insulin + on_antihtn + on_statin + hypo + af
              + rng.poisson(0.4 + 0.05 * np.clip(age - 60, 0, None), n))

    out = d[["patient_id", "visit_date", "checkup_no", "n_checkups_total", "is_index_visit",
             "known_dx_dm", "known_dx_ht", "known_dx_dlp"]].copy()
    out["age_at_visit"] = age.round(1)
    out["height_cm"], out["weight_kg"], out["bmi"] = height.round(1), weight.round(1), bmi.round(1)
    out["waist_cm"] = waist.round(1)
    out["sbp"], out["dbp"], out["hr"] = sbp.round(0), dbp.round(0), hr.round(0)
    out["spo2"] = np.clip(rng.normal(98, 1, n) - 2 * (echo_ef < 40), 85, 100).round(0)
    out["temp_c"] = np.clip(rng.normal(36.7, 0.3, n), 35.8, 38.5).round(1)
    labs = dict(fbs=fbs.round(0), hba1c=a1c.round(1), insulin=insulin.round(1), homa_ir=homa.round(2),
                tc=tc.round(0), ldl=ldl.round(0), hdl=hdl.round(0), tg=tg.round(0), non_hdl=non_hdl.round(0),
                apob=apob.round(0), lpa=lpa.round(1), hscrp=hscrp.round(2), bun=bun.round(0), creatinine=creat.round(2),
                egfr=egfr.round(0), uacr=uacr.round(1), uric_acid=uric.round(1), ast=ast.round(0), alt=alt.round(0),
                alp=alp.round(0), ggt=ggt.round(0), albumin=alb.round(1), tbil=tbil.round(2), hb=hb.round(1),
                hct=hct.round(1), wbc=wbc.round(1), plt=plt_.round(0), mcv=mcv.round(1), ferritin=ferritin.round(0),
                tsh=tsh.round(2), ft4=ft4.round(2), vitd=vitd.round(1), cea=cea.round(2), afp=afp.round(1),
                psa=np.round(psa, 2), ca125=np.round(ca125, 1), ca199=ca199.round(1))
    for k, val in labs.items():
        out[k] = val
    procs = dict(ua_protein=ua_protein, ua_glucose=ua_glucose, hbsag=hbsag, anti_hcv=anti_hcv, fit_result=fit,
                 ekg_result=ekg, cxr_result=cxr, us_fatty_liver_grade=fl_grade, fibroscan_cap=cap.round(0),
                 fibroscan_kpa=lsm.round(1), cac_score=cac, abi=abi.round(2), bapwv=bapwv.round(0),
                 echo_ef=echo_ef.round(0), bmd_tscore=np.round(bmd, 1))
    for k, val in procs.items():
        out[k] = val
    for k, val in sym.items():
        out[k] = val
    out["known_dx_ckd"] = ((d.groupby("patient_id").cumcount() > 0) & (pd.Series(egfr < 60).groupby(d["patient_id"]).shift(1).fillna(False).to_numpy())).astype("int8")
    out["known_dx_cad"] = known_cad.astype("int8")
    out["known_dx_gout"] = known_gout.astype("int8")
    out["known_dx_masld"] = ((fl_grade >= 2) & (rng.random(n) < 0.4)).astype("int8")
    out["known_dx_hypothyroid"] = (hypo & (rng.random(n) < 0.6)).astype("int8")
    out["known_dx_cancer_hx"] = (rng.random(n) < 0.012 + 0.0004 * np.clip(age - 40, 0, None)).astype("int8")
    for c in ["known_dx_cad", "known_dx_gout", "known_dx_masld", "known_dx_hypothyroid", "known_dx_cancer_hx"]:
        out[c] = out.groupby("patient_id")[c].cummax().astype("int8")
    out["med_metformin"], out["med_insulin"] = on_metformin.astype("int8"), on_insulin.astype("int8")
    out["med_antihtn"], out["med_statin"] = on_antihtn.astype("int8"), on_statin.astype("int8")
    out["n_meds"] = n_meds.astype("int16")
    # hidden truth for QA only
    out["_true_cancer"] = any_ca.astype("int8")
    out["_true_cad"] = cad.astype("int8")

    # ── Realistic test ordering (missingness) ──
    if not complete_panel:
        out = apply_test_ordering(out, d, rng)
    return out


def apply_test_ordering(o: pd.DataFrame, d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(o)
    age = o["age_at_visit"].to_numpy()
    male = d["_male"].to_numpy()
    dm, ht = o["known_dx_dm"].to_numpy() == 1, o["known_dx_ht"].to_numpy() == 1
    obese = o["bmi"].to_numpy() >= 25
    risky = dm | ht | obese | (o["fbs"].to_numpy() >= 100)
    smoker = d["smoking_status"].to_numpy() != "never"

    def order(cols, p):
        m = rng.random(n) >= np.clip(p, 0, 1)  # True => not ordered
        for c in cols:
            if o[c].dtype == object or str(o[c].dtype).startswith(("str", "string")):
                o.loc[m, c] = None
            else:
                o.loc[m, c] = np.nan

    order(["hba1c"], np.where(dm | (age >= 35) | (o["fbs"].to_numpy() >= 100), 0.85, 0.30))
    order(["insulin", "homa_ir"], np.where(obese | dm, 0.45, 0.12))
    order(["apob"], np.where(o["ldl"].to_numpy() >= 130, 0.30, 0.08))
    order(["lpa"], 0.07)
    order(["hscrp"], np.where(risky, 0.35, 0.15))
    order(["uacr"], np.where(dm | ht, 0.85, 0.25))
    order(["ggt", "alp", "albumin", "tbil"], 0.55)
    order(["ferritin"], np.where(o["hb"].to_numpy() < 12, 0.55, 0.10))
    order(["tsh", "ft4"], np.where(~male & (age >= 35), 0.45, 0.20))
    order(["vitd"], 0.20)
    order(["cea"], np.where(age >= 40, 0.45, 0.15))
    order(["afp"], np.where(o["hbsag"].to_numpy() == "positive", 0.9, 0.30))
    order(["psa"], np.where(male & (age >= 45), 0.55, 0.0))
    order(["ca125"], np.where(~male & (age >= 35), 0.25, 0.08))
    order(["ca199"], 0.15)
    order(["fit_result"], np.where(age >= 45, 0.40, 0.08))
    order(["hbsag", "anti_hcv"], 0.35)
    order(["ekg_result"], np.where(age >= 35, 0.85, 0.40))
    order(["cxr_result"], 0.92)
    order(["us_fatty_liver_grade"], np.where(age >= 35, 0.50, 0.25))
    order(["fibroscan_cap", "fibroscan_kpa"], np.where(o["alt"].to_numpy() > 40, 0.30, 0.06))
    order(["cac_score"], np.where((age >= 40) & (risky | smoker), 0.18, 0.03))
    order(["abi", "bapwv"], np.where(ht | dm, 0.40, 0.10))
    order(["echo_ef"], np.where((o["sym_dyspnea"].to_numpy() == 1) | (o["sym_chest_pain"].to_numpy() == 1), 0.6, 0.05))
    order(["bmd_tscore"], np.where(~male & (age >= 50), 0.35, np.where(age >= 65, 0.25, 0.0)))
    return o


# ═════════════════════════════════════════════════════════════
# 4) OPD visits, encounters & ICD-10
# ═════════════════════════════════════════════════════════════
ACUTE_DX = [("J06.9", "Acute upper respiratory infection (ไข้หวัด)"), ("K29.7", "Gastritis (กระเพาะอาหารอักเสบ)"),
            ("M54.5", "Low back pain (ปวดหลัง)"), ("L30.9", "Dermatitis (ผื่นผิวหนังอักเสบ)"), ("R51", "Headache (ปวดศีรษะ)"),
            ("N39.0", "Urinary tract infection (กระเพาะปัสสาวะอักเสบ)"), ("K02.9", "Dental caries (ฟันผุ)"),
            ("S93.4", "Ankle sprain (ข้อเท้าแพลง)"), ("A09", "Acute gastroenteritis (ท้องเสีย)"),
            ("H10.9", "Conjunctivitis (ตาแดง)")]
ACUTE_CLINIC = ["อายุรกรรม", "อายุรกรรม", "กระดูกและข้อ (Orthopedics)", "ผิวหนัง", "อายุรกรรม", "อายุรกรรม",
                "ทันตกรรม (Dental)", "กระดูกและข้อ (Orthopedics)", "ฉุกเฉิน-ER", "จักษุ"]
CHRONIC_DX = {"known_dx_dm": ("E11.9", "Type 2 diabetes mellitus (เบาหวาน)"),
              "known_dx_ht": ("I10", "Essential hypertension (ความดันโลหิตสูง)"),
              "known_dx_dlp": ("E78.5", "Dyslipidemia (ไขมันในเลือดสูง)"),
              "known_dx_ckd": ("N18.3", "Chronic kidney disease stage 3 (ไตเสื่อม)"),
              "known_dx_cad": ("I25.1", "Coronary artery disease (หัวใจขาดเลือด)"),
              "known_dx_gout": ("M10.9", "Gout (เกาต์)"),
              "known_dx_masld": ("K76.0", "Fatty liver (ไขมันพอกตับ)"),
              "known_dx_hypothyroid": ("E03.9", "Hypothyroidism (ไทรอยด์ต่ำ)")}


def build_dx_strings(frame: pd.DataFrame, extra_code: str | None, extra_text: str | None):
    codes = pd.Series("" if extra_code is None else extra_code, index=frame.index, dtype=object)
    texts = pd.Series("" if extra_text is None else extra_text, index=frame.index, dtype=object)
    for col, (code, text) in CHRONIC_DX.items():
        if col in frame.columns:
            m = frame[col].fillna(0).astype(int) == 1
            codes = codes.where(~m, codes + ";" + code)
            texts = texts.where(~m, texts + "; " + text)
    return codes.str.strip(";"), texts.str.strip("; ")


def gen_opd_visits(chk: pd.DataFrame, pt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """OPD visits between check-ups; vitals around nearest check-up; dx = chronic (prev known) or acute."""
    last = chk.sort_values("visit_date").groupby("patient_id").tail(1)
    first = chk.sort_values("visit_date").groupby("patient_id").head(1).set_index("patient_id")
    n_chronic = last.set_index("patient_id")[[c for c in CHRONIC_DX if c in last.columns]].sum(axis=1)
    lam_year = 0.8 + 1.5 * n_chronic.clip(0, 3) + 0.5 * (last.set_index("patient_id")["age_at_visit"] >= 60)
    start = (first["visit_date"] - pd.Timedelta(days=365)).clip(lower=CHECKUP_START - pd.Timedelta(days=365))
    span_days = (REF_DATE - start).dt.days.clip(lower=30)
    counts = rng.poisson((lam_year * span_days / 365.25).to_numpy())
    pids = np.repeat(lam_year.index.to_numpy(), counts)
    offs = (rng.random(len(pids)) * np.repeat(span_days.to_numpy(), counts)).astype(int)
    opd = pd.DataFrame({"patient_id": pids,
                        "visit_date": np.repeat(start.to_numpy(), counts) + pd.to_timedelta(offs, unit="D")})
    opd = opd.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    # attach nearest previous (or first) check-up state for vitals + known dx
    keep = ["patient_id", "visit_date", "bmi", "sbp", "dbp", "hr", "height_cm", "waist_cm"] + list(CHRONIC_DX)
    ref = chk[[c for c in keep if c in chk.columns]].sort_values("visit_date")
    m = pd.merge_asof(opd.sort_values("visit_date"), ref.rename(columns={"visit_date": "ref_date"}),
                      left_on="visit_date", right_on="ref_date", by="patient_id", direction="nearest")
    m = m.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    n = len(m)
    m["visit_type"] = "OPD"
    is_chronic = (m[list(CHRONIC_DX)].sum(axis=1) > 0) & (rng.random(n) < 0.55)
    ai = rng.integers(0, len(ACUTE_DX), n)
    acute_code = np.array([c for c, _ in ACUTE_DX], dtype=object)[ai]
    acute_text = np.array([t for _, t in ACUTE_DX], dtype=object)[ai]
    m["clinic_name"] = np.where(is_chronic, np.where(m.get("known_dx_cad", 0) == 1, "Cardiology", "อายุรกรรม"),
                                np.array(ACUTE_CLINIC, dtype=object)[ai])
    chronic_codes, chronic_texts = build_dx_strings(m, None, None)
    m["icd10_codes"] = np.where(is_chronic, chronic_codes, acute_code)
    m["diagnosis_text"] = np.where(is_chronic, chronic_texts, acute_text)
    m["sbp"] = (m["sbp"] + rng.normal(3, 10, n)).clip(85, 230).round(0)   # white-coat noise
    m["dbp"] = (m["dbp"] + rng.normal(1, 6, n)).clip(45, 135).round(0)
    m["hr"] = (m["hr"] + rng.normal(2, 8, n)).clip(45, 150).round(0)
    m["bmi"] = (m["bmi"] + rng.normal(0, 0.4, n)).round(1)
    m["weight_kg"] = (m["bmi"] * (m["height_cm"] / 100) ** 2).round(1)
    m["opd_charge_thb"] = np.clip(np.exp(rng.normal(np.log(1200), 0.6, n)), 200, 30000).round(-1)
    return m.drop(columns=["ref_date"])


# ═════════════════════════════════════════════════════════════
# 5) Purchases
# ═════════════════════════════════════════════════════════════
def simulate_purchases(chk: pd.DataFrame, labels: pd.DataFrame, pt: pd.DataFrame, pkg_dim: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Historical: a check-up visit was a paid package (visit_type='Package') with p≈0.45 (else corporate/insurance
    annual check-up). Package choice = labels of previous check-up (what the clinic would have recommended)."""
    price = pkg_dim.set_index("package_code")["price_thb"].astype(float).to_dict()
    c = chk[["patient_id", "visit_date", "checkup_no", "visit_id"]].join(labels[["label_primary_package_code"]])
    c = c.sort_values(["patient_id", "visit_date"])
    c["prev_rec"] = c.groupby("patient_id")["label_primary_package_code"].shift(1)
    sp = c["patient_id"].map(pt.set_index("patient_id")["spending_tier"])
    p_pkg = np.select([sp == "high", sp == "mid"], [0.6, 0.45], 0.3)
    bought = rng.random(len(c)) < p_pkg
    age = chk.loc[c.index, "age_at_visit"].to_numpy()
    fallback = np.where(age < 30, "L1_BASIC", np.where(age < 35, "L1_STANDARD", "L2_AGE35"))
    follow = rng.random(len(c)) < 0.7
    pkg = np.where(c["prev_rec"].notna() & follow, c["prev_rec"].fillna("L1_BASIC"), fallback)
    hist = pd.DataFrame({"patient_id": c["patient_id"], "visit_id": c["visit_id"], "purchase_date": c["visit_date"],
                         "package_code": pkg, "purchase_type": "historical"})[bought]
    disc = rng.choice([0, 0.1, 0.15, 0.2], len(hist), p=[0.5, 0.25, 0.15, 0.1])
    hist["list_price_thb"] = hist["package_code"].map(price)
    hist["discount_pct"] = disc
    hist["net_amount_thb"] = (hist["list_price_thb"] * (1 - disc)).round(0)
    return hist.reset_index(drop=True)


def simulate_future_purchase(snap: pd.DataFrame, rng: np.random.Generator, price: dict) -> pd.DataFrame:
    n = len(snap)
    tier = snap["label_tier"].to_numpy()
    logit = (-2.4 + 0.9 * (tier >= 3) + 0.5 * (tier == 4) + 0.6 * snap["line_oa_engaged"].to_numpy()
             + 0.6 * (snap["prior_package_count"].to_numpy() > 0) + 0.3 * np.log1p(snap["prior_visits_12m"].to_numpy())
             + np.select([snap["spending_tier"] == "high", snap["spending_tier"] == "low"], [0.6, -0.5], 0)
             + 0.3 * (snap["payer_type"] == "ประกันสุขภาพเอกชน").to_numpy()
             + 0.35 * (snap["n_abnormal_labs"].to_numpy() >= 4)
             + 0.5 * (snap[[c for c in SYM_COLS if c in snap.columns]].sum(axis=1).to_numpy() > 0)
             - 0.4 * (snap["age"].to_numpy() < 30)
             + 0.3 * snap["index_date"].dt.month.isin([8, 9]).to_numpy()      # 9.9 campaign season
             + rng.normal(0, 0.5, n))
    p = 1 / (1 + np.exp(-logit))
    bought = rng.random(n) < p
    # which package: primary 65%, another eligible 20%, cheaper tier-1 15%
    elig_cols = [f"elig_{c}" for c in PACKAGE_CODES]
    elig = snap[elig_cols].to_numpy().astype(bool)
    primary = snap["label_primary_package_code"].to_numpy()
    u = rng.random(n)
    chosen = primary.copy().astype(object)
    for i in np.where(bought & (u >= 0.65))[0]:
        if u[i] < 0.85:
            others = [PACKAGE_CODES[j] for j in np.where(elig[i])[0] if PACKAGE_CODES[j] != primary[i]]
            if others:
                chosen[i] = others[rng.integers(0, len(others))]
        else:
            chosen[i] = rng.choice(["L1_BASIC", "L1_STANDARD", "L1_PREMIUM"], p=[0.45, 0.4, 0.15])
    delay = rng.integers(1, PURCHASE_WINDOW_DAYS + 1, n)
    disc = rng.choice([0, 0.1, 0.15, 0.2], n, p=[0.5, 0.25, 0.15, 0.1])
    lp = pd.Series(chosen).map(price).to_numpy(dtype=float)
    return pd.DataFrame({
        "label_purchased_90d": bought.astype("int8"),
        "label_purchased_package_code": np.where(bought, chosen, None),
        "label_purchase_amount_thb": np.where(bought, (lp * (1 - disc)).round(0), 0.0),
        "_purchase_date": np.where(bought, snap["index_date"].to_numpy() + pd.to_timedelta(delay, unit="D").to_numpy(),
                                   np.datetime64("NaT", "ns")),
        "_discount": disc,
        "_list_price": lp,
    }, index=snap.index)


# ═════════════════════════════════════════════════════════════
# 6) Snapshot (point-in-time features)
# ═════════════════════════════════════════════════════════════
def build_snapshot(chk: pd.DataFrame, opd: pd.DataFrame, hist: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    idx = chk[chk["is_index_visit"] == 1].copy()
    idx = idx.rename(columns={"visit_date": "index_date", "visit_id": "index_visit_id", "age_at_visit": "age"})
    prev = chk[chk["is_index_visit"] == 0].sort_values("visit_date").groupby("patient_id").tail(1)
    prev = prev.set_index("patient_id")
    snap = idx.merge(pt[[c for c in pt.columns if not c.startswith("_")]], on="patient_id", how="left")
    snap = snap.set_index("patient_id", drop=False)

    # trends vs previous check-up
    for col, name in [("hba1c", "delta_hba1c"), ("fbs", "delta_fbs"), ("ldl", "delta_ldl"), ("egfr", "delta_egfr"),
                      ("sbp", "delta_sbp"), ("bmi", "delta_bmi"), ("tg", "delta_tg"), ("alt", "delta_alt")]:
        snap[name] = (snap[col] - prev[col].reindex(snap.index)).round(2)
    snap["months_since_last_checkup"] = ((snap["index_date"] - prev["visit_date"].reindex(snap.index)).dt.days / 30.44).round(1)
    snap["prior_checkups"] = snap["checkup_no"] - 1

    # utilization strictly BEFORE index date (no leakage)
    o = opd.merge(snap[["patient_id", "index_date"]].reset_index(drop=True), on="patient_id")
    o = o[o["visit_date"] < o["index_date"]]
    in12 = o[o["visit_date"] >= o["index_date"] - pd.Timedelta(days=365)]
    in24 = o[o["visit_date"] >= o["index_date"] - pd.Timedelta(days=730)]
    snap["prior_opd_visits_12m"] = in12.groupby("patient_id").size().reindex(snap.index).fillna(0).astype(int)
    snap["prior_opd_visits_total"] = o.groupby("patient_id").size().reindex(snap.index).fillna(0).astype(int)
    c12 = chk.merge(snap[["patient_id", "index_date"]].reset_index(drop=True), on="patient_id")
    c12 = c12[(c12["visit_date"] < c12["index_date"]) & (c12["visit_date"] >= c12["index_date"] - pd.Timedelta(days=365))]
    snap["prior_visits_12m"] = snap["prior_opd_visits_12m"] + c12.groupby("patient_id").size().reindex(snap.index).fillna(0).astype(int)
    snap["opd_spend_24m"] = in24.groupby("patient_id")["opd_charge_thb"].sum().reindex(snap.index).fillna(0)
    h = hist.merge(snap[["patient_id", "index_date"]].reset_index(drop=True), on="patient_id")
    h = h[h["purchase_date"] < h["index_date"]]
    snap["prior_package_count"] = h.groupby("patient_id").size().reindex(snap.index).fillna(0).astype(int)
    lvl = h["package_code"].str.slice(1, 2).astype(int)
    snap["prior_package_max_level"] = lvl.groupby(h["patient_id"]).max().reindex(snap.index).fillna(0).astype(int)
    snap["prior_package_spend_24m"] = (h[h["purchase_date"] >= h["index_date"] - pd.Timedelta(days=730)]
                                       .groupby("patient_id")["net_amount_thb"].sum().reindex(snap.index).fillna(0))
    snap["last_package_code"] = h.sort_values("purchase_date").groupby("patient_id")["package_code"].last().reindex(snap.index)
    snap["total_spend_24m"] = snap["opd_spend_24m"] + snap["prior_package_spend_24m"]

    # derived clinical scores
    male = snap["gender"].eq("ชาย").to_numpy()
    snap["gender_code"] = male.astype("int8")
    snap["waist_height_ratio"] = (snap["waist_cm"] / snap["height_cm"]).round(3)
    snap["tg_hdl_ratio"] = (snap["tg"] / snap["hdl"]).round(2)
    snap["thai_cv_risk_10y"] = thai_cv_risk(snap["age"], male, snap["sbp"],
                                            (snap["known_dx_dm"] == 1) | (snap["hba1c"] >= 6.5) | (snap["fbs"] >= 126),
                                            snap["tc"], snap["smoking_status"].eq("current")).round(4)
    snap["fib4"] = np.round(fib4(snap["age"], snap["ast"], snap["alt"], snap["plt"]), 2)
    ref = {code: (lo, hi) for code, _, _, _, lo, hi, _ in LAB_TESTS}
    abn = np.zeros(len(snap), dtype=int)
    for col, code in LAB_COLS.items():
        lo, hi = ref[code]
        v = snap[col]
        abn += ((v < lo) | (v > hi)).fillna(False).to_numpy().astype(int)
    snap["n_abnormal_labs"] = abn
    snap["n_labs_measured"] = snap[list(LAB_COLS)].notna().sum(axis=1)
    return snap.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════
# 7) Column ordering (recommended layout)
# ═════════════════════════════════════════════════════════════
SNAPSHOT_GROUPS = {
    "keys": ["patient_id", "hn", "index_visit_id", "index_date"],
    "demographic": ["age", "gender", "gender_code", "region", "payer_type", "spending_tier", "occupation"],
    "lifestyle": ["smoking_status", "pack_years", "alcohol_level", "exercise_min_week", "sleep_hours",
                  "stress_score_st5", "diet_score"],
    "family_history": ["fhx_dm", "fhx_ht", "fhx_cad_premature", "fhx_cancer", "fhx_crc"],
    "vitals": ["height_cm", "weight_kg", "bmi", "waist_cm", "waist_height_ratio", "sbp", "dbp", "hr", "spo2", "temp_c"],
    "lab_glycemic": ["fbs", "hba1c", "insulin", "homa_ir"],
    "lab_lipid": ["tc", "ldl", "hdl", "tg", "non_hdl", "tg_hdl_ratio", "apob", "lpa", "hscrp"],
    "lab_kidney": ["bun", "creatinine", "egfr", "uacr", "uric_acid", "ua_protein", "ua_glucose"],
    "lab_liver": ["ast", "alt", "alp", "ggt", "albumin", "tbil", "fib4", "hbsag", "anti_hcv"],
    "lab_cbc": ["hb", "hct", "wbc", "plt", "mcv", "ferritin"],
    "lab_endocrine": ["tsh", "ft4", "vitd"],
    "lab_tumor_marker": ["cea", "afp", "psa", "ca125", "ca199", "fit_result"],
    "imaging_procedure": ["ekg_result", "cxr_result", "us_fatty_liver_grade", "fibroscan_cap", "fibroscan_kpa",
                          "cac_score", "abi", "bapwv", "echo_ef", "bmd_tscore"],
    "symptoms": SYM_COLS,
    "known_diagnosis": ["known_dx_dm", "known_dx_ht", "known_dx_dlp", "known_dx_ckd", "known_dx_cad", "known_dx_gout",
                        "known_dx_masld", "known_dx_hypothyroid", "known_dx_cancer_hx"],
    "medication": ["med_metformin", "med_insulin", "med_antihtn", "med_statin", "n_meds"],
    "trend": ["delta_hba1c", "delta_fbs", "delta_ldl", "delta_tg", "delta_egfr", "delta_sbp", "delta_bmi", "delta_alt",
              "months_since_last_checkup"],
    "utilization": ["prior_checkups", "prior_visits_12m", "prior_opd_visits_12m", "prior_opd_visits_total",
                    "prior_package_count", "prior_package_max_level", "last_package_code", "opd_spend_24m",
                    "prior_package_spend_24m", "total_spend_24m", "line_oa_engaged"],
    "derived_score": ["thai_cv_risk_10y", "mets_count", "n_abnormal_labs", "n_labs_measured"],
    "label_match": ["label_tier", "label_primary_package_code", "label_primary_short_name", "label_rule_id",
                    "label_source", "label_reason", "n_eligible_packages"]
                   + [f"elig_{c}" for c in PACKAGE_CODES],
    "label_addon": ["label_addons"],   # + addon_* columns appended at runtime
    "label_propensity": ["label_purchased_90d", "label_purchased_package_code", "label_purchase_amount_thb"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-patients", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--complete-panel", action="store_true", help="measure every test at every check-up (no missingness)")
    ap.add_argument("--override-rate", type=float, default=0.04, help="share of labels overridden by clinician (label noise)")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] patients: {args.n_patients:,}")
    pt = gen_patients(args.n_patients, rng)
    print("[2/7] check-up schedule + physiology")
    sched = gen_checkup_schedule(pt, rng)
    chk = simulate_checkups(sched, pt, rng, args.complete_panel)
    chk = chk.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    chk["visit_id"] = ["V" + f"{i:08d}" for i in range(1, len(chk) + 1)]
    chk["visit_type"] = "Checkup"
    chk["clinic_name"] = "ศูนย์ตรวจสุขภาพ (Health Checkup)"

    print("[3/7] clinical labels per check-up (package_matching_v2)")
    lab_in = chk.merge(pt[["patient_id", "gender", "smoking_status", "pack_years", "alcohol_level", "spending_tier",
                           "fhx_cad_premature", "fhx_cancer", "fhx_crc"]], on="patient_id", how="left")
    lab_in = lab_in.rename(columns={"age_at_visit": "age"})
    male = lab_in["gender"].eq("ชาย")
    lab_in["thai_cv_risk_10y"] = thai_cv_risk(lab_in["age"], male, lab_in["sbp"],
                                              (lab_in["known_dx_dm"] == 1) | (lab_in["hba1c"] >= 6.5) | (lab_in["fbs"] >= 126),
                                              lab_in["tc"], lab_in["smoking_status"].eq("current"))
    lab_in["fib4"] = fib4(lab_in["age"], lab_in["ast"], lab_in["alt"], lab_in["plt"])
    visit_labels = assign_package_labels(lab_in)
    visit_labels.index = chk.index

    print("[4/7] OPD visits + purchases")
    pkg_dim = build_package_dim()
    price = pkg_dim.set_index("package_code")["price_thb"].astype(float).to_dict()
    hist = simulate_purchases(chk, visit_labels, pt, pkg_dim, rng)
    chk.loc[chk["visit_id"].isin(hist["visit_id"]), "visit_type"] = "Package"
    opd = gen_opd_visits(chk, pt, rng)
    opd["visit_id"] = ["O" + f"{i:08d}" for i in range(1, len(opd) + 1)]
    birth = opd["patient_id"].map(pt.set_index("patient_id")["birth_date"])
    opd["age_at_visit"] = ((opd["visit_date"] - birth).dt.days / 365.25).round(1)

    print("[5/7] snapshot (point-in-time features) + labels")
    snap = build_snapshot(chk, opd, hist, pt)
    snap_lab = assign_package_labels(snap)
    snap = pd.concat([snap.drop(columns=[c for c in snap_lab.columns if c in snap.columns]), snap_lab], axis=1)
    # clinician override noise (realistic label disagreement), only where >=2 eligible
    snap["label_source"] = "rule_v2"
    elig_cols = [f"elig_{c}" for c in PACKAGE_CODES]
    cand = np.where((rng.random(len(snap)) < args.override_rate) & (snap["n_eligible_packages"] >= 2))[0]
    for i in cand:
        options = [PACKAGE_CODES[j] for j, e in enumerate(snap.loc[i, elig_cols].to_numpy()) if e and PACKAGE_CODES[j] != snap.at[i, "label_primary_package_code"]]
        if options:
            new = options[rng.integers(0, len(options))]
            snap.at[i, "label_primary_package_code"] = new
            snap.at[i, "label_tier"] = int(new[1])
            snap.at[i, "label_source"] = "clinician_override"
    from logic.package_matching_v2 import CODE_BY_SHORT_NAME
    short_by_code = {v: k for k, v in CODE_BY_SHORT_NAME.items()}
    snap["label_primary_short_name"] = snap["label_primary_package_code"].map(short_by_code)
    fut = simulate_future_purchase(snap, rng, price)
    snap = pd.concat([snap, fut[["label_purchased_90d", "label_purchased_package_code", "label_purchase_amount_thb"]]], axis=1)
    fut_rows = pd.DataFrame({"patient_id": snap["patient_id"], "visit_id": None, "purchase_date": fut["_purchase_date"],
                             "package_code": snap["label_purchased_package_code"], "purchase_type": "post_index_90d",
                             "list_price_thb": fut["_list_price"], "discount_pct": fut["_discount"],
                             "net_amount_thb": snap["label_purchase_amount_thb"]})[snap["label_purchased_90d"] == 1]
    purchases = pd.concat([hist, fut_rows], ignore_index=True).sort_values(["patient_id", "purchase_date"]).reset_index(drop=True)
    purchases.insert(0, "purchase_id", ["P" + f"{i:08d}" for i in range(1, len(purchases) + 1)])

    print("[6/7] encounters + long lab table")
    chk_codes, chk_texts = build_dx_strings(chk, "Z00.0", "General health examination (ตรวจสุขภาพประจำปี)")
    chk["icd10_codes"], chk["diagnosis_text"] = chk_codes, chk_texts
    enc_cols = (["visit_id", "patient_id", "visit_date", "visit_type", "clinic_name", "checkup_no", "is_index_visit",
                 "age_at_visit", "height_cm", "weight_kg", "bmi", "waist_cm", "sbp", "dbp", "hr", "spo2", "temp_c",
                 "icd10_codes", "diagnosis_text"] + PROC_COLS + SYM_COLS
                + ["known_dx_dm", "known_dx_ht", "known_dx_dlp", "known_dx_ckd", "known_dx_cad", "known_dx_gout",
                   "known_dx_masld", "known_dx_hypothyroid", "known_dx_cancer_hx",
                   "med_metformin", "med_insulin", "med_antihtn", "med_statin", "n_meds"])
    enc_chk = chk[enc_cols].copy()
    enc_chk["opd_charge_thb"] = np.nan
    enc_opd = opd.reindex(columns=enc_cols + ["opd_charge_thb"])
    enc = pd.concat([enc_chk, enc_opd], ignore_index=True)
    enc = enc.sort_values(["patient_id", "visit_date", "visit_id"]).reset_index(drop=True)
    for c in ["visit_type", "clinic_name"]:
        enc[c] = enc[c].astype("category")

    lab_dim = pd.DataFrame(LAB_TESTS, columns=["test_code", "loinc", "test_name", "unit", "ref_low", "ref_high", "panel"])
    long = chk[["visit_id", "patient_id", "visit_date"] + list(LAB_COLS)].melt(
        id_vars=["visit_id", "patient_id", "visit_date"], var_name="col", value_name="value").dropna(subset=["value"])
    long["test_code"] = long["col"].map(LAB_COLS)
    long = long.merge(lab_dim[["test_code", "unit", "ref_low", "ref_high", "panel"]], on="test_code", how="left")
    long["flag"] = np.select([long["value"] > long["ref_high"], long["value"] < long["ref_low"]], ["H", "L"], "N")
    long = long.rename(columns={"visit_date": "collected_date"}).drop(columns="col")
    panel_order = {p: i for i, p in enumerate(dict.fromkeys(lab_dim["panel"]))}
    long["_p"] = long["panel"].map(panel_order)
    long = long.sort_values(["patient_id", "collected_date", "_p", "test_code"]).drop(columns="_p").reset_index(drop=True)
    for c in ["test_code", "unit", "panel", "flag"]:
        long[c] = long[c].astype("category")
    long = long[["patient_id", "visit_id", "collected_date", "panel", "test_code", "value", "unit", "ref_low", "ref_high", "flag"]]

    print("[7/7] write files")
    addon_cols = [c for c in snap.columns if c.startswith("addon_")]
    groups = dict(SNAPSHOT_GROUPS)
    groups["label_addon"] = groups["label_addon"] + addon_cols
    ordered = [c for g in groups.values() for c in g if c in snap.columns]
    snap = snap[ordered].sort_values("patient_id").reset_index(drop=True)
    for c in ["gender", "region", "payer_type", "spending_tier", "occupation", "smoking_status", "alcohol_level",
              "label_primary_package_code", "label_tier", "label_rule_id"]:
        snap[c] = snap[c].astype("category")

    dim_patient = pt[[c for c in pt.columns if not c.startswith("_")]].copy()
    dim_patient.to_parquet(out / "dim_patient.parquet", index=False)
    pkg_dim.to_csv(out / "dim_package.csv", index=False, encoding="utf-8-sig")
    lab_dim.to_csv(out / "dim_lab_test.csv", index=False, encoding="utf-8-sig")
    enc.to_parquet(out / "fact_encounter.parquet", index=False)
    long.to_parquet(out / "fact_lab_result.parquet", index=False)
    purchases.to_parquet(out / "fact_package_purchase.parquet", index=False)
    snap.to_parquet(out / "train_patient_snapshot.parquet", index=False)
    snap.head(300).to_csv(out / "train_patient_snapshot_sample.csv", index=False, encoding="utf-8-sig")
    leakage = (groups["label_match"] + groups["label_addon"] + groups["label_propensity"] + groups["keys"]
               + ["hn"])
    meta = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seed": args.seed, "n_patients": args.n_patients, "complete_panel": args.complete_panel,
        "groups": groups,
        "feature_columns": [c for g, cols in groups.items() if not g.startswith("label") and g != "keys"
                            for c in cols if c in snap.columns],
        "exclude_from_features": sorted(set(leakage)),
        "targets": {
            "multiclass_primary": "label_primary_package_code",
            "multilabel_eligibility": [f"elig_{c}" for c in PACKAGE_CODES],
            "multilabel_addons": addon_cols,
            "binary_propensity": "label_purchased_90d",
            "ordinal_tier": "label_tier",
        },
        "row_counts": {"dim_patient": len(dim_patient), "fact_encounter": len(enc), "fact_lab_result": len(long),
                       "fact_package_purchase": len(purchases), "train_patient_snapshot": len(snap)},
    }
    (out / "feature_groups.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["row_counts"], indent=2))
    print(f"Done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

