# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for Package Matching v2, Synthetic Clinical Data,
and Multi-Task Model Bundle
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from logic.package_matching_v2 import (
    PACKAGE_CODES,
    ADDON_CODES,
    assign_package_labels,
    build_package_dim,
    build_personalized_package,
    ckd_epi_2021,
    predict_patient_packages_v2,
)
from logic.patient_segmentation import segment_patient

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "data" / "synthetic"


class TestPackageMatchingV2(unittest.TestCase):
    def test_package_dim_has_all_13_packages(self):
        dim = build_package_dim()
        self.assertEqual(len(dim), 13)
        self.assertListEqual(list(dim["package_code"]), PACKAGE_CODES)

    def test_ckd_epi_2021_monotonic_with_creatinine(self):
        egfr_low_cr = ckd_epi_2021([0.8], [45], [True])[0]
        egfr_high_cr = ckd_epi_2021([2.2], [45], [True])[0]
        self.assertGreater(egfr_low_cr, 90.0)
        self.assertLess(egfr_high_cr, 45.0)

    def test_all_13_packages_reachable_by_clinical_profiles(self):
        profiles = pd.DataFrame([
            # L1_BASIC: young <30 healthy
            {"age": 24, "gender": "ชาย", "bmi": 21.0, "waist_cm": 76, "sbp": 112, "dbp": 72, "fbs": 86, "hba1c": 5.1, "ldl": 95, "hdl": 55, "tg": 80},
            # L1_STANDARD: 32 healthy mid spender
            {"age": 32, "gender": "หญิง", "bmi": 20.5, "waist_cm": 70, "sbp": 110, "dbp": 70, "fbs": 88, "hba1c": 5.2, "ldl": 100, "hdl": 60, "tg": 75, "spending_tier": "mid"},
            # L1_PREMIUM: 33 healthy high spender
            {"age": 33, "gender": "ชาย", "bmi": 21.5, "waist_cm": 78, "sbp": 114, "dbp": 74, "fbs": 89, "hba1c": 5.2, "ldl": 105, "hdl": 52, "tg": 85, "spending_tier": "high"},
            # L2_AGE35: 40 healthy mid spender
            {"age": 40, "gender": "หญิง", "bmi": 21.0, "waist_cm": 72, "sbp": 112, "dbp": 72, "fbs": 88, "hba1c": 5.3, "ldl": 105, "hdl": 58, "tg": 80, "spending_tier": "mid"},
            # L2_OBESITY: BMI 28 without DM/HT/DLP
            {"age": 31, "gender": "ชาย", "bmi": 28.5, "waist_cm": 96, "sbp": 118, "dbp": 76, "fbs": 92, "hba1c": 5.4, "ldl": 115, "hdl": 48, "tg": 110},
            # L2_METABOLIC: prediabetes (HbA1c 6.1, HOMA-IR 3.2) normal BMI
            {"age": 34, "gender": "หญิง", "bmi": 22.0, "waist_cm": 75, "sbp": 122, "dbp": 78, "fbs": 112, "hba1c": 6.1, "homa_ir": 3.2, "ldl": 120, "hdl": 52, "tg": 140},
            # L3_DM: overt diabetes HbA1c 8.2, FBS 168
            {"age": 48, "gender": "ชาย", "bmi": 24.5, "waist_cm": 86, "sbp": 128, "dbp": 80, "fbs": 168, "hba1c": 8.2, "ldl": 110, "hdl": 45, "tg": 140},
            # L3_HT: stage 1 HT SBP 152/96 without DM/DLP
            {"age": 46, "gender": "ชาย", "bmi": 23.0, "waist_cm": 82, "sbp": 152, "dbp": 96, "fbs": 90, "hba1c": 5.3, "ldl": 110, "hdl": 50, "tg": 100},
            # L3_LIPID: severe dyslipidemia LDL 195
            {"age": 44, "gender": "หญิง", "bmi": 22.5, "waist_cm": 74, "sbp": 118, "dbp": 76, "fbs": 90, "hba1c": 5.3, "ldl": 195, "hdl": 55, "tg": 160},
            # L4_HEART: chest pain + ST-T change on EKG
            {"age": 55, "gender": "ชาย", "bmi": 25.0, "sbp": 148, "dbp": 88, "sym_chest_pain": 1, "ekg_result": "ST-T change"},
            # L4_COLON: FIT positive
            {"age": 52, "gender": "หญิง", "bmi": 23.0, "sbp": 120, "dbp": 76, "fit_result": "positive"},
            # L4_CANCER: elevated PSA 9.5 in male
            {"age": 56, "gender": "ชาย", "bmi": 23.0, "sbp": 122, "dbp": 78, "psa": 9.5},
            # L4_SENIOR: 68 yo with falls & osteoporosis
            {"age": 68, "gender": "หญิง", "bmi": 21.0, "sbp": 124, "dbp": 76, "sym_falls": 1, "bmd_tscore": -2.8},
        ])
        out = assign_package_labels(profiles)
        expected = [
            "L1_BASIC", "L1_STANDARD", "L1_PREMIUM", "L2_AGE35", "L2_OBESITY", "L2_METABOLIC",
            "L3_DM", "L3_HT", "L3_LIPID", "L4_HEART", "L4_COLON", "L4_CANCER", "L4_SENIOR",
        ]
        self.assertListEqual(list(out["label_primary_package_code"]), expected)
        # Verify primary package is always marked eligible
        for idx_i, code in enumerate(expected):
            self.assertEqual(int(out.loc[idx_i, f"elig_{code}"]), 1)

    def test_predict_patient_packages_v2_and_segment_patient_bridge(self):
        pt = {
            "age": 54, "gender": "ชาย", "bmi": 26.8, "waist_cm": 94,
            "sbp": 146, "dbp": 92, "fbs": 138, "hba1c": 7.1, "ldl": 162, "hdl": 38, "tg": 210,
            "egfr": 72, "uacr": 45, "alt": 48, "fib4": 1.45, "us_fatty_liver_grade": 2,
        }
        res = predict_patient_packages_v2(pt)
        self.assertIn(res["primary_package_code"], PACKAGE_CODES)
        self.assertEqual(len(res["top3_packages"]), 3)
        self.assertGreaterEqual(len(res["eligible_packages"]), 3)
        addon_codes = [a["addon_code"] for a in res["recommended_addons"]]
        self.assertIn("ADDON_PSA", addon_codes)
        self.assertIn("ADDON_FUNDUS", addon_codes)
        self.assertIn("ADDON_FIBROSCAN", addon_codes)

        # Also test via segment_patient bridge
        seg = segment_patient(age=54, bmi=26.8, systolic=146, diastolic=92, gender="ชาย", lab_data=pt)
        self.assertEqual(seg["recommended_package"], res["primary_short_name"])
        self.assertEqual(len(seg["top3_packages"]), 3)
        self.assertIn("personalized_package", seg)

    def test_personalized_bespoke_package_for_uncovered_profiles(self):
        # Patient with Renal decline (eGFR 48, UACR 65), Thyroid dysfunction (TSH 6.2), and Anemia (Hb 10.8)
        pt_unmatched = {
            "age": 47, "gender": "หญิง", "bmi": 22.4, "sbp": 124, "dbp": 78,
            "egfr": 48.0, "uacr": 65.0, "tsh": 6.2, "hb": 10.8, "ferritin": 12.0,
        }
        res = predict_patient_packages_v2(pt_unmatched)
        self.assertTrue(res["needs_custom_package"])
        pers = res["personalized_package"]
        self.assertEqual(pers["package_code"], "CUSTOM_BESPOKE")
        self.assertTrue(pers["needs_custom_package"])
        self.assertGreaterEqual(len(pers["uncovered_gaps"]), 3)
        self.assertGreater(pers["price"], 0)
        self.assertLess(pers["price"], pers["full_price"])
        mod_codes = [m["module_code"] for m in pers["selected_modules"]]
        self.assertIn("MOD_RENAL", mod_codes)
        self.assertIn("MOD_THYROID", mod_codes)
        self.assertIn("MOD_ANEMIA", mod_codes)
        self.assertGreaterEqual(len(pers["all_tests"]), 8)

    def test_synthetic_dataset_integrity_and_sorting(self):
        snap = pd.read_parquet(SYNTH_DIR / "train_patient_snapshot.parquet")
        enc = pd.read_parquet(SYNTH_DIR / "fact_encounter.parquet", columns=["patient_id", "visit_date", "visit_id"])
        meta = json.loads((SYNTH_DIR / "feature_groups.json").read_text(encoding="utf-8"))

        self.assertEqual(len(snap), 10000)
        # All 13 packages present in primary labels
        self.assertEqual(snap["label_primary_package_code"].nunique(), 13)
        # Zero leakage between feature_columns and exclude_from_features
        self.assertEqual(len(set(meta["feature_columns"]) & set(meta["exclude_from_features"])), 0)
        # Encounter table sorted by (patient_id, visit_date, visit_id)
        is_sorted = enc.equals(enc.sort_values(["patient_id", "visit_date", "visit_id"]).reset_index(drop=True))
        self.assertTrue(is_sorted)


if __name__ == "__main__":
    unittest.main()

