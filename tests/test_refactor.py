# -*- coding: utf-8 -*-
"""
Unit Tests for Patient Segmentation and Package Service Refactoring
"""

import unittest
import pandas as pd
from logic.patient_segmentation import segment_patient, get_age_band
from logic.package_service import compute_package_stats
from logic.config import PACKAGE_CATALOG, CATEGORY_ORDER


class TestPatientSegmentation(unittest.TestCase):

    def test_age_bands(self):
        self.assertEqual(get_age_band(25), "<30")
        self.assertEqual(get_age_band(30), "30-34")
        self.assertEqual(get_age_band(34.9), "30-34")
        self.assertEqual(get_age_band(35), "35-49")
        self.assertEqual(get_age_band(49), "35-49")
        self.assertEqual(get_age_band(55), "50-59")
        self.assertEqual(get_age_band(65), "60+")

    def test_tier_1_healthy(self):
        # <30 years old -> BASIC
        res = segment_patient(age=24, bmi=20.5, systolic=115, diastolic=75)
        self.assertEqual(res["risk_tier"], 1)
        self.assertEqual(res["recommended_package"], "BASIC")

        # 32 years old -> STANDARD
        res2 = segment_patient(age=32, bmi=21.0, systolic=118, diastolic=78)
        self.assertEqual(res2["risk_tier"], 1)
        self.assertEqual(res2["recommended_package"], "STANDARD")

    def test_tier_2_risk_age_or_weight(self):
        # 38 years old normal weight -> Tier 2, วัย 35+
        res = segment_patient(age=38, bmi=22.0, systolic=118, diastolic=78)
        self.assertEqual(res["risk_tier"], 2)
        self.assertEqual(res["recommended_package"], "วัย 35+")

        # Overweight BMI 26.0 -> Tier 2, คนอ้วน
        res2 = segment_patient(age=28, bmi=26.0, systolic=118, diastolic=78)
        self.assertEqual(res2["risk_tier"], 2)
        self.assertEqual(res2["recommended_package"], "คนอ้วน")

    def test_tier_3_early_disease(self):
        # Stage 1 HT (systolic 145) -> Tier 3
        res = segment_patient(age=40, bmi=22.0, systolic=145, diastolic=85)
        self.assertEqual(res["risk_tier"], 3)
        self.assertEqual(res["recommended_package"], "ความดันโลหิตสูง")

        # History of diabetes in text -> Tier 3, เบาหวาน
        res2 = segment_patient(age=28, bmi=22.0, systolic=115, diastolic=75, diagnosis_text="ตรวจติดตาม โรคเบาหวาน ชนิดที่ 2")
        self.assertEqual(res2["risk_tier"], 3)
        self.assertEqual(res2["recommended_package"], "เบาหวาน")

    def test_tier_4_critical_or_senior(self):
        # Crisis BP (systolic 168) -> Tier 4
        res = segment_patient(age=45, bmi=23.0, systolic=168, diastolic=102)
        self.assertEqual(res["risk_tier"], 4)
        self.assertEqual(res["recommended_package"], "หัวใจ")

        # Senior age 68 -> Tier 4
        res2 = segment_patient(age=68, bmi=22.0, systolic=120, diastolic=80)
        self.assertEqual(res2["risk_tier"], 4)
        self.assertIn(res2["recommended_package"], ["คัดกรองมะเร็ง", "หัวใจ", "ตรวจสุขภาพผู้สูงวัยเชิงลึก"])

    def test_segment_population_dataframe(self):
        from logic.patient_segmentation import segment_population_dataframe
        df = pd.DataFrame([
            {"patient_id": "P1", "age": 24, "bmi": 21.0, "systolic": 115, "diastolic": 75, "diagnosis_text": ""},
            {"patient_id": "P2", "age": 38, "bmi": 22.0, "systolic": 118, "diastolic": 78, "diagnosis_text": ""},
            {"patient_id": "P3", "age": 42, "bmi": 23.0, "systolic": 145, "diastolic": 85, "diagnosis_text": "HT"},
            {"patient_id": "P4", "age": 65, "bmi": 24.0, "systolic": 120, "diastolic": 80, "diagnosis_text": ""},
        ])
        res = segment_population_dataframe(df)
        self.assertListEqual(list(res["risk_tier"]), [1, 2, 3, 4])
        self.assertListEqual(list(res["age_band"]), ["<30", "35-49", "35-49", "60+"])



class TestPackageService(unittest.TestCase):

    def test_compute_package_stats_sorting_and_mismatch(self):
        # Create mock sales data where STANDARD has more VN than BASIC
        mock_sales = pd.DataFrame([
            {"package_name": "แพคเกจตรวจสุขภาพ BASIC", "transactions": 50, "net_amount": 49500},
            {"package_name": "แพคเกจตรวจสุขภาพ STANDARD", "transactions": 200, "net_amount": 458000},
            # PREMIUM not in sales -> should have 0 VN and be tracked in mismatches
        ])

        pkgs, mismatches = compute_package_stats(mock_sales)

        cat1_pkgs = pkgs["1_Basic Health Check"]["packages"]
        # After sorting, STANDARD (200 VN) must come before BASIC (50 VN)
        self.assertEqual(cat1_pkgs[0]["short_name"], "STANDARD")
        self.assertEqual(cat1_pkgs[0]["total_vn"], 200)
        self.assertEqual(cat1_pkgs[1]["short_name"], "BASIC")
        self.assertEqual(cat1_pkgs[1]["total_vn"], 50)
        self.assertEqual(cat1_pkgs[2]["short_name"], "PREMIUM")
        self.assertEqual(cat1_pkgs[2]["total_vn"], 0)

        # Check category summary
        self.assertEqual(pkgs["1_Basic Health Check"]["total_vn"], 250)
        self.assertEqual(pkgs["1_Basic Health Check"]["total_revenue"], 507500.0)

        # PREMIUM should be in mismatches
        mismatch_shorts = [m.get("short_name") for m in mismatches]
        self.assertIn("PREMIUM", mismatch_shorts)

    def test_empty_sales_graceful_fallback(self):
        pkgs, mismatches = compute_package_stats(None)
        self.assertIn("1_Basic Health Check", pkgs)
        for cat in pkgs.values():
            for p in cat["packages"]:
                self.assertEqual(p["total_vn"], 0)


if __name__ == "__main__":
    unittest.main()
