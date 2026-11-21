# -*- coding: utf-8 -*-
"""
Unit tests for View 4: Clinical Tracking & KDIGO Renal Analytics.
"""
import unittest
import pandas as pd
from views.clinical_tracking import categorize_gfr, categorize_uacr, get_kdigo_risk_color


class TestClinicalTracking(unittest.TestCase):
    def test_categorize_gfr_boundaries(self):
        self.assertEqual(categorize_gfr(95.0), "G1 (≥90)")
        self.assertEqual(categorize_gfr(90.0), "G1 (≥90)")
        self.assertEqual(categorize_gfr(89.9), "G2 (60-89)")
        self.assertEqual(categorize_gfr(60.0), "G2 (60-89)")
        self.assertEqual(categorize_gfr(59.9), "G3a (45-59)")
        self.assertEqual(categorize_gfr(45.0), "G3a (45-59)")
        self.assertEqual(categorize_gfr(44.9), "G3b (30-44)")
        self.assertEqual(categorize_gfr(30.0), "G3b (30-44)")
        self.assertEqual(categorize_gfr(29.9), "G4 (15-29)")
        self.assertEqual(categorize_gfr(15.0), "G4 (15-29)")
        self.assertEqual(categorize_gfr(14.9), "G5 (<15)")
        self.assertEqual(categorize_gfr(float("nan")), "Unknown")

    def test_categorize_uacr_boundaries(self):
        self.assertEqual(categorize_uacr(15.0), "A1 (<30)")
        self.assertEqual(categorize_uacr(29.9), "A1 (<30)")
        self.assertEqual(categorize_uacr(30.0), "A2 (30-300)")
        self.assertEqual(categorize_uacr(300.0), "A2 (30-300)")
        self.assertEqual(categorize_uacr(300.1), "A3 (>300)")
        self.assertEqual(categorize_uacr(float("nan")), "A1 (<30)")

    def test_get_kdigo_risk_color_matrix(self):
        # G1 + A1 should be Low Risk (Green)
        level, bg, text = get_kdigo_risk_color("G1 (≥90)", "A1 (<30)")
        self.assertIn("Low", level)
        self.assertEqual(bg, "#DCFCE7")

        # G5 should always be Very High Risk (Red)
        level, bg, text = get_kdigo_risk_color("G5 (<15)", "A1 (<30)")
        self.assertIn("Very High", level)
        self.assertEqual(bg, "#FEE2E2")

        # G4 should always be Very High Risk (Red)
        level, bg, text = get_kdigo_risk_color("G4 (15-29)", "A1 (<30)")
        self.assertIn("Very High", level)

        # G3a + A2 should be High Risk (Orange)
        level, bg, text = get_kdigo_risk_color("G3a (45-59)", "A2 (30-300)")
        self.assertIn("High", level)


if __name__ == "__main__":
    unittest.main()
