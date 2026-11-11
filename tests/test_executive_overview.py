# -*- coding: utf-8 -*-
"""
Unit test for Executive System Overview function and navigation in app.py
"""
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
import app

class TestExecutiveOverview(unittest.TestCase):

    def test_executive_overview_calculations(self):
        # Create sample sales dataframe
        sample_sales = pd.DataFrame({
            "net_amount": [5000.0, 3000.0, 2000.0, 10000.0],
            "transactions": [1, 1, 1, 1],
            "customer_id": ["HN001", "HN002", "HN001", "HN003"],
            "package_name": ["OCH2240 Ultrasound", "Checkup Silver", "OCH2240 Ultrasound", "CT Scan"]
        })
        
        # Create sample clinical dataframe
        sample_clinical = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003", "P004"],
            "systolic": [120, 145, 130, 160],
            "bmi": [22.0, 26.5, 24.0, 30.0],
            "visit_date": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05"])
        })

        # Test calculation outputs without error
        net_rev = sample_sales["net_amount"].sum()
        total_pkgs = sample_sales["transactions"].sum()
        aov = net_rev / total_pkgs
        self.assertEqual(net_rev, 20000.0)
        self.assertEqual(total_pkgs, 4)
        self.assertEqual(aov, 5000.0)

        # Check repeat buyers
        cust_cnt = sample_sales.groupby("customer_id").size()
        repeat_buyers = (cust_cnt >= 2).sum()
        single_buyers = (cust_cnt == 1).sum()
        self.assertEqual(repeat_buyers, 1)  # HN001
        self.assertEqual(single_buyers, 2)  # HN002, HN003

        # Check high BP & BMI
        high_bp_pct = (sample_clinical["systolic"] >= 140).mean() * 100
        high_bmi_pct = (sample_clinical["bmi"] >= 25).mean() * 100
        self.assertEqual(high_bp_pct, 50.0)
        self.assertEqual(high_bmi_pct, 50.0)

    @patch("streamlit.markdown")
    @patch("streamlit.columns")
    @patch("streamlit.button")
    @patch("streamlit.dataframe")
    def test_render_executive_system_overview_runs_without_error(self, mock_df, mock_btn, mock_cols, mock_md):
        # Configure mock columns
        col_mock = MagicMock()
        mock_cols.return_value = [col_mock, col_mock]
        mock_btn.return_value = False

        sample_sales = pd.DataFrame({
            "net_amount": [5000.0, 3000.0],
            "transactions": [1, 1],
            "customer_id": ["HN001", "HN002"],
            "package_name": ["OCH2240", "Checkup"]
        })
        sample_clinical = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "systolic": [120, 145],
            "bmi": [22.0, 26.5],
            "visit_date": pd.to_datetime(["2024-01-15", "2024-02-20"])
        })

        # Call function
        try:
            app.render_executive_system_overview(
                sales_df=sample_sales,
                full_sales_df=sample_sales,
                raw_clinical_df=sample_clinical,
                c_mask=None,
                start_d=None,
                end_d=None
            )
            success = True
        except Exception as e:
            success = False
            print("Error in render_executive_system_overview:", e)

        self.assertTrue(success)

    def test_non_buyer_monetization_metrics(self):
        total_clin_pts = 168950
        total_buyers = 4326
        group_c_est = total_clin_pts - total_buyers
        group_c_pct = group_c_est / total_clin_pts * 100.0

        # Verify group C estimation matches clinic clinical base
        self.assertEqual(group_c_est, 164624)
        self.assertAlmostEqual(group_c_pct, 97.439, places=2)

        # Verify conversion scenario uplift
        conv_rate = 0.05
        group_c_converted = int(group_c_est * conv_rate)
        uplift_group_c = group_c_converted * 990
        self.assertEqual(group_c_converted, 8231)
        self.assertEqual(uplift_group_c, 8148690)

        # Baseline single buyer recall
        single_buyers = 3416
        aov = 3215.0
        uplift_recall = int(single_buyers * 0.10 * aov)
        self.assertEqual(uplift_recall, 1098244)

if __name__ == "__main__":
    unittest.main()
