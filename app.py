# -*- coding: utf-8 -*-
"""
🏥 PrimeCare Wellness Clinic — Main Dashboard Entrypoint
======================================================
Modular Streamlit Application Architecture:
- styles/theme.css: Centralized design system (unified 4-color palette)
- services/data_loader.py: Vectorized pandas loaders with @st.cache_data
- services/ml_engine.py: @st.cache_resource for XGBoost, SHAP, and Prophet
- views/: Sub-modules for Lead Intelligence, Patient Card, Executive Strategy, and Clinical Tracking
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Optional

# Fallback path injection for environments where plotly/openpyxl reside in project venv
_scratch_venv = r"C:\Users\Natcha.L\.gemini\antigravity\brain\0c7bc719-21e5-4a7d-941f-b1e53dd4762b\scratch\.venv\Lib\site-packages"
if os.path.isdir(_scratch_venv) and _scratch_venv not in sys.path:
    sys.path.insert(0, _scratch_venv)

import pandas as pd
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="PrimeCare Wellness Clinic — Clinical Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Services & Views imports
from services.data_loader import (
    load_synthetic_dataset,
    load_real_sales_data,
    resolve_patient_lab_snapshot,
)
from services.ml_engine import (
    get_xgboost_model,
    get_cached_shap_explainer,
    predict_patient_propensity,
)

import views.lead_intelligence as view_lead
import views.patient_card as view_patient
import views.executive_overview as view_exec
import views.clinical_tracking as view_clin

# ── Brand Constants ────────────────────────────────────────────────────────────
DEFAULT_CLINIC_NAME = "PrimeCare Wellness Clinic"
DEFAULT_CLINIC_TAGLINE = "Smart Clinical Intelligence & Healthcare Package Recommendation"


def get_clinic_name() -> str:
    """Retrieves current clinic brand name from session state or default."""
    return st.session_state.get("clinic_brand_name", DEFAULT_CLINIC_NAME)


def inject_css() -> None:
    """Loads centralized theme CSS from styles/theme.css into Streamlit."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            :root { --primary-color: #0F766E; }
            .stApp { background-color: #F8FAFC; }
            </style>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# Main Navigation & Page Controller
# ==============================================================================
def main() -> None:
    inject_css()

    if "clinic_brand_name" not in st.session_state:
        st.session_state["clinic_brand_name"] = DEFAULT_CLINIC_NAME

    cname = get_clinic_name()

    # ── Sidebar Brand Header ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#0F766E 0%,#0D9488 100%); border-radius:14px; padding:16px 18px; color:white; margin-bottom:12px; box-shadow:0 4px 12px rgba(15,118,110,0.18);">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="background:rgba(255,255,255,0.18); width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px;">🏥</div>
                    <div>
                        <div style="font-size:1.02rem; font-weight:800; color:white; line-height:1.25;">{cname}</div>
                        <div style="font-size:0.72rem; color:#CCFBF1; font-weight:500; margin-top:2px;">{DEFAULT_CLINIC_TAGLINE}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ ตั้งค่าชื่อคลินิก (Clinic Brand)", expanded=False):
            new_cname = st.text_input("ชื่อคลินิก / โรงพยาบาล:", value=cname, key="brand_editor_input")
            if new_cname.strip() and new_cname.strip() != cname:
                st.session_state["clinic_brand_name"] = new_cname.strip()
                st.rerun()

        st.divider()

        # ── Navigation Router ──────────────────────────────────────────────────
        if "pending_nav_section" in st.session_state:
            st.session_state["active_nav_page"] = st.session_state.pop("pending_nav_section")

        nav_options = [
            ":material/my_location: AI Lead Intelligence",
            ":material/search: Patient Insight Card",
            ":material/insights: Executive Strategy Overview",
            ":material/vital_signs: Clinical Command Center",
        ]

        # Key compatibility translation
        alias_map = {
            "AI Lead Intelligence": nav_options[0],
            "🎯 AI Lead Intelligence": nav_options[0],
            "Patient Insight Card": nav_options[1],
            "👤 Patient Insight Card": nav_options[1],
            "Executive Overview": nav_options[2],
            "Executive Strategy Overview": nav_options[2],
            "Clinical Command Center": nav_options[3],
        }

        current_nav = st.session_state.get("active_nav_page", nav_options[0])
        if current_nav in alias_map:
            current_nav = alias_map[current_nav]
        if current_nav not in nav_options:
            current_nav = nav_options[0]

        default_idx = nav_options.index(current_nav)
        selected_page = st.radio(
            "📌 ระบบและเมนูการทำงาน:",
            nav_options,
            index=default_idx,
            key="main_navigation_radio",
        )
        st.session_state["active_nav_page"] = selected_page

        st.divider()

        # ── Contextual Sidebar Filter Controls ────────────────────────────────
        st.caption("⚙️ ตัวกรองเฉพาะหน้าการทำงาน:")
        
        # Filters for Executive Overview
        start_date_val = None
        end_date_val = None
        if "Executive Strategy" in selected_page:
            date_range = st.date_input(
                "ช่วงเวลาที่วิเคราะห์ยอดขาย:",
                value=[datetime(2025, 1, 1), datetime(2026, 7, 31)],
                key="exec_date_range",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date_val, end_date_val = date_range[0], date_range[1]
        elif "Lead Intelligence" in selected_page:
            st.info("💡 ตัวกรองเกณฑ์คะแนนและ Journey Stage ถูกรวมไว้ที่แถบควบคุมด้านบนของหน้าจอหลัก")
        elif "Patient Insight Card" in selected_page:
            st.info("💡 สามารถค้นหาผู้ป่วยและเลือกเคสตัวอย่างได้โดยตรงจากแถบค้นหาด้านบน")
        else:
            st.info("💡 หน้าศูนย์ติดตามโรคเรื้อรังแสดงข้อมูลภาพรวมตามคลินิกเฉพาะทาง")

    # ── Breadcrumb Header ──────────────────────────────────────────────────────
    page_clean_title = selected_page.split(":", 2)[-1].strip() if ":" in selected_page else selected_page
    st.markdown(
        f"""
        <div class="breadcrumb-container">
            <span>🏥 {cname}</span>
            <span>&rsaquo;</span>
            <span class="breadcrumb-active">{page_clean_title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Lazy Data Loading & View Dispatching ───────────────────────────────────
    if "Lead Intelligence" in selected_page:
        # Only load encounter or clinical data for Lead Intelligence
        snap_df, enc_df, _ = load_synthetic_dataset()
        clin_base = enc_df if not enc_df.empty else snap_df
        view_lead.render_lead_intelligence_view(raw_clinical_df=clin_base)

    elif "Patient Insight Card" in selected_page:
        # Load synthetic snapshot for patient cards
        snap_df, enc_df, _ = load_synthetic_dataset()
        summary_pts = snap_df.set_index("patient_id") if not snap_df.empty else pd.DataFrame()
        view_patient.render_patient_card_view(
            avail_df=snap_df,
            summary_pts=summary_pts,
            dv=enc_df,
        )

    elif "Executive Strategy" in selected_page:
        # Load sales and clinical overview data
        sales_df = load_real_sales_data()
        snap_df, enc_df, _ = load_synthetic_dataset()
        view_exec.render_executive_overview_view(
            sales_df=sales_df,
            full_sales_df=sales_df,
            raw_clinical_df=enc_df,
            start_d=start_date_val,
            end_d=end_date_val,
        )

    elif "Clinical Command Center" in selected_page:
        # Load clinical tracking data
        snap_df, enc_df, _ = load_synthetic_dataset()
        view_clin.render_clinical_tracking_view(
            dv=enc_df,
            raw_clinical_df=snap_df,
        )


# ==============================================================================
# Backward Compatibility Aliases (Preserving all existing Unit Test Contracts)
# ==============================================================================
render_executive_system_overview = view_exec.render_executive_system_overview
render_lead_scoring_xai_dashboard = view_lead.render_lead_scoring_xai_dashboard
render_patient_profile = view_patient.render_patient_card_view


if __name__ == "__main__":
    main()
