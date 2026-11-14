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
            "📊 ภาพรวมผู้บริหาร",
            "🎯 คัดกรองกลุ่มเป้าหมาย",
            "👤 ข้อมูลผู้รับบริการรายคน",
            "🩺 ติดตามโรคเรื้อรัง",
        ]

        # Key compatibility translation for unit tests & internal redirects
        alias_map = {
            "Executive Overview": nav_options[0],
            "Executive Strategy Overview": nav_options[0],
            "📊 ภาพรวมผู้บริหาร": nav_options[0],
            ":material/insights: Executive Strategy Overview": nav_options[0],
            "AI Lead Intelligence": nav_options[1],
            "🎯 AI Lead Intelligence": nav_options[1],
            "🎯 คัดกรองกลุ่มเป้าหมาย": nav_options[1],
            ":material/my_location: AI Lead Intelligence": nav_options[1],
            "Patient Insight Card": nav_options[2],
            "👤 Patient Insight Card": nav_options[2],
            "👤 ข้อมูลผู้รับบริการรายคน": nav_options[2],
            ":material/search: Patient Insight Card": nav_options[2],
            "Clinical Command Center": nav_options[3],
            "🩺 ติดตามโรคเรื้อรัง": nav_options[3],
            ":material/vital_signs: Clinical Command Center": nav_options[3],
        }

        current_nav = st.session_state.get("active_nav_page", nav_options[2])
        if current_nav in alias_map:
            current_nav = alias_map[current_nav]
        if current_nav not in nav_options:
            current_nav = nav_options[2]

        default_idx = nav_options.index(current_nav)
        selected_page = st.radio(
            "📌 เมนูหลัก (Workflow Navigation):",
            nav_options,
            index=default_idx,
            key="main_navigation_radio",
        )
        st.session_state["active_nav_page"] = selected_page

        st.divider()

        # ── Contextual Sidebar Filter Controls ────────────────────────────────
        st.caption("🎛️ ตัวกรองเฉพาะหน้านี้ (Contextual Filters):")
        
        start_date_val = None
        end_date_val = None

        if selected_page == nav_options[0]:
            # Filters exclusively for Executive Overview
            date_range = st.date_input(
                "📅 ช่วงเวลาที่วิเคราะห์ยอดขาย:",
                value=[datetime(2025, 1, 1), datetime(2026, 7, 31)],
                key="exec_date_range",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date_val, end_date_val = date_range[0], date_range[1]
            st.selectbox(
                "🏥 สาขาบริการ:",
                ["ทุกสาขา (All Branches)", "สาขาหลัก (Main Clinic)", "สาขาในเมือง (Urban Clinic)"],
                key="exec_branch_filter"
            )

        elif selected_page == nav_options[1]:
            # Filters exclusively for AI Lead Intelligence
            lead_sb_score = st.slider(
                "🎯 เกณฑ์คะแนนโอกาสซื้อ (%):",
                min_value=0.0,
                max_value=90.0,
                value=40.0,
                step=5.0,
                key="lead_sb_score",
            )
            lead_sb_stage = st.selectbox(
                "📋 กลุ่มสถานะผู้ป่วย (Journey Stage):",
                ["ทั้งหมด (All Stages)", "Wellness / General OPD", "Chronic Care OPD", "Frequent Active OPD"],
                key="lead_sb_stage",
            )
            lead_sb_prio = st.selectbox(
                "🔥 ระดับความเร่งด่วน:",
                ["ทั้งหมด (All Priorities)", "🔥 High Priority (≥60%)", "⚡ Medium Priority (40-59%)", "🌱 Regular (<40%)"],
                key="lead_sb_prio",
            )

        elif selected_page == nav_options[2]:
            # Quick Jump controls exclusively for Patient Insight Card
            st.caption("🔍 เลือกเคสผู้รับบริการอย่างรวดเร็ว:")
            snap_df_quick, _, _ = load_synthetic_dataset()
            p_list = snap_df_quick["patient_id"].astype(str).tolist() if not snap_df_quick.empty else ["605", "6", "1", "25"]
            p_labels = {
                "605": "ID 605: ไทรอยด์+โลหิตจาง",
                "6": "ID 6: โรคไต+ตับ (Bespoke)",
                "1": "ID 1: ความดัน+เมตาบอลิก",
                "25": "ID 25: สุขภาพทั่วไป (Routine)",
            }
            cur_p = str(st.session_state.get("search_patient_id", "605"))
            cur_p_idx = p_list.index(cur_p) if cur_p in p_list else 0
            quick_sel = st.selectbox(
                "รายชื่อผู้รับบริการในระบบ:",
                p_list,
                index=cur_p_idx,
                format_func=lambda x: p_labels.get(x, f"HN68{int(x):05d}" if x.isdigit() else f"Patient {x}"),
                key="patient_sb_quick_sel",
            )
            if quick_sel != cur_p:
                st.session_state["search_patient_id"] = quick_sel

        elif selected_page == nav_options[3]:
            # Filters exclusively for Clinical Command Center
            st.selectbox(
                "🩺 กลุ่มโรคเรื้อรังหลัก:",
                ["ทั้งหมด (All NCDs)", "🫀 ความดันโลหิตสูง (HT)", "🩸 เบาหวาน (DM)", "🧪 ไขมันในเลือดสูง (DLP)", "🫘 โรคไตเรื้อรัง (CKD)"],
                key="clin_sb_disease",
            )
            st.selectbox(
                "🛡️ ระดับความเสี่ยงประชากร:",
                ["ทุกระดับความเสี่ยง", "Tier 4: วิกฤต/ต้องดูแลใกล้ชิด", "Tier 3: กลุ่มเสี่ยงสูง", "Tier 2: กลุ่มเฝ้าระวัง", "Tier 1: กลุ่มสุขภาพดี"],
                key="clin_sb_tier",
            )

    # ── Breadcrumb Header ──────────────────────────────────────────────────────
    breadcrumb_map = {
        nav_options[0]: "📊 ภาพรวมผู้บริหาร",
        nav_options[1]: "🎯 คัดกรองกลุ่มเป้าหมาย (AI Lead Intelligence)",
        nav_options[2]: "🔍 ค้นหาและดูแลผู้รับบริการรายคน",
        nav_options[3]: "🩺 ศูนย์ติดตามโรคเรื้อรัง (Clinical Command Center)",
    }
    bc_clean = breadcrumb_map.get(selected_page, selected_page)
    st.markdown(
        f"""
        <div class="breadcrumb-container">
            <span>🏥 {cname}</span>
            <span>&rsaquo;</span>
            <span class="breadcrumb-active">{bc_clean}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Lazy Data Loading & View Dispatching ───────────────────────────────────
    if selected_page == nav_options[0] or "Executive" in selected_page or "ภาพรวมผู้บริหาร" in selected_page:
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

    elif selected_page == nav_options[1] or "Lead" in selected_page or "คัดกรอง" in selected_page:
        # Load encounter or clinical data for Lead Intelligence
        snap_df, enc_df, _ = load_synthetic_dataset()
        clin_base = enc_df if not enc_df.empty else snap_df
        min_p = float(st.session_state.get("lead_sb_score", 40.0))
        stg_f = str(st.session_state.get("lead_sb_stage", "All"))
        prio_f = str(st.session_state.get("lead_sb_prio", "All"))
        view_lead.render_lead_intelligence_view(
            raw_clinical_df=clin_base,
            min_propensity=min_p,
            journey_stage_filter=stg_f,
            priority_filter=prio_f,
        )

    elif selected_page == nav_options[2] or "Patient" in selected_page or "ผู้รับบริการ" in selected_page:
        # Load synthetic snapshot for patient cards
        snap_df, enc_df, _ = load_synthetic_dataset()
        summary_pts = snap_df.set_index("patient_id") if not snap_df.empty else pd.DataFrame()
        view_patient.render_patient_card_view(
            avail_df=snap_df,
            summary_pts=summary_pts,
            dv=enc_df,
        )

    elif selected_page == nav_options[3] or "Clinical" in selected_page or "เรื้อรัง" in selected_page:
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
