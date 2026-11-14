# -*- coding: utf-8 -*-
"""
View 1: AI Lead Intelligence & Precision Cross-Sell Dashboard.
Identifies unconverted OPD patients (0 past package purchases), scores package propensity via XGBoost,
explains individual clinical drivers via SHAP, and models revenue uplift with Prophet seasonality.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from services.ml_engine import get_xgboost_model, get_cached_prophet_model
from logic.lead_scoring_xai import render_lead_scoring_xai_dashboard


def render_lead_intelligence_view(
    raw_clinical_df: Optional[pd.DataFrame] = None,
    min_propensity: float = 40.0,
    journey_stage_filter: str = "All",
    priority_filter: str = "All",
) -> None:
    """
    Renders the complete AI Lead Intelligence view.
    Includes contextual filters, executive KPI ribbon, 4 horizontal tabs,
    and business/clinical friendly terminology.
    """
    # ── Header Banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0F766E 0%,#0D9488 100%); border-radius:14px; padding:22px 26px; color:white; margin-bottom:18px; box-shadow:0 4px 14px rgba(15,118,110,0.14);">
            <div style="font-size:0.78rem; font-weight:700; color:#CCFBF1; text-transform:uppercase; letter-spacing:0.06em;">
                🎯 AI Lead Intelligence & Precision Cross-Sell Engine
            </div>
            <div style="font-size:1.65rem; font-weight:800; color:white; margin-top:4px;">
                ระบบคัดกรองและจัดกลุ่มผู้ป่วยเป้าหมายสำหรับแนะนำแพ็กเกจสุขภาพ
            </div>
            <div style="font-size:0.86rem; color:#E6FFFA; opacity:0.95; margin-top:6px; max-width:980px; line-height:1.5;">
                วิเคราะห์ประวัติสุขภาพ ค่าความดัน และประวัติการรับบริการเพื่อค้นหาผู้ป่วยนอกที่ยังไม่เคยซื้อแพ็กเกจตรวจสุขภาพ 
                พร้อมประเมินโอกาสการตอบรับ (Propensity Score) และปัจจัยขับเคลื่อนทางการแพทย์เฉพาะรายบุคคล
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Contextual Filter Bar ──────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([1.5, 1.2, 1.2])
    with col_f1:
        st.caption("🔍 ตัวกรองเฉพาะหน้า: เกณฑ์คะแนนแนวโน้มซื้อ")
        min_score = st.slider(
            "คะแนนโอกาสซื้อขั้นต่ำ (%):",
            min_value=0.0,
            max_value=90.0,
            value=float(min_propensity),
            step=5.0,
            key="lead_view_min_score",
            help="กรองเฉพาะผู้ป่วยที่มีคะแนนคาดการณ์ตั้งแต่เกณฑ์ที่กำหนดขึ้นไป"
        )
    with col_f2:
        st.caption("📋 กลุ่มสถานะผู้ป่วย (Journey Stage)")
        stage_options = ["ทั้งหมด (All Stages)", "Wellness / General OPD", "Chronic Care OPD", "Frequent Active OPD"]
        selected_stage = st.selectbox(
            "สถานะผู้ป่วย:",
            stage_options,
            index=0,
            key="lead_view_stage_sel"
        )
    with col_f3:
        st.caption("🔥 ระดับความเร่งด่วน (Action Priority)")
        prio_options = ["ทั้งหมด (All Priorities)", "🔥 High Priority (≥60%)", "⚡ Medium Priority (40-59%)", "🌱 Regular (ต่ำกว่า 40%)"]
        selected_prio = st.selectbox(
            "ระดับความเร่งด่วน:",
            prio_options,
            index=0,
            key="lead_view_prio_sel"
        )

    st.write("")

    # ── Load Cached ML Models ──────────────────────────────────────────────────
    xgb_model = get_xgboost_model()
    prophet_model = get_cached_prophet_model()

    # ── Delegate to lead_scoring_xai Dashboard ────────────────────────────────
    render_lead_scoring_xai_dashboard(
        raw_clinical_df=raw_clinical_df,
        prophet_model=prophet_model,
        render_section4=True,
        model=xgb_model,
    )
