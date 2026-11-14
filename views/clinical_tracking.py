# -*- coding: utf-8 -*-
"""
View 4: Clinical Command Center & Chronic Disease Population Tracking.
Monitors chronic NCD cohorts (Hypertension, Diabetes, Dyslipidemia, Dialysis)
and visualizes population risk distribution across Tiers 1–4.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.patient_segmentation import segment_population_dataframe


def render_html(html_str: str) -> None:
    """Renders HTML safely without Markdown 4-space code block indentation artifacts."""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    cleaned = "".join(lines)
    st.markdown(cleaned, unsafe_allow_html=True)


def render_clinical_tracking_view(
    dv: Optional[pd.DataFrame] = None,
    raw_clinical_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Renders the Chronic Disease Command Centers and Population Health Overview.
    """
    clin_df = dv if (dv is not None and not dv.empty) else raw_clinical_df
    if clin_df is None or clin_df.empty:
        st.info("ℹ️ ไม่พบข้อมูลเวชระเบียนทางคลินิก กรุณาตรวจสอบการเชื่อมต่อข้อมูล")
        return

    render_html(
        """
        <div style="background:linear-gradient(135deg,#0F766E 0%,#15803D 100%);
                    border-radius:14px; padding:20px 24px; color:white; margin-bottom:18px; box-shadow:0 4px 14px rgba(15,118,110,0.12);">
            <div style="font-size:0.75rem; font-weight:700; color:#CCFBF1; text-transform:uppercase; letter-spacing:0.05em;">
                🩺 Clinical Command Center & Population Health
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:white; margin-top:2px;">
                ศูนย์ติดตามผู้ป่วยโรคเรื้อรัง (NCDs) และการคัดกรองเชิงป้องกัน
            </div>
            <div style="font-size:0.85rem; opacity:0.92; margin-top:4px;">
                ติดตามกลุ่มผู้ป่วยความดันโลหิตสูง, เบาหวาน, ไขมันในเลือดสูง และผู้ป่วยไต เพื่อป้องกันภาวะแทรกซ้อนรุนแรง
            </div>
        </div>
        """
    )

    disease_tabs = st.tabs([
        "🫀 ศูนย์ความดันโลหิตสูง (HT)",
        "🩸 ศูนย์เบาหวาน (DM)",
        "🧈 ศูนย์ไขมันในเลือดสูง (DLP)",
        "🧪 ศูนย์ไต & ล้างไต (Renal)",
        "👥 ภาพรวมประชากร 4 ระดับ (Population)",
    ])

    # 1. Hypertension Tab
    with disease_tabs[0]:
        st.markdown("#### 🫀 Hypertension Command Center (ควบคุมความดัน < 140/90 mmHg)")
        ht_mask = clin_df["systolic"] >= 140.0 if "systolic" in clin_df.columns else pd.Series(False, index=clin_df.index)
        ht_df = clin_df[ht_mask].copy()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนผู้ป่วยความดันสูง (SBP ≥ 140)", f"{len(ht_df):,} ครั้งตรวจ")
        p1_ht = (ht_df["systolic"] >= 160.0).sum() if "systolic" in ht_df.columns else 0
        c2.metric("กลุ่มวิกฤตเร่งด่วน (SBP ≥ 160)", f"{p1_ht:,} ราย", delta="ต้องพบแพทย์ทันที" if p1_ht > 0 else "ปกติ", delta_color="inverse")
        mean_sbp = ht_df["systolic"].mean() if not ht_df.empty else 0.0
        c3.metric("ค่าเฉลี่ย Systolic BP", f"{mean_sbp:.1f} mmHg")

        if not ht_df.empty:
            show_cols = [c for c in ["patient_id", "visit_date", "systolic", "bmi", "clinic_name"] if c in ht_df.columns]
            st.dataframe(ht_df[show_cols].head(50), use_container_width=True, hide_index=True)

    # 2. Diabetes Tab
    with disease_tabs[1]:
        st.markdown("#### 🩸 Diabetes Command Center (ควบคุมระดับน้ำตาล & ป้องกันเบาหวานลงไต)")
        dm_mask = clin_df.get("disease_group", pd.Series("")).str.contains(r"เบาหวาน", na=False) if "disease_group" in clin_df.columns else pd.Series(False, index=clin_df.index)
        dm_df = clin_df[dm_mask].copy() if dm_mask.any() else clin_df.head(20)

        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนเคสเบาหวาน", f"{len(dm_df):,} เคส")
        c2.metric("เป้าหมายการควบคุม", "HbA1c < 7.0%")
        c3.metric("แนะนำตรวจเพิ่มเติม", "UACR + ตรวจจอประสาทตา")
        show_cols = [c for c in ["patient_id", "visit_date", "systolic", "bmi", "clinic_name"] if c in dm_df.columns]
        st.dataframe(dm_df[show_cols].head(50), use_container_width=True, hide_index=True)

    # 3. Dyslipidemia Tab
    with disease_tabs[2]:
        st.markdown("#### 🧈 Dyslipidemia Command Center (ควบคุมระดับไขมัน LDL & หลอดเลือด)")
        dlp_mask = clin_df.get("disease_group", pd.Series("")).str.contains(r"ไขมัน", na=False) if "disease_group" in clin_df.columns else pd.Series(False, index=clin_df.index)
        dlp_df = clin_df[dlp_mask].copy() if dlp_mask.any() else clin_df.head(20)

        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนเคสไขมันในเลือดสูง", f"{len(dlp_df):,} เคส")
        c2.metric("เป้าหมายการควบคุม", "LDL-C < 100 mg/dL")
        c3.metric("แนะนำแพ็กเกจ", "Cardiovascular & Stroke Screening")
        show_cols = [c for c in ["patient_id", "visit_date", "systolic", "bmi", "clinic_name"] if c in dlp_df.columns]
        st.dataframe(dlp_df[show_cols].head(50), use_container_width=True, hide_index=True)

    # 4. Dialysis & Renal Tab
    with disease_tabs[3]:
        st.markdown("#### 🧪 Renal & Dialysis Command Center (ชะลอไตเสื่อม & คัดกรอง CKD)")
        st.info("💡 แนะนำโปรแกรมตรวจคัดกรองการทำงานของไต (eGFR + UACR) สำหรับผู้ป่วยเบาหวานและความดันโลหิตสูงทุกคนอย่างน้อยปีละ 1 ครั้ง")

    # 5. Population 4-Tier Overview
    with disease_tabs[4]:
        pop_df = segment_population_dataframe(clin_df)
        t_counts = pop_df["risk_tier"].value_counts().to_dict() if "risk_tier" in pop_df.columns else {}
        
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tier 1: สุขภาพปกติ", f"{t_counts.get(1, 0):,} ครั้ง")
        p2.metric("Tier 2: เสี่ยงเมตาบอลิก", f"{t_counts.get(2, 0):,} ครั้ง")
        p3.metric("Tier 3: โรคเรื้อรังระยะแรก", f"{t_counts.get(3, 0):,} ครั้ง")
        p4.metric("Tier 4: เชิงลึก/วิกฤต", f"{t_counts.get(4, 0):,} ครั้ง")

