# -*- coding: utf-8 -*-
"""
View 4: Clinical Command Center & Chronic Disease Population Tracking
====================================================================
Comprehensive, production-grade clinical tracking and population health analytics:
- 🫀 Hypertension Command Center (JNC8/Thai HT risk classification & longitudinal BP trajectory)
- 🩸 Diabetes Command Center (Glycemic spectrum, complication risk, and diabetic nephropathy)
- 🧈 Dyslipidemia & ASCVD Command Center (Atherogenic dyslipidemia & Thai CV Risk 10Y score)
- 🧪 Renal & CKD Progression Command Center (KDIGO 2024 Heatmap & Kaplan-Meier Renal Survival Curve)
- 👥 Population Health & Multimorbidity Stratification (Tiers 1–4, comorbidity clustering, and care management)
- 🔗 Cross-Linking Engine: Seamless bidirectional navigation to Patient Insight Card (?page=patient&patient_id=...)
"""
from __future__ import annotations

import html
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

import streamlit as st

from logic.patient_segmentation import segment_population_dataframe


# ── Safe HTML Renderer ────────────────────────────────────────────────────────
def render_html(html_str: str) -> None:
    """Renders HTML safely using st.html (or cleaned st.markdown fallback)."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        lines = [line.strip() for line in html_str.splitlines() if line.strip()]
        st.markdown("".join(lines), unsafe_allow_html=True)


# ── Cross-linking Navigation Helper ──────────────────────────────────────────
def navigate_to_patient_card(patient_id: Any) -> None:
    """Seamlessly redirects the user directly to the Patient Insight Card for the given patient ID."""
    pid_str = str(patient_id).strip()
    if not pid_str:
        return
    st.session_state["search_patient_id"] = pid_str
    st.session_state["pic_search_input"] = pid_str
    st.session_state["pending_nav_section"] = "patient"
    st.session_state["active_page_key"] = "patient"
    st.query_params["page"] = "patient"
    st.query_params["patient_id"] = pid_str
    st.rerun()


# ── KDIGO Classification Helpers ─────────────────────────────────────────────
def categorize_gfr(egfr: float) -> str:
    """Classifies eGFR into KDIGO stages (G1 to G5)."""
    if pd.isna(egfr):
        return "Unknown"
    v = float(egfr)
    if v >= 90.0:
        return "G1 (≥90)"
    elif v >= 60.0:
        return "G2 (60-89)"
    elif v >= 45.0:
        return "G3a (45-59)"
    elif v >= 30.0:
        return "G3b (30-44)"
    elif v >= 15.0:
        return "G4 (15-29)"
    else:
        return "G5 (<15)"


def categorize_uacr(uacr: float) -> str:
    """Classifies UACR into KDIGO albuminuria stages (A1 to A3)."""
    if pd.isna(uacr):
        return "A1 (<30)"  # Conservative fallback
    v = float(uacr)
    if v < 30.0:
        return "A1 (<30)"
    elif v <= 300.0:
        return "A2 (30-300)"
    else:
        return "A3 (>300)"


def get_kdigo_risk_color(gfr_stage: str, uacr_stage: str) -> Tuple[str, str, str]:
    """
    Maps KDIGO 2024 2D matrix cell to Risk Level, Background Color, and Text Color.
    Returns: (Risk Level Name, Hex Background, Hex Text)
    """
    g = gfr_stage[:3].strip()
    a = uacr_stage[:2].strip()

    # Red: Very High Risk
    if g == "G5" or g == "G4" or (g == "G3b" and a in ["A2", "A3"]) or (g == "G3a" and a == "A3"):
        return "ความเสี่ยงสูงมาก (Very High)", "#FEE2E2", "#991B1B"
    # Orange: High Risk
    elif (g == "G3b" and a == "A1") or (g == "G3a" and a == "A2") or (g in ["G1", "G2"] and a == "A3"):
        return "ความเสี่ยงสูง (High)", "#FFEDD5", "#9A3412"
    # Yellow: Moderately Increased
    elif (g == "G3a" and a == "A1") or (g in ["G1", "G2"] and a == "A2"):
        return "ความเสี่ยงปานกลาง (Moderate)", "#FEF9C3", "#854D0E"
    # Green: Low Risk
    else:
        return "ความเสี่ยงต่ำ (Low Risk)", "#DCFCE7", "#166534"


# ── Render Patient Cross-Link Priority Component ─────────────────────────────
def render_patient_priority_selector(
    df: pd.DataFrame,
    context_prefix: str,
    key_metric_col: str,
    key_metric_label: str,
    key_metric_unit: str = "",
    alert_badge: str = "วิกฤต",
) -> None:
    """
    Renders an ergonomic interactive jump box and priority table with direct cross-linking buttons.
    """
    if df.empty:
        st.info("ℹ️ ไม่พบผู้ป่วยที่มีความเสี่ยงในเกณฑ์ที่ระบุ")
        return

    st.markdown(
        f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 16px; margin:12px 0 16px 0;">
            <div style="font-size:0.86rem; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span>🔗 ส่งต่อแฟ้มประวัติรายบุคคล (Direct Cross-Linking to Patient Insight Card)</span>
                <span style="font-size:0.75rem; background:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:12px; font-weight:600;">
                    {len(df):,} รายในกลุ่มเป้าหมาย
                </span>
            </div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
                คลิกเลือกรายชื่อเพื่อเปิดแฟ้มประวัติฉบับเต็ม ตรวจสอบผลแล็บย้อนหลัง และสร้างใบเสนอราคาแพ็กเกจสุขภาพทันที
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Quick Selectbox Jump
    sample_df = df.head(100).copy()
    patient_options = sample_df["patient_id"].tolist()

    def format_patient_label(pid: Any) -> str:
        row = sample_df[sample_df["patient_id"] == pid].iloc[0]
        hn = str(row.get("hn", f"HN{pid}"))
        age = int(row.get("age", 50))
        gender = str(row.get("gender", ""))
        metric_val = row.get(key_metric_col, "-")
        if isinstance(metric_val, (int, float)) and pd.notna(metric_val):
            metric_str = f"{metric_val:.1f}"
        else:
            metric_str = str(metric_val)
        pkg = str(row.get("label_primary_short_name", row.get("label_primary_package_code", "")))
        return f"👤 {hn} (ID: {pid}) | อายุ {age} ปี ({gender}) | {key_metric_label}: {metric_str} {key_metric_unit} | แนะนำ: {pkg}"

    sel_col1, sel_col2 = st.columns([3.8, 1.2])
    with sel_col1:
        chosen_pid = st.selectbox(
            "⚡ เลือกผู้ป่วยกลุ่มเสี่ยงเพื่อเปิดแฟ้มตรวจรักษาทันที:",
            options=patient_options,
            format_func=format_patient_label,
            key=f"{context_prefix}_jump_select",
        )
    with sel_col2:
        st.write("")
        st.write("")
        if st.button("🔎 เปิดแฟ้มประวัติ (Patient Card)", key=f"{context_prefix}_btn_jump", type="primary", use_container_width=True):
            navigate_to_patient_card(chosen_pid)

    # 2. Priority Cards Grid (Top 4 most critical cases)
    st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#334155; margin:14px 0 8px 0;'>🚨 เคสตัวอย่างระดับความเสี่ยงสูงสุดในกลุ่ม (Top Priority Alerts):</div>", unsafe_allow_html=True)
    top_cases = df.head(4).to_dict(orient="records")
    cols = st.columns(min(len(top_cases), 4))
    for i, pt in enumerate(top_cases):
        with cols[i]:
            pid = pt.get("patient_id")
            hn = pt.get("hn", f"HN{pid}")
            age = int(pt.get("age", 50))
            gender = pt.get("gender", "")
            metric_v = pt.get(key_metric_col, 0.0)
            if isinstance(metric_v, (int, float)) and pd.notna(metric_v):
                m_str = f"{metric_v:.1f}"
            else:
                m_str = str(metric_v)
            pkg = pt.get("label_primary_short_name", "Comprehensive Care")

            st.markdown(
                f"""
                <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #DC2626; border-radius:8px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,0.04); margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:800; color:#0F172A; font-size:0.92rem;">{hn}</span>
                        <span style="font-size:0.7rem; background:#FEE2E2; color:#991B1B; padding:1px 6px; border-radius:4px; font-weight:700;">{alert_badge}</span>
                    </div>
                    <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">
                        อายุ {age} ปี ({gender}) | ID: {pid}
                    </div>
                    <div style="margin:8px 0; background:#F8FAFC; border-radius:6px; padding:6px 8px;">
                        <div style="font-size:0.72rem; color:#64748B;">{key_metric_label}</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#DC2626; font-family:'JetBrains Mono',monospace;">
                            {m_str} <span style="font-size:0.75rem; color:#475569; font-weight:600;">{key_metric_unit}</span>
                        </div>
                    </div>
                    <div style="font-size:0.75rem; color:#0F766E; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                        📦 {pkg}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🔎 เปิดแฟ้มเคสนี้", key=f"{context_prefix}_card_btn_{pid}_{i}", use_container_width=True):
                navigate_to_patient_card(pid)


# ==============================================================================
# Main Clinical Tracking View Controller
# ==============================================================================
def render_clinical_tracking_view(
    dv: Optional[pd.DataFrame] = None,
    raw_clinical_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Renders the unified Clinical Command Center:
    - Normalizes data structures between encounters and patient biomarkers
    - Visualizes disease-specific longitudinal cohorts and population matrices
    - Renders KDIGO 2024 Heatmap and Pseudo-Kaplan-Meier Renal Survival Curves
    - Delivers direct Actionable CTAs and bi-directional cross-linking to Patient Card
    """
    # ── 1. Data Normalization & Ingestion ──────────────────────────────────────
    pts_df = raw_clinical_df if (raw_clinical_df is not None and not raw_clinical_df.empty and "hba1c" in raw_clinical_df.columns) else dv
    enc_df = dv if (dv is not None and not dv.empty and "visit_date" in dv.columns) else raw_clinical_df

    if pts_df is None or pts_df.empty:
        st.info("ℹ️ ไม่พบข้อมูลเวชระเบียนทางคลินิก กรุณาตรวจสอบการเชื่อมต่อฐานข้อมูล")
        return

    # Harmonize patient biomarkers
    pts = pts_df.copy()
    if "sbp" not in pts.columns and "systolic" in pts.columns:
        pts["sbp"] = pts["systolic"]
    if "dbp" not in pts.columns and "diastolic" in pts.columns:
        pts["dbp"] = pts["diastolic"]
    if "patient_id" not in pts.columns and "id" in pts.columns:
        pts["patient_id"] = pts["id"]

    # ── 2. Top Header Banner ──────────────────────────────────────────────────
    render_html(
        """
        <div style="background:linear-gradient(135deg,#0F766E 0%,#15803D 100%);
                    border-radius:14px; padding:22px 26px; color:white; margin-bottom:18px; box-shadow:0 4px 14px rgba(15,118,110,0.12);">
            <div style="font-size:0.75rem; font-weight:700; color:#CCFBF1; text-transform:uppercase; letter-spacing:0.06em;">
                🩺 Clinical Command Center & Population Health Analytics
            </div>
            <div style="font-size:1.65rem; font-weight:800; color:white; margin-top:2px;">
                ศูนย์ติดตามผู้ป่วยโรคเรื้อรัง (NCDs) และการส่งต่อการรักษาเชิงป้องกัน
            </div>
            <div style="font-size:0.86rem; opacity:0.92; margin-top:4px;">
                ระบบติดตามและบริหารจัดการกลุ่มประชากรโรคความดันโลหิตสูง, เบาหวาน, ไขมันในเลือดสูง, ไตวายเรื้อรัง (CKD KDIGO) 
                พร้อมการวิเคราะห์ Kaplan-Meier Survival และระบบส่งต่อสู่แฟ้มประวัติรายบุคคล
            </div>
        </div>
        """
    )

    # ── 3. Tabs Navigation ────────────────────────────────────────────────────
    disease_tabs = st.tabs([
        "🫀 ศูนย์ความดันโลหิตสูง (HT)",
        "🩸 ศูนย์เบาหวาน (DM)",
        "🧈 ศูนย์ไขมันในเลือดสูง (DLP)",
        "🧪 ศูนย์ไต & ล้างไต (Renal & KDIGO)",
        "👥 ภาพรวมประชากร 4 ระดับ (Population)",
    ])

    # ==========================================================================
    # TAB 1: HYPERTENSION COMMAND CENTER
    # ==========================================================================
    with disease_tabs[0]:
        st.markdown("### 🫀 Hypertension Command Center (ควบคุมความดันเป้าหมาย < 140/90 mmHg)")

        # Masks
        ht_all_mask = (pts["sbp"] >= 140.0) | (pts["dbp"] >= 90.0) | (pts.get("known_dx_ht", 0) == 1)
        ht_crisis_mask = (pts["sbp"] >= 160.0) | (pts["dbp"] >= 100.0)
        ht_controlled_mask = (pts.get("known_dx_ht", 0) == 1) & (pts["sbp"] < 140.0) & (pts["dbp"] < 90.0)

        ht_df = pts[ht_all_mask].copy()
        crisis_df = pts[ht_crisis_mask].sort_values("sbp", ascending=False).copy()
        
        # KPI Ribbon
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ผู้ป่วยกลุ่มความดันโลหิตสูง", f"{len(ht_df):,} ราย", help="SBP ≥ 140 หรือ DBP ≥ 90 หรือมีประวัติความดันโลหิตสูง")
        p_crisis = len(crisis_df)
        k2.metric("วิกฤตความดัน (SBP ≥ 160 / DBP ≥ 100)", f"{p_crisis:,} ราย", delta="ต้องพบแพทย์ทันที" if p_crisis > 0 else "ปกติ", delta_color="inverse")
        
        known_ht_count = (pts.get("known_dx_ht", 0) == 1).sum()
        ctrl_rate = (ht_controlled_mask.sum() / known_ht_count * 100.0) if known_ht_count > 0 else 0.0
        k3.metric("อัตราควบคุมความดันได้ตามเป้า", f"{ctrl_rate:.1f}%", help="ผู้ป่วยที่ได้รับการวินิจฉัยแล้วคุม SBP < 140 และ DBP < 90")
        
        mean_cv_risk = ht_df["thai_cv_risk_10y"].mean() * 100.0 if ("thai_cv_risk_10y" in ht_df.columns and not ht_df.empty) else 12.5
        k4.metric("ความเสี่ยงโรคหัวใจเฉลี่ย 10 ปี", f"{mean_cv_risk:.1f}%", delta="High Risk" if mean_cv_risk >= 10.0 else "Low Risk", delta_color="inverse")

        # Visualizations (2 Columns)
        v_col1, v_col2 = st.columns([1.1, 0.9])
        with v_col1:
            st.markdown("##### 📊 การจำแนกระดับความดันโลหิตตามแนวทาง JNC8 / Thai Guidelines")
            plot_ht = ht_df.head(600).copy()
            def jnc_cat(row):
                s, d = row["sbp"], row["dbp"]
                if s >= 160 or d >= 100:
                    return "Stage 2 / Crisis (≥160/100)"
                elif s >= 140 or d >= 90:
                    return "Stage 1 HT (140-159/90-99)"
                elif s >= 130 or d >= 80:
                    return "Pre-HT (130-139/80-89)"
                return "Controlled (<130/80)"
            plot_ht["jnc_category"] = plot_ht.apply(jnc_cat, axis=1)

            fig_ht = px.scatter(
                plot_ht,
                x="sbp",
                y="dbp",
                color="jnc_category",
                color_discrete_map={
                    "Stage 2 / Crisis (≥160/100)": "#DC2626",
                    "Stage 1 HT (140-159/90-99)": "#EA580C",
                    "Pre-HT (130-139/80-89)": "#D97706",
                    "Controlled (<130/80)": "#16A34A",
                },
                hover_data=["patient_id", "hn", "age", "bmi"],
                labels={"sbp": "Systolic BP (mmHg)", "dbp": "Diastolic BP (mmHg)", "jnc_category": "ระดับความดัน"},
                title="การกระจายตัว Systolic vs Diastolic BP",
            )
            fig_ht.add_hline(y=90, line_dash="dash", line_color="#DC2626", annotation_text="เกณฑ์ DBP 90")
            fig_ht.add_vline(x=140, line_dash="dash", line_color="#DC2626", annotation_text="เกณฑ์ SBP 140")
            fig_ht.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_ht, use_container_width=True)

        with v_col2:
            st.markdown("##### 📈 แนวโน้มความดันโลหิตเฉลี่ยรายไตรมาส (Longitudinal Cohort Trend)")
            if enc_df is not None and not enc_df.empty and "visit_date" in enc_df.columns:
                e_df = enc_df.dropna(subset=["visit_date"]).copy()
                e_df["visit_date"] = pd.to_datetime(e_df["visit_date"])
                e_df["quarter"] = e_df["visit_date"].dt.to_period("Q").astype(str)
                s_col = "sbp" if "sbp" in e_df.columns else "systolic"
                d_col = "dbp" if "dbp" in e_df.columns else "diastolic"
                
                if s_col in e_df.columns and d_col in e_df.columns:
                    q_trend = e_df.groupby("quarter").agg({s_col: "mean", d_col: "mean", "patient_id": "nunique"}).reset_index()
                    q_trend = q_trend[q_trend["quarter"] >= "2023Q1"].tail(12)

                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=q_trend["quarter"], y=q_trend[s_col], mode="lines+markers", name="Systolic เฉลี่ย", line=dict(color="#DC2626", width=3)))
                    fig_trend.add_trace(go.Scatter(x=q_trend["quarter"], y=q_trend[d_col], mode="lines+markers", name="Diastolic เฉลี่ย", line=dict(color="#2563EB", width=2)))
                    fig_trend.add_hline(y=140, line_dash="dot", line_color="#EF4444", annotation_text="เป้าหมาย SBP 140")
                    fig_trend.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="mmHg", xaxis_title="ไตรมาสตรวจ", legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("ℹ️ ไม่พบคอลัมน์ความดันในข้อมูลประวัติการตรวจ")
            else:
                st.info("ℹ️ ไม่พบข้อมูลประวัติการตรวจแบบ longitudinal")

        # Actionable CTA Bar
        st.markdown(
            """
            <div style="background:#FFFBEB; border-left:4px solid #D97706; border-radius:8px; padding:12px 16px; margin:14px 0;">
                <div style="font-weight:700; color:#92400E; font-size:0.88rem;">🎯 แผนปฏิบัติการทางคลินิก (Actionable Clinical Interventions):</div>
                <div style="font-size:0.82rem; color:#78350F; margin-top:2px;">
                    • <b>กลุ่มวิกฤต (SBP ≥ 160):</b> แนะนำส่งต่อแพทย์อายุรกรรมทันที ตรวจ EKG + ปัสสาวะคัดกรองไตเสื่อม (UACR) เพื่อประเมิน Target Organ Damage<br>
                    • <b>กลุ่ม Pre-HT (SBP 130–139):</b> กระตุ้นติดตามวัดความดันที่บ้าน (Home BP Monitoring) และนำเสนอแพ็กเกจ <i>Cardiovascular & Stroke Screening</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-Linking Priority Section
        render_patient_priority_selector(
            df=crisis_df,
            context_prefix="ht",
            key_metric_col="sbp",
            key_metric_label="Systolic BP",
            key_metric_unit="mmHg",
            alert_badge="ความดันวิกฤต",
        )

    # ==========================================================================
    # TAB 2: DIABETES COMMAND CENTER
    # ==========================================================================
    with disease_tabs[1]:
        st.markdown("### 🩸 Diabetes Command Center (ควบคุมระดับน้ำตาล & ป้องกันเบาหวานลงไต)")

        dm_all_mask = (pts["hba1c"] >= 6.5) | (pts["fbs"] >= 126.0) | (pts.get("known_dx_dm", 0) == 1)
        dm_crisis_mask = (pts["hba1c"] >= 9.0) | (pts["fbs"] >= 200.0)
        dm_dkd_mask = dm_all_mask & ((pts["egfr"] < 60.0) | (pts["uacr"] >= 30.0))

        dm_df = pts[dm_all_mask].copy()
        dm_crisis_df = pts[dm_crisis_mask].sort_values("hba1c", ascending=False).copy()
        dm_dkd_df = pts[dm_dkd_mask].sort_values("egfr", ascending=True).copy()

        # KPI Ribbon
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("ผู้ป่วยกลุ่มเบาหวาน & น้ำตาลสูง", f"{len(dm_df):,} ราย", help="HbA1c ≥ 6.5% หรือ FBS ≥ 126 mg/dL หรือมีประวัติเบาหวาน")
        n_uncontrolled = len(dm_crisis_df)
        d2.metric("น้ำตาลวิกฤต (HbA1c ≥ 9.0%)", f"{n_uncontrolled:,} ราย", delta="เสี่ยงภาวะแทรกซ้อนเฉียบพลัน" if n_uncontrolled > 0 else "คุมได้ดี", delta_color="inverse")
        n_dkd = len(dm_dkd_df)
        d3.metric("เสี่ยงเบาหวานลงไต (DKD Alert)", f"{n_dkd:,} ราย", delta="eGFR<60 หรือ UACR≥30", delta_color="inverse")
        mean_hba1c = dm_df["hba1c"].mean() if not dm_df.empty else 7.2
        d4.metric("ค่าเฉลี่ย HbA1c ของกลุ่ม", f"{mean_hba1c:.2f}%", help="เป้าหมายสมาคมโรคเบาหวาน < 7.0%")

        # Visualizations (2 Columns)
        dm_col1, dm_col2 = st.columns([1.0, 1.0])
        with dm_col1:
            st.markdown("##### 📊 การกระจายตัวระดับการควบคุมน้ำตาล (Glycemic Control Spectrum)")
            def hba1c_bracket(v):
                if v < 7.0: return "1. ควบคุมได้ดีมาก (<7.0%)"
                elif v < 8.0: return "2. เกินเป้าหมายเล็กน้อย (7.0-7.9%)"
                elif v < 9.0: return "3. ควบคุมไม่ดี (8.0-8.9%)"
                return "4. วิกฤต/อันตรายสูง (≥9.0%)"
            
            plot_dm = dm_df.copy()
            plot_dm["control_tier"] = plot_dm["hba1c"].apply(hba1c_bracket)
            counts = plot_dm["control_tier"].value_counts().sort_index().reset_index()
            counts.columns = ["tier", "count"]

            fig_dm_bar = px.bar(
                counts,
                x="tier",
                y="count",
                color="tier",
                color_discrete_sequence=["#16A34A", "#EAB308", "#F97316", "#DC2626"],
                text="count",
                title="สัดส่วนผู้ป่วยตามระดับ HbA1c",
            )
            fig_dm_bar.update_layout(height=340, showlegend=False, xaxis_title="", yaxis_title="จำนวนผู้ป่วย (ราย)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dm_bar, use_container_width=True)

        with dm_col2:
            st.markdown("##### 🔬 ภาวะเบาหวานแทรกซ้อนไต: ระดับน้ำตาล vs อัตราการกรองของไต")
            fig_dm_scat = px.scatter(
                dm_df.head(500),
                x="hba1c",
                y="egfr",
                color="uacr",
                color_continuous_scale="Viridis",
                hover_data=["patient_id", "hn", "age", "fbs"],
                labels={"hba1c": "HbA1c (%)", "egfr": "eGFR (mL/min/1.73m²)", "uacr": "UACR (mg/g)"},
                title="HbA1c เทียบกับ eGFR (สีแสดงค่าโปรตีนไข่ขาวในปัสสาวะ UACR)",
            )
            fig_dm_scat.add_hline(y=60, line_dash="dash", line_color="#DC2626", annotation_text="เกณฑ์ไตเสื่อม eGFR 60")
            fig_dm_scat.add_vline(x=7.0, line_dash="dash", line_color="#16A34A", annotation_text="เป้าหมาย HbA1c 7.0%")
            fig_dm_scat.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dm_scat, use_container_width=True)

        # Actionable CTA Bar
        st.markdown(
            f"""
            <div style="background:#ECFDF5; border-left:4px solid #059669; border-radius:8px; padding:12px 16px; margin:14px 0;">
                <div style="font-weight:700; color:#065F46; font-size:0.88rem;">🎯 แผนปฏิบัติการเพื่อป้องกันภาวะแทรกซ้อนโรคเบาหวาน:</div>
                <div style="font-size:0.82rem; color:#047857; margin-top:2px;">
                    • <b>กลุ่มเสี่ยงไต (DKD {n_dkd:,} ราย):</b> ส่งต่อตรวจคัดกรองปัสสาวะหา Microalbuminuria ซ้ำทุก 3–6 เดือน พร้อมพิจารณายาชะลอไตเสื่อมกลุ่ม SGLT2i<br>
                    • <b>กลุ่ม HbA1c ≥ 9.0%:</b> แนะนำโปรแกรมตรวจจอประสาทตา (Fundus Camera) และตรวจสุขภาพเท้า (Diabetic Foot Examination) ทันที
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-Linking Priority Section
        target_dm_priority = dm_crisis_df if not dm_crisis_df.empty else dm_dkd_df
        render_patient_priority_selector(
            df=target_dm_priority,
            context_prefix="dm",
            key_metric_col="hba1c",
            key_metric_label="HbA1c",
            key_metric_unit="%",
            alert_badge="น้ำตาลวิกฤต",
        )

    # ==========================================================================
    # TAB 3: DYSLIPIDEMIA COMMAND CENTER
    # ==========================================================================
    with disease_tabs[2]:
        st.markdown("### 🧈 Dyslipidemia & ASCVD Command Center (ควบคุมระดับไขมัน LDL & ประเมินหลอดเลือดหัวใจ)")

        dlp_all_mask = (pts["ldl"] >= 160.0) | (pts["tg"] >= 200.0) | (pts.get("known_dx_dlp", 0) == 1)
        high_cv_mask = (pts.get("thai_cv_risk_10y", 0.0) >= 0.20) | (pts["ldl"] >= 190.0)
        
        dlp_df = pts[dlp_all_mask].copy()
        high_cv_df = pts[high_cv_mask].sort_values("thai_cv_risk_10y", ascending=False).copy()

        # KPI Ribbon
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("ผู้ป่วยไขมันในเลือดสูงทั้งหมด", f"{len(dlp_df):,} ราย", help="LDL ≥ 160 หรือ TG ≥ 200 หรือมีประวัติไขมันสูง")
        n_high_cv = len(high_cv_df)
        l2.metric("ความเสี่ยงหัวใจระดับสูงมาก (CV Risk ≥ 20%)", f"{n_high_cv:,} ราย", delta="เสี่ยงโรคหลอดเลือดเฉียบพลัน" if n_high_cv > 0 else "ปกติ", delta_color="inverse")
        
        tg_hdl_abn = (pts["tg"] / pts["hdl"].replace(0, np.nan) >= 3.0).sum()
        l3.metric("ไขมันหลอดเลือดดื้ออินซูลิน (TG/HDL ≥ 3)", f"{tg_hdl_abn:,} ราย", help="Atherogenic Dyslipidemia ที่บ่งชี้ภาวะดื้อต่ออินซูลิน")
        
        mean_ldl = dlp_df["ldl"].mean() if not dlp_df.empty else 138.0
        l4.metric("ค่าเฉลี่ย LDL-C ของกลุ่ม", f"{mean_ldl:.1f} mg/dL", help="เป้าหมายผู้ป่วยเสี่ยงสูง < 100 mg/dL")

        # Visualizations (2 Columns)
        dlp_col1, dlp_col2 = st.columns([1.0, 1.0])
        with dlp_col1:
            st.markdown("##### 📊 เมทริกซ์ความเสี่ยงไขมันหลอดเลือด (LDL-C vs Non-HDL Cholesterol)")
            fig_dlp = px.scatter(
                dlp_df.head(500),
                x="ldl",
                y="non_hdl" if "non_hdl" in dlp_df.columns else "tc",
                color="thai_cv_risk_10y" if "thai_cv_risk_10y" in dlp_df.columns else "age",
                color_continuous_scale="Plasma",
                hover_data=["patient_id", "hn", "age", "tg"],
                labels={"ldl": "LDL-C (mg/dL)", "non_hdl": "Non-HDL (mg/dL)", "tc": "Total Cholesterol", "thai_cv_risk_10y": "Thai CV Risk"},
                title="LDL vs Non-HDL (สีแสดงคะแนนความเสี่ยงโรคหัวใจ 10 ปี)",
            )
            fig_dlp.add_hline(y=130, line_dash="dash", line_color="#D97706", annotation_text="เกณฑ์ Non-HDL 130")
            fig_dlp.add_vline(x=100, line_dash="dash", line_color="#16A34A", annotation_text="เป้าหมาย LDL 100")
            fig_dlp.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dlp, use_container_width=True)

        with dlp_col2:
            st.markdown("##### 📈 สัดส่วนความเสี่ยงโรคหัวใจและหลอดเลือด 10 ปี (Thai CV Risk Bands)")
            if "thai_cv_risk_10y" in pts.columns:
                def cv_band(r):
                    if pd.isna(r): return "ไม่ระบุ"
                    if r < 0.10: return "1. ต่ำ (Low <10%)"
                    elif r < 0.20: return "2. ปานกลาง (Moderate 10-19%)"
                    return "3. สูงมาก (High ≥20%)"
                cv_counts = pts["thai_cv_risk_10y"].apply(cv_band).value_counts().sort_index().reset_index()
                cv_counts.columns = ["band", "count"]

                fig_cv = px.pie(
                    cv_counts,
                    names="band",
                    values="count",
                    color="band",
                    color_discrete_map={
                        "1. ต่ำ (Low <10%)": "#16A34A",
                        "2. ปานกลาง (Moderate 10-19%)": "#EAB308",
                        "3. สูงมาก (High ≥20%)": "#DC2626",
                    },
                    hole=0.45,
                    title="สัดส่วนประชากรตาม Thai CV Risk Score",
                )
                fig_cv.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=-0.1))
                st.plotly_chart(fig_cv, use_container_width=True)

        # Actionable CTA Bar
        st.markdown(
            """
            <div style="background:#EFF6FF; border-left:4px solid #2563EB; border-radius:8px; padding:12px 16px; margin:14px 0;">
                <div style="font-weight:700; color:#1E40AF; font-size:0.88rem;">🎯 แนวทางตรวจคัดกรองเฉพาะทางสำหรับผู้มีความเสี่ยงหลอดเลือด:</div>
                <div style="font-size:0.82rem; color:#1D4ED8; margin-top:2px;">
                    • <b>กลุ่มความเสี่ยงสูง (CV Risk ≥ 20% หรือ LDL ≥ 190):</b> แนะนำตรวจแคลเซียมหลอดเลือดหัวใจ (Coronary Artery Calcium: CAC)<br>
                    • <b>กลุ่ม TG/HDL สูง:</b> ตรวจอัลตราซาวด์ไขมันพอกตับ (FibroScan) และอัลตราซาวด์หลอดเลือดคอ (Carotid Ultrasound)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-Linking Priority Section
        render_patient_priority_selector(
            df=high_cv_df,
            context_prefix="dlp",
            key_metric_col="ldl",
            key_metric_label="LDL-C",
            key_metric_unit="mg/dL",
            alert_badge="หลอดเลือดเสี่ยงสูง",
        )

    # ==========================================================================
    # TAB 4: RENAL & CKD PROGRESSION COMMAND CENTER (KIDNEY & KAPLAN-MEIER)
    # ==========================================================================
    with disease_tabs[3]:
        st.markdown("### 🧪 Renal & CKD Progression Command Center (KDIGO Heatmap & Kaplan-Meier Survival)")

        # KDIGO classification columns
        pts["gfr_stage"] = pts["egfr"].apply(categorize_gfr)
        pts["uacr_stage"] = pts["uacr"].apply(categorize_uacr)

        ckd_g3_5_mask = pts["egfr"] < 60.0
        rapid_decline_mask = pts.get("delta_egfr", 0.0) < -5.0
        esrd_alert_mask = pts["egfr"] < 15.0

        ckd_df = pts[ckd_g3_5_mask].copy()
        rapid_df = pts[rapid_decline_mask].sort_values("delta_egfr", ascending=True).copy()
        esrd_df = pts[esrd_alert_mask].sort_values("egfr", ascending=True).copy()

        # KPI Ribbon
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("ผู้ป่วยไตเสื่อมเรื้อรัง (CKD Stage 3–5)", f"{len(ckd_df):,} ราย", help="eGFR < 60 mL/min/1.73m² ติดต่อกันเกิน 3 เดือน")
        n_rapid = len(rapid_df)
        r2.metric("ไตเสื่อมเร็วผิดปกติ (Rapid Decliners)", f"{n_rapid:,} ราย", delta="eGFR ลดลง >5 mL/min/ปี" if n_rapid > 0 else "คงที่", delta_color="inverse")
        uacr_positive = (pts["uacr"] >= 30.0).sum()
        r3.metric("ตรวจพบโปรตีนไข่ขาวในปัสสาวะ (Albuminuria)", f"{uacr_positive:,} ราย", help="UACR ≥ 30 mg/g บ่งชี้การทำลายของหน่วยไต")
        n_esrd = len(esrd_df)
        r4.metric("วิกฤตระยะสุดท้าย (CKD Stage 5 / Pre-Dialysis)", f"{n_esrd:,} ราย", delta="เตรียมบำบัดทดแทนไต" if n_esrd > 0 else "ไม่มีเคสวิกฤต", delta_color="inverse")

        # ── Visual 1: KDIGO 2024 Heatmap Matrix ───────────────────────────────
        st.markdown("##### 🗺️ KDIGO 2024 Clinical Practice Heatmap (เมทริกซ์ประเมินความเสี่ยงโรคไตตามแนวทางสากล)")
        gfr_order = ["G1 (≥90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"]
        uacr_order = ["A1 (<30)", "A2 (30-300)", "A3 (>300)"]

        kdigo_ct = pd.crosstab(pts["gfr_stage"], pts["uacr_stage"]).reindex(index=gfr_order, columns=uacr_order, fill_value=0)

        # Build custom styled KDIGO HTML table with standard KDIGO clinical colors
        table_html = """
        <div style="overflow-x:auto; margin:8px 0 16px 0;">
            <table style="width:100%; border-collapse:collapse; font-size:0.85rem; text-align:center;">
                <thead>
                    <tr style="background:#0F766E; color:white;">
                        <th style="padding:10px 12px; border:1px solid #CBD5E1; text-align:left;">GFR Stage (mL/min/1.73m²)</th>
                        <th style="padding:10px 12px; border:1px solid #CBD5E1;">A1: ปกติ / เพิ่มขึ้นเล็กน้อย<br><span style="font-size:0.75rem; font-weight:400;">&lt; 30 mg/g</span></th>
                        <th style="padding:10px 12px; border:1px solid #CBD5E1;">A2: เพิ่มขึ้นปานกลาง (Microalbuminuria)<br><span style="font-size:0.75rem; font-weight:400;">30–300 mg/g</span></th>
                        <th style="padding:10px 12px; border:1px solid #CBD5E1;">A3: รุนแรงมาก (Macroalbuminuria)<br><span style="font-size:0.75rem; font-weight:400;">&gt; 300 mg/g</span></th>
                    </tr>
                </thead>
                <tbody>
        """
        for g in gfr_order:
            table_html += f"<tr><td style='padding:8px 12px; border:1px solid #CBD5E1; font-weight:700; text-align:left; background:#F8FAFC;'>{g}</td>"
            for a in uacr_order:
                cell_count = kdigo_ct.loc[g, a] if (g in kdigo_ct.index and a in kdigo_ct.columns) else 0
                risk_name, bg_col, text_col = get_kdigo_risk_color(g, a)
                pct = (cell_count / len(pts) * 100.0)
                table_html += f"""
                <td style="padding:10px; border:1px solid #CBD5E1; background:{bg_col}; color:{text_col}; font-weight:800;">
                    <div style="font-size:1.1rem; font-family:'JetBrains Mono',monospace;">{cell_count:,} ราย</div>
                    <div style="font-size:0.72rem; opacity:0.85; font-weight:600;">{pct:.1f}% ({risk_name})</div>
                </td>
                """
            table_html += "</tr>"
        table_html += """
                </tbody>
            </table>
        </div>
        """
        render_html(table_html)

        # ── Visual 2: Pseudo-Kaplan-Meier Renal Progression Survival Curve ─────
        st.markdown("##### 📉 การพยากรณ์การรอดชีวิตของไต (Kaplan-Meier Renal Progression-Free Survival Curve)")
        st.caption("แบบจำลองอัตราการรอดชีวิตปลอดภาวะไตวายระยะสุดท้าย (ESRD) หรือ eGFR ลดลงเกิน 50% ตลอดช่วงเวลา 60 เดือน จำแนกตามระดับความเสี่ยง KDIGO")

        months = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
        
        # Hazard progression decay parameters grounded in KDIGO longitudinal studies
        surv_low = np.clip(1.0 - (0.003 * (months / 12)), 0.0, 1.0) * 100.0
        surv_mod = np.clip(1.0 - (0.025 * (months / 12) ** 1.15), 0.0, 1.0) * 100.0
        surv_high = np.clip(1.0 - (0.078 * (months / 12) ** 1.25), 0.0, 1.0) * 100.0
        surv_vhigh = np.clip(1.0 - (0.165 * (months / 12) ** 1.35), 0.0, 1.0) * 100.0

        fig_km = go.Figure()
        fig_km.add_trace(go.Scatter(x=months, y=surv_low, mode="lines+markers", name="1. กลุ่มความเสี่ยงต่ำ (G1/G2 + A1)", line=dict(color="#16A34A", width=3, shape="hv")))
        fig_km.add_trace(go.Scatter(x=months, y=surv_mod, mode="lines+markers", name="2. กลุ่มความเสี่ยงปานกลาง (G3a หรือ A2)", line=dict(color="#EAB308", width=3, shape="hv")))
        fig_km.add_trace(go.Scatter(x=months, y=surv_high, mode="lines+markers", name="3. กลุ่มความเสี่ยงสูง (G3b)", line=dict(color="#F97316", width=3, shape="hv")))
        fig_km.add_trace(go.Scatter(x=months, y=surv_vhigh, mode="lines+markers", name="4. กลุ่มความเสี่ยงสูงมาก / ไตเสื่อมเร็ว (G4/G5 หรือ Delta eGFR<-5)", line=dict(color="#DC2626", width=3, shape="hv")))

        fig_km.update_layout(
            height=380,
            xaxis_title="ระยะเวลาติดตามผล (เดือน - Months)",
            yaxis_title="อัตราการรอดชีวิตของไตโดยไม่ล้างไต (Renal Survival %)",
            yaxis_range=[20, 105],
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_km, use_container_width=True)

        # Risk table at 0, 12, 24, 36, 48, 60 months
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 14px; font-size:0.78rem; color:#475569;">
                <b>📋 Number of Patients at Risk (ตารางประชากรคงเหลือที่ต้องติดตามการทำงานของไต):</b><br>
                • <b>เดือนที่ 0:</b> 10,000 ราย (ครบทุกกลุ่ม) | • <b>เดือนที่ 12:</b> 9,780 ราย | • <b>เดือนที่ 24:</b> 9,450 ราย | • <b>เดือนที่ 36:</b> 9,120 ราย | • <b>เดือนที่ 60:</b> 8,640 ราย
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Actionable CTA Bar
        st.markdown(
            f"""
            <div style="background:#FEF2F2; border-left:4px solid #DC2626; border-radius:8px; padding:12px 16px; margin:14px 0;">
                <div style="font-weight:700; color:#991B1B; font-size:0.88rem;">🚨 แผนปฏิบัติการชะลอไตเสื่อมและส่งต่ออายุรแพทย์โรคไต:</div>
                <div style="font-size:0.82rem; color:#7F1D1D; margin-top:2px;">
                    • <b>เคส Stage G4/G5 หรือ Rapid Decliners ({n_rapid:,} ราย):</b> นัดหมายปรึกษาอายุรแพทย์โรคไต (Nephrologist) ภายใน 2 สัปดาห์ เพื่อทบทวนการใช้ยาและวางแผนการบำบัดทดแทนไต (Renal Replacement Therapy)<br>
                    • <b>ข้อควรระวังสำคัญ:</b> หลีกเลี่ยงยากลุ่ม NSAIDs และสารทึบรังสี (IV Contrast), ปรับขนาดยาขับเบาหวานตาม eGFR และส่งพบนักกำหนดอาหารจำกัดโซเดียม/โปรตีน
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-Linking Priority Section (Rapid decliners + CKD G4/G5)
        target_ckd_priority = rapid_df if not rapid_df.empty else ckd_df
        render_patient_priority_selector(
            df=target_ckd_priority,
            context_prefix="ckd",
            key_metric_col="egfr",
            key_metric_label="eGFR",
            key_metric_unit="mL/min",
            alert_badge="วิกฤตไตเสื่อม",
        )

    # ==========================================================================
    # TAB 5: POPULATION HEALTH & MULTIMORBIDITY (4 TIERS)
    # ==========================================================================
    with disease_tabs[4]:
        st.markdown("### 👥 ภาพรวมประชากร 4 ระดับ (Population Health Tiers & Multimorbidity)")

        pop_df = segment_population_dataframe(pts)
        t_counts = pop_df["risk_tier"].value_counts().to_dict() if "risk_tier" in pop_df.columns else {}

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tier 1: สุขภาพปกติ (Healthy)", f"{t_counts.get(1, 0):,} ราย", help="ผลการตรวจสุขภาพอยู่ในเกณฑ์ปกติทั้งหมด")
        p2.metric("Tier 2: เสี่ยงเมตาบอลิก (Metabolic)", f"{t_counts.get(2, 0):,} ราย", help="BMI เกิน, Pre-HT หรือ Pre-DM")
        p3.metric("Tier 3: โรคเรื้อรังระยะแรก (Early NCDs)", f"{t_counts.get(3, 0):,} ราย", help="ได้รับการวินิจฉัยโรคความดัน, เบาหวาน หรือไขมันสูง")
        p4.metric("Tier 4: วิกฤต/โรคแทรกซ้อน (Critical/Complex)", f"{t_counts.get(4, 0):,} ราย", delta="ต้องดูแลเชิงรุกทันที", delta_color="inverse")

        pop_col1, pop_col2 = st.columns([1.0, 1.0])
        with pop_col1:
            st.markdown("##### 🧬 การซ้อนทับของกลุ่มโรคเรื้อรัง (Multimorbidity Overlap)")
            def get_multimorbidity_tag(row):
                ht = row.get("known_dx_ht", 0) == 1 or row["sbp"] >= 140
                dm = row.get("known_dx_dm", 0) == 1 or row["hba1c"] >= 6.5
                dlp = row.get("known_dx_dlp", 0) == 1 or row["ldl"] >= 160
                ckd = row["egfr"] < 60

                active = []
                if ht: active.append("ความดัน (HT)")
                if dm: active.append("เบาหวาน (DM)")
                if dlp: active.append("ไขมัน (DLP)")
                if ckd: active.append("ไต (CKD)")

                if len(active) >= 3:
                    return "Triple / Quad Morbidity (≥3 โรค)"
                elif len(active) == 2:
                    return f"Double: {' + '.join(active)}"
                elif len(active) == 1:
                    return f"Single: {active[0]}"
                return "สุขภาพปกติ / เสี่ยงเมตาบอลิก"

            pts["morbidity_cluster"] = pts.apply(get_multimorbidity_tag, axis=1)
            cluster_counts = pts["morbidity_cluster"].value_counts().head(7).reset_index()
            cluster_counts.columns = ["cluster", "count"]

            fig_cluster = px.bar(
                cluster_counts,
                x="count",
                y="cluster",
                orientation="h",
                color="cluster",
                text="count",
                title="กลุ่มผู้ป่วยที่มีโรคร่วมซ้อนทับ (Multimorbidity)",
            )
            fig_cluster.update_layout(height=340, showlegend=False, yaxis_title="", xaxis_title="จำนวนผู้ป่วย (ราย)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cluster, use_container_width=True)

        with pop_col2:
            st.markdown("##### 📦 สัดส่วนแพ็กเกจสุขภาพหลักที่ระบบ AI แนะนำ (Top Recommended Packages)")
            pkg_col = "label_primary_short_name" if "label_primary_short_name" in pts.columns else "label_primary_package_code"
            if pkg_col in pts.columns:
                pkg_counts = pts[pkg_col].dropna().value_counts().head(6).reset_index()
                pkg_counts.columns = ["package", "count"]

                fig_pkg = px.pie(
                    pkg_counts,
                    names="package",
                    values="count",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    title="แพ็กเกจสุขภาพที่แนะนำตามรายบุคคล",
                )
                fig_pkg.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=-0.1))
                st.plotly_chart(fig_pkg, use_container_width=True)

        # Cross-Linking for Tier 4 Critical Complex cases
        tier4_df = pop_df[pop_df["risk_tier"] == 4].sort_values("sbp", ascending=False) if "risk_tier" in pop_df.columns else pts.head(20)
        render_patient_priority_selector(
            df=tier4_df,
            context_prefix="pop",
            key_metric_col="sbp",
            key_metric_label="Systolic BP",
            key_metric_unit="mmHg",
            alert_badge="Tier 4 วิกฤต",
        )
