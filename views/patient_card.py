# -*- coding: utf-8 -*-
"""
View 2: Patient Insight Card & Modular Custom Package Builder.
Displays individualized patient vitals, 26 standardized lab biomarkers,
AI multi-task package recommendations, modular tailor-made package composer,
and clinical outreach talk tracks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic.patient_segmentation import segment_patient
from logic.package_matching_v2 import (
    predict_patient_packages_v2,
    MODULAR_TEST_CATALOG,
    ADDON_CATALOG,
)
from services.data_loader import resolve_patient_lab_snapshot, load_synthetic_dataset
from services.ml_engine import predict_patient_propensity


def render_patient_card_view(
    avail_df: Optional[pd.DataFrame] = None,
    summary_pts: Optional[pd.DataFrame] = None,
    dv: Optional[pd.DataFrame] = None,
    search_term: str = "",
    selected_pid_input: Optional[str] = None,
) -> None:
    """
    Renders the Patient Insight Card with an organized 3-tab layout:
    - Tab 1: ข้อมูลสุขภาพ & ผลตรวจ Lab (Health Profile & Labs)
    - Tab 2: แพ็กเกจที่แนะนำ & ปรับแต่งเฉพาะบุคคล (Package Recommendation & Bespoke Modular)
    - Tab 3: บทสนทนานำเสนอ & แผนปฏิบัติการ (Outreach Talk Track & Action Plan)
    """
    # ── 1. Resolve Active Patient ID ──────────────────────────────────────────
    snap_df, _, _ = load_synthetic_dataset()

    target_pid = str(selected_pid_input or "").strip()
    if not target_pid and "search_patient_id" in st.session_state:
        target_pid = str(st.session_state["search_patient_id"]).strip()

    if not target_pid and avail_df is not None and not avail_df.empty:
        target_pid = str(avail_df.iloc[0]["patient_id"]).strip()

    if not target_pid:
        target_pid = "6"  # Default sample patient (Renal + Hepatic)

    # ── 2. Top Header & Patient Selector ──────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0F766E 0%,#0D9488 100%); border-radius:14px; padding:20px 24px; color:white; margin-bottom:16px; box-shadow:0 4px 14px rgba(15,118,110,0.14);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <div style="font-size:0.75rem; font-weight:700; color:#CCFBF1; text-transform:uppercase; letter-spacing:0.06em;">
                        👤 Patient Insight Card & Bespoke Precision Care
                    </div>
                    <div style="font-size:1.55rem; font-weight:800; color:white; margin-top:2px;">
                        แฟ้มประวัติสุขภาพและการจัดชุดตรวจเฉพาะบุคคล
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick-Pick Sample Buttons ─────────────────────────────────────────────
    st.caption("⚡ เคสตัวอย่างสำหรับทดสอบระบบ (Quick-Pick Clinical Scenarios):")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("🧪 ID 6: ไตเสื่อม+ตับ (จัดเฉพาะบุคคล)", key="btn_qpick_6", use_container_width=True):
            st.session_state["search_patient_id"] = "6"
            target_pid = "6"
            st.rerun()
    with btn_col2:
        if st.button("🦋 ID 605: ไทรอยด์+โลหิตจาง (จัดเฉพาะบุคคล)", key="btn_qpick_605", use_container_width=True):
            st.session_state["search_patient_id"] = "605"
            target_pid = "605"
            st.rerun()
    with btn_col3:
        if st.button("🫀 ID 1: เมตาบอลิก/ความดัน", key="btn_qpick_1", use_container_width=True):
            st.session_state["search_patient_id"] = "1"
            target_pid = "1"
            st.rerun()
    with btn_col4:
        if st.button("🌱 ID 25: สุขภาพทั่วไป (Routine)", key="btn_qpick_25", use_container_width=True):
            st.session_state["search_patient_id"] = "25"
            target_pid = "25"
            st.rerun()

    # Search Bar
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        entered_pid = st.text_input(
            "🔍 ค้นหาด้วย Patient ID หรือ รหัส HN (เช่น 6, 605, HN6800006):",
            value=target_pid,
            key="pic_search_input",
        )
    with search_col2:
        st.write("")
        st.write("")
        if st.button("ค้นหา", key="btn_search_pid", type="primary", use_container_width=True):
            target_pid = entered_pid.strip()
            st.session_state["search_patient_id"] = target_pid
            st.rerun()

    sel_pid_str = target_pid.strip()

    # ── 3. Resolve Patient Data & Vitals ───────────────────────────────────────
    # Clinical parameters default
    age_val = 35.0
    sys_val = 120.0
    dia_bp_val = 80.0
    bmi_val = 22.0
    visits_val = 1.0
    gender_code = 0.5

    # Check summary_pts or dv if available
    if summary_pts is not None and not summary_pts.empty:
        if sel_pid_str in summary_pts.index:
            pt_row = summary_pts.loc[sel_pid_str]
            age_val = float(pt_row.get("age_at_visit", pt_row.get("age", age_val)))
            sys_val = float(pt_row.get("systolic", pt_row.get("systolic_bp", sys_val)))
            bmi_val = float(pt_row.get("bmi", bmi_val))
            visits_val = float(pt_row.get("visits", visits_val))
            gender_code = float(pt_row.get("gender_code", gender_code))

    # Detect Chronic Conditions
    _is_dm = False
    _is_ht = bool(sys_val >= 140.0)
    _is_dlp = False

    if dv is not None and not dv.empty and "patient_id" in dv.columns:
        p_dv = dv[dv["patient_id"].astype(str).str.strip() == sel_pid_str]
        if not p_dv.empty:
            diag_str = " ".join(p_dv.get("diagnosis_text", pd.Series("")).dropna().astype(str).str.lower())
            if re.search(r"เบาหวาน|diabetes|e10|e11|e12|e13|e14", diag_str):
                _is_dm = True
            if re.search(r"ความดัน|hypertension|i10|i11|i12|i13|i15", diag_str):
                _is_ht = True
            if re.search(r"ไขมัน|lipid|dyslipidemia|e78", diag_str):
                _is_dlp = True

    # Lab Snapshot resolution
    lab_snap = resolve_patient_lab_snapshot(
        sel_pid_str=sel_pid_str,
        age_val=age_val,
        gender_code=gender_code,
        bmi_val=bmi_val,
        sys_val=sys_val,
        dia_bp_val=dia_bp_val,
        is_dm=_is_dm,
        is_ht=_is_ht,
        is_dlp=_is_dlp,
        visits_val=visits_val,
    )

    age_val = float(lab_snap.get("age", age_val))
    bmi_val = float(lab_snap.get("bmi", bmi_val))
    sys_val = float(lab_snap.get("systolic_bp", sys_val))
    dia_bp_val = float(lab_snap.get("diastolic_bp", dia_bp_val))
    gender_str = str(lab_snap.get("gender", "ชาย" if gender_code <= 0.5 else "หญิง"))
    gender_code = 0.0 if gender_str == "ชาย" else 1.0

    # ── 4. Calculate Propensity Score & Package Recommendations ──────────────
    propensity_score, shap_explanation, base_val = predict_patient_propensity(
        age=age_val,
        systolic=sys_val,
        bmi=bmi_val,
        visits_12m=visits_val,
        gender_code=gender_code,
        is_dm=_is_dm,
        is_ht=_is_ht,
        is_dlp=_is_dlp,
    )

    v2_res = predict_patient_packages_v2(lab_snap, top_k=3)
    primary_pkg_name = v2_res.get("primary_short_name", "Essential")
    top3_packages = v2_res.get("top3_packages", [])
    pers_pkg = v2_res.get("personalized_package", {})
    needs_custom = bool(v2_res.get("needs_custom_package", False))

    # Priority Badging
    prio_color = "#10B981" if propensity_score >= 60 else ("#D97706" if propensity_score >= 40 else "#64748B")
    prio_badge = "🔥 High Priority (แนะนำโทรทันที)" if propensity_score >= 60 else ("⚡ Medium Priority" if propensity_score >= 40 else "🌱 Regular Care")

    # ── 5. Compact Hero Ribbon ────────────────────────────────────────────────
    hn_display = lab_snap.get("hn", f"HN68{int(sel_pid_str):05d}" if sel_pid_str.isdigit() else f"HN-{sel_pid_str}")
    st.markdown(
        f"""
        <div style="background:white; border:1px solid #E2E8F0; border-radius:14px; padding:16px 20px; margin-bottom:18px; box-shadow:0 2px 8px rgba(15,23,42,0.03); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.4rem; font-weight:800; color:#0F172A;">{hn_display}</span>
                    <span style="font-size:0.8rem; background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:6px; font-weight:600;">Patient ID: {sel_pid_str}</span>
                </div>
                <div style="font-size:0.85rem; color:#64748B; margin-top:3px;">
                    เพศ: <b>{gender_str}</b> · อายุ: <b>{age_val:.0f} ปี</b> · BMI: <b>{bmi_val:.1f}</b> · ความดัน: <b>{sys_val:.0f}/{dia_bp_val:.0f} mmHg</b>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#64748B; font-weight:600;">โอกาสตอบรับแพ็กเกจสุขภาพ</div>
                    <div style="font-size:1.65rem; font-weight:800; color:{prio_color}; line-height:1.1;">{propensity_score:.1f}%</div>
                </div>
                <span style="background:{prio_color}18; color:{prio_color}; border:1px solid {prio_color}35; font-size:0.8rem; font-weight:700; padding:6px 14px; border-radius:999px;">
                    {prio_badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 6. Three Clean Horizontal Tabs ────────────────────────────────────────
    pic_tabs = st.tabs([
        "📋 1. ข้อมูลสุขภาพ & ผลตรวจ Lab (Health Profile & Labs)",
        "📦 2. แพ็กเกจที่แนะนำ & ปรับแต่งเฉพาะบุคคล (Package Recommendation & Bespoke Modular)",
        "🗣️ 3. บทสนทนานำเสนอ & แผนปฏิบัติการ (Outreach Talk Track & Action Plan)",
    ])

    # ==========================================================================
    # TAB 1: ข้อมูลสุขภาพ & ผลตรวจ Lab
    # ==========================================================================
    with pic_tabs[0]:
        st.markdown("#### 🩺 สัญญาณชีพและผลการตรวจทางคลินิก (Clinical Biomarkers)")
        st.caption("รวบรวมตัวชี้วัดทางสรีรวิทยาและผลตรวจทางห้องปฏิบัติการ เพื่อประเมินความเสี่ยงระดับอวัยวะ")

        # 4 Key Vitals
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            bp_status = "🔴 สูง (Hypertension)" if sys_val >= 140 else ("🟡 เฝ้าระวัง" if sys_val >= 130 else "🟢 ปกติ")
            st.metric("ความดันโลหิต (BP)", f"{sys_val:.0f}/{dia_bp_val:.0f} mmHg", delta=bp_status, delta_color="inverse" if sys_val >= 140 else "normal")
        with v_col2:
            bmi_status = "🔴 อ้วน (Obesity)" if bmi_val >= 25 else ("🟡 ท้วม" if bmi_val >= 23 else "🟢 ปกติ")
            st.metric("ดัชนีมวลกาย (BMI)", f"{bmi_val:.1f} kg/m²", delta=bmi_status, delta_color="inverse" if bmi_val >= 25 else "normal")
        with v_col3:
            fpg = float(lab_snap.get("fpg_mg_dl", 92))
            dm_status = "🔴 เสี่ยงเบาหวาน" if fpg >= 126 else ("🟡 เสี่ยงปานกลาง" if fpg >= 100 else "🟢 ปกติ")
            st.metric("น้ำตาลสะสม (HbA1c / FPG)", f"{lab_snap.get('hba1c_pct', 5.4)}% ({fpg:.0f} mg/dL)", delta=dm_status, delta_color="inverse" if fpg >= 126 else "normal")
        with v_col4:
            egfr = float(lab_snap.get("egfr", 88))
            kidney_status = "🔴 ไตเสื่อม (CKD)" if egfr < 60 else ("🟡 เฝ้าระวัง" if egfr < 90 else "🟢 ปกติ")
            st.metric("การทำงานของไต (eGFR)", f"{egfr:.1f} mL/min", delta=kidney_status, delta_color="inverse" if egfr < 60 else "normal")

        st.markdown("---")

        # Lab Panel Table
        st.markdown("##### 🔬 ตารางวิเคราะห์ผลการตรวจทางห้องปฏิบัติการ (Comprehensive Lab Panels)")
        lab_panels = [
            ("กลุ่มระดับน้ำตาลและการเผาผลาญ (Glycemic Control)", [
                ("fpg_mg_dl", "ระดับน้ำตาลหลังอดอาหาร (FPG)", "mg/dL", 70.0, 99.0),
                ("hba1c_pct", "น้ำตาลสะสมในเม็ดเลือดแดง (HbA1c)", "%", 4.0, 5.6),
            ]),
            ("กลุ่มไขมันในเลือดและหลอดเลือด (Lipid Profile)", [
                ("cholesterol_mg_dl", "คอเลสเตอรอลรวม (Total Cholesterol)", "mg/dL", 100.0, 200.0),
                ("triglyceride_mg_dl", "ไตรกลีเซอไรด์ (Triglyceride)", "mg/dL", 30.0, 150.0),
                ("hdl_mg_dl", "ไขมันดี (HDL-C)", "mg/dL", 40.0, 80.0),
                ("ldl_mg_dl", "ไขมันเลว (LDL-C)", "mg/dL", 50.0, 100.0),
            ]),
            ("กลุ่มการทำงานของไตและเกลือแร่ (Renal Function)", [
                ("creatinine_mg_dl", "ครีเอตินิน (Creatinine)", "mg/dL", 0.6, 1.2),
                ("egfr", "อัตราการกรองของไต (eGFR)", "mL/min/1.73m²", 90.0, 120.0),
                ("uacr_mg_g", "อัลบูมินรั่วในปัสสาวะ (UACR)", "mg/g", 0.0, 30.0),
                ("uric_acid_mg_dl", "กรดยูริก (Uric Acid)", "mg/dL", 3.5, 7.2),
            ]),
            ("กลุ่มการทำงานของตับและการอักเสบ (Hepatic Function)", [
                ("alt_u_l", "เอนไซม์ตับ (ALT/SGPT)", "U/L", 0.0, 40.0),
                ("ast_u_l", "เอนไซม์ตับ (AST/SGOT)", "U/L", 0.0, 40.0),
            ]),
            ("กลุ่มโลหิตวิทยาและไทรอยด์ (Hematology & Thyroid)", [
                ("tsh_miu_l", "ฮอร์โมนกระตุ้นต่อมไทรอยด์ (TSH)", "mIU/L", 0.45, 4.5),
                ("hb_g_dl", "ความเข้มข้นของฮีโมโกลบิน (Hb)", "g/dL", 12.0, 16.0),
                ("hct_pct", "ความเข้มข้นของเม็ดเลือดแดงอัดแน่น (Hct)", "%", 36.0, 48.0),
            ]),
        ]

        lab_records = []
        for panel_name, items in lab_panels:
            for key, name_th, unit, r_min, r_max in items:
                val = lab_snap.get(key)
                if val is not None and pd.notna(val):
                    fval = float(val)
                    status_flag = "🔴 สูงกว่าเกณฑ์" if fval > r_max else ("🟡 ต่ำกว่าเกณฑ์" if fval < r_min else "🟢 ปกติ")
                    lab_records.append({
                        "หมวดการตรวจ": panel_name,
                        "รายการตรวจ": name_th,
                        "ผลตรวจ": f"{fval:.2f}",
                        "หน่วย": unit,
                        "ค่าอ้างอิงปกติ": f"{r_min} - {r_max}",
                        "สถานะ": status_flag,
                    })

        if lab_records:
            st.dataframe(pd.DataFrame(lab_records), use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 2: แพ็กเกจที่แนะนำ & ปรับแต่งเฉพาะบุคคล
    # ==========================================================================
    with pic_tabs[1]:
        st.markdown("#### 📦 ระบบจับคู่แพ็กเกจสุขภาพและการประกอบชุดตรวจเฉพาะบุคคล (Modular Tailor-Made)")
        st.caption("เปรียบเทียบระหว่างแพ็กเกจมาตรฐานที่ดีที่สุด กับ แพ็กเกจประกอบเฉพาะบุคคล (ครอบคลุม 100% ของความเสี่ยงที่ตรวจพบ)")

        comp_col1, comp_col2 = st.columns(2)

        # Standard Package
        top_std = top3_packages[0] if top3_packages else {}
        std_name = top_std.get("name", "แพ็กเกจมาตรฐาน")
        std_price = int(top_std.get("price", 3500))
        std_fit = float(top_std.get("fit_pct", 75.0))
        std_cov = pers_pkg.get("std_coverage_pct", 65)

        with comp_col1:
            st.markdown(
                f"""
                <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #0284C7; border-radius:14px; padding:18px; box-shadow:0 2px 8px rgba(15,23,42,0.03); height:100%;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.75rem; font-weight:700; color:#0369A1; text-transform:uppercase;">📦 แพ็กเกจมาตรฐานอันดับ 1</span>
                        <span style="background:#E0F2FE; color:#0284C7; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px;">ความแม่นยำ {std_fit:.1f}%</span>
                    </div>
                    <div style="font-size:1.15rem; font-weight:800; color:#0F172A; margin:6px 0;">{std_name}</div>
                    <div style="font-size:1.55rem; font-weight:800; color:#0284C7; font-family:JetBrains Mono,monospace;">฿{std_price:,}</div>
                    <div style="margin-top:10px; font-size:0.8rem; color:#64748B;">
                        ครอบคลุมปัญหาสุขภาพคนไข้: <b style="color:#D97706;">{std_cov}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Tailor-Made Package
        bespoke_price = int(pers_pkg.get("price", 4800))
        bespoke_full = int(pers_pkg.get("full_price", 5800))
        bespoke_disc = int(pers_pkg.get("discount_pct", 18))
        bespoke_reason = pers_pkg.get("gap_reason_th", "ประกอบชุดตรวจครอบคลุมทุกความเสี่ยงเฉพาะบุคคล")

        with comp_col2:
            st.markdown(
                f"""
                <div style="background:linear-gradient(180deg,#F0FDF4 0%,#FFFFFF 100%); border:2px solid #0F766E; border-top:4px solid #0F766E; border-radius:14px; padding:18px; box-shadow:0 4px 14px rgba(15,118,110,0.08); height:100%;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.75rem; font-weight:700; color:#0F766E; text-transform:uppercase;">🧬 แนะนำแพ็กเกจเฉพาะบุคคล (Tailor-Made)</span>
                        <span style="background:#CCFBF1; color:#0F766E; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px;">ครอบคลุม 100%</span>
                    </div>
                    <div style="font-size:1.15rem; font-weight:800; color:#0F172A; margin:6px 0;">{pers_pkg.get('name', 'แพ็กเกจเฉพาะบุคคล')}</div>
                    <div style="display:flex; align-items:baseline; gap:8px;">
                        <span style="font-size:1.55rem; font-weight:800; color:#0F766E; font-family:JetBrains Mono,monospace;">฿{bespoke_price:,}</span>
                        <span style="text-decoration:line-through; color:#94A3B8; font-size:0.85rem;">฿{bespoke_full:,}</span>
                        <span style="background:#FEF3C7; color:#B45309; font-size:0.75rem; font-weight:700; padding:2px 6px; border-radius:4px;">ลด {bespoke_disc}%</span>
                    </div>
                    <div style="margin-top:10px; font-size:0.78rem; color:#047857; line-height:1.4;">
                        💡 {bespoke_reason}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Interactive Modular Composer
        st.markdown("##### 🛠️ ปรับแต่งโมดูลการตรวจเฉพาะบุคคล (Interactive Modular Composer)")
        st.caption("สามารถเลือกเพิ่มหรือตัดรายการตรวจตามดุลยพินิจของแพทย์หรือความประสงค์ของผู้ป่วย โดยระบบจะคำนวณส่วนลดชุดเหมาจ่ายอัตโนมัติ")

        all_mod_options = {code: f"{spec['organ']}: {spec['name']} (฿{spec['price']:,})" for code, spec in MODULAR_TEST_CATALOG.items()}
        selected_mod_default = [m["module_code"] for m in pers_pkg.get("selected_modules", []) if m["module_code"] in all_mod_options]

        chosen_modules = st.multiselect(
            "เลือกโมดูลการตรวจที่ต้องการรวมในแพ็กเกจ:",
            options=list(all_mod_options.keys()),
            default=selected_mod_default,
            format_func=lambda x: all_mod_options.get(x, x),
            key=f"mod_selector_{sel_pid_str}",
        )

        if chosen_modules:
            total_raw = sum(int(MODULAR_TEST_CATALOG[c]["price"]) for c in chosen_modules if c in MODULAR_TEST_CATALOG)
            disc_rate = 18 if len(chosen_modules) >= 4 else 15
            net_custom_price = int(round((total_raw * (100 - disc_rate) / 100.0) / 10.0) * 10)
            save_amount = total_raw - net_custom_price

            m_c1, m_c2, m_c3 = st.columns(3)
            m_c1.metric("จำนวนโมดูลที่เลือก", f"{len(chosen_modules)} โมดูล")
            m_c2.metric("ราคารวมก่อนลด", f"฿{total_raw:,}")
            m_c3.metric("ราคาสุทธิ (ส่วนลดเหมาจ่าย)", f"฿{net_custom_price:,}", delta=f"-฿{save_amount:,} ({disc_rate}%)")

            st.write("")
            if st.button("✨ บันทึกและเลือกใช้แพ็กเกจเฉพาะบุคคลนี้ (Prescribe Custom Package)", key=f"btn_prescribe_{sel_pid_str}", type="primary", use_container_width=True):
                st.session_state["prescribed_pkg"] = {
                    "patient_id": sel_pid_str,
                    "package_name": f"เฉพาะบุคคล ({len(chosen_modules)} โมดูล)",
                    "price": net_custom_price,
                    "modules": chosen_modules,
                }
                st.success(f"✅ บันทึกแพ็กเกจเฉพาะบุคคลสำหรับผู้ป่วย {hn_display} สำเร็จแล้ว (฿{net_custom_price:,})")

    # ==========================================================================
    # TAB 3: บทสนทนานำเสนอ & แผนปฏิบัติการ
    # ==========================================================================
    with pic_tabs[2]:
        st.markdown("#### 🗣️ แนวทางการสื่อสารและสคริปต์พูดคุยกับผู้ป่วย (Outreach Talk Track)")
        st.caption("สคริปต์บทสนทนาที่เน้นความห่วงใย จริยธรรมทางการแพทย์ และประโยชน์สูงสุดของผู้รับบริการ")

        top_driver_name = pers_pkg.get("selected_modules", [{}])[1].get("organ", "การทำงานของอวัยวะสำคัญ") if len(pers_pkg.get("selected_modules", [])) > 1 else "สุขภาพทั่วไป"

        talk_track_text = (
            f"สวัสดีครับ/ค่ะ วันนี้จากผลการตรวจติดตามสุขภาพเบื้องต้นของคุณ {hn_display} "
            f"ทางทีมแพทย์สังเกตเห็นตัวชี้วัดด้าน {top_driver_name} ที่เริ่มมีความเสี่ยงหรือควรเฝ้าระวังเพิ่มเติม "
            f"เพื่อความอุ่นใจและป้องกันก่อนเกิดอาการ ทางคลินิกจึงขอแนะนำโปรแกรมตรวจสุขภาพเฉพาะบุคคล "
            f"ที่รวมรายการตรวจที่จำเป็นโดยตรง ช่วยให้คุณได้รับการดูแลที่ตรงจุดและคุ้มค่าที่สุดครับ/ค่ะ"
        )

        st.markdown(
            f"""
            <div style="background:#F0FDF4; border:1px solid #86EFAC; border-left:5px solid #16A34A; border-radius:10px; padding:18px 22px; margin-bottom:16px;">
                <div style="font-weight:800; color:#166534; font-size:1.02rem; margin-bottom:8px;">
                    💬 บทสนทนาแนะนำสำหรับเจ้าหน้าที่หน้างาน (Caring & Service-Oriented Talk Track)
                </div>
                <div style="font-size:0.95rem; color:#14532D; line-height:1.6; font-style:italic; background:rgba(255,255,255,0.8); padding:14px 18px; border-radius:8px;">
                    "{talk_track_text}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:14px 18px; font-size:0.82rem; color:#475569; line-height:1.55;">
                🛡️ <b>หลักจริยธรรมและการดูแลสุขภาพเชิงป้องกัน:</b><br>
                • หลีกเลี่ยงการสร้างความตื่นตระหนก หรือใช้คำว่า <i>"ระบบ AI สั่งให้ซื้อ"</i><br>
                • มุ่งเน้นการให้ความรู้ สร้างความเข้าใจในผล Lab และอธิบายถึงความสำคัญของการตรวจคัดกรองระยะแรกเริ่ม
            </div>
            """,
            unsafe_allow_html=True,
        )

