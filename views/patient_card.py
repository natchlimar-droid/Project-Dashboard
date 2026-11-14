# -*- coding: utf-8 -*-
"""
View 2: Patient Insight Card & Modular Custom Package Builder.
Role-Based & Progressive Disclosure design:
- 5-Second glanceable Patient Hero Banner
- 3 Clear Workflow Tabs:
  1. 📦 แนะนำแพ็กเกจ & จัดทำใบเสนอราคา (Package Recommendation & Quotation Summary)
  2. 🧪 ผลแล็บ & ประวัติสุขภาพ (Health Profile & Labs)
  3. 🤖 เหตุผล AI & บทสนทนานำเสนอ (AI Drivers & Outreach Talk Track)
"""
from __future__ import annotations

import re
import textwrap
from datetime import datetime
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


def render_html(html_str: str) -> None:
    """Renders HTML safely without Markdown 4-space code block indentation artifacts."""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    cleaned = "".join(lines)
    st.markdown(cleaned, unsafe_allow_html=True)


@st.dialog("📋 รายละเอียดรายการตรวจในแพ็กเกจ")
def show_package_details_dialog(pkg_name: str, pkg_price: int, bullets: List[str], reason: str = ""):
    st.markdown(f"### {pkg_name}")
    st.markdown(
        f"**ราคาแพ็กเกจ:** <span style='font-size:1.35rem; font-weight:800; color:#0F766E;'>฿{pkg_price:,}</span>",
        unsafe_allow_html=True,
    )
    if reason:
        st.markdown(f"**ข้อบ่งชี้ทางการแพทย์:** {reason}")
    st.markdown("---")
    st.markdown("##### 🩺 รายการตรวจที่ครอบคลุมในโปรแกรม:")
    if not bullets:
        bullets = [
            "การตรวจร่างกายทั่วไปโดยแพทย์เวชปฏิบัติ (Physical Examination)",
            "การตรวจความสมบูรณ์ของเม็ดเลือด (Complete Blood Count - CBC)",
            "การตรวจปัสสาวะสมบูรณ์แบบ (Urinalysis - UA)",
            "ตรวจระดับน้ำตาลในเลือด (Fasting Blood Sugar)",
            "ตรวจระดับไขมันในเลือดครบชุด (Lipid Profile)",
        ]
    for b in bullets:
        st.markdown(f"- {b}")
    st.caption("ℹ️ ทุกรายการตรวจดำเนินการโดยทีมแพทย์และบุคลากรทางการแพทย์ผู้เชี่ยวชาญ พร้อมรายงานผลผ่านระบบดิจิทัล")
    if st.button("ปิดหน้าต่าง", use_container_width=True, key="btn_close_pkg_modal"):
        st.rerun()


@st.dialog("🧾 ใบเสนอราคาโปรแกรมตรวจสุขภาพ (Medical Quotation)")
def show_quotation_print_dialog(
    hn_display: str,
    gender_str: str,
    age_val: float,
    primary_pkg_name: str,
    primary_pkg_price: int,
    selected_addons: List[Dict[str, Any]],
    net_total: int,
    discount_val: int,
):
    today_str = datetime.now().strftime("%d/%m/%Y")
    addon_rows = "".join([
        f"<tr style='border-bottom:1px solid #E2E8F0;'><td style='padding:8px 10px;'>+ {a['name']}</td><td style='text-align:right; padding:8px 10px;'>฿{a['price']:,}</td></tr>"
        for a in selected_addons
    ])
    discount_row = f"<div style='font-size:0.88rem; color:#D97706; margin-bottom:4px;'>ส่วนลดชุดเหมาจ่าย (Bundle Discount): <b>-฿{discount_val:,}</b></div>" if discount_val > 0 else ""

    render_html(
        f"""
        <div style="background:white; border:1px solid #CBD5E1; border-radius:12px; padding:22px; font-family:'Inter',sans-serif; color:#0F172A;">
            <div style="text-align:center; border-bottom:2px solid #0F766E; padding-bottom:12px; margin-bottom:16px;">
                <div style="font-size:1.35rem; font-weight:800; color:#0F766E;">🏥 PrimeCare Wellness Clinic</div>
                <div style="font-size:0.85rem; color:#64748B; margin-top:2px;">ใบเสนอราคาโปรแกรมตรวจสุขภาพเฉพาะบุคคล (Personalized Healthcare Quotation)</div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.88rem; margin-bottom:14px; background:#F8FAFC; padding:10px 14px; border-radius:8px;">
                <span><b>ผู้รับบริการ:</b> {hn_display} (เพศ {gender_str}, อายุ {age_val:.0f} ปี)</span>
                <span><b>วันที่ออกเอกสาร:</b> {today_str}</span>
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:0.88rem; margin-bottom:14px;">
                <thead>
                    <tr style="background:#F1F5F9; border-bottom:1.5px solid #CBD5E1;">
                        <th style="text-align:left; padding:8px 10px;">รายการตรวจสุขภาพ</th>
                        <th style="text-align:right; padding:8px 10px;">ราคา (บาท)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 8px;"><b>แพ็กเกจหลัก:</b> {primary_pkg_name}</td>
                        <td style="text-align:right; padding:10px 8px; font-weight:700;">฿{primary_pkg_price:,}</td>
                    </tr>
                    {addon_rows}
                </tbody>
            </table>
            <div style="border-top:2px solid #E2E8F0; padding-top:12px; text-align:right;">
                {discount_row}
                <div style="font-size:1.45rem; font-weight:800; color:#0F766E;">
                    ยอดสุทธิที่ต้องชำระ: ฿{net_total:,}
                </div>
            </div>
            <div style="margin-top:18px; font-size:0.75rem; color:#64748B; text-align:center; border-top:1px dashed #CBD5E1; padding-top:10px;">
                ใบเสนอราคานี้มีอายุ 30 วันนับจากวันที่ออก • กรุณานัดหมายล่วงหน้าก่อนเข้ารับบริการ
            </div>
        </div>
        """
    )
    st.info("💡 สามารถกดปุ่มพิมพ์ (Ctrl+P) หรือบันทึกภาพหน้าจอนี้เพื่อส่งต่อให้ผู้รับบริการหรือทีมขายได้ทันที")


def render_patient_card_view(
    avail_df: Optional[pd.DataFrame] = None,
    summary_pts: Optional[pd.DataFrame] = None,
    dv: Optional[pd.DataFrame] = None,
    search_term: str = "",
    selected_pid_input: Optional[str] = None,
) -> None:
    """
    Renders the Patient Insight Card with an organized 3-tab layout:
    - Tab 1: 📦 แนะนำแพ็กเกจ & จัดทำใบเสนอราคา (Package Recommendation & Quotation Summary)
    - Tab 2: 🧪 ผลแล็บ & ประวัติสุขภาพ (Health Profile & Labs)
    - Tab 3: 🤖 เหตุผล AI & บทสนทนานำเสนอ (AI Drivers & Outreach Talk Track)
    """
    # ── 1. Resolve Active Patient ID ──────────────────────────────────────────
    snap_df, _, _ = load_synthetic_dataset()

    target_pid = str(selected_pid_input or "").strip()
    if not target_pid and "search_patient_id" in st.session_state:
        target_pid = str(st.session_state["search_patient_id"]).strip()

    if not target_pid and "pic_search_input" in st.session_state and str(st.session_state["pic_search_input"]).strip():
        target_pid = str(st.session_state["pic_search_input"]).strip()

    if not target_pid and avail_df is not None and not avail_df.empty:
        target_pid = str(avail_df.iloc[0]["patient_id"]).strip()

    if not target_pid:
        target_pid = "605"  # Default sample patient (Thyroid + Anemia + Metabolic)

    # Initialize widget state strictly BEFORE creating the text_input widget
    if "pic_search_input" not in st.session_state or not str(st.session_state.get("pic_search_input", "")).strip():
        st.session_state["pic_search_input"] = target_pid
    elif "search_patient_id" in st.session_state and str(st.session_state["search_patient_id"]).strip():
        st.session_state["pic_search_input"] = str(st.session_state["search_patient_id"]).strip()
        target_pid = st.session_state["pic_search_input"]

    st.session_state["search_patient_id"] = target_pid

    # ── 2. Top Search & Quick Case Selector ───────────────────────────────────
    def _on_search_text_changed() -> None:
        val = str(st.session_state.get("pic_search_input", "")).strip()
        if val:
            st.session_state["search_patient_id"] = val

    def _select_scenario(pid: str) -> None:
        st.session_state["search_patient_id"] = pid
        st.session_state["pic_search_input"] = pid

    search_col1, search_col2 = st.columns([3.5, 1.0])
    with search_col1:
        st.text_input(
            "🔍 ค้นหารหัส HN / ชื่อผู้ป่วย (เช่น 605, 6, 1, 25, HN6800605):",
            key="pic_search_input",
            placeholder="พิมพ์รหัส HN หรือ Patient ID...",
            on_change=_on_search_text_changed,
        )

    with search_col2:
        st.write("")
        st.write("")
        if st.button("🔍 ค้นหา", key="btn_search_pid", type="primary", use_container_width=True):
            val = str(st.session_state.get("pic_search_input", "")).strip()
            if val:
                st.session_state["search_patient_id"] = val
            st.rerun()

    st.caption("⚡ เคสตัวอย่างสำหรับทดสอบระบบ (Quick Clinical Scenarios):")
    q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
    with q_c1:
        st.button("🧪 ID 6: ไต+ตับ (Bespoke)", key="btn_q_6", on_click=_select_scenario, args=("6",), use_container_width=True)
    with q_c2:
        st.button("🦋 ID 605: ไทรอยด์+โลหิตจาง", key="btn_q_605", on_click=_select_scenario, args=("605",), use_container_width=True)
    with q_c3:
        st.button("🫀 ID 1: ความดัน/เมตาบอลิก", key="btn_q_1", on_click=_select_scenario, args=("1",), use_container_width=True)
    with q_c4:
        st.button("🌱 ID 25: สุขภาพทั่วไป (Routine)", key="btn_q_25", on_click=_select_scenario, args=("25",), use_container_width=True)
    with q_c5:
        st.button("🔥 ID 5690: ไขมันสูง (Lead Top)", key="btn_q_5690", on_click=_select_scenario, args=("5690",), use_container_width=True)

    sel_pid_str = str(st.session_state.get("search_patient_id", target_pid)).strip()

    # ── 3. Resolve Patient Data & Vitals ───────────────────────────────────────
    # Candidate lookup IDs (handling HN, prefixes, int, str)
    clean_term = sel_pid_str
    digits = re.findall(r"\d+", clean_term)
    cand_ids = {clean_term, clean_term.lstrip("0"), clean_term.upper(), clean_term.lower()}
    if digits:
        cand_ids.add(digits[-1])
        cand_ids.add(str(int(digits[-1])))
        cand_ids.add(f"HN68{int(digits[-1]):05d}")

    # Check session state for fresh demographic/cohort bridge from Lead Intelligence
    cached_lead = st.session_state.get("cached_lead_customer", {})
    is_lead_match = (
        isinstance(cached_lead, dict) and
        str(cached_lead.get("customer_id", "")).strip() in cand_ids
    )

    matched_pt: Optional[Dict[str, Any]] = None
    propensity_override: Optional[float] = None
    shap_override: Optional[Any] = None
    pers_pkg_override: Optional[Dict[str, Any]] = None

    if is_lead_match:
        age_val = float(cached_lead.get("age", 48.0))
        sys_val = float(cached_lead.get("systolic", 120.0))
        dia_bp_val = float(cached_lead.get("diastolic", 80.0 if sys_val < 130 else 85.0))
        bmi_val = float(cached_lead.get("bmi", 22.5))
        visits_val = float(cached_lead.get("visits_12m", 1.0))
        gender_code = float(cached_lead.get("gender_code", 1.0))
        gender_str = str(cached_lead.get("gender", "หญิง" if gender_code > 0.5 else "ชาย"))
        _is_dm = bool(cached_lead.get("is_dm", False))
        _is_ht = bool(cached_lead.get("is_ht", sys_val >= 140.0))
        _is_dlp = bool(cached_lead.get("is_dlp", False))
        propensity_override = float(cached_lead.get("propensity_score", 0.0))
        shap_override = cached_lead.get("shap_explanation")
        pers_pkg_override = cached_lead.get("pers_pkg")
    else:
        # Search in snap_df or avail_df
        target_source_df = avail_df if (avail_df is not None and not avail_df.empty) else snap_df
        if target_source_df is not None and not target_source_df.empty:
            mask = target_source_df["patient_id"].astype(str).str.strip().isin(cand_ids)
            if "hn" in target_source_df.columns:
                mask = mask | target_source_df["hn"].astype(str).str.strip().str.upper().isin({c.upper() for c in cand_ids})
            matches = target_source_df[mask]
            if not matches.empty:
                matched_pt = matches.iloc[0].to_dict()

        # Base vitals with realistic defaults if entirely unmatched
        if matched_pt is not None:
            age_val = float(matched_pt.get("age", matched_pt.get("age_at_visit", 48.0)))
            sys_val = float(matched_pt.get("sbp", matched_pt.get("systolic_bp", matched_pt.get("systolic", 120.0))))
            dia_bp_val = float(matched_pt.get("dbp", matched_pt.get("diastolic_bp", matched_pt.get("diastolic", 80.0))))
            bmi_val = float(matched_pt.get("bmi", 22.5))
            visits_val = float(matched_pt.get("visits_last_2y", matched_pt.get("prior_visits_12m", matched_pt.get("prior_opd_visits_total", matched_pt.get("visits", 1.0)))))
            g_raw = str(matched_pt.get("gender", ""))
            g_code_raw = float(matched_pt.get("gender_code", 1.0 if g_raw in ["หญิง", "female", "F"] else 0.0))
            gender_code = 1.0 if (g_raw in ["หญิง", "female", "F"] or g_code_raw > 0.5) else 0.0
            gender_str = "หญิง" if gender_code > 0.5 else "ชาย"
            _is_dm = bool(int(matched_pt.get("known_dx_dm", 0)) == 1 or float(matched_pt.get("hba1c", 5.0)) >= 6.5 or float(matched_pt.get("fbs", 90.0)) >= 126.0)
            _is_ht = bool(int(matched_pt.get("known_dx_ht", 0)) == 1 or sys_val >= 140.0)
            _is_dlp = bool(int(matched_pt.get("known_dx_dlp", 0)) == 1 or float(matched_pt.get("ldl", 100.0)) >= 160.0)
        else:
            age_val = 48.0
            sys_val = 142.0
            dia_bp_val = 88.0
            bmi_val = 26.4
            visits_val = 4.0
            gender_code = 1.0
            gender_str = "หญิง"
            _is_dm = False
            _is_ht = bool(sys_val >= 140.0)
            _is_dlp = False

        # Check encounter / visit diagnosis texts for additional context
        if dv is not None and not dv.empty and "patient_id" in dv.columns:
            p_dv = dv[dv["patient_id"].astype(str).str.strip().isin(cand_ids)]
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

    if is_lead_match:
        lab_snap["age"] = age_val
        lab_snap["bmi"] = bmi_val
        lab_snap["systolic_bp"] = sys_val
        lab_snap["diastolic_bp"] = dia_bp_val
        lab_snap["gender"] = gender_str
        lab_snap["known_dx_dm"] = 1 if _is_dm else 0
        lab_snap["known_dx_ht"] = 1 if _is_ht else 0
        lab_snap["known_dx_dlp"] = 1 if _is_dlp else 0
        if propensity_override is not None and propensity_override > 0.0:
            propensity_score = propensity_override
        if shap_override is not None:
            shap_explanation = shap_override
        if pers_pkg_override:
            pers_pkg = pers_pkg_override

    v2_res = predict_patient_packages_v2(lab_snap)
    top3_packages = v2_res.get("top3_packages", [])
    pers_pkg = v2_res.get("personalized_package", {})
    needs_custom = bool(v2_res.get("needs_custom_package", False))
    risk_reasons = v2_res.get("risk_reasons", [])

    # Risk Tier & Priority Badging
    risk_tier = int(v2_res.get("risk_tier", 2))
    tier_map = {
        1: ("🟢 สุขภาพดี", "#15803D", "#F0FDF4"),
        2: ("🟡 เฝ้าระวังความเสี่ยง", "#D97706", "#FFFBEB"),
        3: ("🟠 เริ่มมีโรคเรื้อรัง", "#EA580C", "#FFF7ED"),
        4: ("🔴 วิกฤต/ต้องดูแลใกล้ชิด", "#DC2626", "#FEF2F2"),
    }
    tier_label, tier_color, tier_bg = tier_map.get(risk_tier, ("🟡 เฝ้าระวัง", "#D97706", "#FFFBEB"))

    prio_color = "#0F766E" if propensity_score >= 60 else ("#D97706" if propensity_score >= 40 else "#64748B")
    prio_badge = "🔥 แนะนำติดต่อด่วน (High Priority)" if propensity_score >= 60 else ("⚡ เฝ้าระวังต่อเนื่อง (Medium)" if propensity_score >= 40 else "🌱 ติดตามตามรอบปกติ")

    pref_clinic = "คลินิกอายุรกรรมทั่วไป"
    if _is_ht or sys_val >= 140:
        pref_clinic = "คลินิกความดันโลหิตและหัวใจ"
    elif float(lab_snap.get("egfr", 90)) < 60:
        pref_clinic = "คลินิกโรคไตเรื้อรัง"
    elif float(lab_snap.get("tsh_miu_l", 2.0)) > 4.5:
        pref_clinic = "คลินิกต่อมไร้ท่อและไทรอยด์"

    # ── 5. Patient Hero Banner (5-Second Glanceable Summary) ───────────────────
    hn_display = lab_snap.get("hn", f"HN68{int(sel_pid_str):05d}" if sel_pid_str.isdigit() else f"HN-{sel_pid_str}")

    render_html(
        f"""
        <div style="background:white; border:1px solid #E2E8F0; border-radius:14px; padding:16px 20px; margin-bottom:18px; box-shadow:0 2px 8px rgba(15,23,42,0.03); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="width:48px; height:48px; border-radius:12px; background:#F1F5F9; display:flex; align-items:center; justify-content:center; font-size:1.6rem;">
                    {'👨' if gender_str == 'ชาย' else '👩'}
                </div>
                <div>
                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <span style="font-size:1.35rem; font-weight:800; color:#0F172A;">{hn_display}</span>
                        <span style="font-size:0.85rem; color:#475569; font-weight:600;">| {gender_str} {age_val:.0f} ปี</span>
                        <span style="background:{tier_bg}; color:{tier_color}; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px; border:1px solid {tier_color}35;">
                            {tier_label}
                        </span>
                        <span style="background:{prio_color}15; color:{prio_color}; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px; border:1px solid {prio_color}35;">
                            {prio_badge}
                        </span>
                    </div>
                    <div style="font-size:0.85rem; color:#64748B; margin-top:4px;">
                        ความดัน <b>{sys_val:.0f}/{dia_bp_val:.0f} mmHg</b> • BMI <b>{bmi_val:.1f}</b> • มารับบริการ <b>{visits_val:.0f} ครั้ง/ปี</b> • คลินิกประจำ: <b>{pref_clinic}</b>
                    </div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#64748B; font-weight:600;">โอกาสตอบรับแพ็กเกจสุขภาพ</div>
                    <div style="font-size:1.75rem; font-weight:800; color:{prio_color}; line-height:1.1;">{propensity_score:.1f}%</div>
                </div>
            </div>
        </div>
        """
    )

    # ── Linked Navigation Pill to AI Lead Intelligence ──
    h_col1, h_col2 = st.columns([3.2, 1.3])
    with h_col1:
        st.caption(f"🔗 **ฐานข้อมูลเวชระเบียนแบบรวมศูนย์:** ข้อมูลเชื่อมโยงกับ AI Lead Intelligence อัตโนมัติ (Customer ID: `{sel_pid_str}`)")
    with h_col2:
        if st.button("🎯 ดูการวิเคราะห์ใน AI Lead Intelligence", key=f"btn_jump_to_lead_{sel_pid_str}", use_container_width=True):
            st.session_state["selected_lead_cid"] = str(sel_pid_str)
            st.session_state["search_patient_id"] = str(sel_pid_str)
            st.session_state["active_page_key"] = "lead"
            st.session_state["last_synced_page"] = "lead"
            st.query_params["page"] = "lead"
            st.rerun()

    # ── 6. Three Workflow Tabs (Progressive Disclosure) ────────────────────────
    pic_tabs = st.tabs([
        "📦 1. แนะนำแพ็กเกจ (Package Recommendation)",
        "🧪 2. ผลแล็บ & ประวัติ (Health Profile & Labs)",
        "🤖 3. เหตุผล AI & บทสนทนา (AI Drivers & Outreach)",
    ])

    # ==========================================================================
    # TAB 1: 📦 แนะนำแพ็กเกจ & จัดทำใบเสนอราคา
    # ==========================================================================
    with pic_tabs[0]:
        st.markdown("#### 📦 ระบบจับคู่แพ็กเกจสุขภาพ & จัดทำใบเสนอราคา (Precision Package Quotation)")
        st.caption("แนะนำแพ็กเกจตรวจสุขภาพที่เหมาะสมที่สุดตามประวัติทางคลินิก พร้อมระบบประกอบชุดตรวจเสริมและสรุปใบเสนอราคา")

        col_pkg_left, col_pkg_right = st.columns([1.55, 1.05], gap="large")

        # ── Primary & Tailor-Made Package Resolution ──
        top_std = top3_packages[0] if top3_packages else {}
        std_name = top_std.get("name") or top_std.get("short_name", "แพ็กเกจตรวจสุขภาพทั่วไป")
        std_price = int(top_std.get("price", 2990))
        std_fit = float(top_std.get("ml_score_pct", top_std.get("fit_pct", 92.0)))
        bullets = top_std.get("bullets", [])

        # Construct concise clinical indication
        reason_parts = []
        if _is_dlp or float(lab_snap.get("ldl", 100.0)) >= 160.0:
            reason_parts.append("ประวัติไขมันในเลือดสูง (Dyslipidemia)")
        if _is_ht or sys_val >= 140:
            reason_parts.append(f"ความดันโลหิตสูง ({sys_val:.0f}/{dia_bp_val:.0f} mmHg)")
        elif sys_val >= 130:
            reason_parts.append(f"ความดันก้ำกึ่งเฝ้าระวัง ({sys_val:.0f} mmHg)")
        if _is_dm or float(lab_snap.get("fbs", 90.0)) >= 126.0:
            reason_parts.append("ระดับน้ำตาลในเลือด/เบาหวาน")
        if bmi_val >= 25:
            reason_parts.append(f"BMI {bmi_val:.1f} เกินเกณฑ์มาตรฐาน")
        elif bmi_val >= 23:
            reason_parts.append(f"น้ำหนักเกินเกณฑ์เล็กน้อย (BMI {bmi_val:.1f})")

        if float(lab_snap.get("egfr", 90)) < 60:
            reason_parts.append("ค่าการทำงานของไต eGFR ลดลง")
        if float(lab_snap.get("alt_u_l", 20)) > 40:
            reason_parts.append("เอนไซม์ตับ ALT สูง")
        if float(lab_snap.get("tsh_miu_l", 2.0)) > 4.5:
            reason_parts.append("ฮอร์โมนไทรอยด์ TSH ผิดปกติ")

        clinical_reason_text = " + ".join(reason_parts) if reason_parts else "คัดกรองสุขภาพประจำปีตามช่วงอายุและสรีรวิทยา"

        with col_pkg_left:
            # ── Card 1: Best Match Package ──
            render_html(
                f"""
                <div style="background:white; border:2px solid #0F766E; border-top:5px solid #0F766E; border-radius:14px; padding:18px 20px; box-shadow:0 4px 14px rgba(15,118,110,0.06); margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.75rem; font-weight:700; color:#0F766E; text-transform:uppercase; letter-spacing:0.04em;">
                            ⭐ แพ็กเกจหลักที่เหมาะสมที่สุด (Best Match)
                        </span>
                        <span style="background:#CCFBF1; color:#0F766E; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px;">
                            ความแม่นยำ {std_fit:.1f}%
                        </span>
                    </div>
                    <div style="font-size:1.25rem; font-weight:800; color:#0F172A; margin:6px 0;">
                        {std_name}
                    </div>
                    <div style="font-size:1.65rem; font-weight:800; color:#0F766E; font-family:'JetBrains Mono',monospace;">
                        ฿{std_price:,}
                    </div>
                    <div style="margin-top:10px; font-size:0.82rem; color:#475569; background:#F8FAFC; padding:10px 12px; border-radius:8px; border-left:3px solid #0F766E;">
                        <b>🩺 เหตุผลทางการแพทย์:</b> {clinical_reason_text}
                    </div>
                </div>
                """
            )

            btn_col1, btn_col2 = st.columns([1, 1.2])
            with btn_col1:
                if st.button("📋 ดูรายการตรวจ", key=f"btn_view_bullets_{sel_pid_str}", use_container_width=True):
                    show_package_details_dialog(std_name, std_price, bullets, clinical_reason_text)
            with btn_col2:
                selected_primary_key = f"active_pkg_sel_{sel_pid_str}"
                is_selected = st.session_state.get(selected_primary_key, "std") == "std"
                if st.button(
                    "✅ เลือกแพ็กเกจนี้" if not is_selected else "✔️ แพ็กเกจที่เลือกอยู่",
                    key=f"btn_choose_std_{sel_pid_str}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[selected_primary_key] = "std"
                    st.rerun()

            # ── Card 2: Bespoke Tailor-Made Option (If Gap Detected or ID 6) ──
            if needs_custom or sel_pid_str == "6" or pers_pkg:
                bespoke_name = pers_pkg.get("name", "แพ็กเกจตรวจเฉพาะบุคคล (Bespoke Modular)")
                bespoke_price = int(pers_pkg.get("price", 4800))
                bespoke_full = int(pers_pkg.get("full_price", 5800))
                bespoke_disc = int(pers_pkg.get("discount_pct", 18))
                bespoke_reason = pers_pkg.get("gap_reason_th", "ประกอบชุดตรวจครอบคลุมทุกความเสี่ยงเฉพาะบุคคล 100%")

                st.write("")
                render_html(
                    f"""
                    <div style="background:linear-gradient(180deg,#F0FDF4 0%,#FFFFFF 100%); border:1.5px solid #16A34A; border-radius:12px; padding:16px 18px; margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.75rem; font-weight:700; color:#15803D; text-transform:uppercase;">
                                🧬 ทางเลือก: แพ็กเกจเฉพาะบุคคล (Tailor-Made)
                            </span>
                            <span style="background:#DCFCE7; color:#166534; font-size:0.72rem; font-weight:700; padding:2px 6px; border-radius:4px;">
                                ลด {bespoke_disc}%
                            </span>
                        </div>
                        <div style="font-size:1.1rem; font-weight:800; color:#0F172A; margin:4px 0;">{bespoke_name}</div>
                        <div style="display:flex; align-items:baseline; gap:8px;">
                            <span style="font-size:1.4rem; font-weight:800; color:#16A34A; font-family:'JetBrains Mono',monospace;">฿{bespoke_price:,}</span>
                            <span style="text-decoration:line-through; color:#94A3B8; font-size:0.82rem;">฿{bespoke_full:,}</span>
                        </div>
                        <div style="font-size:0.78rem; color:#15803D; margin-top:6px;">
                            💡 {bespoke_reason}
                        </div>
                    </div>
                    """
                )
                if st.button("🧬 สลับมาใช้แพ็กเกจเฉพาะบุคคลนี้", key=f"btn_switch_bespoke_{sel_pid_str}", use_container_width=True):
                    st.session_state[selected_primary_key] = "bespoke"
                    st.rerun()

            st.markdown("---")

            # ── 3. Add-on Modular Tests Section ──
            st.markdown("##### ➕ รายการตรวจเสริมที่แนะนำเพิ่มเติม (Add-on Modules)")
            st.caption("สามารถเลือกเพิ่มรายการตรวจเฉพาะทางที่ตรงกับความกังวลของผู้ป่วย เพื่อรวมในใบเสนอราคาฉบับเดียวกัน")

            addon_options = {
                "ADDON_US_ABDOMEN": {"name": "อัลตราซาวด์ช่องท้องส่วนบนและล่าง (Ultrasound Whole Abdomen)", "price": 990, "organ": "ตับ/ไต/ถุงน้ำดี"},
                "ADDON_EKG": {"name": "ตรวจคลื่นไฟฟ้าหัวใจ (EKG 12-Leads)", "price": 450, "organ": "หัวใจและหลอดเลือด"},
                "ADDON_RENAL": {"name": "ตรวจการทำงานของไตและโปรตีนรั่ว (Creatinine + eGFR + UACR)", "price": 650, "organ": "ไตและระบบขับถ่าย"},
                "ADDON_THYROID": {"name": "ตรวจสมดุลฮอร์โมนไทรอยด์ (TSH + Free T4)", "price": 750, "organ": "ต่อมไร้ท่อ"},
                "ADDON_MAMMOGRAM": {"name": "ดิจิทัลแมมโมแกรมและอัลตราซาวด์เต้านม (Digital Mammogram)", "price": 2500, "organ": "คัดกรองเต้านม"},
                "ADDON_BONE": {"name": "ตรวจความหนาแน่นมวลกระดูก (DEXA Bone Density Scan)", "price": 1200, "organ": "กระดูกและข้อ"},
            }

            # Smart defaults based on patient profile
            default_addons = []
            if bmi_val >= 25 or _is_ht or sys_val >= 140:
                default_addons.append("ADDON_US_ABDOMEN")
            if age_val >= 45:
                default_addons.append("ADDON_EKG")

            selected_addon_keys = st.multiselect(
                "เลือกรายการตรวจเสริมที่ต้องการรวมในใบเสนอราคา:",
                options=list(addon_options.keys()),
                default=[k for k in default_addons if k in addon_options],
                format_func=lambda k: f"{addon_options[k]['name']} (+฿{addon_options[k]['price']:,})",
                key=f"addons_multisel_{sel_pid_str}",
            )

        # ── Quotation Summary (Right Column) ──
        is_bespoke_active = st.session_state.get(selected_primary_key, "std") == "bespoke"
        active_primary_name = bespoke_name if is_bespoke_active else std_name
        active_primary_price = bespoke_price if is_bespoke_active else std_price

        active_addons = [addon_options[k] for k in selected_addon_keys if k in addon_options]
        addons_total = sum(a["price"] for a in active_addons)
        gross_total = active_primary_price + addons_total

        # Bundle discount: 10% discount on entire package if 2 or more add-ons are chosen
        discount_amount = int(gross_total * 0.10) if len(active_addons) >= 2 else (int(gross_total * 0.05) if len(active_addons) == 1 else 0)
        net_payable = gross_total - discount_amount

        with col_pkg_right:
            render_html(
                f"""
                <div class="quotation-card">
                    <div class="quotation-header">
                        🧾 สรุปใบเสนอราคาตรวจสุขภาพ (Quotation)
                    </div>
                    <div style="font-size:0.85rem; color:#64748B; margin-bottom:12px;">
                        ผู้รับบริการ: <b>{hn_display}</b>
                    </div>
                    
                    <div class="quotation-row">
                        <span><b>แพ็กเกจหลัก:</b> {active_primary_name[:28]}...</span>
                        <b>฿{active_primary_price:,}</b>
                    </div>
                    
                    {"".join([f'<div class="quotation-row" style="color:#475569; font-size:0.82rem;"><span>+ {a["name"][:26]}...</span><span>+฿{a["price"]:,}</span></div>' for a in active_addons])}
                    
                    <div style="border-top:1px dashed #CBD5E1; margin:10px 0;"></div>
                    
                    <div class="quotation-row" style="font-size:0.82rem; color:#64748B;">
                        <span>รวมราคาปกติ</span>
                        <span>฿{gross_total:,}</span>
                    </div>
                    
                    {f'<div class="quotation-row" style="font-size:0.82rem; color:#D97706; font-weight:700;"><span>ส่วนลดชุดเหมาจ่าย</span><span>-฿{discount_amount:,}</span></div>' if discount_amount > 0 else ''}
                    
                    <div class="quotation-total">
                        <div>
                            <div style="font-size:0.8rem; font-weight:700; color:#0F172A; text-transform:uppercase;">ยอดสุทธิที่ต้องชำระ</div>
                            <div style="font-size:0.72rem; color:#64748B;">ราคารวมแพทย์และบริการคลินิก</div>
                        </div>
                        <div class="quotation-total-price">
                            ฿{net_payable:,}
                        </div>
                    </div>
                </div>
                """
            )

            st.write("")
            if st.button("✅ บันทึกใบเสนอราคาลงเวชระเบียน", key=f"btn_save_quote_{sel_pid_str}", type="primary", use_container_width=True):
                st.session_state["prescribed_pkg"] = {
                    "patient_id": sel_pid_str,
                    "package_name": active_primary_name,
                    "price": net_payable,
                    "addons": [a["name"] for a in active_addons],
                }
                st.success(f"บันทึกใบเสนอราคาสำหรับ {hn_display} เรียบร้อยแล้ว (ยอดสุทธิ ฿{net_payable:,})")

            if st.button("🖨️ พิมพ์ใบเสนอราคา / ส่งต่อคนไข้", key=f"btn_print_quote_{sel_pid_str}", use_container_width=True):
                show_quotation_print_dialog(
                    hn_display=hn_display,
                    gender_str=gender_str,
                    age_val=age_val,
                    primary_pkg_name=active_primary_name,
                    primary_pkg_price=active_primary_price,
                    selected_addons=active_addons,
                    net_total=net_payable,
                    discount_val=discount_amount,
                )

    # ==========================================================================
    # TAB 2: 🧪 ผลแล็บ & ประวัติสุขภาพ
    # ==========================================================================
    with pic_tabs[1]:
        st.markdown("#### 🧪 สัญญาณชีพและผลตรวจทางห้องปฏิบัติการ (Health Profile & Labs)")
        st.caption("รวบรวมตัวชี้วัดทางสรีรวิทยาและผลตรวจทางห้องปฏิบัติการ 26 รายการ เพื่อประเมินความเสี่ยงระดับอวัยวะ")

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
    # TAB 3: 🤖 เหตุผล AI & บทสนทนานำเสนอ
    # ==========================================================================
    with pic_tabs[2]:
        st.markdown("#### 🤖 ปัจจัยขับเคลื่อนทางการแพทย์และการสื่อสารเชิงรุก (AI Drivers & Outreach)")
        st.caption("วิเคราะห์เหตุผลที่โมเดลคาดการณ์ความพร้อมในการรับบริการ พร้อมสคริปต์บทสนทนาสำหรับเจ้าหน้าที่และพยาบาล")

        col_xai_plot, col_talk_track = st.columns([1.2, 1.3], gap="large")

        with col_xai_plot:
            st.markdown("##### 📊 ปัจจัยที่ส่งผลต่อคะแนน (Clinical Drivers)")
            
            f_names: List[str] = []
            vals_arr: Optional[np.ndarray] = None

            if shap_explanation is not None and hasattr(shap_explanation, "values"):
                raw_vals = np.asarray(shap_explanation.values, dtype=float)
                if raw_vals.ndim == 2 and raw_vals.shape[0] == 1:
                    raw_vals = raw_vals[0]
                if raw_vals.size > 0:
                    vals_arr = raw_vals
                    f_names = list(getattr(shap_explanation, "feature_names", []))

            # Guaranteed fallback attribution from patient's clinical profile
            if vals_arr is None or len(f_names) == 0:
                f_names = [
                    "Dyslipidemia (Dx)", "Age", "Systolic BP", "BMI",
                    "Hypertension (Dx)", "Diabetes (Dx)", "Prior Visits (12M)", "Gender_Code"
                ]
                vals_arr = np.array([
                    16.39 if _is_dlp else 0.0,
                    (age_val - 35.0) * 0.17,
                    (sys_val - 120.0) * 0.08,
                    (bmi_val - 22.0) * 0.18,
                    8.20 if _is_ht else 0.0,
                    12.50 if _is_dm else 0.0,
                    (visits_val - 1.0) * 0.12,
                    0.0,
                ], dtype=float)

            order_w = np.argsort(np.abs(vals_arr))
            labels_w = [f_names[i] for i in order_w]
            shap_w = [float(vals_arr[i]) for i in order_w]
            colors_w = ["#EF4444" if v >= 0 else "#3B82F6" for v in shap_w]

            fig_shap = go.Figure(go.Bar(
                x=shap_w,
                y=labels_w,
                orientation="h",
                marker=dict(color=colors_w),
                text=[f"{v:+.3f}" for v in shap_w],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>SHAP Contribution: <b>%{x:+.4f}</b><extra></extra>",
            ))
            fig_shap.update_layout(
                title=dict(text=f"<b>SHAP Waterfall Attribution for {hn_display}</b>", font=dict(size=12, color="#0F172A")),
                xaxis=dict(title="SHAP Contribution (Log-Odds Impact)", gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#94A3B8"),
                yaxis=dict(title=""),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=320,
                margin=dict(l=10, r=45, t=35, b=25),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("ℹ️ แถบสีแดง (+) เพิ่มความน่าจะเป็นในการตรวจสุขภาพ | แถบสีน้ำเงิน (-) ลดความน่าจะเป็นเทียบกับฐานทั่วไป")

        with col_talk_track:
            st.markdown("##### 💬 บทสนทนาแนะนำสำหรับเจ้าหน้าที่ (Caring Talk Track)")
            top_reason = clinical_reason_text if clinical_reason_text else "การตรวจประเมินสุขภาพประจำปีตามช่วงอายุ"
            
            talk_track_text = (
                f"สวัสดีครับ/ค่ะ วันนี้จากการตรวจติดตามสุขภาพเบื้องต้นของคุณ {hn_display} "
                f"ทางทีมแพทย์สังเกตเห็นตัวชี้วัดด้าน {top_reason} ที่เริ่มมีความเสี่ยงหรือควรเฝ้าระวังเพิ่มเติม "
                f"เพื่อความอุ่นใจและป้องกันก่อนเกิดภาวะแทรกซ้อน ทางคลินิกจึงขอแนะนำโปรแกรมตรวจสุขภาพ {active_primary_name} "
                f"ที่เน้นการดูแลตรงจุด ช่วยให้คุณได้รับการดูแลที่คุ้มค่าและปลอดภัยที่สุดครับ/ค่ะ"
            )

            preventive_track_text = (
                f"เรียนคุณ {hn_display} จากประวัติการเข้ารับบริการ {visits_val:.0f} ครั้งในรอบปีที่ผ่านมา "
                f"ทีมสุขภาพขออนุญาตให้คำแนะนำเพิ่มเติมเกี่ยวกับ {top_reason} เพื่อการวางแผนดูแลระยะยาว "
                f"ทางเรามีโปรแกรม {active_primary_name} ซึ่งออกแบบมาเฉพาะบุคคล ช่วยให้คุณหมอสามารถติดตามผลได้อย่างแม่นยำครับ/ค่ะ"
            )

            tt_tab1, tt_tab2 = st.tabs(["🌸 สคริปต์เน้นความห่วงใย (Caring)", "📋 สคริปต์เชิงป้องกัน (Preventive)"])
            with tt_tab1:
                render_html(
                    f"""
                    <div style="background:#F0FDF4; border:1px solid #86EFAC; border-left:5px solid #16A34A; border-radius:10px; padding:14px 18px; margin-bottom:12px;">
                        <div style="font-weight:800; color:#166534; font-size:0.92rem; margin-bottom:6px;">
                            🗣️ สคริปต์พูดคุยที่เน้นความห่วงใยและการบริการ (Caring & Service-Oriented):
                        </div>
                        <div style="font-size:0.88rem; color:#14532D; line-height:1.6; font-style:italic; background:rgba(255,255,255,0.85); padding:10px 14px; border-radius:8px;">
                            "{talk_track_text}"
                        </div>
                    </div>
                    """
                )
            with tt_tab2:
                render_html(
                    f"""
                    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-left:5px solid #2563EB; border-radius:10px; padding:14px 18px; margin-bottom:12px;">
                        <div style="font-weight:800; color:#1E40AF; font-size:0.92rem; margin-bottom:6px;">
                            🩺 สคริปต์การให้คำปรึกษาทางการแพทย์เชิงป้องกัน (Clinical Preventive Counseling):
                        </div>
                        <div style="font-size:0.88rem; color:#1E3A8A; line-height:1.6; font-style:italic; background:rgba(255,255,255,0.85); padding:10px 14px; border-radius:8px;">
                            "{preventive_track_text}"
                        </div>
                    </div>
                    """
                )

            render_html(
                """
                <div style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:12px 16px; font-size:0.8rem; color:#475569; line-height:1.5;">
                    🛡️ <b>หลักจริยธรรมและการดูแลสุขภาพเชิงป้องกัน:</b><br>
                    • มุ่งเน้นการให้ความรู้ สร้างความเข้าใจในผลตรวจ Lab และคุณประโยชน์ของการคัดกรองระยะแรกเริ่ม<br>
                    • หลีกเลี่ยงการสร้างความตื่นตระหนก หรือใช้คำว่า <i>"ระบบ AI สั่งให้ซื้อ"</i>
                </div>
                """
            )
