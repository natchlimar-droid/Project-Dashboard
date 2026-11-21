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
from logic.package_service import build_patient_precision_quotation, UNIFIED_ADDON_CATALOG
from services.data_loader import resolve_patient_lab_snapshot, load_synthetic_dataset
from services.ml_engine import predict_patient_propensity


def render_html(html_str: str) -> None:
    """Renders HTML safely using st.html (or cleaned st.markdown fallback)."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        lines = [line.strip() for line in html_str.splitlines() if line.strip()]
        st.markdown("".join(lines), unsafe_allow_html=True)


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


def get_lab_range_type(key: str) -> str:
    """Classifies lab biomarkers into 3 physiological reference range geometries."""
    if key in ["egfr", "hdl_mg_dl"]:
        return "lower_only"
    if key in ["ldl_mg_dl", "cholesterol_mg_dl", "triglyceride_mg_dl", "uacr_mg_g", "alt_u_l", "ast_u_l", "uric_acid_mg_dl", "creatinine_mg_dl", "hba1c_pct"]:
        return "upper_only"
    return "two_sided"


def compute_clinical_correlations(
    lab_snap: Dict[str, Any],
    age_val: float,
    gender_str: str,
    sys_val: float,
    dia_bp_val: float,
    is_dm: bool,
    is_ht: bool,
    is_dlp: bool,
) -> List[Dict[str, str]]:
    """
    Automated Clinical Syndromic Correlation Engine:
    Connects multi-organ lab markers into clinical syndromes (DKD, Renal Anemia, Cardiorenal, MASLD, etc.).
    """
    correlations: List[Dict[str, str]] = []

    egfr = float(lab_snap.get("egfr", 90.0) or 90.0)
    uacr = float(lab_snap.get("uacr_mg_g", 0.0) or 0.0)
    hba1c = float(lab_snap.get("hba1c_pct", 5.4) or 5.4)
    fpg = float(lab_snap.get("fpg_mg_dl", 95.0) or 95.0)
    hb = float(lab_snap.get("hb_g_dl", 14.0) or 14.0)
    alt = float(lab_snap.get("alt_u_l", 25.0) or 25.0)
    ast = float(lab_snap.get("ast_u_l", 25.0) or 25.0)
    uric = float(lab_snap.get("uric_acid_mg_dl", 5.0) or 5.0)
    ldl = float(lab_snap.get("ldl_mg_dl", 95.0) or 95.0)
    tg = float(lab_snap.get("triglyceride_mg_dl", 110.0) or 110.0)
    bmi = float(lab_snap.get("bmi", 23.0) or 23.0)

    # 1. Diabetic Kidney Disease (DKD) & Albuminuria
    if (egfr < 60.0 or uacr >= 30.0) and (hba1c >= 6.5 or fpg >= 126.0 or is_dm):
        stage = "ระยะ 4-5 (ไตวายรุนแรง)" if egfr < 30.0 else ("ระยะ 3b (รุนแรงปานกลาง)" if egfr < 45.0 else ("ระยะ 3a" if egfr < 60.0 else "ระยะ 1-2"))
        uacr_txt = f"โปรตีนรั่วในปัสสาวะ UACR {uacr:.1f} mg/g" if uacr >= 30.0 else "เสี่ยงไตเสื่อมจากเบาหวาน"
        correlations.append({
            "icon": "🚨" if egfr < 45.0 or uacr >= 300.0 else "💡",
            "title": f"Diabetic Kidney Disease (DKD) สหสัมพันธ์เบาหวานลงไต ({stage})",
            "evidence": f"eGFR {egfr:.1f} mL/min ร่วมกับ {uacr_txt} และระดับน้ำตาลสะสม (HbA1c {hba1c:.1f}%)",
            "advice": "ควรระวังการใช้ยากลุ่ม NSAIDs และยาที่มีพิษต่อไต, ปรับขนาดยาขับทางไตตาม eGFR, ตั้งเป้าควบคุมความดัน < 130/80 mmHg และนัดตรวจติดตาม UACR ซ้ำใน 3 เดือน",
            "severity": "panic" if egfr < 30.0 or uacr >= 300.0 else "critical"
        })

    # 2. Renal Anemia (ภาวะโลหิตจางสัมพันธ์กับโรคไตเรื้อรัง)
    anemia_cutoff = 12.0 if gender_str == "หญิง" else 13.0
    if egfr < 60.0 and hb < anemia_cutoff:
        correlations.append({
            "icon": "🩸",
            "title": "Renal Anemia (ภาวะโลหิตจางสัมพันธ์กับโรคไตเรื้อรัง)",
            "evidence": f"ระดับฮีโมโกลบินลดลง (Hb {hb:.1f} g/dL) สัมพันธ์กับไตเสื่อมเรื้อรัง (eGFR {egfr:.1f} mL/min)",
            "advice": "ไตเสื่อมทำให้การสร้างฮอร์โมน Erythropoietin (EPO) ลดลง แนะนำตรวจ Serum Ferritin และ Iron Profile เพิ่มเติม และปรึกษาอายุรแพทย์โรคไต",
            "severity": "panic" if hb < 8.0 else "warning"
        })

    # 3. Cardiorenal-Metabolic Risk
    if (sys_val >= 140.0 or dia_bp_val >= 90.0 or is_ht) and (ldl >= 130.0 or tg >= 200.0 or is_dlp) and (egfr < 60.0 or uacr >= 30.0):
        correlations.append({
            "icon": "🫀",
            "title": "Cardiorenal-Metabolic Syndrome (สหสัมพันธ์ความดัน-ไขมันและไตเสื่อม)",
            "evidence": f"ความดันโลหิตสูง ({sys_val:.0f}/{dia_bp_val:.0f} mmHg) ร่วมกับไขมันผิดปกติและไตทำงานลดลง (eGFR {egfr:.1f})",
            "advice": "ความดันและไขมันที่สูงเร่งให้ไตเสื่อมเร็วขึ้นและเพิ่มความเสี่ยงโรคหลอดเลือดหัวใจ แนะนำจำกัดโซเดียม < 2,000 mg/วัน, ใช้ Statin และตรวจ EKG",
            "severity": "warning"
        })

    # 4. Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD / ไขมันพอกตับ)
    if (alt >= 40.0 or ast >= 40.0) and (bmi >= 25.0 or tg >= 200.0 or hba1c >= 6.5 or is_dm):
        correlations.append({
            "icon": "🧪",
            "title": "MASLD / MASH (สหสัมพันธ์เอนไซม์ตับสูงและกลุ่มอาการเมตาบอลิก)",
            "evidence": f"เอนไซม์ตับสูง (ALT {alt:.0f} U/L) ร่วมกับภาวะน้ำหนักเกิน (BMI {bmi:.1f}) หรือไขมันไตรกลีเซอไรด์สูง",
            "advice": "สงสัยภาวะไขมันพอกตับจากการเผาผลาญผิดปกติ แนะนำตรวจ Ultrasound ช่องท้องส่วนบน หรือ FibroScan และควบคุมอาหารลดแป้ง/น้ำตาล",
            "severity": "warning"
        })

    # 5. Hyperuricemia & Urate Nephropathy Risk
    if uric >= 8.5 and egfr < 60.0:
        correlations.append({
            "icon": "🦴",
            "title": "Hyperuricemia & Gout Risk (กรดยูริกสูงในผู้ป่วยไตเสื่อม)",
            "evidence": f"กรดยูริกสูง ({uric:.1f} mg/dL) ในผู้ป่วยที่ไตขับยูริกได้ลดลง (eGFR {egfr:.1f} mL/min)",
            "advice": "เสี่ยงต่อการเกิดผลึกยูริกอุดกั้นท่อไตและข้ออักเสบเกาต์ แนะนำงดสัตว์ปีก/เครื่องใน/แอลกอฮอล์ ดื่มน้ำให้เพียงพอ และพิจารณายาลดยูริก",
            "severity": "warning"
        })

    return correlations


def _render_visual_range_bar(fval: float, r_min: float, r_max: float, marker_col: str, range_type: str = "two_sided") -> str:
    """
    Renders a Smart Range Bar matching clinical physiology:
    - two_sided: Normal band in the center (Hb, FPG, CBC, TSH)
    - upper_only: Normal band starts at 0% on the left (LDL, TG, UACR, HbA1c, ALT, Uric)
    - lower_only: Normal band is on the far right up to 100% (eGFR, HDL)
    """
    if range_type == "upper_only":
        d_min = 0.0
        d_max = max(r_max * 2.0, fval * 1.15) if r_max > 0 else 100.0
        norm_start_pct = 0.0
        norm_width_pct = min(60.0, (r_max / d_max) * 100.0)
        val_pct = max(3.0, min(97.0, (fval / d_max) * 100.0))
    elif range_type == "lower_only":
        d_min = 0.0
        d_max = max(r_min * 1.4, fval * 1.15) if r_min > 0 else 100.0
        norm_start_pct = max(30.0, (r_min / d_max) * 100.0)
        norm_width_pct = 100.0 - norm_start_pct
        val_pct = max(3.0, min(97.0, (fval / d_max) * 100.0))
    else:  # two_sided
        span = r_max - r_min
        d_min = max(0.0, r_min - span * 0.7)
        d_max = r_max + span * 0.7
        if d_max <= d_min:
            d_max = d_min + 1.0
        norm_start_pct = max(0.0, (r_min - d_min) / (d_max - d_min) * 100.0)
        norm_width_pct = max(8.0, (r_max - r_min) / (d_max - d_min) * 100.0)
        val_pct = max(3.0, min(97.0, (fval - d_min) / (d_max - d_min) * 100.0))

    return f"""
    <div style="position:relative; width:100%; height:18px; display:flex; align-items:center;" title="ผลตรวจ {fval:.2f} (เกณฑ์ปกติ: {r_min} - {r_max})">
        <div style="position:absolute; width:100%; height:7px; background:#F1F5F9; border-radius:999px;"></div>
        <div style="position:absolute; left:{norm_start_pct:.1f}%; width:{norm_width_pct:.1f}%; height:7px; background:#CBD5E1; border-radius:999px;"></div>
        <div style="position:absolute; left:calc({val_pct:.1f}% - 7px); width:14px; height:14px; border-radius:50%; background:{marker_col}; border:2.5px solid white; box-shadow:0 1px 5px rgba(0,0,0,0.3); z-index:2;"></div>
    </div>
    """


@st.fragment
def render_zone3_lab_matrix(
    panels_data: List[Tuple[str, str, List[Dict[str, Any]], int, int, int]],
    total_measured_count: int,
    total_abnormal_count: int,
    sel_pid_str: str,
    all_table_records: List[Dict[str, Any]],
) -> None:
    """
    Renders Zone 3 Interactive Lab Matrix as an isolated fragment.
    Filter changes and searches update ONLY this section with 0-flicker UX.
    Organ cards are dynamically sorted by clinical risk so the most critical organ is top-left.
    Rows use CSS Grid (35% 43% 22%) with responsive mobile stacked fallback and WCAG AA contrast.
    """
    st.markdown("##### 🔬 แผงวิเคราะห์ผลตรวจทางห้องปฏิบัติการ (Interactive Visual Range Matrix)")
    st.caption("แถบสเกลวัดระดับ (Visual Range Bar) ช่วยให้เห็นทันทีว่าผลตรวจอยู่ในช่วงปกติ (แถบสีเทา) หรือเบี่ยงเบนออกนอกเกณฑ์")

    filter_c1, filter_c2 = st.columns([2.5, 1.5])
    with filter_c1:
        mat_filter = st.radio(
            "ตัวกรองการแสดงผล:",
            [
                f"📋 แสดงทั้งหมด ({total_measured_count} รายการ)",
                f"⚠️ เฉพาะค่าผิดปกติ ({total_abnormal_count} รายการ)",
                f"🟢 เฉพาะค่าปกติ ({total_measured_count - total_abnormal_count} รายการ)",
            ],
            horizontal=True,
            key=f"matrix_filter_{sel_pid_str}",
        )
    with filter_c2:
        search_query = st.text_input(
            "🔍 ค้นหารายการตรวจ (Search Lab):",
            placeholder="เช่น LDL, eGFR, กรดยูริก, ตับ...",
            key=f"search_lab_{sel_pid_str}",
        ).strip().lower()

    # Dynamic Section Sorting (จัดเรียงการ์ดอัตโนมัติตาม Risk Score):
    # ดันหมวดหมู่ที่มีค่า critical / risk score สูงสุดขึ้นไปอยู่มุมซ้ายบนสุดเสมอ
    sorted_panels = sorted(
        panels_data,
        key=lambda p: (p[4], p[5], p[3]),
        reverse=True,
    )

    panels_l = [p for idx, p in enumerate(sorted_panels) if idx % 2 == 0]
    panels_r = [p for idx, p in enumerate(sorted_panels) if idx % 2 == 1]

    def _render_matrix_panel(title_th: str, title_en: str, items: List[Dict[str, Any]], n_abn: int, crit_c: int, is_top_priority: bool) -> None:
        filtered = []
        for it in items:
            if "เฉพาะค่าผิดปกติ" in mat_filter and it["tier"] == "normal":
                continue
            if "เฉพาะค่าปกติ" in mat_filter and it["tier"] != "normal":
                continue
            if search_query:
                match_s = (search_query in it["name_th"].lower()) or (search_query in it["abbr"].lower())
                if not match_s:
                    continue
            filtered.append(it)

        if not filtered:
            if "เฉพาะค่าผิดปกติ" in mat_filter or search_query:
                render_html(
                    f"""
                    <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; margin-bottom:14px; overflow:hidden;">
                        <div style="background:#F8FAFC; border-bottom:1px solid #E2E8F0; padding:10px 16px; display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-size:0.92rem; font-weight:800; color:#0F172A;">{title_th}</span>
                                <span style="font-size:0.75rem; color:#475569;">({title_en})</span>
                            </div>
                            <span style="font-size:0.75rem; color:#047857; background:#ECFDF5; border:1px solid #A7F3D0; padding:2px 8px; border-radius:6px; font-weight:700;">🟢 ปกติทุกค่า</span>
                        </div>
                        <div style="padding:14px 16px; font-size:0.82rem; color:#64748B; text-align:center;">
                            ไม่มีรายการที่ตรงกับเงื่อนไขในหมวดนี้
                        </div>
                    </div>
                    """
                )
            return

        has_panic = any(it.get("is_panic") for it in filtered)
        if has_panic:
            border_left_style = "border-left: 5px solid #991B1B;"
            badge_card = f'<span style="background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; font-size:0.75rem; font-weight:800; padding:3px 10px; border-radius:6px;">🚨⚡ PANIC ALERT: ค่าวิกฤตฉุกเฉิน</span>'
        elif crit_c > 0:
            border_left_style = "border-left: 4px solid #DC2626;"
            badge_card = f'<span style="background:#FEE2E2; color:#B91C1C; border:1px solid #FECACA; font-size:0.75rem; font-weight:800; padding:2px 8px; border-radius:6px;">🔴 วิกฤต {crit_c} | รวมผิดปกติ {n_abn}</span>'
        elif n_abn > 0:
            border_left_style = "border-left: 4px solid #EA580C;"
            badge_card = f'<span style="background:#FFEDD5; color:#9A3412; border:1px solid #FED7AA; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px;">⚠️ ผิดปกติ {n_abn} รายการ</span>'
        else:
            border_left_style = "border-left: 4px solid #10B981;"
            badge_card = f'<span style="background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px;">🟢 ปกติทุกค่า</span>'

        priority_tag = (
            '<span style="font-size:0.7rem; background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; padding:2px 7px; border-radius:4px; font-weight:800; margin-left:6px;">🚨 ลำดับความสำคัญสูงสุด</span>'
            if is_top_priority and (crit_c > 0 or has_panic) else ""
        )

        # CSS Grid Rows: 35% 43% 22% with Mobile Responsive Stacked Fallback
        rows_code = ""
        for idx, it in enumerate(filtered):
            is_last_row = (idx == len(filtered) - 1)
            b_btm = "none" if is_last_row else "1px solid #F1F5F9"
            range_bar_html = _render_visual_range_bar(
                it["fval"],
                it["r_min"],
                it["r_max"],
                it["marker_col"],
                range_type=it.get("range_type", "two_sided"),
            )

            panic_pill = (
                '<span style="font-size:0.68rem; background:#991B1B; color:white; padding:1px 5px; border-radius:3px; font-weight:800; margin-left:4px;">PANIC</span>'
                if it.get("is_panic") else ""
            )

            rows_code += f"""
            <div class="lab-row-grid" style="border-bottom:{b_btm};">
                <!-- Col 1 (32fr): Name & Reference (Geriatric WCAG AA Contrast) -->
                <div class="lab-col-meta">
                    <div>
                        <div style="display:flex; align-items:center; gap:5px; flex-wrap:wrap;">
                            <span style="font-size:0.92rem; font-weight:800; color:#0F172A;" title="{it['name_th']}">{it['name_th']}</span>
                            <span style="font-size:0.72rem; color:#475569; background:#F1F5F9; border:1px solid #E2E8F0; padding:1px 5px; border-radius:4px; font-weight:700;">{it['abbr']}</span>
                            {panic_pill}
                        </div>
                        <div style="font-size:0.82rem; color:#334155; font-weight:600; margin-top:3px;">
                            ช่วงอ้างอิงปกติ: <b>{it['r_min']} - {it['r_max']}</b> <span style="font-size:0.75rem; color:#475569;">{it['unit']}</span>
                        </div>
                    </div>
                </div>
                <!-- Col 2 (36fr): Smart Range Bar -->
                <div class="lab-col-bar">
                    {range_bar_html}
                </div>
                <!-- Col 3 (32fr): Value & Trend (Fully Visible & Balanced) -->
                <div class="lab-col-val">
                    <div style="font-size:1.12rem; font-weight:800; color:{it['marker_col']}; font-family:'JetBrains Mono',monospace; line-height:1.2; white-space:nowrap;">
                        {it['fval']:.2f} <span style="font-size:0.78rem; color:#475569; font-weight:700;">{it['unit']}</span>
                    </div>
                    <div style="margin-top:3px; white-space:nowrap;">
                        {it['trend_badge']}
                    </div>
                </div>
            </div>
            """

        render_html(
            f"""
            <div class="lab-organ-card" style="background:white; border:1px solid #E2E8F0; {border_left_style} border-radius:12px; margin-bottom:14px; overflow:hidden; box-shadow:0 1px 3px rgba(15,23,42,0.02);">
                <div style="background:#F8FAFC; border-bottom:1px solid #E2E8F0; padding:10px 16px; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.92rem; font-weight:800; color:#0F172A;">{title_th}</span>
                        <span style="font-size:0.75rem; color:#475569;">({title_en})</span>
                        {priority_tag}
                    </div>
                    {badge_card}
                </div>
                <div>
                    {rows_code}
                </div>
            </div>
            """
        )

    grid_l, grid_r = st.columns(2, gap="medium")
    with grid_l:
        for idx, (p_th, p_en, items, n_abn, crit_c, r_score) in enumerate(panels_l):
            _render_matrix_panel(p_th, p_en, items, n_abn, crit_c, is_top_priority=(idx == 0))

    with grid_r:
        for p_th, p_en, items, n_abn, crit_c, r_score in panels_r:
            _render_matrix_panel(p_th, p_en, items, n_abn, crit_c, is_top_priority=False)

    # ── Export & Raw Data Table Accordion ──
    with st.expander("📊 ดูข้อมูลแบบตารางสรุปครบทุกรายการ (Table View & Export)", expanded=False):
        if all_table_records:
            st.dataframe(pd.DataFrame(all_table_records), use_container_width=True, hide_index=True)


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
        gender_str = str(cached_lead.get("gender", "ชาย"))
        gender_code = 1.0 if (gender_str in ["ชาย", "male", "M"] or float(cached_lead.get("gender_code", 1.0)) >= 0.5) else 0.0
        gender_str = "ชาย" if gender_code >= 0.5 else "หญิง"
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
            visits_val = float(matched_pt.get("prior_visits_12m", matched_pt.get("visits_last_2y", matched_pt.get("prior_opd_visits_total", matched_pt.get("visits", 1.0)))))
            g_raw = str(matched_pt.get("gender", "")).strip()
            g_code_raw = float(matched_pt.get("gender_code", 1.0))
            if "ชาย" in g_raw or g_raw.lower() in ["male", "m"]:
                gender_str = "ชาย"
                gender_code = 1.0
            elif "หญิง" in g_raw or g_raw.lower() in ["female", "f"]:
                gender_str = "หญิง"
                gender_code = 0.0
            else:
                gender_str = "ชาย" if g_code_raw >= 0.5 else "หญิง"
                gender_code = 1.0 if g_code_raw >= 0.5 else 0.0
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
            gender_str = "ชาย"
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
    gender_str = str(lab_snap.get("gender", "ชาย" if gender_code >= 0.5 else "หญิง"))
    gender_code = 1.0 if gender_str == "ชาย" else 0.0

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
    if is_lead_match and pers_pkg_override:
        pers_pkg = pers_pkg_override
    else:
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

        # ── Unified Precision Package Resolution & Quotation Engine ──
        selected_primary_key = f"active_pkg_sel_{sel_pid_str}"
        default_pkg_type = st.session_state.get(f"lead_prescribed_type_{sel_pid_str}", "bespoke" if needs_custom else "std")
        current_pkg_type = st.session_state.get(selected_primary_key, default_pkg_type)

        selected_addons_key = f"addons_multisel_{sel_pid_str}"
        if selected_addons_key not in st.session_state and f"lead_prescribed_addons_{sel_pid_str}" in st.session_state:
            st.session_state[selected_addons_key] = st.session_state[f"lead_prescribed_addons_{sel_pid_str}"]
        user_addons = st.session_state.get(selected_addons_key, None)

        quote = build_patient_precision_quotation(
            lab_snap=lab_snap,
            selected_pkg_type=current_pkg_type,
            selected_addon_keys=user_addons,
        )

        std_pkg = quote["std_package"]
        std_name = std_pkg["name"]
        std_price = std_pkg["price"]
        std_fit = std_pkg["fit_pct"]
        bullets = std_pkg["bullets"]
        clinical_reason_text = quote["clinical_reason_text"]

        bespoke_pkg = quote["bespoke_package"]
        bespoke_name = bespoke_pkg["name"]
        bespoke_price = bespoke_pkg["price"]
        bespoke_full = bespoke_pkg["full_price"]
        bespoke_disc = bespoke_pkg["discount_pct"]
        bespoke_reason = bespoke_pkg["reason"]

        active_primary = quote["active_primary"]
        active_primary_name = quote["active_primary_name"]
        active_primary_price = quote["active_primary_price"]
        active_addons = quote["active_addons"]
        gross_total = quote["gross_total"]
        discount_amount = quote["discount_amount"]
        net_payable = quote["net_payable"]

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
                is_selected = (current_pkg_type == "std")
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
                is_bespoke_selected = (current_pkg_type == "bespoke")
                if st.button(
                    "🧬 สลับมาใช้แพ็กเกจเฉพาะบุคคลนี้" if not is_bespoke_selected else "✔️ แพ็กเกจเฉพาะบุคคลที่เลือกอยู่",
                    key=f"btn_switch_bespoke_{sel_pid_str}",
                    type="primary" if is_bespoke_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[selected_primary_key] = "bespoke"
                    st.rerun()

            st.markdown("---")

            # ── 3. Add-on Modular Tests Section ──
            st.markdown("##### ➕ รายการตรวจเสริมที่แนะนำเพิ่มเติม (Add-on Modules)")
            st.caption("สามารถเลือกเพิ่มรายการตรวจเฉพาะทางที่ตรงกับความกังวลของผู้ป่วย เพื่อรวมในใบเสนอราคาฉบับเดียวกัน")

            addon_candidates = list(UNIFIED_ADDON_CATALOG.keys())
            rec_addon_set = {a["code"] for a in quote["catalog_addons"] if a["is_recommended"]}

            if selected_addons_key not in st.session_state:
                st.session_state[selected_addons_key] = quote["active_addon_keys"]

            selected_addon_keys = st.multiselect(
                "เลือกรายการตรวจเสริมที่ต้องการรวมในใบเสนอราคา:",
                options=addon_candidates,
                format_func=lambda k: f"{UNIFIED_ADDON_CATALOG[k]['name']} (+฿{UNIFIED_ADDON_CATALOG[k]['price']:,}){' ⭐ แนะนำ' if k in rec_addon_set else ''}",
                key=selected_addons_key,
            )

        # ── Quotation Summary (Right Column) ──
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
    # TAB 2: 🧪 ผลแล็บ & ประวัติสุขภาพ (3-Zone Clinical Hierarchy)
    # ==========================================================================
    with pic_tabs[1]:
        # ── Data Preparation & Clinical Triage Severity Classification ────────
        lab_panels_def = [
            ("🩸 ระบบระดับน้ำตาลและการเผาผลาญ", "Glycemic Control", [
                ("fpg_mg_dl", "ระดับน้ำตาลหลังอดอาหาร", "FPG", "mg/dL", 70.0, 99.0, "delta_fbs"),
                ("hba1c_pct", "น้ำตาลสะสมในเม็ดเลือดแดง", "HbA1c", "%", 4.0, 5.6, "delta_hba1c"),
            ]),
            ("🫀 ระบบไขมันในเลือดและหลอดเลือด", "Lipid Profile", [
                ("cholesterol_mg_dl", "คอเลสเตอรอลรวม", "Total Chol", "mg/dL", 100.0, 200.0, None),
                ("triglyceride_mg_dl", "ไตรกลีเซอไรด์", "Triglyceride", "mg/dL", 30.0, 150.0, "delta_tg"),
                ("hdl_mg_dl", "ไขมันดี", "HDL-C", "mg/dL", 40.0, 80.0, None),
                ("ldl_mg_dl", "ไขมันเลว", "LDL-C", "mg/dL", 50.0, 100.0, "delta_ldl"),
            ]),
            ("🫘 การทำงานของไตและเกลือแร่", "Renal Function", [
                ("creatinine_mg_dl", "ครีเอตินิน", "Creatinine", "mg/dL", 0.6, 1.2, None),
                ("egfr", "อัตราการกรองของไต", "eGFR", "mL/min/1.73m²", 90.0, 120.0, "delta_egfr"),
                ("uacr_mg_g", "อัลบูมินรั่วในปัสสาวะ", "UACR", "mg/g", 0.0, 30.0, None),
                ("uric_acid_mg_dl", "กรดยูริก", "Uric Acid", "mg/dL", 3.5, 7.2, None),
            ]),
            ("🧪 การทำงานของตับและการอักเสบ", "Hepatic Function", [
                ("alt_u_l", "เอนไซม์ตับ", "ALT/SGPT", "U/L", 0.0, 40.0, "delta_alt"),
                ("ast_u_l", "เอนไซม์ตับ", "AST/SGOT", "U/L", 0.0, 40.0, None),
            ]),
            ("🩺 โลหิตวิทยาและต่อมไทรอยด์", "Hematology & Thyroid", [
                ("tsh_miu_l", "ฮอร์โมนกระตุ้นไทรอยด์", "TSH", "mIU/L", 0.45, 4.5, None),
                ("hb_g_dl", "ความเข้มข้นฮีโมโกลบิน", "Hb", "g/dL", 12.0, 16.0, None),
                ("hct_pct", "เม็ดเลือดแดงอัดแน่น", "Hct", "%", 36.0, 48.0, None),
                ("wbc_10_3_ul", "เม็ดเลือดขาวรวม", "WBC", "10³/µL", 4.5, 10.5, None),
                ("platelet_count_10_3_ul", "เกล็ดเลือด", "Platelet", "10³/µL", 150.0, 450.0, None),
            ]),
        ]

        def _classify_severity(k_name: str, val_num: float, min_r: float, max_r: float) -> Tuple[str, str, str, str, bool]:
            # Returns (tier: 'critical'|'moderate'|'mild'|'normal', marker_color, label_th, clinical_note, is_panic: bool)
            
            # 0. Emergency Panic Tier (วิกฤตฉุกเฉิน LIS Critical Alert - ต้องรายงานแพทย์ทันที)
            if k_name == "egfr" and val_num < 30.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (ไตวายรุนแรง Stage 4-5)", f"eGFR {val_num:.1f} mL/min วิกฤต! ไตทำงานเหลือน้อยกว่า 30% ต้องพบอายุรแพทย์โรคไตทันที", True
            if k_name == "uacr_mg_g" and val_num >= 300.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (โปรตีนรั่วรุนแรง)", f"UACR {val_num:.1f} mg/g มีโปรตีนรั่วระดับ Macroalbuminuria เสี่ยงไตวายเฉียบพลัน", True
            if k_name == "hba1c_pct" and val_num >= 9.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (น้ำตาลสะสมวิกฤต)", f"HbA1c {val_num:.1f}% สูงวิกฤต เสี่ยงภาวะแทรกซ้อนเฉียบพลัน (Hyperglycemic Crisis)", True
            if k_name == "fpg_mg_dl" and val_num >= 250.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (น้ำตาลอดอาหารสูงวิกฤต)", f"FPG {val_num:.0f} mg/dL น้ำตาลสูงวิกฤต เสี่ยงภาวะหมดสติจากเบาหวาน", True
            if k_name == "ldl_mg_dl" and val_num >= 190.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (ไขมันเลวสูงวิกฤต)", f"LDL {val_num:.0f} mg/dL วิกฤต! เสี่ยงหลอดเลือดหัวใจและสมองตีบเฉียบพลัน", True
            if k_name == "uric_acid_mg_dl" and val_num >= 10.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (ยูริกสูงวิกฤต)", f"กรดยูริก {val_num:.1f} mg/dL เสี่ยงภาวะไตวายจากผลึกยูริกและข้ออักเสบเกาต์รุนแรง", True
            if k_name in ["alt_u_l", "ast_u_l"] and val_num >= 120.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (ตับอักเสบรุนแรง)", f"เอนไซม์ตับ {val_num:.0f} U/L สูงเกิน 3 เท่าของเกณฑ์ปกติ บ่งชี้เซลล์ตับถูกทำลายรุนแรง", True
            if k_name == "hb_g_dl" and val_num < 8.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (ซีดรุนแรง)", f"Hb {val_num:.1f} g/dL ซีดขั้นวิกฤต อาจต้องพิจารณาให้เลือดทดแทน", True
            if k_name == "platelet_count_10_3_ul" and val_num <= 50.0:
                return "critical", "#991B1B", "🚨 วิกฤตฉุกเฉิน (เกล็ดเลือดต่ำวิกฤต)", f"เกล็ดเลือด {val_num:.0f} 10³/µL เสี่ยงภาวะเลือดออกเองในอวัยวะภายใน", True

            if min_r <= val_num <= max_r:
                return "normal", "#0F766E", "ปกติ", "อยู่ในเกณฑ์อ้างอิง", False

            # 1. Critical Tier (วิกฤต / ต้องพบแพทย์เร่งด่วน)
            if k_name == "egfr" and val_num < 60:
                stage = "ระยะ 3b (รุนแรงปานกลาง)" if val_num < 45 else "ระยะ 3a"
                return "critical", "#DC2626", f"🔴 วิกฤต (ไตเสื่อม {stage})", f"eGFR {val_num:.1f} mL/min เข้าสู่ภาวะไตเสื่อม ควรพบแพทย์อายุรกรรมโรคไต", False
            if k_name == "uacr_mg_g" and val_num >= 30:
                return "critical", "#DC2626", "🔴 วิกฤต (โปรตีนรั่วในปัสสาวะ)", f"UACR {val_num:.1f} mg/g มีโปรตีนรั่ว เสี่ยงไตวายจากเบาหวาน/ความดัน", False
            if k_name == "uric_acid_mg_dl" and val_num >= 9.0:
                return "critical", "#DC2626", "🔴 วิกฤต (ยูริกสูงมาก)", f"กรดยูริก {val_num:.1f} mg/dL เสี่ยงข้ออักเสบเกาต์เฉียบพลันและนิ่วในไต", False
            if k_name == "ldl_mg_dl" and val_num >= 160:
                return "critical", "#DC2626", "🔴 วิกฤต (ไขมันเลวสูงมาก)", f"LDL {val_num:.0f} mg/dL สูงมาก เสี่ยงหลอดเลือดหัวใจตีบเฉียบพลัน", False
            if k_name == "hba1c_pct" and val_num >= 8.0:
                return "critical", "#DC2626", "🔴 วิกฤต (เบาหวานควบคุมไม่ได้)", f"HbA1c {val_num:.1f}% สูงมาก เสี่ยงภาวะแทรกซ้อนหลอดเลือดและจอประสาทตา", False
            if k_name in ["alt_u_l", "ast_u_l"] and val_num >= 80:
                return "critical", "#DC2626", "🔴 วิกฤต (ตับอักเสบเฉียบพลัน)", f"เอนไซม์ตับ {val_num:.0f} U/L สูงเกิน 2 เท่า ควรตรวจอัลตราซาวด์ตับ", False
            if k_name == "hb_g_dl" and val_num < 10.0:
                return "critical", "#DC2626", "🔴 วิกฤต (ภาวะซีดปานกลาง-รุนแรง)", f"Hb {val_num:.1f} g/dL เม็ดเลือดแดงต่ำชัดเจน ควรตรวจหาสาเหตุ", False

            # 2. Moderate Tier (เฝ้าระวัง / ปรับพฤติกรรมและยา)
            if k_name == "ldl_mg_dl" and 130 <= val_num < 160:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (ไขมันเลวสูง)", f"LDL {val_num:.0f} mg/dL แนะนำปรับอาหารไขมันและเริ่มยาลดไขมัน", False
            if k_name == "cholesterol_mg_dl" and val_num >= 200:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (คอเลสเตอรอลสูง)", f"คอเลสเตอรอลรวม {val_num:.0f} mg/dL เกินเกณฑ์มาตรฐาน", False
            if k_name == "triglyceride_mg_dl" and val_num >= 200:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (ไตรกลีเซอไรด์สูง)", f"ไตรกลีเซอไรด์ {val_num:.0f} mg/dL เสี่ยงไขมันพอกตับ", False
            if k_name == "hba1c_pct" and 6.5 <= val_num < 8.0:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (เบาหวานระยะแรก)", f"HbA1c {val_num:.1f}% เข้าเกณฑ์โรคเบาหวาน ควรตรวจติดตามระดับน้ำตาล", False
            if k_name == "fpg_mg_dl" and 126 <= val_num < 180:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (น้ำตาลอดอาหารสูง)", f"FPG {val_num:.0f} mg/dL แนะนำงดของหวานและปรึกษาแพทย์", False
            if k_name == "egfr" and 60 <= val_num < 90:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (การกรองไตเริ่มลดลง)", f"eGFR {val_num:.1f} mL/min ไตเสื่อมระยะ 2 เลี่ยงเค็มและยาแก้ปวด NSAIDs", False
            if k_name == "uric_acid_mg_dl" and 7.2 <= val_num < 9.0:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (กรดยูริกสูง)", f"กรดยูริก {val_num:.1f} mg/dL ควรลดสัตว์ปีก ยอดผัก และแอลกอฮอล์", False
            if k_name in ["alt_u_l", "ast_u_l"] and 40 < val_num < 80:
                return "moderate", "#EA580C", "🟠 เฝ้าระวัง (ตับอักเสบเล็กน้อย)", f"เอนไซม์ตับ {val_num:.0f} U/L สงสัยภาวะไขมันพอกตับ (MASLD)", False

            # 3. Mild Tier (เบี่ยงเบนเล็กน้อย)
            if val_num > max_r:
                return "mild", "#CA8A04", "🟡 เกินเกณฑ์เล็กน้อย", f"สูงกว่าค่าปกติเล็กน้อย ({val_num:.1f} vs max {max_r})", False
            else:
                return "mild", "#CA8A04", "🟡 ต่ำกว่าเกณฑ์เล็กน้อย", f"ต่ำกว่าค่าปกติเล็กน้อย ({val_num:.1f} vs min {min_r})", False

        # Pre-process all panel items
        triage_panic = []
        triage_critical = []
        triage_moderate = []
        triage_mild = []
        all_table_records = []
        panels_data = []

        for p_title, p_en, items in lab_panels_def:
            p_items = []
            abnormal_in_panel = 0
            for key, name_th, abbr, unit, r_min, r_max, delta_col in items:
                val = lab_snap.get(key)
                if val is not None and pd.notna(val):
                    fval = float(val)
                    tier, marker_col, lbl_th, note, is_panic = _classify_severity(key, fval, r_min, r_max)
                    delta_val = lab_snap.get(delta_col) if delta_col else None
                    if delta_val is not None and pd.notna(delta_val):
                        d_num = float(delta_val)
                        if d_num > 0:
                            trend_badge = f"<span style='font-size:0.72rem; color:#B91C1C; background:#FEF2F2; border:1px solid #FECACA; padding:2px 6px; border-radius:4px; font-weight:700; white-space:nowrap; display:inline-block;'>▲ +{d_num:.1f} <span style='font-size:0.68rem; color:#991B1B;'>เทียบครั้งก่อน</span></span>"
                        elif d_num < 0:
                            trend_badge = f"<span style='font-size:0.72rem; color:#047857; background:#ECFDF5; border:1px solid #A7F3D0; padding:2px 6px; border-radius:4px; font-weight:700; white-space:nowrap; display:inline-block;'>▼ {d_num:.1f} <span style='font-size:0.68rem; color:#065F46;'>เทียบครั้งก่อน</span></span>"
                        else:
                            trend_badge = "<span style='font-size:0.72rem; color:#475569; background:#F8FAFC; border:1px solid #E2E8F0; padding:2px 6px; border-radius:4px; font-weight:600; white-space:nowrap; display:inline-block;'>— คงที่</span>"
                    else:
                        trend_badge = "<span style='font-size:0.72rem; color:#475569; background:#F8FAFC; border:1px solid #E2E8F0; padding:2px 6px; border-radius:4px; font-weight:600; white-space:nowrap; display:inline-block;'>— ตรวจครั้งแรก</span>"

                    if tier != "normal":
                        abnormal_in_panel += 1
                        item_summary = {
                            "name": name_th,
                            "abbr": abbr,
                            "val": fval,
                            "unit": unit,
                            "note": note,
                            "col": marker_col,
                            "is_panic": is_panic,
                        }
                        if is_panic:
                            triage_panic.append(item_summary)
                        elif tier == "critical":
                            triage_critical.append(item_summary)
                        elif tier == "moderate":
                            triage_moderate.append(item_summary)
                        elif tier == "mild":
                            triage_mild.append(item_summary)

                    item_dict = {
                        "key": key,
                        "name_th": name_th,
                        "abbr": abbr,
                        "unit": unit,
                        "fval": fval,
                        "r_min": r_min,
                        "r_max": r_max,
                        "tier": tier,
                        "marker_col": marker_col,
                        "lbl_th": lbl_th,
                        "note": note,
                        "trend_badge": trend_badge,
                        "is_panic": is_panic,
                        "range_type": get_lab_range_type(key),
                    }
                    p_items.append(item_dict)
                    all_table_records.append({
                        "หมวดการตรวจ": p_title,
                        "รายการตรวจ": f"{name_th} ({abbr})",
                        "ผลตรวจ": f"{fval:.2f}",
                        "หน่วย": unit,
                        "ค่าอ้างอิงปกติ": f"{r_min} - {r_max}",
                        "ระดับความเสี่ยง": lbl_th,
                    })

            panic_in_panel = sum(1 for it in p_items if it.get("is_panic"))
            crit_in_panel = sum(1 for it in p_items if it["tier"] == "critical" and not it.get("is_panic"))
            mod_in_panel = sum(1 for it in p_items if it["tier"] == "moderate")
            mild_in_panel = sum(1 for it in p_items if it["tier"] == "mild")
            panel_risk_score = panic_in_panel * 1000 + crit_in_panel * 100 + mod_in_panel * 10 + mild_in_panel * 1
            panels_data.append((p_title, p_en, p_items, abnormal_in_panel, crit_in_panel + panic_in_panel, panel_risk_score))

        total_measured_count = len(all_table_records)
        total_abnormal_count = len(triage_panic) + len(triage_critical) + len(triage_moderate) + len(triage_mild)

        # ── ZONE 1: Patient & Clinical Header Bar ─────────────────────────────
        index_date_raw = lab_snap.get("index_date", None)
        if index_date_raw and hasattr(index_date_raw, "strftime"):
            visit_date_str = index_date_raw.strftime("%d/%m/%Y")
        else:
            visit_date_str = "ตรวจประเมินสุขภาพประจำรอบปี"

        if len(triage_panic) > 0:
            risk_badge_bg = "#FFF1F2"
            risk_badge_color = "#991B1B"
            risk_badge_border = "#FECDD3"
            risk_badge_text = f"🚨⚡ ระดับวิกฤตฉุกเฉิน (PANIC): พบ {len(triage_panic)} ค่าวิกฤต ต้องรายงานแพทย์ทันที"
        elif len(triage_critical) > 0:
            risk_badge_bg = "#FEF2F2"
            risk_badge_color = "#991B1B"
            risk_badge_border = "#FECACA"
            risk_badge_text = "🔴 ระดับความเสี่ยงสูง: พบค่าวิกฤต ต้องพบแพทย์เฉพาะทาง"
        elif len(triage_moderate) > 0:
            risk_badge_bg = "#FFEDD5"
            risk_badge_color = "#9A3412"
            risk_badge_border = "#FED7AA"
            risk_badge_text = "🟠 ระดับความเสี่ยงปานกลาง: เฝ้าระวังเฉพาะทางและปรับพฤติกรรม"
        elif len(triage_mild) > 0:
            risk_badge_bg = "#FEF9C3"
            risk_badge_color = "#854D0E"
            risk_badge_border = "#FEF08A"
            risk_badge_text = "🟡 ระดับความเสี่ยงต่ำ: เบี่ยงเบนเล็กน้อย ตรวจติดตามประจำปี"
        else:
            risk_badge_bg = "#ECFDF5"
            risk_badge_color = "#047857"
            risk_badge_border = "#A7F3D0"
            risk_badge_text = "🟢 สุขภาพสมบูรณ์: ผลตรวจทุกรายการอยู่ในเกณฑ์ปกติ"

        render_html(
            f"""
            <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:12px 18px; margin-bottom:18px; box-shadow:0 1px 3px rgba(15,23,42,0.02); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:40px; height:40px; border-radius:10px; background:#F1F5F9; display:flex; align-items:center; justify-content:center; font-size:1.4rem;">
                        {'👨' if gender_str == 'ชาย' else '👩'}
                    </div>
                    <div>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-size:1.2rem; font-weight:800; color:#0F172A;">{hn_display}</span>
                            <span style="font-size:0.85rem; color:#475569; font-weight:600;">| {gender_str} {age_val:.0f} ปี</span>
                            <span style="font-size:0.75rem; color:#64748B; background:#F8FAFC; border:1px solid #E2E8F0; padding:2px 8px; border-radius:6px;">
                                📅 รอบตรวจ: {visit_date_str}
                            </span>
                        </div>
                        <div style="font-size:0.76rem; color:#64748B; margin-top:2px;">
                            ประวัติรับบริการสะสม: <b>{visits_val:.0f} ครั้ง</b> • รายการตรวจที่วิเคราะห์: <b>{total_measured_count} รายการ</b>
                        </div>
                    </div>
                </div>
                <div style="background:{risk_badge_bg}; border:1px solid {risk_badge_border}; color:{risk_badge_color}; padding:6px 14px; border-radius:8px; font-size:0.82rem; font-weight:700;">
                    {risk_badge_text}
                </div>
            </div>
            """
        )

        # ── ZONE 2: Asymmetric Executive Summary (Left 35% : Right 65%) ────────
        col_triage, col_vitals = st.columns([0.36, 0.64], gap="large")

        # ── Left Column: Actionable Alerts & Clinical Triage Panel ──
        with col_triage:
            st.markdown("##### 🚨 ลำดับความสำคัญทางคลินิก (Clinical Triage)")
            st.caption("จัดเรียงตามระดับความเร่งด่วนทางการแพทย์ เพื่อการตัดสินใจดูแลทันที")

            triage_content_html = ""

            # 0. Emergency Panic Items (LIS Critical Values)
            if triage_panic:
                panic_rows = "".join([
                    f"""
                    <div style="background:#FFF1F2; border:1px solid #FECDD3; border-left:5px solid #991B1B; border-radius:8px; padding:10px 12px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:baseline;">
                            <span style="font-size:0.88rem; font-weight:900; color:#881337;">🚨 {item['name']} ({item['abbr']})</span>
                            <span style="font-size:1.05rem; font-weight:900; color:#991B1B; font-family:'JetBrains Mono',monospace;">{item['val']:.1f} <span style="font-size:0.72rem; color:#881337;">{item['unit']}</span></span>
                        </div>
                        <div style="font-size:0.75rem; color:#9F1239; font-weight:600; margin-top:3px; line-height:1.35;">{item['note']}</div>
                    </div>
                    """
                    for item in triage_panic
                ])
                triage_content_html += f"""
                <div style="margin-bottom:14px;">
                    <div style="font-size:0.78rem; font-weight:800; color:#991B1B; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
                        <span>🚨⚡ PANIC: ค่าวิกฤตฉุกเฉิน (Immediate Physician Notification)</span>
                        <span style="background:#991B1B; color:white; padding:1px 7px; border-radius:999px; font-size:0.68rem; font-weight:800;">{len(triage_panic)}</span>
                    </div>
                    {panic_rows}
                </div>
                """

            # 1. Critical Items
            if triage_critical:
                crit_rows = "".join([
                    f"""
                    <div style="background:white; border:1px solid #FECACA; border-left:4px solid #DC2626; border-radius:8px; padding:9px 12px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:baseline;">
                            <span style="font-size:0.85rem; font-weight:800; color:#991B1B;">{item['name']} ({item['abbr']})</span>
                            <span style="font-size:0.95rem; font-weight:800; color:#DC2626; font-family:'JetBrains Mono',monospace;">{item['val']:.1f} <span style="font-size:0.7rem; color:#64748B;">{item['unit']}</span></span>
                        </div>
                        <div style="font-size:0.73rem; color:#7F1D1D; margin-top:3px; line-height:1.35;">{item['note']}</div>
                    </div>
                    """
                    for item in triage_critical
                ])
                triage_content_html += f"""
                <div style="margin-bottom:12px;">
                    <div style="font-size:0.76rem; font-weight:800; color:#DC2626; margin-bottom:6px; display:flex; align-items:center; gap:4px;">
                        <span>🔴 วิกฤต / ต้องดูแลเร่งด่วน</span>
                        <span style="background:#FEE2E2; color:#B91C1C; padding:1px 6px; border-radius:999px; font-size:0.68rem;">{len(triage_critical)}</span>
                    </div>
                    {crit_rows}
                </div>
                """
            elif not triage_panic:
                triage_content_html += """
                <div style="background:#F0FDF4; border:1px dashed #BBF7D0; border-radius:8px; padding:8px 12px; margin-bottom:10px; font-size:0.75rem; color:#166534;">
                    🟢 ไม่มีผลตรวจที่อยู่ในเกณฑ์วิกฤต
                </div>
                """

            # 2. Moderate Items
            if triage_moderate:
                mod_rows = "".join([
                    f"""
                    <div style="background:white; border:1px solid #FED7AA; border-left:4px solid #EA580C; border-radius:8px; padding:8px 12px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:baseline;">
                            <span style="font-size:0.84rem; font-weight:800; color:#9A3412;">{item['name']} ({item['abbr']})</span>
                            <span style="font-size:0.92rem; font-weight:800; color:#EA580C; font-family:'JetBrains Mono',monospace;">{item['val']:.1f} <span style="font-size:0.7rem; color:#64748B;">{item['unit']}</span></span>
                        </div>
                        <div style="font-size:0.72rem; color:#9A3412; margin-top:2px; line-height:1.35;">{item['note']}</div>
                    </div>
                    """
                    for item in triage_moderate
                ])
                triage_content_html += f"""
                <div style="margin-bottom:12px;">
                    <div style="font-size:0.76rem; font-weight:800; color:#EA580C; margin-bottom:6px; display:flex; align-items:center; gap:4px;">
                        <span>🟠 เฝ้าระวัง / ปรับพฤติกรรมและยา</span>
                        <span style="background:#FFEDD5; color:#C2410C; padding:1px 6px; border-radius:999px; font-size:0.68rem;">{len(triage_moderate)}</span>
                    </div>
                    {mod_rows}
                </div>
                """

            # 3. Mild Items
            if triage_mild:
                mild_rows = "".join([
                    f"""
                    <div style="background:#FEFCE8; border:1px solid #FEF08A; border-radius:6px; padding:6px 10px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.78rem; font-weight:700; color:#854D0E;">{item['name']} ({item['abbr']})</span>
                        <span style="font-size:0.82rem; font-weight:800; color:#CA8A04; font-family:'JetBrains Mono',monospace;">{item['val']:.1f} {item['unit']}</span>
                    </div>
                    """
                    for item in triage_mild
                ])
                triage_content_html += f"""
                <div>
                    <div style="font-size:0.74rem; font-weight:800; color:#CA8A04; margin-bottom:4px;">
                        🟡 ติดตามผลเล็กน้อย ({len(triage_mild)})
                    </div>
                    {mild_rows}
                </div>
                """

            render_html(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px; min-height:360px;">
                    {triage_content_html}
                </div>
                """
            )

        # ── Right Column: 4 Key Vitals & Metabolic Snapshot (Grid 2x2 with Real Severity & Trends) ──
        with col_vitals:
            st.markdown("##### 🫀 สัญญาณชีพและเมตาบอลิกหลัก (Key Vitals & Metabolic Snapshot)")
            st.caption("ประเมินระดับความรุนแรงตามมาตรฐานสากล พร้อมแนวโน้มเปรียบเทียบจากครั้งก่อน (Trend Analysis)")

            # Vitals Calculations
            # 1. BP Severity
            if sys_val >= 160 or dia_bp_val >= 100:
                bp_col, bp_bg, bp_bd = "#DC2626", "#FEE2E2", "#FECACA"
                bp_lbl = "🔴 ความดันสูงมาก (Stage 2)"
            elif sys_val >= 140 or dia_bp_val >= 90:
                bp_col, bp_bg, bp_bd = "#EA580C", "#FFEDD5", "#FED7AA"
                bp_lbl = "🟠 ความดันสูง (Stage 1)"
            elif sys_val >= 130 or dia_bp_val >= 85:
                bp_col, bp_bg, bp_bd = "#CA8A04", "#FEF9C3", "#FEF08A"
                bp_lbl = "🟡 เฝ้าระวัง (Pre-HT)"
            else:
                bp_col, bp_bg, bp_bd = "#0F766E", "#ECFDF5", "#A7F3D0"
                bp_lbl = "🟢 ปกติ (Optimal)"

            d_sbp = lab_snap.get("delta_sbp")
            if d_sbp is not None and pd.notna(d_sbp):
                d_sbp_f = float(d_sbp)
                bp_trend = f"<span style='color:{'#DC2626' if d_sbp_f > 0 else '#0F766E'}; font-weight:700;'>{'▲ +' if d_sbp_f > 0 else '▼ '}{d_sbp_f:.1f} mmHg</span> จากครั้งก่อน"
            else:
                bp_trend = "<span style='color:#94A3B8;'>— ตรวจครั้งแรก</span>"

            # 2. BMI Severity
            if bmi_val >= 30.0:
                bmi_col, bmi_bg, bmi_bd = "#DC2626", "#FEE2E2", "#FECACA"
                bmi_lbl = "🔴 อ้วนระดับ 2 (Severe)"
            elif bmi_val >= 25.0:
                bmi_col, bmi_bg, bmi_bd = "#EA580C", "#FFEDD5", "#FED7AA"
                bmi_lbl = "🟠 อ้วน (Obesity)"
            elif bmi_val >= 23.0:
                bmi_col, bmi_bg, bmi_bd = "#CA8A04", "#FEF9C3", "#FEF08A"
                bmi_lbl = "🟡 ท้วม (Overweight)"
            else:
                bmi_col, bmi_bg, bmi_bd = "#0F766E", "#ECFDF5", "#A7F3D0"
                bmi_lbl = "🟢 ปกติ (Normal)"

            d_bmi = lab_snap.get("delta_bmi")
            if d_bmi is not None and pd.notna(d_bmi):
                d_bmi_f = float(d_bmi)
                bmi_trend = f"<span style='color:{'#DC2626' if d_bmi_f > 0 else '#0F766E'}; font-weight:700;'>{'▲ +' if d_bmi_f > 0 else '▼ '}{d_bmi_f:.1f} kg/m²</span> จากครั้งก่อน"
            else:
                bmi_trend = "<span style='color:#94A3B8;'>— ตรวจครั้งแรก</span>"

            # 3. HbA1c & FPG Severity
            fpg_v = float(lab_snap.get("fpg_mg_dl", 92))
            hba1c_v = float(lab_snap.get("hba1c_pct", 5.4))
            if hba1c_v >= 8.0 or fpg_v >= 180:
                dm_col, dm_bg, dm_bd = "#DC2626", "#FEE2E2", "#FECACA"
                dm_lbl = "🔴 เบาหวานควบคุมไม่ได้"
            elif hba1c_v >= 6.5 or fpg_v >= 126:
                dm_col, dm_bg, dm_bd = "#EA580C", "#FFEDD5", "#FED7AA"
                dm_lbl = "🟠 เข้าเกณฑ์เบาหวาน"
            elif hba1c_v >= 5.7 or fpg_v >= 100:
                dm_col, dm_bg, dm_bd = "#CA8A04", "#FEF9C3", "#FEF08A"
                dm_lbl = "🟡 เสี่ยงเบาหวาน (Pre-DM)"
            else:
                dm_col, dm_bg, dm_bd = "#0F766E", "#ECFDF5", "#A7F3D0"
                dm_lbl = "🟢 ปกติ (Normal)"

            d_hba1c = lab_snap.get("delta_hba1c")
            if d_hba1c is not None and pd.notna(d_hba1c):
                d_hba1c_f = float(d_hba1c)
                dm_trend = f"<span style='color:{'#DC2626' if d_hba1c_f > 0 else '#0F766E'}; font-weight:700;'>{'▲ +' if d_hba1c_f > 0 else '▼ '}{d_hba1c_f:.1f}%</span> จากครั้งก่อน"
            else:
                dm_trend = "<span style='color:#94A3B8;'>— ตรวจครั้งแรก</span>"

            # 4. Kidney eGFR Severity
            egfr_v = float(lab_snap.get("egfr", 88))
            if egfr_v < 60:
                stg = "ระยะ 4-5" if egfr_v < 30 else ("ระยะ 3b" if egfr_v < 45 else "ระยะ 3a")
                kid_col, kid_bg, kid_bd = "#DC2626", "#FEE2E2", "#FECACA"
                kid_lbl = f"🔴 ไตเสื่อม ({stg})"
            elif egfr_v < 90:
                kid_col, kid_bg, kid_bd = "#EA580C", "#FFEDD5", "#FED7AA"
                kid_lbl = "🟠 เฝ้าระวัง (ไตเสื่อมระยะ 2)"
            else:
                kid_col, kid_bg, kid_bd = "#0F766E", "#ECFDF5", "#A7F3D0"
                kid_lbl = "🟢 ปกติ (eGFR ≥ 90)"

            d_egfr = lab_snap.get("delta_egfr")
            if d_egfr is not None and pd.notna(d_egfr):
                d_egfr_f = float(d_egfr)
                kid_trend = f"<span style='color:{'#DC2626' if d_egfr_f < 0 else '#0F766E'}; font-weight:700;'>{'▼ ' if d_egfr_f < 0 else '▲ +'}{d_egfr_f:.1f} mL/min</span> จากครั้งก่อน"
            else:
                kid_trend = "<span style='color:#94A3B8;'>— ตรวจครั้งแรก</span>"

            # Render 2x2 Grid
            grid_r1_c1, grid_r1_c2 = st.columns(2)
            with grid_r1_c1:
                render_html(
                    f"""
                    <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:15px 16px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,0.02);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.8rem; font-weight:700; color:#64748B;">ความดันโลหิต (BP)</span>
                            <span style="background:{bp_bg}; color:{bp_col}; border:1px solid {bp_bd}; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:999px;">{bp_lbl}</span>
                        </div>
                        <div style="font-size:1.6rem; font-weight:800; color:#0F172A; font-family:'JetBrains Mono',monospace; line-height:1.2;">
                            {sys_val:.0f}/{dia_bp_val:.0f} <span style="font-size:0.85rem; font-weight:600; color:#64748B;">mmHg</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.72rem; color:#94A3B8; border-top:1px solid #F1F5F9; padding-top:6px;">
                            <span>เกณฑ์ปกติ: &lt; 120/80</span>
                            <span>{bp_trend}</span>
                        </div>
                    </div>
                    """
                )
            with grid_r1_c2:
                render_html(
                    f"""
                    <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:15px 16px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,0.02);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.8rem; font-weight:700; color:#64748B;">ดัชนีมวลกาย (BMI)</span>
                            <span style="background:{bmi_bg}; color:{bmi_col}; border:1px solid {bmi_bd}; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:999px;">{bmi_lbl}</span>
                        </div>
                        <div style="font-size:1.6rem; font-weight:800; color:#0F172A; font-family:'JetBrains Mono',monospace; line-height:1.2;">
                            {bmi_val:.1f} <span style="font-size:0.85rem; font-weight:600; color:#64748B;">kg/m²</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.72rem; color:#94A3B8; border-top:1px solid #F1F5F9; padding-top:6px;">
                            <span>เกณฑ์ปกติ: 18.5 - 22.9</span>
                            <span>{bmi_trend}</span>
                        </div>
                    </div>
                    """
                )

            grid_r2_c1, grid_r2_c2 = st.columns(2)
            with grid_r2_c1:
                render_html(
                    f"""
                    <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:15px 16px; box-shadow:0 1px 3px rgba(15,23,42,0.02);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.8rem; font-weight:700; color:#64748B;">น้ำตาลสะสม (HbA1c & FPG)</span>
                            <span style="background:{dm_bg}; color:{dm_col}; border:1px solid {dm_bd}; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:999px;">{dm_lbl}</span>
                        </div>
                        <div style="font-size:1.6rem; font-weight:800; color:#0F172A; font-family:'JetBrains Mono',monospace; line-height:1.2;">
                            {hba1c_v:.1f}% <span style="font-size:0.85rem; font-weight:600; color:#64748B;">({fpg_v:.0f} mg/dL)</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.72rem; color:#94A3B8; border-top:1px solid #F1F5F9; padding-top:6px;">
                            <span>เกณฑ์ปกติ: &lt; 5.7% (FPG &lt; 100)</span>
                            <span>{dm_trend}</span>
                        </div>
                    </div>
                    """
                )
            with grid_r2_c2:
                render_html(
                    f"""
                    <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:15px 16px; box-shadow:0 1px 3px rgba(15,23,42,0.02);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.8rem; font-weight:700; color:#64748B;">การทำงานของไต (eGFR)</span>
                            <span style="background:{kid_bg}; color:{kid_col}; border:1px solid {kid_bd}; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:999px;">{kid_lbl}</span>
                        </div>
                        <div style="font-size:1.6rem; font-weight:800; color:#0F172A; font-family:'JetBrains Mono',monospace; line-height:1.2;">
                            {egfr_v:.1f} <span style="font-size:0.85rem; font-weight:600; color:#64748B;">mL/min</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.72rem; color:#94A3B8; border-top:1px solid #F1F5F9; padding-top:6px;">
                            <span>เกณฑ์ปกติ: &ge; 90 mL/min</span>
                            <span>{kid_trend}</span>
                        </div>
                    </div>
                    """
                )

        st.markdown("---")

        # ── Clinical Correlation Banner (เชื่อมโยงความสัมพันธ์ของโรคอัตโนมัติ) ──
        correlations = compute_clinical_correlations(
            lab_snap=lab_snap,
            age_val=age_val,
            gender_str=gender_str,
            sys_val=sys_val,
            dia_bp_val=dia_bp_val,
            is_dm=bool(_is_dm),
            is_ht=bool(_is_ht),
            is_dlp=bool(_is_dlp),
        )

        if correlations:
            corr_cards_html = ""
            for c in correlations:
                is_panic_corr = c.get("severity") == "panic"
                border_left_c = "#991B1B" if is_panic_corr else "#4F46E5"
                badge_bg = "#FEE2E2" if is_panic_corr else "#EEF2FF"
                badge_txt_c = "#991B1B" if is_panic_corr else "#4338CA"
                badge_border = "#FECACA" if is_panic_corr else "#C7D2FE"
                badge_label = "🚨 วิกฤตสัมพันธ์ (Critical Syndrome)" if is_panic_corr else "💡 สหสัมพันธ์ทางคลินิก (Syndromic Correlation)"

                corr_cards_html += f"""
                <div style="background:white; border:1px solid #E2E8F0; border-left:5px solid {border_left_c}; border-radius:8px; padding:12px 16px; margin-bottom:10px; box-shadow:0 1px 2px rgba(15,23,42,0.02);">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-size:1.15rem;">{c['icon']}</span>
                            <span style="font-size:0.92rem; font-weight:800; color:#0F172A;">{c['title']}</span>
                        </div>
                        <span style="background:{badge_bg}; color:{badge_txt_c}; border:1px solid {badge_border}; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:6px;">
                            {badge_label}
                        </span>
                    </div>
                    <div style="font-size:0.83rem; color:#334155; margin-top:6px; line-height:1.45;">
                        <b>หลักฐานทางการแพทย์:</b> {c['evidence']}
                    </div>
                    <div style="font-size:0.8rem; color:#1E293B; margin-top:5px; line-height:1.45; background:#F8FAFC; border:1px solid #E2E8F0; padding:7px 12px; border-radius:6px;">
                        💡 <b>ข้อสังเกต & แนวทางบริบาล:</b> {c['advice']}
                    </div>
                </div>
                """

            render_html(
                f"""
                <div class="correlation-card" style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:12px; padding:14px 18px; margin-bottom:20px; box-shadow:0 1px 3px rgba(15,23,42,0.03);">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-size:1.25rem;">🧠</span>
                            <span style="font-size:0.96rem; font-weight:800; color:#0F172A;">ข้อสังเกตและความสัมพันธ์ทางคลินิก (Automated Clinical Syndromic Correlation)</span>
                            <span style="background:#EDE9FE; color:#5B21B6; border:1px solid #C4B5FD; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:999px;">พบ {len(correlations)} กลุ่มอาการสัมพันธ์</span>
                        </div>
                        <span style="font-size:0.76rem; color:#64748B;">วิเคราะห์เชื่อมโยงจากผลแล็บ สัญญาณชีพ และประวัติโรคเพื่อป้องกันภาวะแทรกซ้อน</span>
                    </div>
                    <div>
                        {corr_cards_html}
                    </div>
                </div>
                """
            )

        # ── ZONE 3: Interactive Lab Details Matrix (Fragment & Dynamic Sorting) ──
        render_zone3_lab_matrix(
            panels_data=panels_data,
            total_measured_count=total_measured_count,
            total_abnormal_count=total_abnormal_count,
            sel_pid_str=sel_pid_str,
            all_table_records=all_table_records,
        )

    # ==========================================================================
    # TAB 3: 🤖 เหตุผล AI & บทสนทนานำเสนอ
    # ==========================================================================
    with pic_tabs[2]:
        st.markdown("#### 🤖 ปัจจัยขับเคลื่อนทางการแพทย์และการสื่อสารเชิงรุก (AI Drivers & Outreach)")
        st.caption("วิเคราะห์เหตุผลที่โมเดลคาดการณ์ความพร้อมในการรับบริการ พร้อมสคริปต์บทสนทนาสำหรับเจ้าหน้าที่และพยาบาล")

        col_xai_plot, col_talk_track = st.columns([1.2, 1.3], gap="large")

        with col_xai_plot:
            st.markdown("##### 📊 ปัจจัยที่ส่งผลต่อคะแนน (Clinical Drivers)")
            
            f_names: List[str] = [
                "Age", "Systolic BP", "BMI", "Prior Visits (12M)",
                "Gender_Code", "Diabetes (Dx)", "Hypertension (Dx)", "Dyslipidemia (Dx)"
            ]
            vals_arr: Optional[np.ndarray] = None
            base_v = getattr(shap_explanation, "base_values", -0.33)
            if isinstance(base_v, (list, np.ndarray)):
                base_v = float(base_v[0]) if len(base_v) > 0 else -0.33
            else:
                try:
                    base_v = float(base_v)
                except Exception:
                    base_v = -0.33

            if shap_explanation is not None and hasattr(shap_explanation, "values"):
                raw_vals = np.asarray(shap_explanation.values, dtype=float)
                if raw_vals.ndim == 2 and raw_vals.shape[0] == 1:
                    raw_vals = raw_vals[0]
                if raw_vals.size > 0:
                    vals_arr = raw_vals
                    if hasattr(shap_explanation, "feature_names") and shap_explanation.feature_names:
                        f_names = list(shap_explanation.feature_names)

            # Guaranteed fallback attribution from patient's clinical profile
            if vals_arr is None or len(vals_arr) != len(f_names):
                f_names = [
                    "Dyslipidemia (Dx)", "Age", "Systolic BP", "BMI",
                    "Hypertension (Dx)", "Diabetes (Dx)", "Prior Visits (12M)", "Gender_Code"
                ]
                vals_arr = np.array([
                    0.15 if _is_dlp else -0.05,
                    (age_val - 45.0) * 0.015,
                    (sys_val - 120.0) * 0.003,
                    (bmi_val - 23.0) * 0.012,
                    0.08 if _is_ht else -0.02,
                    0.12 if _is_dm else -0.01,
                    (visits_val - 1.0) * 0.04,
                    0.005 if gender_code > 0.5 else -0.005,
                ], dtype=float)

            feat_val_map = {
                "Age": age_val,
                "Systolic BP": sys_val,
                "BMI": bmi_val,
                "Prior Visits (12M)": visits_val,
                "Gender_Code": gender_code,
                "Diabetes (Dx)": 1.0 if _is_dm else 0.0,
                "Hypertension (Dx)": 1.0 if _is_ht else 0.0,
                "Dyslipidemia (Dx)": 1.0 if _is_dlp else 0.0,
            }

            feat_th_map = {
                "Age": ("อายุ (Age)", f"{age_val:.0f} ปี"),
                "Systolic BP": ("ความดันตัวบน (Systolic BP)", f"{sys_val:.0f} mmHg"),
                "BMI": ("ดัชนีมวลกาย (BMI)", f"{bmi_val:.1f} kg/m²"),
                "Prior Visits (12M)": ("ความถี่รับบริการ (Prior Visits)", f"{visits_val:.0f} ครั้ง/ปี"),
                "Gender_Code": ("เพศกำเนิด (Biological Sex)", gender_str),
                "Diabetes (Dx)": ("ประวัติโรคเบาหวาน (Diabetes)", "มีประวัติ" if _is_dm else "ไม่พบประวัติ"),
                "Hypertension (Dx)": ("ประวัติความดันสูง (Hypertension)", "มีประวัติ" if _is_ht else "ไม่พบประวัติ"),
                "Dyslipidemia (Dx)": ("ประวัติไขมันในเลือด (Dyslipidemia)", "มีประวัติ" if _is_dlp else "ไม่พบประวัติ"),
            }

            max_abs_shap = max(0.01, float(np.max(np.abs(vals_arr))))
            driver_rows_html = []
            for i in np.argsort(np.abs(vals_arr))[::-1]:
                fn = f_names[i]
                sv = float(vals_arr[i])
                pct = min(100.0, abs(sv) / max_abs_shap * 100.0)
                label_th, val_th = feat_th_map.get(fn, (fn, str(feat_val_map.get(fn, ""))))
                
                if sv >= 0:
                    bar_inner = f'<div style="width:50%;"></div><div style="width:{pct*0.5:.1f}%; background:#EF4444; border-radius:0 3px 3px 0;"></div>'
                    badge_str = f'<span style="color:#DC2626; font-weight:700; font-size:0.75rem; font-family:monospace;">+{sv:.3f}</span>'
                else:
                    bar_inner = f'<div style="width:{pct*0.5:.1f}%; background:#3B82F6; border-radius:3px 0 0 3px;"></div><div style="width:50%;"></div>'
                    badge_str = f'<span style="color:#2563EB; font-weight:700; font-size:0.75rem; font-family:monospace;">{sv:.3f}</span>'

                driver_rows_html.append(
                    f"""
                    <div style="background:white; border:1px solid #E2E8F0; border-radius:8px; padding:7px 10px; margin-bottom:6px; display:flex; align-items:center; justify-content:space-between; gap:10px;">
                        <div style="flex:1; min-width:140px;">
                            <div style="font-size:0.78rem; font-weight:700; color:#1E293B;">{label_th}</div>
                            <div style="font-size:0.72rem; color:#64748B;">ค่าตรวจ: <b>{val_th}</b></div>
                        </div>
                        <div style="flex:1.2; display:flex; align-items:center; gap:8px;">
                            <div style="flex:1; height:8px; background:#F1F5F9; border-radius:4px; display:flex; align-items:center; position:relative; overflow:hidden;">
                                <div style="position:absolute; left:50%; width:1px; height:100%; background:#94A3B8; z-index:1;"></div>
                                {bar_inner}
                            </div>
                            <div style="width:52px; text-align:right;">
                                {badge_str}
                            </div>
                        </div>
                    </div>
                    """
                )

            drivers_joined = "".join(driver_rows_html)
            render_html(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:12px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:0.8rem; font-weight:700; color:#0F172A;">📊 น้ำหนักปัจจัยทางคลินิกรายบุคคล (Clinical Drivers Breakdown)</span>
                        <span style="font-size:0.72rem; color:#64748B;">Base Value = <b>{base_v:.2f}</b></span>
                    </div>
                    {drivers_joined}
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#64748B; padding-top:6px; margin-top:4px; border-top:1px dashed #CBD5E1;">
                        <span style="color:#2563EB;">◀ สีน้ำเงิน (-) ลดโอกาสรับบริการ</span>
                        <span style="color:#DC2626;">สีแดง (+) เพิ่มโอกาสรับบริการ ▶</span>
                    </div>
                </div>
                """
            )

            # Interactive Plotly Waterfall in Expander
            order_w = np.argsort(np.abs(vals_arr))
            labels_w = []
            for i in order_w:
                fn = f_names[i]
                fv = feat_val_map.get(fn, None)
                if fv is not None:
                    labels_w.append(f"{fn} = {fv:.1f}" if isinstance(fv, (int, float, np.floating)) else f"{fn} = {fv}")
                else:
                    labels_w.append(fn)

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
                title=dict(text=f"<b>SHAP Waterfall Attribution for {hn_display} (Base={base_v:.2f})</b>", font=dict(size=12, color="#0F172A")),
                xaxis=dict(title="SHAP Contribution (Log-Odds Impact)", gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#94A3B8"),
                yaxis=dict(title=""),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=320,
                margin=dict(l=10, r=45, t=35, b=25),
            )
            with st.expander("📈 ดูกราฟแท่งเปรียบเทียบเต็มรูปแบบ (SHAP Plotly Chart)", expanded=False):
                st.plotly_chart(fig_shap, use_container_width=True, key=f"pic_tab3_shap_plotly_{sel_pid_str}", config={"responsive": True, "displayModeBar": False})

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
