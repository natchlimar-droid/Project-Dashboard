# -*- coding: utf-8 -*-
"""
🏥 PrimeCare Wellness Clinic — Main Dashboard Entrypoint
======================================================
High-Performance, Secure, and Production-Grade Streamlit Architecture:
- Lazy Loading: Views and heavy ML models loaded on-demand via Page Registry
- Security: HTML-escaped clinic branding, zero hardcoded local paths
- Robustness: Page Registry, Deep Linking (?page=...), and Error Boundary per page
- Date Presets: Dynamic fiscal/operational date ranges with incomplete-range guard
- Caching: Theme & Shell CSS cached via @st.cache_data
"""
from __future__ import annotations

import html
import importlib
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional external site-packages path injection via environment variable
_extra_site_packages = os.environ.get("PRIMECARE_EXTRA_SITE_PACKAGES")
if _extra_site_packages and os.path.isdir(_extra_site_packages) and _extra_site_packages not in sys.path:
    sys.path.insert(0, _extra_site_packages)

import pandas as pd
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="PrimeCare Wellness Clinic — Clinical Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand Constants & Defaults ────────────────────────────────────────────────
DEFAULT_CLINIC_NAME = "PrimeCare Wellness Clinic"
DEFAULT_CLINIC_TAGLINE = "Smart Clinical Intelligence & Healthcare Package Recommendation"
ANALYSIS_START = date(2025, 1, 1)


# ── Cached CSS Injection ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_bundled_css() -> str:
    """Reads and bundles styles/app_shell.css and styles/theme.css once."""
    base_dir = os.path.dirname(__file__)
    css_files = [
        os.path.join(base_dir, "styles", "app_shell.css"),
        os.path.join(base_dir, "styles", "theme.css"),
    ]
    parts: List[str] = []
    for fp in css_files:
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    parts.append(f.read())
            except Exception:
                pass
    if not parts:
        return ":root { --primary-color: #0F766E; } .stApp { background-color: #F8FAFC; }"
    return "\n".join(parts)


def inject_css() -> None:
    """Injects cached CSS stylesheets into Streamlit head."""
    bundled = load_bundled_css()
    st.markdown(f"<style>{bundled}</style>", unsafe_allow_html=True)


def get_clinic_name() -> str:
    """Retrieves current clinic brand name safely from session state."""
    raw = st.session_state.get("clinic_brand_name", DEFAULT_CLINIC_NAME)
    cleaned = str(raw).strip()
    return cleaned if cleaned else DEFAULT_CLINIC_NAME


# ── Page Registry & Route Definitions ─────────────────────────────────────────
PAGES: Dict[str, Dict[str, Any]] = {
    "exec": {
        "key": "exec",
        "title": "ภาพรวมผู้บริหาร",
        "full_label": "📊 ภาพรวมผู้บริหาร (Executive Overview)",
        "icon": "📊",
        "breadcrumb": "📊 ภาพรวมผู้บริหาร",
        "module": "views.executive_overview",
        "func": "render_executive_overview_view",
    },
    "lead": {
        "key": "lead",
        "title": "คัดกรองกลุ่มเป้าหมาย",
        "full_label": "🎯 คัดกรองกลุ่มเป้าหมาย (AI Lead Intelligence)",
        "icon": "🎯",
        "breadcrumb": "🎯 คัดกรองกลุ่มเป้าหมาย (AI Lead Intelligence)",
        "module": "views.lead_intelligence",
        "func": "render_lead_intelligence_view",
    },
    "patient": {
        "key": "patient",
        "title": "ข้อมูลผู้รับบริการรายคน",
        "full_label": "👤 ข้อมูลผู้รับบริการรายคน (Patient Insight Card)",
        "icon": "👤",
        "breadcrumb": "🔍 ค้นหาและดูแลผู้รับบริการรายคน",
        "module": "views.patient_card",
        "func": "render_patient_card_view",
    },
    "clinical": {
        "key": "clinical",
        "title": "ติดตามโรคเรื้อรัง",
        "full_label": "🩺 ติดตามโรคเรื้อรัง (Clinical Command Center)",
        "icon": "🩺",
        "breadcrumb": "🩺 ศูนย์ติดตามโรคเรื้อรัง (Clinical Command Center)",
        "module": "views.clinical_tracking",
        "func": "render_clinical_tracking_view",
    },
}

PAGE_ORDER = ["exec", "lead", "patient", "clinical"]

PAGE_ALIASES: Dict[str, str] = {
    "exec": "exec",
    "executive": "exec",
    "executive overview": "exec",
    "executive strategy overview": "exec",
    "ภาพรวมผู้บริหาร": "exec",
    "📊 ภาพรวมผู้บริหาร": "exec",
    ":material/insights: executive strategy overview": "exec",
    "lead": "lead",
    "lead intelligence": "lead",
    "ai lead intelligence": "lead",
    "คัดกรองกลุ่มเป้าหมาย": "lead",
    "🎯 คัดกรองกลุ่มเป้าหมาย": "lead",
    "🎯 ai lead intelligence": "lead",
    ":material/my_location: ai lead intelligence": "lead",
    "patient": "patient",
    "patient insight card": "patient",
    "ข้อมูลผู้รับบริการรายคน": "patient",
    "👤 ข้อมูลผู้รับบริการรายคน": "patient",
    "👤 patient insight card": "patient",
    ":material/search: patient insight card": "patient",
    "clinical": "clinical",
    "clinical command center": "clinical",
    "ติดตามโรคเรื้อรัง": "clinical",
    "🩺 ติดตามโรคเรื้อรัง": "clinical",
    ":material/vital_signs: clinical command center": "clinical",
}


def resolve_page_key(raw_input: Optional[str]) -> str:
    """Resolves arbitrary page names, query params, or aliases to a canonical page key."""
    if not raw_input:
        return "patient"
    norm = str(raw_input).strip().lower()
    return PAGE_ALIASES.get(norm, "patient")


# ── Date Filter Helpers ───────────────────────────────────────────────────────
def get_date_preset_range(preset_name: str) -> Tuple[date, date]:
    """Computes dynamic operational date ranges based on chosen preset."""
    today = date.today()
    if preset_name == "ปีนี้ (This Year)":
        return date(today.year, 1, 1), today
    elif preset_name == "ไตรมาสนี้ (This Quarter)":
        q_month = ((today.month - 1) // 3) * 3 + 1
        return date(today.year, q_month, 1), today
    elif preset_name == "90 วันล่าสุด (Last 90 Days)":
        return today - timedelta(days=90), today
    elif preset_name == "30 วันล่าสุด (Last 30 Days)":
        return today - timedelta(days=30), today
    # Default: ทั้งหมด (All Time)
    return ANALYSIS_START, today


# ── Error Boundary Wrapper ────────────────────────────────────────────────────
def run_with_error_boundary(page_title: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Executes a view function protected by an isolated visual error boundary."""
    try:
        func(*args, **kwargs)
    except Exception as exc:
        escaped_title = html.escape(page_title)
        st.markdown(
            f"""
            <div class="error-boundary-card">
                <div class="error-boundary-title">⚠️ เกิดข้อผิดพลาดในการโหลดหน้า {escaped_title}</div>
                <div class="error-boundary-desc">
                    ระบบไม่สามารถประมวลผลข้อมูลในหน้านี้ได้ชั่วคราว ข้อมูลในส่วนอื่นของระบบยังคงทำงานได้ตามปกติ 
                    กรุณากดปุ่ม <b>"รีเฟรชข้อมูล"</b> ในแถบเมนูซ้าย หรือตรวจสอบรายละเอียดข้อผิดพลาดด้านล่าง
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🔍 ดูรายละเอียดข้อผิดพลาดทางเทคนิค (Technical Error Details)"):
            st.exception(exc)


# ==============================================================================
# Main Controller
# ==============================================================================
def main() -> None:
    inject_css()

    if "clinic_brand_name" not in st.session_state:
        st.session_state["clinic_brand_name"] = DEFAULT_CLINIC_NAME

    cname = get_clinic_name()
    escaped_cname = html.escape(cname)
    escaped_tagline = html.escape(DEFAULT_CLINIC_TAGLINE)

    # ── 1. Resolve Navigation State (Deep Link & Priority Router) ─────────────
    # Priority: pending_nav_section > external URL change > active session state > default
    query_page = st.query_params.get("page")
    if "pending_nav_section" in st.session_state:
        target_page_key = resolve_page_key(st.session_state.pop("pending_nav_section"))
        st.session_state["active_page_key"] = target_page_key
        st.session_state["main_navigation_radio"] = PAGES[target_page_key]["full_label"]
        st.session_state["last_synced_page"] = target_page_key
        st.query_params["page"] = target_page_key
    elif "main_navigation_radio" not in st.session_state:
        # First load of session
        initial_key = resolve_page_key(query_page) if query_page else "patient"
        st.session_state["active_page_key"] = initial_key
        st.session_state["main_navigation_radio"] = PAGES[initial_key]["full_label"]
        st.session_state["last_synced_page"] = initial_key
        st.query_params["page"] = initial_key
        target_page_key = initial_key
    else:
        # Check if URL parameter changed externally (e.g. browser back/forward or direct bookmark)
        if query_page and query_page != st.session_state.get("last_synced_page"):
            target_page_key = resolve_page_key(query_page)
            st.session_state["active_page_key"] = target_page_key
            st.session_state["main_navigation_radio"] = PAGES[target_page_key]["full_label"]
            st.session_state["last_synced_page"] = target_page_key
        else:
            target_page_key = st.session_state.get("active_page_key", "patient")

    if target_page_key not in PAGES:
        target_page_key = "patient"

    # ── 2. Sidebar Brand Header & Clinic Name Customizer ──────────────────────
    with st.sidebar:
        st.markdown(
            f"""
            <div class="clinic-brand-card">
            <div class="clinic-brand-header">
                <div class="clinic-brand-icon">🏥</div>
                <div>
                    <div class="clinic-brand-title">{escaped_cname}</div>
                    <div class="clinic-brand-tagline">{escaped_tagline}</div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ ตั้งค่าชื่อคลินิก (Clinic Brand)", expanded=False):
            with st.form("form_brand_update", clear_on_submit=False):
                new_cname = st.text_input("ชื่อคลินิก / โรงพยาบาล:", value=cname)
                submitted = st.form_submit_button("บันทึกชื่อคลินิก", use_container_width=True)
                if submitted and new_cname.strip() and new_cname.strip() != cname:
                    st.session_state["clinic_brand_name"] = new_cname.strip()
                    st.toast("✅ บันทึกชื่อคลินิกเรียบร้อยแล้ว")
                    st.rerun()

        st.divider()

        # ── 3. Role-Based Navigation Menu ─────────────────────────────────────
        nav_display_options = [PAGES[k]["full_label"] for k in PAGE_ORDER]

        def on_nav_change():
            chosen = st.session_state.get("main_navigation_radio")
            for k in PAGE_ORDER:
                if PAGES[k]["full_label"] == chosen:
                    st.session_state["active_page_key"] = k
                    st.session_state["last_synced_page"] = k
                    st.query_params["page"] = k
                    break

        st.radio(
            "📌 เมนูหลัก (Workflow Navigation):",
            nav_display_options,
            key="main_navigation_radio",
            on_change=on_nav_change,
        )

        # Reflect any navigation change in active page
        target_page_key = st.session_state.get("active_page_key", target_page_key)

        # ── 4. Contextual Sidebar Filters (Only Rendered When Relevant) ───────
        start_date_val: Optional[date] = None
        end_date_val: Optional[date] = None

        if target_page_key == "exec":
            st.divider()
            st.caption("🎛️ ตัวกรองข้อมูลผู้บริหาร (Executive Filters):")
            preset_choice = st.selectbox(
                "ช่วงเวลาที่วิเคราะห์ (Preset):",
                ["ทั้งหมด (All)", "ปีนี้ (This Year)", "ไตรมาสนี้ (This Quarter)", "90 วันล่าสุด (Last 90 Days)", "30 วันล่าสุด (Last 30 Days)", "กำหนดเอง (Custom)"],
                index=0,
                key="exec_date_preset",
            )
            if preset_choice == "กำหนดเอง (Custom)":
                custom_range = st.date_input(
                    "เลือกช่วงวันที่เริ่มต้น - สิ้นสุด:",
                    value=(ANALYSIS_START, date.today()),
                    key="exec_custom_date_range",
                )
                if isinstance(custom_range, (list, tuple)):
                    if len(custom_range) == 2:
                        start_date_val, end_date_val = custom_range[0], custom_range[1]
                    elif len(custom_range) == 1:
                        start_date_val = custom_range[0]
                        st.caption("ℹ️ กรุณาเลือกวันสิ้นสุดช่วงเวลาในปฏิทิน")
            else:
                start_date_val, end_date_val = get_date_preset_range(preset_choice)

            if start_date_val and end_date_val:
                st.caption(f"📅 ช่วงข้อมูล: **{start_date_val.strftime('%d/%m/%Y')}** ถึง **{end_date_val.strftime('%d/%m/%Y')}**")

            st.selectbox(
                "🏥 สาขาบริการ:",
                ["ทุกสาขา (All Branches)", "สาขาหลัก (Main Clinic)", "สาขาในเมือง (Urban Clinic)"],
                key="exec_branch_filter",
            )

        elif target_page_key == "lead":
            st.divider()
            st.caption("🎛️ ตัวกรองการคัดกรอง (Lead Filters):")
            st.slider(
                "🎯 เกณฑ์คะแนนโอกาสซื้อ (%):",
                min_value=0.0,
                max_value=90.0,
                value=40.0,
                step=5.0,
                key="lead_sb_score",
            )
            st.selectbox(
                "📋 กลุ่มสถานะผู้ป่วย (Journey Stage):",
                ["ทั้งหมด (All Stages)", "Wellness / General OPD", "Chronic Care OPD", "Frequent Active OPD"],
                key="lead_sb_stage",
            )
            st.selectbox(
                "🔥 ระดับความเร่งด่วน:",
                ["ทั้งหมด (All Priorities)", "🔥 High Priority (≥60%)", "⚡ Medium Priority (40-59%)", "🌱 Regular (<40%)"],
                key="lead_sb_prio",
            )

        elif target_page_key == "clinical":
            st.divider()
            st.caption("🎛️ ตัวกรองศูนย์โรคเรื้อรัง (Clinical Filters):")
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

        # ── 5. System Utilities (Clear Cache Button) ──────────────────────────
        st.divider()
        if st.button("🔄 รีเฟรชข้อมูล (Clear Cache)", use_container_width=True, help="ล้างข้อมูลแคชในหน่วยความจำและโหลดข้อมูลใหม่ล่าสุด"):
            st.cache_data.clear()
            st.toast("✅ ล้างข้อมูลแคชเรียบร้อยแล้ว กำลังโหลดข้อมูลใหม่...")
            st.rerun()

    # ── 6. Accessible Breadcrumb Navigation ────────────────────────────────────
    active_page_meta = PAGES[target_page_key]
    breadcrumb_label = html.escape(active_page_meta["breadcrumb"])

    st.markdown(
        f"""
        <nav class="app-breadcrumb" aria-label="breadcrumb">
            <ol>
                <li><span>🏥 {escaped_cname}</span></li>
                <li><span>&rsaquo;</span></li>
                <li class="breadcrumb-item-active" aria-current="page"><span>{breadcrumb_label}</span></li>
            </ol>
        </nav>
        """,
        unsafe_allow_html=True,
    )

    # ── 7. Lazy Data Loading & View Dispatching ────────────────────────────────
    if target_page_key == "exec":
        from services.data_loader import load_real_sales_data, load_synthetic_dataset
        with st.spinner("กำลังโหลดข้อมูลภาพรวมผู้บริหารและยอดขาย..."):
            sales_df = load_real_sales_data()
            snap_df, enc_df, _ = load_synthetic_dataset()
        
        mod = importlib.import_module("views.executive_overview")
        run_with_error_boundary(
            active_page_meta["title"],
            mod.render_executive_overview_view,
            sales_df=sales_df,
            full_sales_df=sales_df,
            raw_clinical_df=enc_df,
            start_d=start_date_val,
            end_d=end_date_val,
        )

    elif target_page_key == "lead":
        from services.data_loader import load_synthetic_dataset
        with st.spinner("กำลังประมวลผลโมเดล AI Lead Scoring & Precision Cross-Sell..."):
            snap_df, enc_df, _ = load_synthetic_dataset()
            clin_base = enc_df if not enc_df.empty else snap_df

        min_p = float(st.session_state.get("lead_sb_score", 40.0))
        stg_f = str(st.session_state.get("lead_sb_stage", "All"))
        prio_f = str(st.session_state.get("lead_sb_prio", "All"))

        mod = importlib.import_module("views.lead_intelligence")
        run_with_error_boundary(
            active_page_meta["title"],
            mod.render_lead_intelligence_view,
            raw_clinical_df=clin_base,
            min_propensity=min_p,
            journey_stage_filter=stg_f,
            priority_filter=prio_f,
        )

    elif target_page_key == "patient":
        from services.data_loader import load_synthetic_dataset
        with st.spinner("กำลังโหลดแฟ้มประวัติสุขภาพผู้รับบริการ..."):
            snap_df, enc_df, _ = load_synthetic_dataset()
            if not snap_df.empty:
                summary_pts = snap_df.copy()
                summary_pts.index = summary_pts["patient_id"].astype(str)
            else:
                summary_pts = pd.DataFrame()

        mod = importlib.import_module("views.patient_card")
        run_with_error_boundary(
            active_page_meta["title"],
            mod.render_patient_card_view,
            avail_df=snap_df,
            summary_pts=summary_pts,
            dv=enc_df,
        )

    elif target_page_key == "clinical":
        from services.data_loader import load_synthetic_dataset
        with st.spinner("กำลังรวบรวมข้อมูลศูนย์ติดตามโรคเรื้อรัง (Clinical Command Center)..."):
            snap_df, enc_df, _ = load_synthetic_dataset()

        mod = importlib.import_module("views.clinical_tracking")
        run_with_error_boundary(
            active_page_meta["title"],
            mod.render_clinical_tracking_view,
            dv=enc_df,
            raw_clinical_df=snap_df,
        )


# ==============================================================================
# Backward Compatibility Wrappers (Zero-Regression Contract with Unit Tests)
# ==============================================================================
def render_executive_system_overview(*args: Any, **kwargs: Any) -> Any:
    mod = importlib.import_module("views.executive_overview")
    return mod.render_executive_system_overview(*args, **kwargs)


def render_lead_scoring_xai_dashboard(*args: Any, **kwargs: Any) -> Any:
    mod = importlib.import_module("views.lead_intelligence")
    return mod.render_lead_scoring_xai_dashboard(*args, **kwargs)


def render_patient_profile(*args: Any, **kwargs: Any) -> Any:
    mod = importlib.import_module("views.patient_card")
    return mod.render_patient_card_view(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Lazy module-level attribute loader for zero-startup-penalty backwards compatibility."""
    if name == "view_lead":
        return importlib.import_module("views.lead_intelligence")
    elif name == "view_patient":
        return importlib.import_module("views.patient_card")
    elif name == "view_exec":
        return importlib.import_module("views.executive_overview")
    elif name == "view_clin":
        return importlib.import_module("views.clinical_tracking")
    elif name == "get_xgboost_model":
        from services.ml_engine import get_xgboost_model
        return get_xgboost_model
    elif name == "get_cached_shap_explainer":
        from services.ml_engine import get_cached_shap_explainer
        return get_cached_shap_explainer
    elif name == "predict_patient_propensity":
        from services.ml_engine import predict_patient_propensity
        return predict_patient_propensity
    elif name == "resolve_patient_lab_snapshot":
        from services.data_loader import resolve_patient_lab_snapshot
        return resolve_patient_lab_snapshot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
