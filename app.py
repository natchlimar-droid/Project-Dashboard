# -*- coding: utf-8 -*-
"""
🏥 PrimeCare Wellness Clinic — Main Dashboard Entrypoint
======================================================
High-Performance, Secure, and Production-Grade Streamlit Architecture:
- Lazy Loading: Views and heavy ML models loaded on-demand via Page Registry
- Security: HTML-escaped clinic branding, zero hardcoded local paths, PHI-safe error boundaries
- Robustness: Page Registry, Deep Linking (?page=...), Dynamic filter signature inspection
- Date Presets: Dynamic fiscal/operational date ranges with incomplete-range guard
- Caching: Theme & Shell CSS cached with mtime invalidation
- Compatibility: Top-level data loader exports and lazy __getattr__ for unit tests
"""
from __future__ import annotations

import html
import importlib
import inspect
import logging
import os
import sys
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional external site-packages path injection via environment variable
_extra_site_packages = os.environ.get("PRIMECARE_EXTRA_SITE_PACKAGES")
if _extra_site_packages and os.path.isdir(_extra_site_packages) and _extra_site_packages not in sys.path:
    sys.path.insert(0, _extra_site_packages)

import pandas as pd
import streamlit as st

# Top-level exports preserving contract for tests (e.g. unittest.mock.patch("app.load_synthetic_dataset"))
from services.data_loader import load_real_sales_data, load_synthetic_dataset

# Configure Streamlit page settings
st.set_page_config(
    page_title="PrimeCare Wellness Clinic — Clinical Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("primecare.app")

# ── Brand Constants & Defaults ────────────────────────────────────────────────
DEFAULT_CLINIC_NAME = "PrimeCare Wellness Clinic"
DEFAULT_CLINIC_TAGLINE = "Smart Clinical Intelligence & Healthcare Package Recommendation"
DEFAULT_PAGE_KEY = "exec"
NAV_KEY = "active_page_key"
ANALYSIS_START = date(2025, 1, 1)
PRIMECARE_DEBUG = os.environ.get("PRIMECARE_DEBUG", "0").strip() == "1"

THEME_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
APP_SHELL_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "app_shell.css")


# ── Cached CSS Injection with mtime Invalidation ─────────────────────────────
def get_css_mtime() -> float:
    """Calculates max modification time of stylesheet files for auto-cache invalidation."""
    mtimes: List[float] = []
    for fp in [THEME_CSS_PATH, APP_SHELL_CSS_PATH]:
        if os.path.exists(fp):
            try:
                mtimes.append(os.path.getmtime(fp))
            except Exception:
                pass
    return max(mtimes) if mtimes else 0.0


@st.cache_data(show_spinner=False)
def load_bundled_css(_mtime: float) -> Tuple[str, str]:
    """Reads styles/theme.css then styles/app_shell.css with font import block hoisted to top."""
    font_block = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;800&display=swap');"
    css_parts: List[str] = []
    # Load theme.css first, app_shell.css second so shell overrides theme rules
    for fp in [THEME_CSS_PATH, APP_SHELL_CSS_PATH]:
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                    content = content.replace(font_block, "")
                    css_parts.append(content)
            except Exception:
                pass
    if not css_parts:
        return font_block, ":root { --primary-teal: #0F766E; } .stApp { background-color: #F8FAFC; }"
    return font_block, "\n".join(css_parts)


def inject_css() -> None:
    """Injects cached CSS stylesheets into Streamlit head."""
    font_block, bundled = load_bundled_css(get_css_mtime())
    st.markdown(f"<style>{font_block}\n{bundled}</style>", unsafe_allow_html=True)


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
        return DEFAULT_PAGE_KEY
    norm = str(raw_input).strip().lower()
    return PAGE_ALIASES.get(norm, DEFAULT_PAGE_KEY)


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
        return today - timedelta(days=89), today
    elif preset_name == "30 วันล่าสุด (Last 30 Days)":
        return today - timedelta(days=29), today
    # Default: ทั้งหมด (All Time)
    return ANALYSIS_START, today


# ── Safe View Dispatcher & Error Boundary ─────────────────────────────────────
def dispatch_view(func: Callable[..., Any], **kwargs: Any) -> None:
    """Inspects view function signature and passes only declared keyword arguments."""
    sig = inspect.signature(func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    func(**filtered_kwargs)


def run_with_error_boundary(page_title: str, func: Callable[..., Any], **kwargs: Any) -> None:
    """Executes a view function protected by a PHI-safe error boundary."""
    try:
        dispatch_view(func, **kwargs)
    except Exception as exc:
        exc_type = type(exc).__name__
        logger.exception("Error executing view '%s': %s", page_title, exc)
        print(f"[ERROR] Exception in view '{page_title}': {exc_type}: {exc}", file=sys.stderr)

        escaped_title = html.escape(page_title)
        st.markdown(
            f"""
            <div class="pc-error-boundary error-boundary-card">
                <div class="pc-error-title error-boundary-title">⚠️ เกิดข้อผิดพลาดในการโหลดหน้า {escaped_title}</div>
                <div class="pc-error-desc error-boundary-desc">
                    ระบบไม่สามารถประมวลผลข้อมูลในหน้านี้ได้ชั่วคราว ข้อมูลในส่วนอื่นของระบบยังคงทำงานได้ตามปกติ 
                    กรุณากดปุ่ม <b>"รีเฟรชข้อมูล"</b> ในแถบเมนูซ้าย หรือติดต่อผู้ดูแลระบบ
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if PRIMECARE_DEBUG:
            st.caption("🔧 ข้อมูลทางเทคนิคสำหรับนักพัฒนา (PRIMECARE_DEBUG=1):")
            st.exception(exc)
        else:
            with st.expander("ℹ️ ดูข้อมูลเบื้องต้นสำหรับแจ้งทีมพัฒนาระบบ"):
                st.code(f"Module: {getattr(func, '__module__', 'unknown')}\nException: {exc_type}", language="text")


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

    # ── 1. Resolve Navigation State ───────────────────────────────────────────
    # Priority: pending_nav_section > active_nav_page > URL query param > NAV_KEY in state > DEFAULT_PAGE_KEY
    query_page = st.query_params.get("page")
    if "pending_nav_section" in st.session_state:
        target_page_key = resolve_page_key(st.session_state.pop("pending_nav_section"))
    elif "active_nav_page" in st.session_state:
        target_page_key = resolve_page_key(st.session_state.pop("active_nav_page"))
    elif query_page and query_page != st.session_state.get("last_synced_page"):
        target_page_key = resolve_page_key(query_page)
    else:
        target_page_key = resolve_page_key(st.session_state.get(NAV_KEY, DEFAULT_PAGE_KEY))

    if target_page_key not in PAGES:
        target_page_key = DEFAULT_PAGE_KEY

    st.session_state[NAV_KEY] = target_page_key
    st.session_state["last_synced_page"] = target_page_key
    st.query_params["page"] = target_page_key

    # ── 2. Sidebar Brand Header & Clinic Name Customizer ──────────────────────
    with st.sidebar:
        st.markdown(
            f"""
            <div class="pc-brand-card clinic-brand-card">
                <div class="pc-brand-header clinic-brand-header">
                    <div class="pc-brand-icon clinic-brand-icon">🏥</div>
                    <div>
                        <div class="pc-brand-title clinic-brand-title">{escaped_cname}</div>
                        <div class="pc-brand-tagline clinic-brand-tagline">{escaped_tagline}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def save_clinic_brand_name() -> None:
            new_val = str(st.session_state.get("input_brand_name", "")).strip()
            if new_val:
                st.session_state["clinic_brand_name"] = new_val
                st.toast("✅ บันทึกชื่อคลินิกเรียบร้อยแล้ว")

        with st.expander("⚙️ ตั้งค่าชื่อคลินิก (Clinic Brand)", expanded=False):
            st.text_input("ชื่อคลินิก / โรงพยาบาล:", value=cname, key="input_brand_name")
            st.button("บันทึกชื่อคลินิก", on_click=save_clinic_brand_name, use_container_width=True)

        st.divider()

        # ── 3. Role-Based Navigation Menu ─────────────────────────────────────
        nav_display_options = [PAGES[k]["full_label"] for k in PAGE_ORDER]
        current_idx = PAGE_ORDER.index(target_page_key)

        def on_nav_change():
            chosen = st.session_state.get("sidebar_nav_radio")
            for k in PAGE_ORDER:
                if PAGES[k]["full_label"] == chosen:
                    st.session_state[NAV_KEY] = k
                    st.session_state["last_synced_page"] = k
                    st.query_params["page"] = k
                    break

        st.radio(
            "📌 เมนูหลัก (Workflow Navigation):",
            nav_display_options,
            index=current_idx,
            key="sidebar_nav_radio",
            on_change=on_nav_change,
        )

        # Reflect any navigation change
        target_page_key = st.session_state.get(NAV_KEY, target_page_key)

        # ── 4. Contextual Sidebar Filters (Only Rendered When View Accepts Them)
        start_date_val: Optional[date] = None
        end_date_val: Optional[date] = None

        if target_page_key == "exec":
            st.divider()
            st.caption("🎛️ ตัวกรองช่วงเวลาผู้บริหาร (Executive Filters):")
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
                if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
                    start_date_val, end_date_val = custom_range[0], custom_range[1]
                else:
                    st.caption("ℹ️ กรุณาเลือกวันสิ้นสุดช่วงเวลาในปฏิทิน")
                    start_date_val = None
                    end_date_val = None
            else:
                start_date_val, end_date_val = get_date_preset_range(preset_choice)

            if start_date_val and end_date_val:
                st.caption(f"📅 ช่วงข้อมูล: **{start_date_val.strftime('%d/%m/%Y')}** ถึง **{end_date_val.strftime('%d/%m/%Y')}**")

        # ── 5. System Utilities (Clear Cache Button) ──────────────────────────
        st.divider()
        if st.button("🔄 รีเฟรชข้อมูล (Clear Cache)", use_container_width=True, help="ล้างข้อมูลแคชในหน่วยความจำและโหลดข้อมูลใหม่ล่าสุด"):
            st.cache_data.clear()
            st.toast("✅ ล้างข้อมูลแคชเรียบร้อยแล้ว กำลังโหลดข้อมูลใหม่...")
            st.rerun()
        st.caption("🚀 Version: **v2.4.0** | Clinical KDIGO & KM Analytics")

    # ── 6. Accessible Breadcrumb Navigation ────────────────────────────────────
    active_page_meta = PAGES[target_page_key]
    breadcrumb_label = html.escape(active_page_meta["breadcrumb"])

    st.markdown(
        f"""
        <nav class="pc-breadcrumb app-breadcrumb" aria-label="breadcrumb">
            <ol>
                <li><span>🏥 {escaped_cname}</span></li>
                <li><span>&rsaquo;</span></li>
                <li class="breadcrumb-item-active" aria-current="page"><span>{breadcrumb_label}</span></li>
            </ol>
        </nav>
        """,
        unsafe_allow_html=True,
    )

    # ── 7. Lazy Data Loading & View Dispatching (Wrapped in Spinner) ───────────
    if target_page_key == "exec":
        with st.spinner("กำลังโหลดข้อมูลภาพรวมผู้บริหารและยอดขาย..."):
            sales_df = load_real_sales_data()
            snap_df, enc_df, _ = load_synthetic_dataset()
            mod = importlib.import_module("views.executive_overview")
            mod = importlib.reload(mod)
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
        with st.spinner("กำลังประมวลผลโมเดล AI Lead Scoring & Precision Cross-Sell..."):
            snap_df, enc_df, _ = load_synthetic_dataset()
            clin_base = snap_df if not snap_df.empty else enc_df
            mod = importlib.import_module("views.lead_intelligence")
            mod = importlib.reload(mod)
            run_with_error_boundary(
                active_page_meta["title"],
                mod.render_lead_intelligence_view,
                raw_clinical_df=clin_base,
            )

    elif target_page_key == "patient":
        with st.spinner("กำลังโหลดแฟ้มประวัติสุขภาพผู้รับบริการ..."):
            snap_df, enc_df, _ = load_synthetic_dataset()
            if not snap_df.empty:
                summary_pts = snap_df.copy()
                summary_pts.index = summary_pts["patient_id"].astype(str)
            else:
                summary_pts = pd.DataFrame()

            mod = importlib.import_module("views.patient_card")
            mod = importlib.reload(mod)
            run_with_error_boundary(
                active_page_meta["title"],
                mod.render_patient_card_view,
                avail_df=snap_df,
                summary_pts=summary_pts,
                dv=enc_df,
            )

    elif target_page_key == "clinical":
        with st.spinner("กำลังรวบรวมข้อมูลศูนย์ติดตามโรคเรื้อรัง (Clinical Command Center)..."):
            snap_df, enc_df, _ = load_synthetic_dataset()
            mod = importlib.import_module("views.clinical_tracking")
            mod = importlib.reload(mod)
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
