# -*- coding: utf-8 -*-
"""
Package Service
Handles merging real sales data with the Package Catalog, sorting by VN,
and identifying mismatches.
"""

import copy
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from logic.config import PACKAGE_CATALOG, PACKAGE_KEYWORD_ALIASES, CATEGORY_ORDER, is_excluded_package


def compute_package_stats(
    sales_df: Optional[pd.DataFrame],
    packages_dict: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Groups sales_df by package_name to sum total_vn and total_revenue,
    merges stats into packages_dict, sorts packages inside each category
    by total_vn descending, and tracks any unmatched packages.

    Returns:
        (updated_packages_dict, mismatches_list)
    """
    if packages_dict is None:
        pkgs = copy.deepcopy(PACKAGE_CATALOG)
    else:
        pkgs = copy.deepcopy(packages_dict)

    mismatches: List[Dict[str, Any]] = []

    # If sales_df is empty or None, return packages with 0 stats
    if sales_df is None or sales_df.empty:
        for cat_key in CATEGORY_ORDER:
            if cat_key in pkgs:
                cat = pkgs[cat_key]
                for p in cat.get("packages", []):
                    p["total_vn"] = 0
                    p["total_revenue"] = 0.0
                cat["total_vn"] = 0
                cat["total_revenue"] = 0.0
        return pkgs, [{"reason": "Sales dataframe is empty or unavailable"}]

    # Normalize sales columns
    sdf = sales_df.copy()
    col_name_map = {}
    for c in sdf.columns:
        c_str = str(c).strip().lower()
        if c_str in ["package_name", "group request"]:
            col_name_map[c] = "package_name"
        elif c_str in ["transactions", "count of vn", "vn", "total_vn"]:
            col_name_map[c] = "transactions"
        elif c_str in ["net_amount", "sum of สุทธิ", "revenue", "sum of revenue", "total_revenue"]:
            col_name_map[c] = "net_amount"

    sdf = sdf.rename(columns=col_name_map)

    if "package_name" not in sdf.columns:
        return pkgs, [{"reason": "Missing package_name column in sales data"}]

    if "transactions" not in sdf.columns:
        sdf["transactions"] = 1
    if "net_amount" not in sdf.columns:
        sdf["net_amount"] = 0.0

    sdf["package_name"] = sdf["package_name"].astype(str).str.strip()
    sdf = sdf[~sdf["package_name"].apply(is_excluded_package)].copy()
    sdf["transactions"] = pd.to_numeric(sdf["transactions"], errors="coerce").fillna(0).astype(int)
    sdf["net_amount"] = pd.to_numeric(sdf["net_amount"], errors="coerce").fillna(0.0).astype(float)

    # Group sales
    grouped = sdf.groupby("package_name").agg(
        total_vn=("transactions", "sum"),
        total_revenue=("net_amount", "sum")
    ).reset_index()

    used_sales_indices = set()

    for cat_key in CATEGORY_ORDER:
        cat = pkgs.get(cat_key)
        if not cat:
            continue

        for p in cat.get("packages", []):
            short_name = p.get("short_name", "")
            full_name = p.get("name", "")

            # 1. Exact match on short_name or full_name
            exact_mask = (grouped["package_name"].str.lower() == short_name.lower()) | \
                         (grouped["package_name"].str.lower() == full_name.lower())

            matched_rows = grouped[exact_mask]

            # 2. If no exact match, use alias keyword matching
            if matched_rows.empty:
                aliases = PACKAGE_KEYWORD_ALIASES.get(short_name, [short_name.lower()])
                pattern = "|".join([a.lower() for a in aliases])
                keyword_mask = grouped["package_name"].str.lower().str.contains(pattern, na=False, regex=True)
                matched_rows = grouped[keyword_mask]

            if not matched_rows.empty:
                p["total_vn"] = int(matched_rows["total_vn"].sum())
                p["total_revenue"] = float(matched_rows["total_revenue"].sum())
                used_sales_indices.update(matched_rows.index.tolist())
            else:
                p["total_vn"] = 0
                p["total_revenue"] = 0.0
                mismatches.append({
                    "category": cat_key,
                    "package_name": full_name,
                    "short_name": short_name,
                    "reason": "ไม่พบรายการขายในฐานข้อมูล RAWDATA Excel"
                })

        # Sort packages within category by total_vn descending
        cat["packages"].sort(key=lambda x: (x.get("total_vn", 0), x.get("total_revenue", 0)), reverse=True)

        # Category level aggregation
        cat["total_vn"] = sum(p.get("total_vn", 0) for p in cat["packages"])
        cat["total_revenue"] = sum(p.get("total_revenue", 0.0) for p in cat["packages"])

    return pkgs, mismatches


# ==============================================================================
# Unified Precision Package Matching & Quotation Engine
# ==============================================================================

UNIFIED_ADDON_CATALOG: Dict[str, Dict[str, Any]] = {
    "ADDON_US_ABDOMEN": {
        "code": "ADDON_US_ABDOMEN",
        "name": "อัลตราซาวด์ช่องท้องส่วนบนและล่าง (Ultrasound Whole Abdomen)",
        "short_name": "อัลตราซาวด์ช่องท้อง",
        "price": 990,
        "organ": "ตับ/ไต/ถุงน้ำดี/ช่องท้อง",
        "indication": "ตรวจคัดกรองนิ่ว ไขมันพอกตับ และก้อนในช่องท้อง",
    },
    "ADDON_EKG": {
        "code": "ADDON_EKG",
        "name": "ตรวจคลื่นไฟฟ้าหัวใจ (EKG 12-Leads)",
        "short_name": "คลื่นไฟฟ้าหัวใจ EKG",
        "price": 450,
        "organ": "หัวใจและหลอดเลือด",
        "indication": "ประเมินจังหวะการเต้นของหัวใจและภาวะกล้ามเนื้อหัวใจขาดเลือด",
    },
    "ADDON_RENAL": {
        "code": "ADDON_RENAL",
        "name": "ตรวจการทำงานของไตและอัลบูมินรั่ว (Creatinine + eGFR + UACR)",
        "short_name": "ตรวจไตเชิงลึก & UACR",
        "price": 650,
        "organ": "ไตและระบบขับถ่าย",
        "indication": "ตรวจคัดกรองภาวะไตเสื่อมระยะเริ่มต้นและไข่ขาวรั่วในปัสสาวะ",
    },
    "ADDON_THYROID": {
        "code": "ADDON_THYROID",
        "name": "ตรวจสมดุลฮอร์โมนไทรอยด์ (TSH + Free T4)",
        "short_name": "ฮอร์โมนไทรอยด์ครบชุด",
        "price": 750,
        "organ": "ต่อมไร้ท่อและไทรอยด์",
        "indication": "ตรวจภาวะไทรอยด์ทำงานเกินหรือทำงานต่ำ เหนื่อยง่าย น้ำหนักเปลี่ยน",
    },
    "ADDON_MAMMOGRAM": {
        "code": "ADDON_MAMMOGRAM",
        "name": "ดิจิทัลแมมโมแกรมและอัลตราซาวด์เต้านม (Digital Mammogram)",
        "short_name": "ดิจิทัลแมมโมแกรม",
        "price": 2500,
        "organ": "คัดกรองเต้านม",
        "indication": "คัดกรองมะเร็งเต้านมระยะแรกเริ่มในสตรีอายุ 40 ปีขึ้นไป",
    },
    "ADDON_BONE": {
        "code": "ADDON_BONE",
        "name": "ตรวจความหนาแน่นมวลกระดูก (DEXA Bone Density Scan)",
        "short_name": "มวลกระดูก DEXA Scan",
        "price": 1200,
        "organ": "กระดูกและข้อ",
        "indication": "ตรวจภาวะกระดูกพรุนและกระดูกบางในผู้สูงวัยหรือวัยหมดประจำเดือน",
    },
    "ADDON_FIT": {
        "code": "ADDON_FIT",
        "name": "ตรวจเลือดแฝงในอุจจาระ (FIT Test)",
        "short_name": "ตรวจเลือดแฝง FIT Test",
        "price": 350,
        "organ": "ทางเดินอาหารและลำไส้",
        "indication": "คัดกรองมะเร็งลำไส้ใหญ่ระยะแรกเริ่มโดยไม่ต้องส่องกล้อง",
    },
    "ADDON_PSA": {
        "code": "ADDON_PSA",
        "name": "ตรวจสารบ่งชี้มะเร็งต่อมลูกหมาก (PSA)",
        "short_name": "สารบ่งชี้ PSA ต่อมลูกหมาก",
        "price": 650,
        "organ": "ต่อมลูกหมาก",
        "indication": "คัดกรองมะเร็งต่อมลูกหมากในเพศชายอายุ 50 ปีขึ้นไป",
    },
    "ADDON_CAC": {
        "code": "ADDON_CAC",
        "name": "ตรวจคราบหินปูนหลอดเลือดหัวใจ (CT Coronary Calcium Score)",
        "short_name": "แคลเซียมหัวใจ CAC Score",
        "price": 2500,
        "organ": "หัวใจและหลอดเลือด",
        "indication": "ประเมินความเสี่ยงโรคหลอดเลือดหัวใจตีบและหัวใจวายเฉียบพลัน",
    },
    "ADDON_FIBROSCAN": {
        "code": "ADDON_FIBROSCAN",
        "name": "ตรวจวัดไขมันและพังผืดตับ (FibroScan)",
        "short_name": "ตรวจตับ FibroScan",
        "price": 2200,
        "organ": "ตับและทางเดินน้ำดี",
        "indication": "ตรวจปริมาณไขมันสะสมในตับและพังผืดตับโดยไม่ต้องเจาะตับ",
    },
}


def build_patient_precision_quotation(
    lab_snap: Dict[str, Any],
    selected_pkg_type: str = "std",
    selected_addon_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Unified Precision Package Matching & Quotation Engine.
    Used by both:
      1. AI Lead Intelligence ('วิเคราะห์การตัดสินใจรายบุคคล & แนะนำแพ็กเกจเฉพาะคนไข้')
      2. Patient Insight Card ('ระบบจับคู่แพ็กเกจสุขภาพ & จัดทำใบเสนอราคา')

    Ensures 100% single source of truth for:
      - Best match standard package vs. tailor-made bespoke package
      - Itemized clinical indications & test bullets
      - Add-on modular catalog & smart recommendations
      - Commercial bundling discount & transparent net quotation math
    """
    from logic.package_matching_v2 import predict_patient_packages_v2

    v2_res = predict_patient_packages_v2(lab_snap)
    top3 = v2_res.get("top3_packages", [])
    top_std = top3[0] if top3 else {}
    pers_pkg = v2_res.get("personalized_package", {})
    needs_custom = bool(v2_res.get("needs_custom_package", False))

    # Patient Vitals & Clinical Indicators
    age_val = float(lab_snap.get("age", 48.0))
    bmi_val = float(lab_snap.get("bmi", 24.0))
    sys_val = float(lab_snap.get("systolic_bp", lab_snap.get("sbp", 120.0)))
    dia_val = float(lab_snap.get("diastolic_bp", lab_snap.get("dbp", 80.0)))
    gender_str = str(lab_snap.get("gender", "หญิง"))
    is_male = gender_str in ["ชาย", "male", "M"]

    is_dlp = bool(int(lab_snap.get("known_dx_dlp", 0)) == 1 or float(lab_snap.get("ldl_mg_dl", lab_snap.get("ldl", 110.0))) >= 160.0)
    is_ht = bool(int(lab_snap.get("known_dx_ht", 0)) == 1 or sys_val >= 140.0)
    is_dm = bool(int(lab_snap.get("known_dx_dm", 0)) == 1 or float(lab_snap.get("fpg_mg_dl", lab_snap.get("fbs", 95.0))) >= 126.0)
    alt_val = float(lab_snap.get("alt_u_l", lab_snap.get("alt", 25.0)))
    egfr_val = float(lab_snap.get("egfr", 88.0))
    tsh_val = float(lab_snap.get("tsh_miu_l", lab_snap.get("tsh", 2.1)))

    # 1. Standard Best Match Package Specification
    std_name = str(top_std.get("name") or top_std.get("short_name", "แพ็กเกจตรวจสุขภาพทั่วไป"))
    std_short_name = str(top_std.get("short_name", std_name))
    std_price = int(top_std.get("price", 2990))
    std_fit = float(top_std.get("ml_score_pct", top_std.get("fit_pct", 92.0)))
    bullets = list(top_std.get("bullets", []))
    pkg_code = str(top_std.get("package_code", "L1_STANDARD"))

    # Clinical indication text
    reason_parts: List[str] = []
    if is_dlp:
        reason_parts.append("ประวัติไขมันในเลือดสูง (Dyslipidemia)")
    if is_ht:
        reason_parts.append(f"ความดันโลหิตสูง ({sys_val:.0f}/{dia_val:.0f} mmHg)")
    elif sys_val >= 130:
        reason_parts.append(f"ความดันก้ำกึ่งเฝ้าระวัง ({sys_val:.0f} mmHg)")
    if is_dm:
        reason_parts.append("ระดับน้ำตาลในเลือด/เบาหวาน")
    if bmi_val >= 25:
        reason_parts.append(f"BMI {bmi_val:.1f} เกินเกณฑ์มาตรฐาน")
    elif bmi_val >= 23:
        reason_parts.append(f"น้ำหนักเกินเกณฑ์เล็กน้อย (BMI {bmi_val:.1f})")
    if egfr_val < 60:
        reason_parts.append("ค่าการทำงานของไต eGFR ลดลง")
    if alt_val > 40:
        reason_parts.append("เอนไซม์ตับ ALT สูง")
    if tsh_val > 4.5:
        reason_parts.append("ฮอร์โมนไทรอยด์ TSH ผิดปกติ")

    clinical_reason_text = " + ".join(reason_parts) if reason_parts else "คัดกรองสุขภาพประจำปีตามช่วงอายุและสรีรวิทยา"

    std_package = {
        "code": pkg_code,
        "name": std_name,
        "short_name": std_short_name,
        "price": std_price,
        "fit_pct": std_fit,
        "bullets": bullets,
        "clinical_reason": clinical_reason_text,
    }

    # 2. Bespoke Tailor-Made Package Specification
    bespoke_name = str(pers_pkg.get("name", "แพ็กเกจตรวจเฉพาะบุคคล (Tailor-Made Modular)"))
    bespoke_price = int(pers_pkg.get("price", 4800))
    bespoke_full = int(pers_pkg.get("full_price", int(bespoke_price * 1.2)))
    bespoke_disc = int(pers_pkg.get("discount_pct", 18))
    bespoke_savings = int(pers_pkg.get("savings_baht", max(0, bespoke_full - bespoke_price)))
    bespoke_reason = str(pers_pkg.get("gap_reason_th", "ประกอบชุดตรวจครอบคลุมทุกความเสี่ยงเฉพาะบุคคล 100%"))
    bespoke_gaps = list(pers_pkg.get("uncovered_gaps", []))
    bespoke_bullets = list(pers_pkg.get("bullets", bullets))
    bespoke_all_tests = list(pers_pkg.get("all_tests", []))

    bespoke_package = {
        "code": str(pers_pkg.get("package_code", "CUSTOM_BESPOKE")),
        "name": bespoke_name,
        "price": bespoke_price,
        "full_price": bespoke_full,
        "discount_pct": bespoke_disc,
        "savings_baht": bespoke_savings,
        "reason": bespoke_reason,
        "uncovered_gaps": bespoke_gaps,
        "bullets": bespoke_bullets,
        "all_tests": bespoke_all_tests,
    }

    # 3. Addon Catalog with Smart Patient-Specific Indication Badging
    catalog_addons: List[Dict[str, Any]] = []
    smart_defaults: List[str] = []

    for code, spec in UNIFIED_ADDON_CATALOG.items():
        is_rec = False
        rec_reason = ""

        if code == "ADDON_US_ABDOMEN" and (bmi_val >= 25 or is_ht or alt_val > 40 or is_dlp):
            is_rec = True
            rec_reason = "แนะนำสำหรับผู้มีภาวะน้ำหนักเกิน/ความดัน หรือค่าตับ/ไขมัน เพื่อตรวจไขมันพอกตับ"
        elif code == "ADDON_EKG" and (age_val >= 45 or is_ht):
            is_rec = True
            rec_reason = "แนะนำสำหรับผู้มีอายุตั้งแต่ 45 ปี หรือมีความดัน เพื่อตรวจจังหวะการเต้นของหัวใจ"
        elif code == "ADDON_RENAL" and (egfr_val < 90 or is_ht or is_dm):
            is_rec = True
            rec_reason = "แนะนำสำหรับผู้มีความดัน/เบาหวาน เพื่อตรวจกรองไตและไข่ขาวรั่ว"
        elif code == "ADDON_THYROID" and (tsh_val > 4.5 or (not is_male and age_val >= 40)):
            is_rec = True
            rec_reason = "แนะนำสำหรับสตรีอายุ 40 ปีขึ้นไป หรือค่า TSH ก้ำกึ่ง เพื่อตรวจการทำงานของไทรอยด์"
        elif code == "ADDON_MAMMOGRAM" and (not is_male and age_val >= 40):
            is_rec = True
            rec_reason = "แนวทางเวชปฏิบัติแนะนำคัดกรองมะเร็งเต้านมทุก 1-2 ปี ในสตรีอายุ 40+"
        elif code == "ADDON_BONE" and (age_val >= 50 or (not is_male and age_val >= 45)):
            is_rec = True
            rec_reason = "แนะนำตรวจความหนาแน่นมวลกระดูกป้องกันภาวะกระดูกพรุนในกลุ่ม 50+"
        elif code == "ADDON_FIT" and age_val >= 45:
            is_rec = True
            rec_reason = "แนะนำตรวจคัดกรองมะเร็งลำไส้ใหญ่ในผู้มีอายุ 45 ปีขึ้นไป"
        elif code == "ADDON_PSA" and is_male and age_val >= 50:
            is_rec = True
            rec_reason = "แนะนำคัดกรองต่อมลูกหมากในสุภาพบุรุษอายุ 50 ปีขึ้นไป"
        elif code == "ADDON_CAC" and (is_dlp or is_ht or age_val >= 55):
            is_rec = True
            rec_reason = "ประเมินคราบหินปูนเกาะหลอดเลือดหัวใจในผู้มีความเสี่ยงหลอดเลือด"
        elif code == "ADDON_FIBROSCAN" and (bmi_val >= 28 or alt_val > 40):
            is_rec = True
            rec_reason = "ประเมินพังผืดและความยืดหยุ่นของตับในผู้มีไขมันสะสมในตับ"

        if is_rec and len(smart_defaults) < 2:
            smart_defaults.append(code)

        catalog_addons.append({
            "code": code,
            "name": spec["name"],
            "short_name": spec["short_name"],
            "price": int(spec["price"]),
            "organ": spec["organ"],
            "indication": spec["indication"],
            "is_recommended": is_rec,
            "recommendation_reason": rec_reason,
        })

    # 4. Resolve Active Addons
    if selected_addon_keys is None:
        active_keys = smart_defaults
    else:
        active_keys = [k for k in selected_addon_keys if k in UNIFIED_ADDON_CATALOG]

    active_addons = [UNIFIED_ADDON_CATALOG[k] for k in active_keys]
    addons_total = sum(a["price"] for a in active_addons)

    # 5. Resolve Active Primary Package
    active_is_bespoke = (selected_pkg_type == "bespoke")
    active_primary = bespoke_package if active_is_bespoke else std_package
    active_primary_price = active_primary["price"]

    # 6. Quotation Financial Math
    gross_total = active_primary_price + addons_total
    num_addons = len(active_addons)
    if num_addons >= 2:
        discount_pct = 10
        discount_amount = int(gross_total * 0.10)
    elif num_addons == 1:
        discount_pct = 5
        discount_amount = int(gross_total * 0.05)
    else:
        discount_pct = 0
        discount_amount = 0

    net_payable = max(0, gross_total - discount_amount)
    total_savings = discount_amount + (bespoke_savings if active_is_bespoke else 0)

    # 7. AI Outreach Pitch
    top_risk_driver = reason_parts[0] if reason_parts else "การตรวจประเมินสุขภาพประจำปี"
    ai_pitch = (
        f"แนะนำโปรแกรมตรวจสุขภาพ {active_primary['name']} "
        f"โดยมุ่งเน้นการตรวจติดตาม {top_risk_driver} เพื่อเสริมการดูแลเชิงป้องกัน "
        f"พร้อมสิทธิประโยชน์ชุดตรวจเสริมครบวงจร (ประหยัดรวม ฿{total_savings:,})"
    )

    return {
        "std_package": std_package,
        "bespoke_package": bespoke_package,
        "needs_custom": needs_custom,
        "active_is_bespoke": active_is_bespoke,
        "active_primary": active_primary,
        "active_primary_name": active_primary["name"],
        "active_primary_price": active_primary_price,
        "catalog_addons": catalog_addons,
        "smart_defaults": smart_defaults,
        "active_addon_keys": active_keys,
        "active_addons": active_addons,
        "addons_total": addons_total,
        "gross_total": gross_total,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "net_payable": net_payable,
        "total_savings": total_savings,
        "clinical_reason_text": clinical_reason_text,
        "ai_pitch": ai_pitch,
    }

