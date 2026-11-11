# -*- coding: utf-8 -*-
"""
Central Configuration & Single Source of Truth
Constants, Clinical Cutoffs, Category Orders, and Package Catalog
"""
import re
from typing import Any, Set


CATEGORY_ORDER = [
    "1_Basic Health Check",
    "2_Health Risk Check",
    "3_Early Disease Check",
    "4_Deep Health & Symptoms",
]

PATIENT_TABLE_COLUMNS = [
    ("patient_id", "Patient ID"),
    ("visit_date", "วันที่รับบริการ"),
    ("clinic_name", "คลินิก"),
    ("diagnosis_text", "ผลการวินิจฉัย"),
    ("systolic", "ความดันตัวบน (mmHg)"),
    ("diastolic", "ความดันตัวล่าง (mmHg)"),
    ("bmi", "BMI"),
]

CLINICAL_THRESHOLDS = {
    "age_bands": [
        ("<30", 0, 30),
        ("30-34", 30, 35),
        ("35-49", 35, 50),
        ("50-59", 50, 60),
        ("60+", 60, 150),
    ],
    "bp": {
        "crisis_sys": 160,
        "crisis_dia": 100,
        "stage1_sys": 140,
        "stage1_dia": 90,
        "elevated_sys": 120,
        "elevated_dia": 80,
    },
    "bmi": {
        "severe_obese": 32.0,
        "obese": 27.5,
        "overweight": 23.0,
        "underweight": 18.5,
    },
    "keywords": {
        "chronic_crisis": [
            "หัวใจ", "heart", "แน่นหน้าอก", "chest pain", "มะเร็ง",
            "cancer", "tumor", "ก้อน", "stemi", "nstemi", "cad", "chf"
        ],
        "diabetes": ["เบาหวาน", "diabetes", "dm", "fbs", "hba1c"],
        "hypertension": ["ความดัน", "hypertension", "ht"],
        "lipid": ["ไขมัน", "lipid", "cholesterol", "dyslipidemia"],
        "gout": ["เกาต์", "gout", "uric", "เก๊า"],
    }
}

# 4-Level Package Catalog: Single Source of Truth
# Note: total_vn and total_revenue are dynamically computed via compute_package_stats()
PACKAGE_CATALOG = {
    "1_Basic Health Check": {
        "level": 1,
        "label": "1️⃣ ตรวจสุขภาพทั่วไป",
        "subtitle": "สำหรับคนที่ใส่ใจสุขภาพ อยากรู้พื้นฐานของตัวเอง",
        "slogan": "เช็กสุขภาพวันนี้ เพื่อความมั่นใจในอนาคต",
        "color": "#0E7055",
        "bg": "#DCFCE7",
        "header_bg": "linear-gradient(135deg,#0E7055,#15a070)",
        "icon": "🟢",
        "packages": [
            {
                "name": "แพคเกจตรวจสุขภาพ BASIC",
                "short_name": "BASIC",
                "price": 990,
                "category": "1_Basic Health Check",
                "level": 1,
                "bullets": ["ตรวจร่างกายทั่วไป", "ตรวจเลือดพื้นฐาน", "ตรวจปัสสาวะ, เอกซเรย์ปอด"],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ STANDARD",
                "short_name": "STANDARD",
                "price": 2290,
                "category": "1_Basic Health Check",
                "level": 1,
                "bullets": ["ตรวจร่างกายทั่วไป", "ตรวจเลือดครบชุด", "เอกซเรย์ปอด, คลื่นไฟฟ้าหัวใจ<br>(Ultrasound ช่องท้องส่วนบน)"],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ PREMIUM",
                "short_name": "PREMIUM",
                "price": 4990,
                "category": "1_Basic Health Check",
                "level": 1,
                "bullets": ["ตรวจร่างกายละเอียด", "ตรวจเลือดครบชุด", "เอกซเรย์ปอด, คลื่นไฟฟ้าหัวใจ", "อัลตราซาวด์ช่องท้องส่วนบน-ล่าง"],
            }
        ]
    },
    "2_Health Risk Check": {
        "level": 2,
        "label": "2️⃣ ยังไม่พบโรค แต่มีความเสี่ยง",
        "subtitle": "สำหรับผู้ที่มีปัจจัยเสี่ยงจากอายุ น้ำหนัก พฤติกรรม หรือประวัติครอบครัว",
        "slogan": "รู้ความเสี่ยงก่อนเกิดโรค ป้องกันได้...ให้ชีวิตยืนยาว",
        "color": "#B45309",
        "bg": "#FEF3C7",
        "header_bg": "linear-gradient(135deg,#D97706,#F59E0B)",
        "icon": "🟡",
        "packages": [
            {
                "name": "แพคเกจตรวจสุขภาพ 9.9 METABOLIC HEALTH MONTH",
                "short_name": "9.9 METABOLIC",
                "price": 2990,
                "category": "2_Health Risk Check",
                "level": 2,
                "bullets": [
                    "ตรวจระดับน้ำตาลและดื้ออินซูลิน (FBS, HbA1c, Fasting Insulin)",
                    "ตรวจไขมันในเลือดครบชุด (Lipid Profile & Non-HDL)",
                    "ตรวจการทำงานของตับ ไต และประเมินภาวะอ้วนลงพุง",
                    "ประเมิน 6 เสาหลักวิถีชีวิต (Lifestyle Assessment) และให้คำปรึกษา",
                ],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)",
                "short_name": "คนอ้วน",
                "price": 2990,
                "category": "2_Health Risk Check",
                "level": 2,
                "bullets": ["ตรวจสุขภาพสำหรับผู้มีภาวะน้ำหนักเกิน", "ประเมินภาวะแทรกซ้อนจากโรคอ้วน", "ตรวจระดับน้ำตาล ไขมัน ตับ ไต", "ให้คำแนะนำการดูแลน้ำหนักโดยแพทย์"],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ วัย 35+",
                "short_name": "วัย 35+",
                "price": 3990,
                "category": "2_Health Risk Check",
                "level": 2,
                "bullets": ["ตรวจคัดกรองโรคที่มักพบในวัยทำงาน", "ตรวจระดับน้ำตาล ไขมัน ความดัน", "ตรวจการทำงานของตับ ไต", "เอกซเรย์ปอด อัลตราซาวด์ช่องท้อง"],
            }
        ]
    },
    "3_Early Disease Check": {
        "level": 3,
        "label": "3️⃣ เริ่มมีความผิดปกติ / โรคระยะเริ่มต้น",
        "subtitle": "สำหรับผู้ที่มีค่าผลตรวจเริ่มผิดปกติ หรือได้รับการวินิจฉัยโรคระยะเริ่มต้น",
        "slogan": "ตรวจให้ชัด...ดูแลได้เร็ว ลดความเสี่ยงในอนาคต",
        "color": "#D97706",
        "bg": "#FFFBEB",
        "header_bg": "linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%)",
        "header_text": "#92400E",
        "header_sub": "#B45309",
        "header_border": "1px solid #FDE68A",
        "icon": "⚠️",
        "packages": [
            {
                "name": "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)",
                "short_name": "เบาหวาน",
                "price": 3290,
                "category": "3_Early Disease Check",
                "level": 3,
                "bullets": [
                    "ตรวจระดับน้ำตาลเชิงลึก (FBS, HbA1c, HOMA-IR)",
                    "ประเมินตับและไตระยะแรก (eGFR, UACR ไมโครอัลบูมิน)",
                    "คัดกรองภาวะแทรกซ้อน (Fundus จอประสาทตา & ตรวจเท้า)",
                    "วางแผนโภชนบำบัดเฉพาะบุคคลโดยนักกำหนดอาหาร (MDT)",
                ],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)",
                "short_name": "ความดันโลหิตสูง",
                "price": 2990,
                "category": "3_Early Disease Check",
                "level": 3,
                "bullets": [
                    "ตรวจคัดกรองความดัน (EKG & แนะนำ Home BP Monitoring)",
                    "ตรวจสมรรถภาพหลอดเลือดและหัวใจ (baPWV/ABI, Echo เบื้องต้น)",
                    "ตรวจไตและไขมันหลอดเลือด (UACR ไมโครอัลบูมิน, Lipid, eGFR)",
                    "โปรแกรมออกกำลังกายและปรับวิถีชีวิตโดยนักกายภาพบำบัด (MDT)",
                ],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)",
                "short_name": "ไขมันในเลือดสูง",
                "price": 2990,
                "category": "3_Early Disease Check",
                "level": 3,
                "bullets": [
                    "ตรวจไขมันละเอียด (Lipid Profile, ApoB, Non-HDL)",
                    "ประเมินการอักเสบหลอดเลือดและหัวใจ (hs-CRP, ออปชัน CAC Score)",
                    "ตรวจตับและภาวะไขมันพอกตับ (Liver Panel, ออปชัน FibroScan)",
                    "วางแผนโภชนาการลดไขมันหลอดเลือดโดยนักกำหนดอาหาร (MDT)",
                ],
            }
        ]
    },
    "4_Deep Health & Symptoms": {
        "level": 4,
        "label": "4️⃣ มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม",
        "subtitle": "สำหรับผู้ที่มีอาการผิดปกติซ้ำๆ หรือมีความกังวลและต้องการตรวจเชิงลึก",
        "slogan": "หาสาเหตุให้ชัดเจน เพื่อความสบายใจและการดูแลที่ตรงจุด",
        "color": "#5B21B6",
        "bg": "#F3E8FF",
        "header_bg": "linear-gradient(135deg,#6941C6,#7c3aed)",
        "icon": "🟣",
        "packages": [
            {
                "name": "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)",
                "short_name": "คัดกรองมะเร็ง",
                "price": 4990,
                "category": "4_Deep Health & Symptoms",
                "level": 4,
                "bullets": ["ตรวจสารบ่งชี้มะเร็ง (Tumor Marker)", "เอกซเรย์ปอด / อัลตราซาวด์", "ตรวจอวัยวะตามความเสี่ยง (ผู้ชาย/ผู้หญิง)", "พบแพทย์ให้คำแนะนำ"],
            },
            {
                "name": "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)",
                "short_name": "หัวใจ",
                "price": 4990,
                "category": "4_Deep Health & Symptoms",
                "level": 4,
                "bullets": ["ตรวจคลื่นไฟฟ้าหัวใจ (EKG)", "ตรวจสมรรถภาพหัวใจ (EST)", "อัลตราซาวด์หัวใจ (Echo)", "ตรวจไขมัน และปัจจัยเสี่ยงอื่นๆ"],
            },
            {
                "name": "แพ็กเกจตรวจสุขภาพผู้สูงวัยเชิงลึก (Senior Health Check)",
                "short_name": "ตรวจสุขภาพผู้สูงวัยเชิงลึก",
                "price": 5900,
                "category": "4_Deep Health & Symptoms",
                "level": 4,
                "bullets": ["ตรวจการทำงานของระบบประสาทและสมองเบื้องต้น", "ตรวจมวลกระดูกและกล้ามเนื้อ (Bone & Muscle)", "ตรวจความเสื่อมของอวัยวะภายในตามวัย", "ให้คำปรึกษาการดูแลสุขภาพโดยแพทย์ผู้สูงอายุ"],
            },
            {
                "name": "แพคเกจตรวจคัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)",
                "short_name": "Colonoscopy",
                "price": 8900,
                "category": "4_Deep Health & Symptoms",
                "level": 4,
                "bullets": ["ส่องกล้องตรวจลำไส้ใหญ่", "ตรวจหาติ่งเนื้อ (Polyp)", "โดยทีมแพทย์เฉพาะทาง", "พร้อมการดูแลหลังการตรวจ"],
            }
        ]
    }
}

PACKAGE_KEYWORD_ALIASES = {
    "BASIC": ["basic", "สุดคุ้ม 990", "990 บาท"],
    "STANDARD": ["standard"],
    "PREMIUM": ["premium"],
    "9.9 METABOLIC": ["9.9", "metabolic"],
    "คนอ้วน": ["คนอ้วน", "obesity", "เสี่ยงโรคอ้วน"],
    "วัย 35+": ["วัย 35+", "35+"],
    "เบาหวาน": ["เบาหวาน", "diabetes"],
    "ความดันโลหิตสูง": ["ความดันโลหิตสูง", "ความดันสูง", "hypertension"],
    "ไขมันในเลือดสูง": ["ไขมันในเลือดสูง", "ไขมันในเลือด", "lipid check"],
    "คัดกรองมะเร็ง": ["คัดกรองมะเร็ง", "cancer screening", "มะเร็งปากมดลูก", "mammogram"],
    "หัวใจ": ["ตรวจสุขภาพหัวใจ", "heart check", "echo", "est", "แคลเซียมในหลอดเลือดหัวใจ"],
    "ตรวจสุขภาพผู้สูงวัยเชิงลึก": ["ตรวจสุขภาพผู้สูงวัยเชิงลึก", "senior", "ผู้สูงอายุ", "ผู้สูงวัย"],
    "Colonoscopy": ["colonoscopy", "ส่องกล้องตรวจลำไส้ใหญ่"],
}

# ============================================================
# Excluded Non-Checkup Packages (Vaccines, Covid tests, Certificates)
# ============================================================
EXCLUDED_PACKAGE_CODES: Set[str] = {
    "OCH2172", "OCH2418", "OCH2516", "OCH2527", "OCH2137", "OCH2139", "OCH2141",
    "OCH2219", "OCH2173", "OCH2684", "OCH2603", "OCH2602", "OCH2537", "OCH2530",
    "OCH2529", "OCH2528", "OCH2524", "OCH2523", "OCH2522", "OCH2520", "OCH2518",
    "OCH2517", "OCH2513", "OCH2508", "OCH2436", "OCH2414", "OCH2403", "OCH2406",
    "OCH2405", "OCH2320B", "OCH2320A", "OCH2320", "OCH2316", "OCH2315", "OCH2314",
    "OCH2313", "OCH2312", "OCH2311", "OCH2309", "OCH2308", "OCH2307", "OCH2306",
    "OCH2301A", "OCH2302", "OCH2301", "OCH2249", "OCH2138", "OCH2136", "OCH1925"
}

def is_excluded_package(pkg_name: Any) -> bool:
    """
    Returns True if package is in the exclusion list (vaccines, rapid tests, employee specials, etc.)
    """
    if not pkg_name:
        return False
    p = str(pkg_name).strip()
    if not p or p.lower() == "nan":
        return False
    m = re.match(r"^(OCH[0-9A-Za-z]+)", p)
    if m and m.group(1).upper() in EXCLUDED_PACKAGE_CODES:
        return True
    for code in EXCLUDED_PACKAGE_CODES:
        if p.startswith(code):
            return True
    return False

