# 🏥 PrimeCare Wellness Clinic — Clinical Command Center & AI Package Intelligence

แดชบอร์ดวิเคราะห์ข้อมูลสุขภาพผู้ป่วย พร้อมระบบ **AI Multi-Task Package Matcher v2**, **Dynamic Personalized / Tailor-Made Package Builder (`CUSTOM_BESPOKE`)**, และ **Lead Scoring & Explainable AI (XAI)** สำหรับคัดกรองความเสี่ยงและแนะนำแพ็กเกจตรวจสุขภาพรายบุคคล

---

## 🏗️ โครงสร้างโปรเจกต์ (Project Structure)

```text
Project Dashboard/
├── app.py                                      # Main Streamlit Application Entrypoint
├── custom_ui.py                                # Custom UI & Glassmorphic Component Helpers
├── logic/
│   ├── config.py                               # Package Catalogs, Clinical Thresholds & Constants
│   ├── package_matching_v2.py                  # Multi-Task Package Matcher v2 + Modular Tailor-Made Package Engine
│   ├── patient_segmentation.py                 # Clinical Tier 1–4 Segmentation & Personalized Bridge
│   ├── package_service.py                      # Package Sales Analytics & Mismatch Detection
│   └── lead_scoring_xai.py                     # XGBoost Propensity Scoring, SHAP Explainability & Prophet Revenue Projection
├── data/
│   ├── sample_hospital_transactions.csv        # Sample Hospital Transactions
│   └── synthetic/                              # 10,000-Patient Synthetic Clinical & Lab Star-Schema Dataset
│       ├── dim_patient.parquet                 # Patient Demographics & Baseline Profiles (10,000 patients)
│       ├── dim_package.csv                     # 13 Standard Health Checkup Packages (4 Tiers)
│       ├── dim_lab_test.csv                    # 26 Standardized Lab Biomarkers (LOINC + Reference Ranges)
│       ├── fact_encounter.parquet              # Longitudinal Visit History (Sorted by patient_id, visit_date, visit_id)
│       ├── fact_lab_result.parquet             # Longitudinal Lab Results (Long Format)
│       ├── fact_package_purchase.parquet       # Historical Package Purchases & Conversion Labels
│       ├── train_patient_snapshot.parquet      # Wide Feature Matrix (83 Features + Multi-Task Labels)
│       └── feature_groups.json                 # Feature Schema & Zero-Leakage Metadata
├── models/
│   ├── package_matcher_v2_bundle.npz           # Multi-Task Package Matcher v2 Weights (NumPy Portable Bundle)
│   ├── package_matcher_v2_meta.json            # Model Metrics & Evaluation Metadata
│   └── xgboost_package_upsell_model.json       # Pre-trained XGBoost Propensity Model
├── data_generation/
│   ├── generate_synthetic_patients.py          # Synthetic Clinical & Lab Data Generator (10,000 Patients)
│   └── train_package_matcher.py                # Multi-Task Model Training Pipeline
├── tests/
│   ├── test_executive_overview.py              # Executive Overview Unit Tests
│   ├── test_package_matching_v2.py             # Package Matching v2 & Tailor-Made Package Unit Tests
│   └── test_refactor.py                        # Segmentation & Package Service Unit Tests
├── .streamlit/
│   └── config.toml                             # Streamlit Theme & Server Configuration
├── requirements.txt                            # Python Dependencies (Streamlit Cloud Ready)
└── .gitignore                                  # PDPA & Large Raw Data Exclusions
```

---

## ✨ ฟีเจอร์หลัก (Key Capabilities)

### 1. 🧬 ระบบจัดแพ็กเกจเฉพาะคนไข้ (Dynamic Personalized / Tailor-Made Package — `CUSTOM_BESPOKE`)
- ตรวจจับอัตโนมัติเมื่อคนไข้มีผล Lab หรือความเสี่ยงเฉพาะอวัยวะที่แพ็กเกจมาตรฐานทั้ง 13 รายการไม่ครอบคลุมครบ 100% หรือมีความเสี่ยงซ้อนทับหลายระบบ:
  - **ไตเสื่อม (`MOD_RENAL`):** `eGFR < 60` หรือ `UACR ≥ 30 mg/g`
  - **ไทรอยด์ผิดปกติ (`MOD_THYROID`):** `TSH < 0.45` หรือ `TSH > 4.5 mIU/L`
  - **โลหิตจาง/ขาดธาตุเหล็ก (`MOD_ANEMIA`):** `Hb` ต่ำกว่าเกณฑ์ หรือ `Ferritin < 30 ng/mL`
  - **กรดยูริกสูง/เกาต์ (`MOD_GOUT`):** `Uric Acid > 7.0` (ชาย) หรือ `> 6.0` (หญิง)
  - **ตับอักเสบ/พังผืดตับ (`MOD_HEPATIC`):** `ALT > 40 U/L`, `FIB-4 ≥ 1.30`, หรือพบไขมันพอกตับ
- รองรับการปรับเพิ่ม/ลดโมดูลการตรวจแบบ Interactive (`🛠️ ปรับแต่งโมดูลการตรวจเฉพาะบุคคล`) พร้อมคำนวณราคาเต็มและส่วนลด Bundle (**15%–18%**) อัตโนมัติ

### 2. 🤖 AI Multi-Task Package Matcher v2 (4 Tiers · 13 Standard Packages + 9 Add-ons)
- จับคู่คนไข้เข้ากับแพ็กเกจมาตรฐาน 4 ระดับ (Tier 1 ตรวจสุขภาพพื้นฐาน → Tier 4 คัดกรองโรคเฉพาะทางเชิงลึก)
- แสดงเหตุผลประกอบรายบุคคลจากค่า Lab จริง (Top Biomarker Drivers) และแนะนำรายการตรวจเสริม (Add-ons) เช่น `PSA`, `Mammogram`, `FibroScan`, `Fundus Exam`, `Low-Dose CT Lung`

### 3. 🎯 AI Lead Intelligence (XGBoost Propensity + SHAP XAI + Prophet Forecast)
- แบ่งหน้าจอออกเป็น **4 แท็บแนวนอน** ที่อ่านง่ายและไม่รกตา:
  1. **🎯 รายชื่อกลุ่มเป้าหมาย & วิเคราะห์รายบุคคล:** การ์ดโปรไฟล์คนไข้ + แนะนำแพ็กเกจเฉพาะบุคคล + กราฟ SHAP Waterfall + ตารางรายชื่อผู้มุ่งหวังพร้อมดาวน์โหลด CSV
  2. **🌐 ปัจจัยขับเคลื่อนระดับประชากร (Global SHAP):** กราฟลำดับความสำคัญของตัวแปรระดับประชากร
  3. **🔄 โครงสร้างประชากร & Funnel:** วิเคราะห์สัดส่วนผู้เคยซื้อ vs ผู้ยังไม่เคยซื้อแพ็กเกจ (Anti-Join Funnel)
  4. **📈 พยากรณ์รายได้ & จำลองกลยุทธ์:** จำลองเป้าหมายรายได้ 6–24 เดือนข้างหน้า

### 4. ⚙️ ปรับเปลี่ยนชื่อคลินิกแบบ Real-time
- ค่าเริ่มต้นคือ **PrimeCare Wellness Clinic** และสามารถเปลี่ยนชื่อคลินิกหรือคำโปรย (Tagline) ได้ทันทีผ่านเมนู **`⚙️ ตั้งค่าชื่อคลินิก (Clinic Brand)`** บนแถบ Sidebar ด้านซ้าย

---

## 🚀 การติดตั้งและรันบนเครื่อง (Local Setup)

```bash
# 1. ติดตั้งไลบรารีที่จำเป็น
pip install -r requirements.txt

# 2. รันชุดทดสอบ (Unit Tests)
python -m unittest discover -s tests -v

# 3. เปิดใช้งาน Streamlit Dashboard
streamlit run app.py
```

---

## ☁️ การนำขึ้น GitHub และ Deploy บน Streamlit Community Cloud

1. **สร้าง Repository บน GitHub** และรันคำสั่งต่อไปนี้ในโฟลเดอร์โปรเจกต์:
   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "feat: PrimeCare Wellness Clinic dashboard with personalized package engine"
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. **Deploy บน [Streamlit Community Cloud](https://share.streamlit.io/):**
   - เลือก Repository ที่เพิ่ง Push ขึ้นไป
   - ตั้งค่า **Branch:** `main`
   - ตั้งค่า **Main file path:** `app.py`
   - กด **Deploy!** (ไฟล์ข้อมูลจำลอง `data/synthetic/*.parquet` และโมเดล `models/*` มีขนาดรวมเพียง ~6.7 MB จึงพร้อมทำงานบน Cloud ทันทีโดยไม่ต้องตั้งค่าเพิ่ม)
