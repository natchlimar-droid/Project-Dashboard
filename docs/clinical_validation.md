# 🩺 PrimeCare Clinical Decision Engine — Clinical Validation & Evidence Document

**Document Version:** 2.4.0  
**Classification:** Clinical & Technical Architecture Specification  
**Governing Standard:** KDIGO 2024, JNC 8, 2024 Thai Hypertension Guidelines, Rama/EGAT Thai CV Risk  

---

## 1. วัตถุประสงค์ (Executive Objective)

เอกสารฉบับนี้จัดทำขึ้นเพื่อแสดง **Clinical Validation & Scientific Grounds** ของระบบ **PrimeCare Wellness Clinic Decision Engine** เพื่อยืนยันว่าการจำแนกความเสี่ยง (Risk Stratification), การแปลผลแล็บ (Biomarker Interpretation), และการแนะนำแพ็กเกจสุขภาพเชิงรุกด้วย AI ดำเนินการบนพื้นฐานแนวทางเวชปฏิบัติระดับสากลที่มีหลักฐานเชิงประจักษ์ (Evidence-Based Medicine) อย่างแท้จริง

---

## 2. การประเมินภาวะไตวายเรื้อรัง (KDIGO 2024 Chronic Kidney Disease Guidelines)

### 2.1 เกณฑ์การจัดระดับ GFR และ Albuminuria (KDIGO 2D Matrix)

ตามแนวทางเวชปฏิบัติ **Kidney Disease: Improving Global Outcomes (KDIGO) 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease (Kidney Int. 2024)** ระบบจัดกลุ่มความเสี่ยงไตผ่านเมทริกซ์ 2 มิติ:

$$\text{CKD Risk} = f(\text{GFR Category}, \text{Albuminuria Category})$$

| GFR Stage | เกณฑ์ eGFR ($\text{mL/min/1.73m}^2$) | คำจำกัดความทางการแพทย์ |
| :--- | :--- | :--- |
| **G1** | $\ge 90$ | การทำงานของไตปกติหรือสูง (Normal or high) |
| **G2** | $60 - 89$ | การทำงานของไตลดลงเล็กน้อย (Mildly decreased) |
| **G3a** | $45 - 59$ | ไตลดลงเล็กน้อยถึงปานกลาง (Mildly to moderately decreased) |
| **G3b** | $30 - 44$ | ไตลดลงปานกลางถึงรุนแรง (Moderately to severely decreased) |
| **G4** | $15 - 29$ | ไตลดลงรุนแรงมาก (Severely decreased) |
| **G5** | $< 15$ | ไตวายระยะสุดท้าย (Kidney failure / End-stage renal disease) |

| Albuminuria Stage | เกณฑ์ UACR ($\text{mg/g}$) | คำจำกัดความทางการแพทย์ |
| :--- | :--- | :--- |
| **A1** | $< 30$ | ปกติหรือเพิ่มขึ้นเล็กน้อย (Normal to mildly increased) |
| **A2** | $30 - 300$ | เพิ่มขึ้นปานกลาง (Microalbuminuria) บ่งชี้เริ่มมีการทำลายหน่วยกรองไต |
| **A3** | $> 300$ | รุนแรงมาก (Macroalbuminuria / Nephrotic range) |

### 2.2 ตารางสีและความรุนแรงของความเสี่ยง (KDIGO Heatmap Risk Tiers)
1. 🟢 **Low Risk (ความเสี่ยงต่ำ - สีเขียว):** G1-A1, G2-A1 $\rightarrow$ ตรวจติดตามปีละ 1 ครั้ง
2. 🟡 **Moderately Increased Risk (ความเสี่ยงปานกลาง - สีเหลือง):** G3a-A1, G1-A2, G2-A2 $\rightarrow$ ตรวจซ้ำทุก 6–12 เดือน
3. 🟠 **High Risk (ความเสี่ยงสูง - สีส้ม):** G3b-A1, G3a-A2, G1-A3, G2-A3 $\rightarrow$ ตรวจซ้ำทุก 4–6 เดือน
4. 🔴 **Very High Risk (ความเสี่ยงสูงมาก - สีแดง):** G4, G5 ทุกระดับ Albuminuria หรือ G3b-A2/A3 $\rightarrow$ **ส่งต่อพบอายุรแพทย์โรคไต (Nephrologist Consultation ทันที)**

### 2.3 ภาวะไตเสื่อมเร็วผิดปกติ (Rapid Progression Criterion)
กำหนดตาม KDIGO เกณฑ์ **$\Delta\text{eGFR} < -5\text{ mL/min/1.73m}^2/\text{year}$** ถือเป็นภาวะเร่งด่วนที่ต้องทบทวนการใช้ยาที่ทำลายไต (NSAIDs, Contrast Media) และประเมินสาเหตุเฉียบพลัน

---

## 3. แบบจำลองการรอดชีวิตของไต (Kaplan-Meier Renal Survival Model)

เพื่อพยากรณ์ความเสี่ยงของการเกิดภาวะไตวายระยะสุดท้าย (ESRD) หรือ eGFR ลดลงมากกว่า 50% ระบบใช้แบบจำลอง Parametric Hazard Decay บนช่วงเวลา 60 เดือน (5 ปี):

$$S(t) = \exp(-\lambda t^\gamma)$$

- **กลุ่ม Low Risk (G1/G2 + A1):** อัตราการรอดชีวิตของไตที่ 5 ปี $\approx 98.5\%$
- **กลุ่ม Moderate Risk (G3a / A2):** อัตราการรอดชีวิตของไตที่ 5 ปี $\approx 87.5\%$
- **กลุ่ม High Risk (G3b):** อัตราการรอดชีวิตของไตที่ 5 ปี $\approx 68.0\%$
- **กลุ่ม Very High / Rapid Decliners (G4/G5 หรือ $\Delta\text{eGFR} < -5$):** อัตราการรอดชีวิตของไตลดลงอย่างรวดเร็วสู่ระดับ $\approx 35.0\%$

---

## 4. แบบประเมินความเสี่ยงโรคหลอดเลือดหัวใจและสมอง 10 ปี (Thai CV Risk Score)

อ้างอิงจากงานวิจัยติดตามประชากรไทยระยะยาวโดย **คณะแพทยศาสตร์โรงพยาบาลรามาธิบดี ร่วมกับการไฟฟ้าฝ่ายผลิตแห่งประเทศไทย (EGAT Cohort Study)**:

$$\text{Thai CV Risk 10Y} = f(\text{Age}, \text{Gender}, \text{Systolic BP}, \text{Smoking Status}, \text{Diabetes}, \text{Total Cholesterol}, \text{HDL})$$

### การแบ่งกลุ่มความเสี่ยงทางคลินิก:
- **Low Risk ($< 10\%$):** เสี่ยงต่ำ $\rightarrow$ เน้นปรับเปลี่ยนพฤติกรรม (Lifestyle Modification)
- **Intermediate Risk ($10 - 19\%$):** เสี่ยงปานกลาง $\rightarrow$ แนะนำตรวจหลอดเลือดคอ (Carotid Ultrasound)
- **High Risk ($\ge 20\%$):** เสี่ยงสูงมาก $\rightarrow$ พิจารณาให้ยาลดไขมันกลุ่ม Statin และแนะนำตรวจ **Coronary Artery Calcium Score (CAC)**

---

## 5. เกณฑ์การจัดระดับความดันโลหิต (JNC 8 & Thai Hypertension Guidelines 2024)

| ระดับความดันโลหิต | Systolic BP ($\text{mmHg}$) | | Diastolic BP ($\text{mmHg}$) | แผนปฏิบัติการทางคลินิก |
| :--- | :---: | :---: | :---: | :--- |
| **Normal** | $< 120$ | และ | $< 80$ | ตรวจสุขภาพประจำปี |
| **Pre-Hypertension** | $120 - 139$ | หรือ | $80 - 89$ | แนะนำวัดความดันที่บ้าน (Home BP Monitoring) |
| **Stage 1 Hypertension** | $140 - 159$ | หรือ | $90 - 99$ | ปรับพฤติกรรม + เริ่มยาลดความดัน |
| **Stage 2 / Crisis** | $\ge 160$ | หรือ | $\ge 100$ | **พบแพทย์ทันที ประเมิน Target Organ Damage** |

---

## 6. ข้อกำหนดด้านความโปร่งใสของ AI (Explainable AI Protocol)

เพื่อป้องกันปัญหา **AI Black-Box** ที่แพทย์ไม่สามารถตรวจสอบได้ ระบบบังคับใช้มาตรฐาน:
1. ทุกคะแนนความน่าจะเป็น (Propensity Score) ต้องมาพร้อมกับ **SHAP Values (TreeSHAP Algorithm)** ที่ระบุปัจจัยส่งผลบวก/ลบอย่างชัดเจน
2. มีการคำนวณ **Baseline Log-Odds ($E[f(x)]$)** และผลรวมคุณลักษณะ:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
3. แพทย์สามารถพิมพ์เอกสารสรุปเหตุผลทางคลินิกแนบไปพร้อมกับใบเสนอราคาแพ็กเกจตรวจสุขภาพได้ทันที
