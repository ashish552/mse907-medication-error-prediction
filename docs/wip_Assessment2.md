# MSE907 – Assessment 2 (Work in Progress Report)

**Project Title:** Explainable Machine Learning for High-Risk Prescription (Medication Error Risk) Prediction using MIMIC-IV Demo  
**Student Name:** Ashish KC  
**Student ID:** 270592890  
**Course:** MSE907  
**Assessment:** Assessment 2 – Work in Progress (WIP)  
**Date:** 2026-01-30  

---

## 1. Project overview (short)
Medication errors (wrong dose, wrong drug, wrong timing, or unsafe combinations) are a major cause of avoidable harm in hospitals. The goal of this project is to develop an **explainable machine learning model** that predicts whether a prescription is **high-risk** (proxy for potential medication error risk) using patient context such as demographics, admission context, laboratory results, and prescription patterns.

Because MIMIC-IV does not provide direct “medication error” labels, this project will create **proxy labels** using transparent, clinically motivated rule-based safety criteria (e.g., renal risk + medication risk, contraindicated combinations, polypharmacy risk). These proxy labels will be used for supervised learning and compared against a rule-based baseline.

---

## LO1 (25%) – Literature progress and refined research gap (WIP)

### LO1.1 What I have done so far
- I have confirmed the project direction and research gap based on my proposal: medication safety prediction is less explored compared to disease prediction, and hospitals need explainable risk scoring for prescriptions.
- I have prepared to expand my literature review with additional journal papers focusing on medication safety prediction, ADE/DDI detection, and explainable ML in healthcare.

### LO1.2 Updated research gap (current WIP)
The project addresses these gaps:
- Limited research predicting **medication error risk before administration** (near real-time flagging).
- Lack of models combining **patient-specific context** (labs/vitals/age/polypharmacy) with medication patterns.
- Limited **explainability** suitable for clinical decision support.
- Lack of datasets with direct medication error labels, requiring **proxy rule-based labelling**.

### LO1.3 Next actions for LO1 (Week 7–8)
- Add **4–8 journal articles** (not arXiv) and write short critical notes for each:
  - What method they used
  - What data they used
  - Their limitations (e.g., no real-time scoring, limited context features, no explainability, label problems)
- Update the gap statement using those limitations.

---

## LO2 (25%) – Refined research questions, methodology, and progress

### LO2.1 Refined research questions
**RQ1:** Can patient context (demographics + labs/vitals + polypharmacy/prescription patterns) improve detection of high-risk prescriptions compared with a traditional rule-based approach?  
**RQ2:** Which factors contribute most to predicted risk, and can explainability methods provide clinician-friendly reasons for predictions?

### LO2.2 Dataset
- Dataset: **MIMIC-IV Clinical Database Demo (v2.2)**
- Available domains: hospital tables (`hosp/`) and ICU tables (`icu/`) including prescriptions, admissions, patients, labs, and vitals.

### LO2.3 Unit of prediction
- **One row = one prescription order** from `hosp/prescriptions.csv.gz`, linked to admission and patient information.

### LO2.4 Proposed pipeline (methodology)
1. **Data extraction & integration:** extract dataset; map tables; join prescriptions + admissions + patients.  
2. **Cleaning & preprocessing:** convert timestamps, handle missing values, remove unusable rows, avoid leakage.  
3. **Feature engineering:** polypharmacy count; renal/hepatic lab features; later add vitals/DDI features if feasible.  
4. **Proxy label creation:** transparent rule-based “high-risk vs low-risk” prescription labels.  
5. **Model training:** Logistic Regression baseline, Random Forest, XGBoost.  
6. **Explainability:** feature importance and SHAP-style local explanations to support clinician interpretation.  
7. **Evaluation:** AUROC, precision/recall/F1, confusion matrix; compare ML models vs a rule-based baseline.

### LO2.5 Implementation progress completed so far (Week 7 WIP)

**Data extraction and table mapping**
- Extracted the dataset zip into a usable folder and confirmed the presence of `hosp/` and `icu/` tables.
- Verified key tables exist: admissions, patients, prescriptions, labevents, chartevents.

**Core table loading**
- Successfully loaded `patients`, `admissions`, and `prescriptions` and recorded row counts:
  - patients: 100 (unique subject_id: 100)
  - admissions: 275 (unique hadm_id: 275)
  - prescriptions: 18,087 (unique hadm_id: 250)

**Base dataset creation**
- Created and saved a joined “base cleaned dataset”:
  - Output: `data/processed/base_rx_clean.csv`
  - Rows: 18,087
  - Columns: 16
- Cleaning included converting time fields to datetime and standardizing drug name text.

**Feature engineering (Polypharmacy)**
- Implemented polypharmacy feature: count of concurrent active medications at prescription time.
- Saved dataset with feature:
  - Output: `data/processed/base_rx_polypharm.csv`
- Verified polypharmacy sanity check:
  - min = 1 (expected, because at least the current prescription is active)

**Lab feature engineering (Week 7)**
- Added latest prior labs per prescription (no leakage): creatinine, BUN, ALT, AST, bilirubin_total
- Missing rates:
  - creatinine missing = 17.55%
  - bun missing = 17.56%
  - alt missing = 44.97%
  - ast missing = 40.48%
  - bilirubin_total missing = 44.41%
- Output file: `data/processed/base_rx_with_labs.csv`

 Cleaning v2 (Week 7):
- Saved: data/processed/model_dataset_clean_v1.csv
- Rows: 18087 -> 15484
- Duplicates removed (hadm_id, starttime, drug): 2603
- Dose numeric coverage: 94.54%


Data quality summary (Week 7):
- Lab missingness: creatinine 17.93%, BUN 17.94%, ALT 45.20%, AST 40.80%, bilirubin_total 44.60%.
- Top drugs by frequency include insulin (693), 0.9% sodium chloride (557), furosemide (462), heparin (261), vancomycin (239).
- Polypharmacy distribution: median 26, 75% 37, max 80.
- Dose numeric coverage: 94.54% non-missing; dose max outlier observed (8500).


### LO2.6 Evidence (console outputs)
(These are copied from development logs.)

```text
✅ Using dataset root: data/raw/mimic_demo_2.2/mimic-iv-clinical-database-demo-2.2

--- Row counts ---
patients: 100 | unique subject_id: 100
admissions: 275 | unique hadm_id: 275
prescriptions: 18087 | unique hadm_id: 250

✅ Saved: data/processed/base_rx_clean.csv
Final rows: 18087 | cols: 16

✅ Saved: data/processed/base_rx_polypharm.csv

Polypharmacy summary:
count    18087.000000
mean        28.767070
std         15.094545
min          1.000000
25%         17.000000
50%         27.000000
75%         38.000000
max         80.000000

✅ Filtered labevents rows: 7342
✅ Added feature: creatinine (missing=17.55%)
✅ Added feature: bun (missing=17.56%)
✅ Added feature: alt (missing=44.97%)
✅ Added feature: ast (missing=40.48%)
✅ Added feature: bilirubin_total (missing=44.41%)

✅ Saved: data/processed/base_rx_with_labs.csv
 
 Baseline Logistic Regression (v1) on proxy labels:
- AUROC: 0.9972
- AUPRC: 0.9932
- Precision: 0.8883 | Recall: 0.9910 | F1: 0.9368
- Confusion matrix [[TN FP],[FN TP]] = [[2348 83],[6 660]]
Artifacts saved:
- reports/metrics_logreg_baseline_v1.txt
- reports/figures/confusion_matrix_logreg_baseline_v1.png

