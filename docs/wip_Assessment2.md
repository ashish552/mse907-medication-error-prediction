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

@@ -131,26 +131,218 @@ count    18087.000000
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
```

### LO2.7 Detailed implementation log (what has been completed)

This section documents each implemented script in execution order and what evidence/artifact it produced.

#### A) Data pipeline scripts (`src/pipeline/`)

1. `01_extract.py`
   - Purpose: unpack and validate the MIMIC-IV demo dataset path.
   - Outcome: dataset root confirmed and used in subsequent scripts.

2. `02_list_files.py`
   - Purpose: verify available tables and folder layout (`hosp/`, `icu/`).
   - Outcome: key source files confirmed for admissions, prescriptions, labs, and ICU chart events.

3. `03_load_core_tables.py`
   - Purpose: load and profile core tables (`patients`, `admissions`, `prescriptions`).
   - Outcome: baseline row counts captured and used to sanity-check joins.

4. `04_build_base_rx_clean.py`
   - Purpose: build the primary prescription-level dataset.
   - Join logic: prescriptions linked to admissions and patient demographics.
   - Cleaning done: datetime conversion, drug text normalization.
   - Artifact: `data/processed/base_rx_clean.csv`.

5. `05_add_polypharmacy.py`
   - Purpose: compute concurrent active medication count at each prescription timestamp.
   - Feature produced: `polypharmacy_active_meds`.
   - Artifact: `data/processed/base_rx_polypharm.csv`.

6. `06_fix_polypharmacy.py`
   - Purpose: correction pass for edge cases in polypharmacy counting.
   - Outcome: feature consistency improved before lab integration.

7. `07_add_lab_features.py`
   - Purpose: add latest prior labs before prescription time (anti-leakage design).
   - Features added: `creatinine`, `bun`, `alt`, `ast`, `bilirubin_total`.
   - Artifact: `data/processed/base_rx_with_labs.csv`.

8. `08_clean_model_dataset.py`
   - Purpose: final modeling table cleanup.
   - Outcome: duplicate removal and better numeric readiness (e.g., `dose_val_rx_num`).
   - Artifact: `data/processed/model_dataset_clean_v1.csv`.

9. `09_data_quality_summary.py`
   - Purpose: generate summary stats for missingness, top drugs, and feature distributions.
   - Outcome: used for WIP report quality evidence and risk discussion.

#### B) Labeling script (`src/labels/`)

10. `10_create_proxy_labels_v1.py`
   - Purpose: create binary high-risk proxy label because direct medication-error labels are unavailable in MIMIC-IV demo.
   - Labeling basis: transparent rule-based safety logic (renal risk / medication context / polypharmacy conditions).
   - Artifact dependency: labeling run on cleaned feature-rich dataset.

#### C) Modeling scripts (`src/models/`)

11. `11_train_logreg_baseline.py`
   - Model: Logistic Regression baseline.
   - Role: interpretable benchmark and baseline metric reference.

13. `13_train_logreg_group_split.py`
   - Model: Logistic Regression with `hadm_id` grouped splitting.
   - Role: reduce admission-level leakage risk in validation.

16. `16_train_rf_group_split.py`
   - Model: Random Forest with grouped split.
   - Role: non-linear ensemble benchmark.

20. `20_train_xgb_group_split.py`
   - Model: XGBoost with grouped split.
   - Role: gradient boosting baseline for high performance.

23. `23_train_xgb_mlp_stacking_group_split.py`
   - Model: Hybrid stacking (`XGBoost + MLP`).
   - Role: advanced ensemble to combine tree and neural representations.

#### D) Evaluation and explainability scripts (`src/eval/`)

12. `12_evaluate_baseline.py`
   - Purpose: evaluate non-group baseline logistic regression.

14. `14_evaluate_group_baseline.py`
   - Purpose: evaluate grouped logistic baseline.

15. `15_explain_logreg_coefficients.py`
   - Purpose: coefficient-based interpretability for baseline model.

17. `17_evaluate_rf_group.py`
   - Purpose: grouped RF metrics and confusion matrix.

18. `18_rf_feature_importance.py`
   - Purpose: global feature importance plot for RF.

19. `19_model_comparison_table_v1.py`
   - Purpose: initial model comparison table generation (v1).

21. `21_evaluate_xgb_group.py`
   - Purpose: grouped XGBoost metrics and confusion matrix.

22. `22_xgb_feature_importance.py`
   - Purpose: global feature importance plot for XGBoost.

24. `24_evaluate_xgb_mlp_stacking_group.py`
   - Purpose: grouped evaluation for hybrid stacking model.

25. `25_model_comparison_all_models_v3.py`
   - Purpose: final combined model comparison and v3 reporting assets.

### LO2.8 Consolidated results snapshot (current)

- **LogReg baseline (non-group):** AUROC 0.9972, AUPRC 0.9932, F1 0.9368.
- **LogReg group-split:** AUROC 0.9972, AUPRC 0.9900, F1 0.9617.
- **Random Forest group-split:** AUROC 1.0000, AUPRC 1.0000, F1 0.9854.
- **XGBoost group-split:** AUROC 1.0000, AUPRC 1.0000, F1 1.0000.
- **XGB + MLP stacking group-split:** AUROC 1.0000, AUPRC 1.0000, F1 1.0000.

Interpretation at WIP stage:
- Group-based splitting improved methodological rigor compared to random split.
- Very high/near-perfect scores are promising but require explicit leakage-risk discussion and robustness checks in final report.
- Hybrid model currently matches top XGBoost performance on this dataset version.

### LO2.9 Explainability and presentation progress

- Implemented global explainability outputs:
  - Logistic regression coefficients (`src/eval/15_explain_logreg_coefficients.py`)
  - RF importance plot (`src/eval/18_rf_feature_importance.py`)
  - XGBoost importance plot (`src/eval/22_xgb_feature_importance.py`)
- Built a Streamlit demonstration app (`src/app/streamlit_demo_ui.py`) for:
  - interactive probability prediction,
  - threshold-based risk class adjustment,
  - hybrid snapshot probabilities (xgb/mlp/final where available),
  - local input-context explanation using median and standardized distance.

### LO2.10 Artifacts produced and report readiness

#### Metrics text artifacts
- `reports/metrics_logreg_baseline_v1.txt`
- `reports/metrics_logreg_group_baseline_v1.txt`
- `reports/metrics_rf_group_v1.txt`
- `reports/metrics_xgb_group_v1.txt`
- `reports/metrics_xgb_mlp_stacking_group_v1.txt`

#### Comparison report artifacts
- `reports/model_comparison_v1.md`
- `reports/model_comparison_v3.md`

#### Figure artifacts (selected)
- confusion matrices for baseline/group models,
- RF and XGBoost feature importance plots,
- model-comparison charts (bar/line/hbar/heatmap).

These outputs demonstrate that the project has progressed from proposal-level planning to implementation, evaluation, and demo-level communication.

---

## LO3 (25%) – Current risks, limitations, and mitigation (WIP)

### LO3.1 Key limitations currently acknowledged
- **Proxy label limitation:** labels are rule-based surrogates, not directly observed medication-error events.
- **Demo dataset size limitation:** MIMIC-IV demo is small, potentially making classification easier and less generalizable.
- **Potential leakage concern:** near-perfect metrics require strong temporal and grouping validation discussion.
- **Missingness limitation:** some liver-related labs remain highly missing (around 40–45%).

### LO3.2 Mitigation already performed
- Group-based split by admission (`hadm_id`) in major model runs.
- “Latest prior lab” feature extraction to reduce look-ahead leakage.
- Data cleaning and duplicate handling before training.

### LO3.3 Planned mitigation before final submission
- Add stronger robustness section: sensitivity to threshold, alternative split seeds, and calibration checks.
- Add explicit leakage audit table in final report.
- Include model-card style limitations and intended-use statement.

---

## LO4 (25%) – Remaining tasks and timeline to final submission

### Week 8–9 (near-term)
- Expand literature review with 4–8 journal sources and critical gap mapping.
- Finalize methods narrative with reproducibility details.
- Add explicit risk/ethics discussion (proxy labels, bias, deployment caution).

### Week 9–10 (finalization)
- Complete comparative discussion (why simple vs advanced models differ).
- Strengthen explainability section with representative case examples.
- Polish figures/tables and align all numbering with final report template.

### Week 10+ (submission preparation)
- Final consistency pass (metrics in text vs tables).
- Presentation rehearsal using Streamlit demo and key evidence slides.
- Package repository + report appendix with script-to-artifact mapping.