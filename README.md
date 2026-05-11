 Medication Error Risk Prediction

A reproducible machine-learning pipeline for predicting medication-error risk from the MIMIC-IV Clinical Database Demo. The project builds a prescription-level modelling dataset, creates transparent proxy risk labels, trains multiple tabular classifiers, compares model performance, and provides a Streamlit demo UI for interactive prediction.

> **Important:** This repository is for coursework, research, and demonstration purposes only. The labels are rule-based proxy labels, not clinician-adjudicated medication-error outcomes. Do not use this project for clinical decision-making without formal validation, governance, and review.
>
> **Documentation note:** This is the single README for the project. Streamlit UI instructions are included in the [Streamlit demo UI](#streamlit-demo-ui) section instead of a separate app README.

## Table of contents

- [Project goals](#project-goals)
- [Repository structure](#repository-structure)
- [Data requirements](#data-requirements)
- [Environment setup](#environment-setup)
- [Quick start: full pipeline](#quick-start-full-pipeline)
- [End-to-end workflow](#end-to-end-workflow)
- [Model training](#model-training)
- [Evaluation and reports](#evaluation-and-reports)
- [Streamlit demo UI](#streamlit-demo-ui)
- [Current benchmark results](#current-benchmark-results)
- [Generated artifacts](#generated-artifacts)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Project goals

This project explores how structured hospital data can be transformed into medication-risk features and used for supervised risk prediction. The workflow focuses on:

- Extracting and validating MIMIC-IV demo files.
- Building a cleaned prescription/admission/patient dataset.
- Engineering medication and laboratory features, including active-medication polypharmacy counts.
- Creating explainable proxy labels for high-risk medication situations.
- Training baseline and stronger ML models with admission-level group splits.
- Evaluating models with AUROC, AUPRC, precision, recall, F1, and confusion matrices.
- Serving the best-performing model through a local Streamlit demonstration UI.

## Repository structure

```text
.
├── README.md
├── docs/
│   └── wip_Assessment2.md
├── reports/
│   ├── metrics_*.txt
│   ├── model_comparison_v*.md
│   └── figures/
├── src/
│   ├── app/
│   │   ├── streamlit_demo_ui.css
│   │   └── streamlit_demo_ui.py
│   ├── eval/
│   │   └── evaluation, feature-importance, and comparison scripts
│   ├── labels/
│   │   └── proxy label generation script
│   ├── models/
│   │   └── model training scripts
│   └── pipeline/
│       └── data extraction, cleaning, and feature engineering scripts
└── gantt_chart_weeks_clean.png
```

Large/generated assets are intentionally not tracked by Git:

- `data/raw/`
- `data/processed/`
- `models/*.joblib`
- `*.csv`
- `*.zip`

## Data requirements

The pipeline expects the **MIMIC-IV Clinical Database Demo v2.2** ZIP file at:

```text
data/raw/mimic-iv-clinical-database-demo-2.2.zip
```

After extraction, scripts look under:

```text
data/raw/mimic_demo_2.2/
```

The scripts handle the common case where the ZIP extracts into an additional nested folder before `hosp/` and `icu/`.

Expected source tables include, at minimum:

- `hosp/patients.csv.gz`
- `hosp/admissions.csv.gz`
- `hosp/prescriptions.csv.gz`
- `hosp/labevents.csv.gz`
- `hosp/d_labitems.csv.gz`

## Environment setup

Use Python 3.10+ if possible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn xgboost joblib matplotlib tabulate streamlit
```

If you only want to run the Streamlit UI after a model has already been trained, install:

```bash
python -m pip install streamlit joblib pandas numpy scikit-learn xgboost
```

## Quick start: full pipeline

After placing the MIMIC-IV demo ZIP at `data/raw/mimic-iv-clinical-database-demo-2.2.zip`, run this sequence from the repository root to rebuild the main dataset, train the default stacking model, evaluate it, and launch the UI:

```bash
python src/pipeline/01_extract.py
python src/pipeline/02_list_files.py
python src/pipeline/03_load_core_tables.py
python src/pipeline/04_build_base_rx_clean.py
python src/pipeline/06_fix_polypharmacy.py
python src/pipeline/07_add_lab_features.py
python src/pipeline/08_clean_model_dataset.py
python src/pipeline/09_data_quality_summary.py
python src/labels/10_create_proxy_labels_v1.py
python src/models/23_train_xgb_mlp_stacking_group_split.py
python src/eval/24_evaluate_xgb_mlp_stacking_group.py
streamlit run src/app/streamlit_demo_ui.py
```

Run the additional training and evaluation commands below if you want to reproduce all model-comparison reports.

## End-to-end workflow

Run commands from the repository root.

### 1. Extract the raw demo dataset

```bash
python src/pipeline/01_extract.py
```

### 2. Confirm available files

```bash
python src/pipeline/02_list_files.py
```

### 3. Inspect core table row counts and columns

```bash
python src/pipeline/03_load_core_tables.py
```

### 4. Build the cleaned prescription base table

```bash
python src/pipeline/04_build_base_rx_clean.py
```

Creates:

```text
data/processed/base_rx_clean.csv
```

### 5. Add polypharmacy features

Use the fixed implementation:

```bash
python src/pipeline/06_fix_polypharmacy.py
```

Creates:

```text
data/processed/base_rx_polypharm.csv
```

> `src/pipeline/05_add_polypharmacy.py` is an earlier version kept for traceability. Prefer `src/pipeline/06_fix_polypharmacy.py` for the main pipeline.

### 6. Add laboratory features

```bash
python src/pipeline/07_add_lab_features.py
```

Creates:

```text
data/processed/base_rx_with_labs.csv
```

The current lab feature targets are:

- Creatinine
- Blood urea nitrogen / BUN
- ALT
- AST
- Total bilirubin

For each lab feature, the pipeline attaches the latest available lab value before the prescription `starttime`, plus missingness and time-since-lab indicators.

### 7. Clean the modelling dataset

```bash
python src/pipeline/08_clean_model_dataset.py
```

Creates:

```text
data/processed/model_dataset_clean_v1.csv
```

### 8. Review data quality

```bash
python src/pipeline/09_data_quality_summary.py
```

### 9. Create proxy labels

```bash
python src/labels/10_create_proxy_labels_v1.py
```

Creates:

```text
data/processed/model_dataset_labeled_v1.csv
```

The current high-risk proxy label is triggered when either rule is true:

1. **Polypharmacy rule:** active medication count meets or exceeds the configured threshold.
2. **Renal-risk rule:** creatinine is elevated and the medication name matches a renal-risk keyword list.

The label script also writes helper columns such as rule indicators and a readable `risk_reason`.

## Model training

Train one or more models after creating `data/processed/model_dataset_labeled_v1.csv`.

### Logistic regression baseline with random row split

```bash
python src/models/11_train_logreg_baseline.py
```

Creates:

```text
models/logreg_baseline_v1.joblib
```

### Logistic regression with admission-level group split

```bash
python src/models/13_train_logreg_group_split.py
```

Creates:

```text
models/logreg_group_baseline_v1.joblib
```

### Random forest with admission-level group split

```bash
python src/models/16_train_rf_group_split.py
```

Creates:

```text
models/rf_group_v1.joblib
```

### XGBoost with admission-level group split

```bash
python src/models/20_train_xgb_group_split.py
```

Creates:

```text
models/xgb_group_v1.joblib
```

### Hybrid XGBoost + MLP stacking model

```bash
python src/models/23_train_xgb_mlp_stacking_group_split.py
```

Creates:

```text
models/xgb_mlp_stacking_group_v1.joblib
```

This is the default model used by the Streamlit demo UI.

## Evaluation and reports

Evaluate trained models with:

```bash
python src/eval/12_evaluate_baseline.py
python src/eval/14_evaluate_group_baseline.py
python src/eval/17_evaluate_rf_group.py
python src/eval/21_evaluate_xgb_group.py
python src/eval/24_evaluate_xgb_mlp_stacking_group.py
```

Generate explainability artifacts and comparison reports with:

```bash
python src/eval/15_explain_logreg_coefficients.py
python src/eval/18_rf_feature_importance.py
python src/eval/22_xgb_feature_importance.py
python src/eval/19_model_comparison_table_v1.py
python src/eval/25_model_comparison_all_models_v3.py
```

Typical report outputs include:

- `reports/metrics_logreg_baseline_v1.txt`
- `reports/metrics_logreg_group_baseline_v1.txt`
- `reports/metrics_rf_group_v1.txt`
- `reports/metrics_xgb_group_v1.txt`
- `reports/metrics_xgb_mlp_stacking_group_v1.txt`
- `reports/model_comparison_v1.md`
- `reports/model_comparison_v3.md`
- `reports/figures/*.png`
- `reports/figures/*.svg`

## Streamlit demo UI

The app provides a presentation-friendly interface for entering patient, medication, and lab values and viewing a risk prediction.

Run from the repository root:

```bash
streamlit run src/app/streamlit_demo_ui.py
```

Default inputs expected by the app:

```text
models/xgb_mlp_stacking_group_v1.joblib
data/processed/model_dataset_labeled_v1.csv
```

The app shows:

- Predicted high-risk probability.
- UI risk band: low, medium, or high.
- Threshold-based class prediction.
- Hybrid stacking snapshot where available.
- Input-context explanation comparing submitted numeric values with dataset medians.

### Interpreting Streamlit outputs

The Streamlit interface has one source of documentation in this root README so users do not need to check a second README file. Key outputs are:

1. **Predicted risk probability:** estimated probability that the case is high risk.
2. **Risk band:** a UI-only readability label: low when probability is below `0.30`, medium from `0.30` to `0.69`, and high at `0.70` or above.
3. **Predicted class:** threshold-based class label using the sidebar threshold. If `probability >= threshold`, the UI displays `HIGH RISK (1)`; otherwise it displays `LOW RISK (0)`.
4. **Hybrid snapshot explanation:** when available, the app shows XGBoost, MLP, and final stacked probabilities. The final stacked probability remains the primary model output.
5. **Explainability table and bar chart:** numeric inputs are compared with dataset medians, and larger standardized distances indicate values that are more unusual relative to the reference dataset. This is input-context explainability, not a causal attribution method.

## Current benchmark results

The latest committed comparison table reports:

| Model | AUROC | AUPRC | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|--:|
| LogReg (Group Split) | 0.9972 | 0.9900 | 0.9408 | 0.9835 | 0.9617 |
| Random Forest (Group Split) | 1.0000 | 1.0000 | 1.0000 | 0.9711 | 0.9854 |
| XGBoost (Group Split) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost + MLP Stacking (Group Split) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

These results are based on rule-derived proxy labels. Very high scores are expected when labels are generated from features that are also available to the models, so interpret performance as a pipeline demonstration rather than proof of clinical predictive validity.

## Generated artifacts

### Data pipeline outputs

```text
data/processed/base_rx_clean.csv
data/processed/base_rx_polypharm.csv
data/processed/base_rx_with_labs.csv
data/processed/model_dataset_clean_v1.csv
data/processed/model_dataset_labeled_v1.csv
```

### Model outputs

```text
models/logreg_baseline_v1.joblib
models/logreg_group_baseline_v1.joblib
models/rf_group_v1.joblib
models/xgb_group_v1.joblib
models/xgb_mlp_stacking_group_v1.joblib
```

### Report outputs

```text
reports/metrics_*.txt
reports/model_comparison_*.md
reports/figures/*
```

## Limitations

- Uses the MIMIC-IV demo dataset, which is small and not representative of full production-scale data.
- Labels are proxy labels derived from rules, not confirmed medication-error events.
- Rule-derived labels can make model results look artificially strong when rule input features are included in training.
- The project uses prescription-level rows, so repeated admissions/patients require careful group splitting to avoid leakage.
- The Streamlit explanations are input-context summaries, not causal explanations.
- Any clinical use would require expert validation, calibration, fairness review, and prospective evaluation.

## Troubleshooting

### `FileNotFoundError` for raw data

Confirm the ZIP exists at:

```text
data/raw/mimic-iv-clinical-database-demo-2.2.zip
```

Then run:

```bash
python src/pipeline/01_extract.py
python src/pipeline/02_list_files.py
```

### Missing model for Streamlit

Train the stacking model first:

```bash
python src/models/23_train_xgb_mlp_stacking_group_split.py
```

Then start the app:

```bash
streamlit run src/app/streamlit_demo_ui.py
```

### Missing `tabulate` when creating Markdown comparison tables

Install `tabulate`:

```bash
python -m pip install tabulate
```

Then rerun:

```bash
python src/eval/25_model_comparison_all_models_v3.py
```