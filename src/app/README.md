# Streamlit Demo UI

A presentation-friendly UI for the hybrid stacking model (`XGBoost + MLP`) used in this repository.

## Run

From repository root:

```bash
pip install streamlit joblib pandas scikit-learn xgboost
streamlit run src/app/streamlit_demo_ui.py
```

## What each output means

### 1) Prediction output
- **Predicted risk probability (P=High Risk):**
  The model's estimated probability that the case is high risk.
- **Risk band:**
  UI-only bucket for readability:
  - `Low` if probability < 0.30
  - `Medium` if 0.30–0.69
  - `High` if >= 0.70
- **Predicted class (threshold):**
  Converts probability to class label using your sidebar threshold.
  - If `probability >= threshold` -> `HIGH RISK (1)`
  - Else -> `LOW RISK (0)`

### 2) Hybrid snapshot explanation (Stacking)
This section tries to show three probabilities:
- **XGBoost probability** (base model)
- **MLP probability** (base model)
- **Final stacked probability** (meta-learner output)

Interpretation:
- The two base models each estimate risk.
- The stacking meta-learner combines base outputs (and passthrough feature signal) into the final probability.
- If extraction fails for base probabilities, the final stacked probability is still valid.

### 3) Explainability table
The table compares your numeric input values against dataset medians:
- **feature:** numeric feature name (e.g., creatinine, bun)
- **value:** your submitted input
- **reference_median:** median from reference dataset
- **direction_hint:** whether your value is higher/lower than typical

This is a transparent *input-context* explanation, not causal attribution.

### 4) Explainability bar chart
The bar chart uses a standardized distance (`|z-score|`) from the dataset median:
- Larger bar -> value is more unusual vs reference data.
- Smaller bar -> value is closer to typical range.

Use it to explain **which numeric inputs are most atypical** in the current case.

## Notes and caveats
- This local explainability panel is demo-friendly and fast, but not a full causal explanation method.
- For global explainability across the whole dataset, use scripts in `src/eval/` (feature importances/coefficients).