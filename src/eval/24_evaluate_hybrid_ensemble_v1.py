import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

LOGREG_BUNDLE = Path("models/logreg_group_baseline_v1.joblib")
RF_BUNDLE = Path("models/rf_group_v1.joblib")
XGB_BUNDLE = Path("models/xgb_group_v1.joblib")

METRICS_OUT = Path("reports/metrics_hybrid_ensemble_v1.txt")
FIG_OUT = Path("reports/figures/confusion_matrix_hybrid_ensemble_v1.png")

def main():
    for p in [LOGREG_BUNDLE, RF_BUNDLE, XGB_BUNDLE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model bundle: {p}")

    # Load bundles
    logreg = joblib.load(LOGREG_BUNDLE)
    rf = joblib.load(RF_BUNDLE)
    xgb = joblib.load(XGB_BUNDLE)

    # Use LogReg test set as the canonical test set
    X_test = logreg["X_test"]
    y_test = logreg["y_test"].astype(int)

    m1 = logreg["model"]
    m2 = rf["model"]
    m3 = xgb["model"]

    # Predict probabilities
    p1 = m1.predict_proba(X_test)[:, 1]
    p2 = m2.predict_proba(X_test)[:, 1]
    p3 = m3.predict_proba(X_test)[:, 1]

    # Soft voting: average probability
    y_prob = (p1 + p2 + p3) / 3.0
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Hybrid Ensemble (Soft Voting) v1 ===")
    print("Models: LogReg + RF + XGBoost")
    print("AUROC:", round(auroc, 4))
    print("AUPRC:", round(auprc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1:", round(f1, 4))
    print("\nConfusion matrix [ [TN FP], [FN TP] ]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save metrics
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        f.write("Hybrid Ensemble (Soft Voting) v1\n")
        f.write("Models: LogReg + RF + XGBoost\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"AUPRC: {auprc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n\n")
        f.write("Confusion matrix [[TN FP],[FN TP]]:\n")
        f.write(np.array2string(cm))
        f.write("\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Save confusion matrix figure
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix - Hybrid Ensemble v1")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=200)

    print("\n✅ Saved metrics:", METRICS_OUT.resolve())
    print("✅ Saved figure:", FIG_OUT.resolve())

if __name__ == "__main__":
    main()
