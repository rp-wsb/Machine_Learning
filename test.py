import pickle
import numpy as np
from sklearn.metrics import classification_report, roc_curve

RESULTS_DIR = "./wyniki"
LABELS = ["NORMAL", "PNEUMONIA"]

with open(f"{RESULTS_DIR}/cnn_results.pkl", "rb") as f:
    res = pickle.load(f)

# for threshold in [0.40, 0.45, 0.50]:
#     y_pred = (res["y_prob"] > threshold).astype(int)
#     print(f"\n── Próg: {threshold} ──")
#     print(classification_report(res["y_true"], y_pred, target_names=LABELS))

y_true = res["y_true"]
y_prob = res["y_prob"]

# Znajdź próg optymalny (kryterium Youdena)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
optimal_idx = np.argmax(tpr - fpr)
threshold_youden = thresholds[optimal_idx]

# Próg dla sensitivity >= 0.90 (priorytet medyczny)
sensitivity_target = 0.90
idx_90 = np.where(tpr >= sensitivity_target)[0][0]
threshold_sens90 = thresholds[idx_90]

for name, thr in [("Youden", threshold_youden), ("Sens≥0.90", threshold_sens90)]:
    y_pred = (y_prob > thr).astype(int)
    print(f"\n── Próg {name}: {thr:.4f} ──")
    print(classification_report(y_true, y_pred, target_names=LABELS))