import pickle
import numpy as np
from sklearn.metrics import classification_report

RESULTS_DIR = "./wyniki"
LABELS = ["NORMAL", "PNEUMONIA"]

with open(f"{RESULTS_DIR}/cnn_results.pkl", "rb") as f:
    res = pickle.load(f)

for threshold in [0.40, 0.45, 0.50]:
    y_pred = (res["y_prob"] > threshold).astype(int)
    print(f"\n── Próg: {threshold} ──")
    print(classification_report(res["y_true"], y_pred, target_names=LABELS))