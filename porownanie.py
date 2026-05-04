"""
Porównanie wyników CNN vs CapsNet
Wczytuje pliki: ./wyniki/cnn_results.pkl i ./wyniki/capsnet_results.pkl
Uruchom po zakończeniu obu modeli (01_cnn.py i 02_capsnet.py)
"""

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
RESULTS_DIR = "./wyniki"
LABELS      = ["NORMAL", "PNEUMONIA"]
COLORS      = {"CNN": "#2196F3", "CapsNet": "#FF5722"}


# ─────────────────────────────────────────────
# WCZYTANIE DANYCH
# ─────────────────────────────────────────────
def load_results():
    paths = {
        "CNN":     os.path.join(RESULTS_DIR, "cnn_results.pkl"),
        "CapsNet": os.path.join(RESULTS_DIR, "capsnet_results.pkl"),
    }
    results = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Brak pliku: {path}\n"
                f"Najpierw uruchom {'01_cnn.py' if name == 'CNN' else '02_capsnet.py'}"
            )
        with open(path, "rb") as f:
            results[name] = pickle.load(f)
        print(f"✓ Wczytano wyniki {name}")
    return results


# ─────────────────────────────────────────────
# WYKRESY
# ─────────────────────────────────────────────
def plot_training_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Krzywe uczenia: CNN vs CapsNet", fontsize=15, fontweight="bold")

    metrics = [("accuracy", "Dokładność (Accuracy)"),
               ("loss",     "Strata (Loss)"),
               ("auc",      "AUC")]

    for ax, (metric, title) in zip(axes, metrics):
        for name, res in results.items():
            hist = res["history"]
            c    = COLORS[name]
            if metric in hist:
                epochs = range(1, len(hist[metric]) + 1)
                ax.plot(epochs, hist[metric],            color=c, lw=2,
                        label=f"{name} trening")
                ax.plot(epochs, hist[f"val_{metric}"],   color=c, lw=2,
                        linestyle="--", label=f"{name} walidacja")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoka")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_krzywe_uczenia.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {path}")


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Macierze pomyłek", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="Blues", xticklabels=LABELS, yticklabels=LABELS,
                    annot_kws={"size": 16, "weight": "bold"})
        acc = res["report"]["accuracy"]
        ax.set_title(f"{name}  (Accuracy: {acc:.4f})", fontsize=12,
                     color=COLORS[name], fontweight="bold")
        ax.set_xlabel("Predykcja", fontsize=11)
        ax.set_ylabel("Rzeczywistość", fontsize=11)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_macierze_pomylek.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {path}")


def plot_roc_pr(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, res in results.items():
        c = COLORS[name]
        y_true, y_prob = res["y_true"], res["y_prob"]

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=c, lw=2.5, label=f"{name}  AUC = {roc_auc:.4f}")

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc        = auc(rec, prec)
        ax2.plot(rec, prec, color=c, lw=2.5, label=f"{name}  AUC = {pr_auc:.4f}")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Losowy klasyfikator")
    ax1.set(title="Krzywa ROC", xlabel="False Positive Rate",
            ylabel="True Positive Rate")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set(title="Krzywa Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_roc_pr_krzywe.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {path}")


def plot_metrics_bar(results):
    metryki = ["precision", "recall", "f1-score"]
    klasy   = ["NORMAL", "PNEUMONIA", "macro avg"]
    x       = np.arange(len(klasy))
    width   = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Porównanie metryk: CNN vs CapsNet", fontsize=14, fontweight="bold")

    for ax, metr in zip(axes, metryki):
        for i, (name, res) in enumerate(results.items()):
            vals = [res["report"][k][metr] for k in klasy]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=name,
                          color=COLORS[name], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", fontsize=8)

        ax.set_title(metr.capitalize(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(klasy, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_porownanie_metryk.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {path}")


def plot_time_and_epochs(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Czas treningu i liczba epok", fontsize=13, fontweight="bold")

    names  = list(results.keys())
    colors = [COLORS[n] for n in names]

    # Czas
    times = [results[n]["time"] / 60 for n in names]
    bars1 = ax1.bar(names, times, color=colors, alpha=0.85, width=0.4)
    ax1.set_ylabel("Czas (minuty)")
    ax1.set_title("Czas treningu")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f} min", ha="center", fontweight="bold")

    # Epoki
    epochs = [results[n]["epochs"] for n in names]
    bars2  = ax2.bar(names, epochs, color=colors, alpha=0.85, width=0.4)
    ax2.set_ylabel("Liczba epok")
    ax2.set_title("Wykonane epoki")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, epochs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(val), ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "05_czas_epoki.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {path}")


# ─────────────────────────────────────────────
# TABELA PODSUMOWANIA
# ─────────────────────────────────────────────
def save_summary(results):
    rows = []
    for name, res in results.items():
        rep = res["report"]
        rows.append({
            "Model":                  name,
            "Accuracy":               round(rep["accuracy"], 4),
            "Precision (PNEUMONIA)":  round(rep["PNEUMONIA"]["precision"], 4),
            "Recall / Sensitivity":   round(rep["PNEUMONIA"]["recall"], 4),
            "Specificity (NORMAL)":   round(rep["NORMAL"]["recall"], 4),
            "F1 (PNEUMONIA)":         round(rep["PNEUMONIA"]["f1-score"], 4),
            "F1 (macro)":             round(rep["macro avg"]["f1-score"], 4),
            "Czas treningu (min)":    round(res["time"] / 60, 1),
            "Liczba epok":            res["epochs"],
        })

    df = pd.DataFrame(rows)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "podsumowanie.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Zapisano: {csv_path}")

    # TXT
    txt_path = os.path.join(RESULTS_DIR, "podsumowanie.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  RAPORT PORÓWNAWCZY: CNN vs CapsNet\n")
        f.write("  Klasyfikacja zapalenia płuc (Chest X-Ray)\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        winner_acc = max(results, key=lambda n: results[n]["report"]["accuracy"])
        winner_f1  = max(results, key=lambda n: results[n]["report"]["macro avg"]["f1-score"])
        winner_sen = max(results, key=lambda n: results[n]["report"]["PNEUMONIA"]["recall"])
        f.write(f"Najwyższa Accuracy:    {winner_acc}\n")
        f.write(f"Najwyższe F1 (macro):  {winner_f1}\n")
        f.write(f"Najwyższa Sensitivity: {winner_sen}\n")
    print(f"✓ Zapisano: {txt_path}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Porównanie wyników: CNN vs CapsNet")
    print("=" * 60)

    results = load_results()

    print("\nGenerowanie wykresów...")
    plot_training_curves(results)
    plot_confusion_matrices(results)
    plot_roc_pr(results)
    plot_metrics_bar(results)
    plot_time_and_epochs(results)

    print("\nGenerowanie podsumowania...")
    df = save_summary(results)

    print("\n" + "=" * 60)
    print("WYNIKI KOŃCOWE")
    print("=" * 60)
    print(df.to_string(index=False))

    print(f"\n✓ Wszystkie pliki zapisane w: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()