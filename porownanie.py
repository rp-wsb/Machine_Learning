"""
Porównanie wyników: CNN vs CapsNet (BCE) vs CapsNet+Focal (Youden) vs CapsNet+Focal (Sens>=0.92)
Wczytuje: ./wyniki/cnn_results.pkl
          ./wyniki/capsnet_results.pkl
          ./wyniki/capsnet_focal_results.pkl
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
    precision_recall_curve, classification_report
)

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
RESULTS_DIR = "./wyniki"
LABELS      = ["NORMAL", "PNEUMONIA"]

COLORS = {
    "CNN":                    "#2196F3",
    "CapsNet (BCE)":          "#FF5722",
    "CapsNet+Focal (Youden)": "#4CAF50",
    "CapsNet+Focal (Sens≥0.92)": "#9C27B0",
}


# ─────────────────────────────────────────────
# WCZYTANIE DANYCH
# ─────────────────────────────────────────────
def load_results():
    """
    Zwraca slownik z czterema modelami:
      CNN                       – z cnn_results.pkl
      CapsNet (BCE)             – z capsnet_results.pkl  (prog 0.5)
      CapsNet+Focal (Youden)    – z capsnet_focal_results.pkl
      CapsNet+Focal (Sens>=0.92)– z capsnet_focal_results.pkl
    """
    # CNN
    cnn_path = os.path.join(RESULTS_DIR, "cnn_results.pkl")
    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"Brak: {cnn_path}")
    with open(cnn_path, "rb") as f:
        cnn = pickle.load(f)
    print("✓ Wczytano: CNN")

    # CapsNet BCE
    caps_path = os.path.join(RESULTS_DIR, "capsnet_results.pkl")
    if not os.path.exists(caps_path):
        raise FileNotFoundError(f"Brak: {caps_path}")
    with open(caps_path, "rb") as f:
        caps = pickle.load(f)
    print("✓ Wczytano: CapsNet (BCE)")

    # CapsNet Focal
    focal_path = os.path.join(RESULTS_DIR, "capsnet_focal_results.pkl")
    if not os.path.exists(focal_path):
        raise FileNotFoundError(f"Brak: {focal_path}")
    with open(focal_path, "rb") as f:
        focal = pickle.load(f)
    print("✓ Wczytano: CapsNet+Focal")

    # Wyciagnij raporty z obu progow focal
    focal_y  = focal["ft_results"]["Youden"]
    focal_s  = focal["ft_results"]["Sens>=0.92"]

    results = {
        "CNN": {
            "y_true":  cnn["y_true"],
            "y_pred":  cnn["y_pred"],
            "y_prob":  cnn["y_prob"],
            "report":  cnn["report"],
            "history": cnn["history"],
            "time":    cnn["time"],
            "epochs":  cnn["epochs"],
        },
        "CapsNet (BCE)": {
            "y_true":  caps["y_true"],
            "y_pred":  caps["y_pred"],
            "y_prob":  caps["y_prob"],
            "report":  caps["report"],
            "history": caps["history"],
            "time":    caps["time"],
            "epochs":  caps["epochs"],
        },
        "CapsNet+Focal (Youden)": {
            "y_true":  focal["y_true"],
            "y_pred":  focal_y["y_pred"],
            "y_prob":  focal["y_prob"],
            "report":  focal_y["report"],
            "history": focal["history"],
            "time":    focal["time"],
            "epochs":  focal["epochs"],
            "threshold": focal_y["threshold"],
        },
        "CapsNet+Focal (Sens≥0.92)": {
            "y_true":  focal["y_true"],
            "y_pred":  focal_s["y_pred"],
            "y_prob":  focal["y_prob"],
            "report":  focal_s["report"],
            "history": focal["history"],
            "time":    focal["time"],
            "epochs":  focal["epochs"],
            "threshold": focal_s["threshold"],
        },
    }
    return results


# ─────────────────────────────────────────────
# 1. KRZYWE UCZENIA (CNN + CapsNet BCE + CapsNet Focal)
#    Focal i CapsNet BCE maja te same wagi wejsciowe –
#    pokazujemy historie fine-tuningu jako osobna sekcje
# ─────────────────────────────────────────────
def plot_training_curves(results):
    # Pokaz krzywe dla CNN i CapsNet BCE (pełny trening)
    # oraz fine-tuning focal osobno
    models_full  = ["CNN", "CapsNet (BCE)"]
    models_focal = ["CapsNet+Focal (Youden)"]   # historia identyczna dla obu progow

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Krzywe uczenia", fontsize=15, fontweight="bold")

    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]

    for col, (metric, title) in enumerate(metrics):
        # Górny rząd – pełny trening
        ax = axes[0, col]
        ax.set_title(f"{title} – pełny trening", fontsize=11)
        for name in models_full:
            res  = results[name]
            hist = res["history"]
            c    = COLORS[name]
            if metric in hist:
                ep = range(1, len(hist[metric]) + 1)
                ax.plot(ep, hist[metric],          color=c, lw=2,
                        label=f"{name} train")
                ax.plot(ep, hist[f"val_{metric}"], color=c, lw=2,
                        linestyle="--", label=f"{name} val")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoka")

        # Dolny rząd – fine-tuning focal
        ax2 = axes[1, col]
        ax2.set_title(f"{title} – fine-tuning Focal Loss", fontsize=11)
        res  = results["CapsNet+Focal (Youden)"]
        hist = res["history"]
        c    = COLORS["CapsNet+Focal (Youden)"]
        if metric in hist:
            ep = range(1, len(hist[metric]) + 1)
            ax2.plot(ep, hist[metric],          color=c, lw=2, label="train")
            ax2.plot(ep, hist[f"val_{metric}"], color=c, lw=2,
                     linestyle="--", label="val")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Epoka")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_krzywe_uczenia.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ─────────────────────────────────────────────
# 2. MACIERZE POMYŁEK (wszystkie 4 modele)
# ─────────────────────────────────────────────
def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Macierze pomyłek", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="Blues", xticklabels=LABELS, yticklabels=LABELS,
                    annot_kws={"size": 14, "weight": "bold"})
        acc  = res["report"]["accuracy"]
        sens = res["report"]["PNEUMONIA"]["recall"]
        thr  = res.get("threshold", 0.5)
        ax.set_title(
            f"{name}\nAcc={acc:.3f}  Sens={sens:.3f}  thr={thr:.3f}",
            fontsize=9, color=COLORS[name], fontweight="bold"
        )
        ax.set_xlabel("Predykcja", fontsize=10)
        ax.set_ylabel("Rzeczywistość", fontsize=10)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_macierze_pomylek.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ─────────────────────────────────────────────
# 3. ROC + Precision-Recall
# ─────────────────────────────────────────────
def plot_roc_pr(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Modele z unikalnym y_prob (CNN i CapsNet maja inne y_prob niz Focal)
    # Focal oba progi dzielą ten sam y_prob – rysujemy krzywą raz
    plotted_probs = set()

    for name, res in results.items():
        c      = COLORS[name]
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        prob_id = id(y_prob.tobytes()) if hasattr(y_prob, "tobytes") else id(y_prob)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc        = auc(rec, prec)

        # Dla focal – rysuj krzywa tylko raz (współdzielona), próg jako punkt
        if "Focal" in name:
            label_roc = f"{name}  AUC={roc_auc:.4f}"
            thr = res.get("threshold", 0.5)
            # Zaznacz punkt operacyjny
            fpr_op = 1 - res["report"]["NORMAL"]["recall"]
            tpr_op = res["report"]["PNEUMONIA"]["recall"]
            ax1.scatter([fpr_op], [tpr_op], color=c, s=100, zorder=5)
            ax2.scatter(
                [res["report"]["PNEUMONIA"]["recall"]],
                [res["report"]["PNEUMONIA"]["precision"]],
                color=c, s=100, zorder=5,
                label=f"{name}  thr={thr:.3f}"
            )

        ax1.plot(fpr, tpr, color=c, lw=2.5, label=f"{name}  AUC={roc_auc:.4f}")
        ax2.plot(rec, prec, color=c, lw=2.5, label=f"{name}  AUC={pr_auc:.4f}")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Losowy")
    ax1.set(title="Krzywa ROC", xlabel="False Positive Rate",
            ylabel="True Positive Rate (Sensitivity)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set(title="Krzywa Precision-Recall",
            xlabel="Recall (Sensitivity)", ylabel="Precision")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_roc_pr_krzywe.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ─────────────────────────────────────────────
# 4. SŁUPKI METRYK
# ─────────────────────────────────────────────
def plot_metrics_bar(results):
    metryki = [
        ("PNEUMONIA", "recall",    "Sensitivity\n(PNEUMONIA recall)"),
        ("NORMAL",    "recall",    "Specificity\n(NORMAL recall)"),
        ("PNEUMONIA", "f1-score",  "F1\n(PNEUMONIA)"),
        ("accuracy",  None,        "Accuracy"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Porównanie metryk: wszystkie modele", fontsize=14, fontweight="bold")

    names  = list(results.keys())
    x      = np.arange(len(names))
    colors = [COLORS[n] for n in names]

    for ax, (key, subkey, title) in zip(axes, metryki):
        if subkey:
            vals = [results[n]["report"][key][subkey] for n in names]
        else:
            vals = [results[n]["report"][key] for n in names]

        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" (", "\n(").replace("+", "+\n") for n in names],
            fontsize=8
        )
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_porownanie_metryk.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ─────────────────────────────────────────────
# 5. CZAS I EPOKI
# ─────────────────────────────────────────────
def plot_time_and_epochs(results):
    # Dla focal – czas to fine-tuning, nie pelny trening
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Czas treningu i liczba epok", fontsize=13, fontweight="bold")

    names  = list(results.keys())
    colors = [COLORS[n] for n in names]
    x      = np.arange(len(names))

    times  = [results[n]["time"] / 60 for n in names]
    epochs = [results[n]["epochs"] for n in names]

    bars1 = ax1.bar(x, times, color=colors, alpha=0.85, width=0.5)
    ax1.set_ylabel("Czas (minuty)")
    ax1.set_title("Czas treningu / fine-tuningu")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [n.replace(" (", "\n(").replace("+", "+\n") for n in names], fontsize=8
    )
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.1f} min", ha="center", fontweight="bold", fontsize=9)

    bars2 = ax2.bar(x, epochs, color=colors, alpha=0.85, width=0.5)
    ax2.set_ylabel("Liczba epok")
    ax2.set_title("Wykonane epoki")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [n.replace(" (", "\n(").replace("+", "+\n") for n in names], fontsize=8
    )
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, epochs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "05_czas_epoki.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ─────────────────────────────────────────────
# 6. TABELA PODSUMOWANIA
# ─────────────────────────────────────────────
def save_summary(results):
    rows = []
    for name, res in results.items():
        rep = res["report"]
        rows.append({
            "Model":                  name,
            "Próg decyzyjny":         round(res.get("threshold", 0.5), 4),
            "Accuracy":               round(rep["accuracy"], 4),
            "Precision (PNEUMONIA)":  round(rep["PNEUMONIA"]["precision"], 4),
            "Sensitivity":            round(rep["PNEUMONIA"]["recall"], 4),
            "Specificity":            round(rep["NORMAL"]["recall"], 4),
            "F1 (PNEUMONIA)":         round(rep["PNEUMONIA"]["f1-score"], 4),
            "F1 (macro)":             round(rep["macro avg"]["f1-score"], 4),
            "Czas (min)":             round(res["time"] / 60, 1),
            "Epoki":                  res["epochs"],
        })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(RESULTS_DIR, "podsumowanie.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ {csv_path}")

    txt_path = os.path.join(RESULTS_DIR, "podsumowanie.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("  RAPORT PORÓWNAWCZY: CNN vs CapsNet (BCE) vs CapsNet+Focal\n")
        f.write("  Klasyfikacja zapalenia płuc (Chest X-Ray)\n")
        f.write("=" * 90 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        def best(col):
            return df.loc[df[col].idxmax(), "Model"]

        f.write(f"Najwyższa Accuracy:      {best('Accuracy')}\n")
        f.write(f"Najwyższa Sensitivity:   {best('Sensitivity')}\n")
        f.write(f"Najwyższa Specificity:   {best('Specificity')}\n")
        f.write(f"Najwyższe F1 (PNEUMONIA):{best('F1 (PNEUMONIA)')}\n")
        f.write(f"Najwyższe F1 (macro):    {best('F1 (macro)')}\n")

    print(f"✓ {txt_path}")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Porównanie: CNN vs CapsNet BCE vs CapsNet+Focal (oba progi)")
    print("=" * 70)

    results = load_results()

    print("\nGenerowanie wykresów...")
    plot_training_curves(results)
    plot_confusion_matrices(results)
    plot_roc_pr(results)
    plot_metrics_bar(results)
    plot_time_and_epochs(results)

    print("\nGenerowanie podsumowania...")
    df = save_summary(results)

    print("\n" + "=" * 90)
    print("WYNIKI KOŃCOWE")
    print("=" * 90)
    print(df.to_string(index=False))
    print(f"\n✓ Wszystkie pliki w: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()