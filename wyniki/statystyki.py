"""
Analiza zbioru danych – Chest X-Ray Pneumonia
================================================
Generuje 3 wizualizacje:
  1. Przykładowe zdjęcia RTG (Normal vs Pneumonia)
  2. Wykres słupkowy rozkładu klas
  3. Tabela z podziałem liczbowym i procentowym

Uruchom przed treningiem modeli.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.table import Table
import warnings
warnings.filterwarnings("ignore")

from PIL import Image

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
DATA_DIR    = r"C:\Users\troch\PycharmProjects\Pneumonia\chest_xray"
RESULTS_DIR = "./wyniki"
os.makedirs(RESULTS_DIR, exist_ok=True)

SPLITS = ["train", "val", "test"]
KLASY  = ["NORMAL", "PNEUMONIA"]
KOLORY = {"NORMAL": "#4CAF50", "PNEUMONIA": "#F44336"}


# ─────────────────────────────────────────────
# LICZENIE OBRAZÓW
# ─────────────────────────────────────────────
def policz_obrazy(data_dir):
    stats = {}
    for split in SPLITS:
        stats[split] = {}
        for klasa in KLASY:
            path = os.path.join(data_dir, split, klasa)
            if os.path.exists(path):
                n = len([f for f in os.listdir(path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            else:
                n = 0
            stats[split][klasa] = n
    return stats


def znajdz_przyklad(data_dir, split, klasa):
    """Zwraca ścieżkę do pierwszego obrazu danej klasy."""
    path = os.path.join(data_dir, split, klasa)
    for f in os.listdir(path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(path, f)
    return None


# ─────────────────────────────────────────────
# 1. PRZYKŁADOWE ZDJĘCIA RTG
# ─────────────────────────────────────────────
def plot_przyklady(data_dir, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0D1117")

    # opisy = {
    #     "NORMAL":    "Płuca zdrowe. Pola płucne są czyste,\nprzejrzyste, bez zagęszczeń ani nacieków.",
    #     "PNEUMONIA": "Zapalenie płuc. Widoczne zagęszczenia\ni nacieki w polach płucnych."
    # }
    ramki = {"NORMAL": "#4CAF50", "PNEUMONIA": "#F44336"}

    for ax, klasa in zip(axes, KLASY):
        sciezka = znajdz_przyklad(data_dir, "train", klasa)
        img     = Image.open(sciezka).convert("L")
        img_arr = np.array(img)

        ax.imshow(img_arr, cmap="gray", aspect="auto")
        ax.set_facecolor("#0D1117")

        # Kolorowa ramka
        for spine in ax.spines.values():
            spine.set_edgecolor(ramki[klasa])
            spine.set_linewidth(3)

        ax.set_xticks([])
        ax.set_yticks([])

        # Etykieta klasy
        ax.set_title(klasa, fontsize=18, fontweight="bold",
                     color=ramki[klasa], pad=12, fontfamily="monospace")

        # Opis poniżej
        # ax.set_xlabel(opisy[klasa], fontsize=10, color="#CCCCCC",
        #               labelpad=10, linespacing=1.6)

    # fig.suptitle("Przykładowe radiogramy klatki piersiowej",
    #              fontsize=16, fontweight="bold", color="white", y=1.01)

    plt.tight_layout()
    path = os.path.join(save_dir, "01_przyklady_rtg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Zapisano: {path}")


# ─────────────────────────────────────────────
# 2. WYKRES SŁUPKOWY ROZKŁADU KLAS
# ─────────────────────────────────────────────
def plot_rozklad(stats, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    etykiety_split = {
        "train": "Zbiór treningowy",
        "val":   "Zbiór walidacyjny",
        "test":  "Zbiór testowy"
    }

    x       = np.arange(len(SPLITS))
    width   = 0.35
    offsets = [-width/2, width/2]

    for i, klasa in enumerate(KLASY):
        wartosci = [stats[s][klasa] for s in SPLITS]
        bars = ax.bar(x + offsets[i], wartosci, width,
                      label=klasa, color=KOLORY[klasa], alpha=0.88,
                      edgecolor="white", linewidth=1.2)

        for bar, val in zip(bars, wartosci):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{val:,}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=KOLORY[klasa]
            )

    ax.set_xticks(x)
    ax.set_xticklabels([etykiety_split[s] for s in SPLITS], fontsize=12)
    ax.set_ylabel("Liczba obrazów", fontsize=12)
    # ax.set_title("Rozkład klas w zbiorach danych",
    #              fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adnotacja o nierównowadze
    total_train = sum(stats["train"].values())
    pneu_pct    = stats["train"]["PNEUMONIA"] / total_train * 100
    # ax.annotate(
    #     f"Nierównowaga klas w treningu:\n{pneu_pct:.0f}% PNEUMONIA vs {100-pneu_pct:.0f}% NORMAL",
    #     xy=(0, stats["train"]["PNEUMONIA"]),
    #     xytext=(1.2, stats["train"]["PNEUMONIA"] * 0.85),
    #     fontsize=9, color="#555555",
    #     arrowprops=dict(arrowstyle="->", color="#999999"),
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
    #               edgecolor="#CCCCCC", alpha=0.9)
    # )

    plt.tight_layout()
    path = os.path.join(save_dir, "02_rozklad_klas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Zapisano: {path}")


# ─────────────────────────────────────────────
# 3. TABELA Z PODZIAŁEM LICZBOWYM
# ─────────────────────────────────────────────
def plot_tabela(stats, save_dir):
    etykiety_split = {
        "train": "Treningowy",
        "val":   "Walidacyjny",
        "test":  "Testowy"
    }

    # Przygotowanie danych
    wiersze = []
    for split in SPLITS:
        n_norm  = stats[split]["NORMAL"]
        n_pneu  = stats[split]["PNEUMONIA"]
        n_total = n_norm + n_pneu
        if n_total > 0:
            pct_norm = n_norm  / n_total * 100
            pct_pneu = n_pneu / n_total * 100
        else:
            pct_norm = pct_pneu = 0
        wiersze.append([
            etykiety_split[split],
            f"{n_norm:,}",
            f"{pct_norm:.1f}%",
            f"{n_pneu:,}",
            f"{pct_pneu:.1f}%",
            f"{n_total:,}",
        ])

    # Suma
    total_norm = sum(stats[s]["NORMAL"]    for s in SPLITS)
    total_pneu = sum(stats[s]["PNEUMONIA"] for s in SPLITS)
    total_all  = total_norm + total_pneu
    wiersze.append([
        "ŁĄCZNIE",
        f"{total_norm:,}",
        f"{total_norm/total_all*100:.1f}%",
        f"{total_pneu:,}",
        f"{total_pneu/total_all*100:.1f}%",
        f"{total_all:,}",
    ])

    naglowki = [
        "Podzbiór",
        "NORMAL\n(liczba)",
        "NORMAL\n(%)",
        "PNEUMONIA\n(liczba)",
        "PNEUMONIA\n(%)",
        "Łącznie",
    ]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    ax.axis("off")

    tabela = ax.table(
        cellText=wiersze,
        colLabels=naglowki,
        cellLoc="center",
        loc="center",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1, 2.2)

    # Stylizacja nagłówka
    for j in range(len(naglowki)):
        cell = tabela[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Kolory kolumn NORMAL / PNEUMONIA
    kol_norm  = [1, 2]
    kol_pneu  = [3, 4]
    for i in range(1, len(wiersze) + 1):
        for j in kol_norm:
            tabela[i, j].set_facecolor("#E8F5E9")
        for j in kol_pneu:
            tabela[i, j].set_facecolor("#FFEBEE")

    # Wiersz ŁĄCZNIE
    for j in range(len(naglowki)):
        cell = tabela[len(wiersze), j]
        cell.set_facecolor("#ECF0F1")
        cell.set_text_props(fontweight="bold")

    # Obramowania
    for (i, j), cell in tabela.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.8)

    # ax.set_title("Podział liczbowy zbioru danych",
    #              fontsize=13, fontweight="bold", pad=20, color="#2C3E50")

    # Legenda kolorów
    patch_n = mpatches.Patch(color="#E8F5E9", label="NORMAL")
    patch_p = mpatches.Patch(color="#FFEBEE", label="PNEUMONIA")
    ax.legend(handles=[patch_n, patch_p], loc="lower right",
              fontsize=9, framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(save_dir, "03_tabela_podzialu.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Zapisano: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Analiza zbioru danych – Chest X-Ray Pneumonia")
    print("=" * 55)

    print("\n[1/4] Liczenie obrazów w zbiorach...")
    stats = policz_obrazy(DATA_DIR)
    for split in SPLITS:
        n = stats[split]["NORMAL"]
        p = stats[split]["PNEUMONIA"]
        print(f"  {split:6s}: NORMAL={n:4d}  PNEUMONIA={p:4d}  ŁĄCZNIE={n+p:4d}")

    print("\n[2/4] Generowanie przykładów RTG...")
    plot_przyklady(DATA_DIR, RESULTS_DIR)

    print("\n[3/4] Generowanie wykresu rozkładu klas...")
    plot_rozklad(stats, RESULTS_DIR)

    print("\n[4/4] Generowanie tabeli podziału...")
    plot_tabela(stats, RESULTS_DIR)

    print(f"\n✓ Wszystkie pliki zapisane w: {os.path.abspath(RESULTS_DIR)}/")
    print("  01_przyklady_rtg.png")
    print("  02_rozklad_klas.png")
    print("  03_tabela_podzialu.png")


if __name__ == "__main__":
    main()