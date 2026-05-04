"""
Model CNN – klasyfikacja zapalenia płuc
Dataset: chest_xray (Kaggle)
Wyniki zapisywane do: ./wyniki/cnn_*
"""

import os
import time
import warnings
import numpy as np
import pickle

from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
DATA_DIR    = r"C:\Users\troch\PycharmProjects\Pneumonia\chest_xray"
IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 15
SEED        = 42
RESULTS_DIR = "./wyniki"

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

LABELS = ["NORMAL", "PNEUMONIA"]


# ─────────────────────────────────────────────
# DANE
# ─────────────────────────────────────────────
def build_generators(data_dir, img_size, batch_size):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.05,
        validation_split=0.2,
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size, batch_size=batch_size,
        class_mode="binary", seed=SEED, shuffle=True,
        subset="training"
    )
    val = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size, batch_size=batch_size,
        class_mode="binary", seed=SEED, shuffle=False,
        subset="validation"
    )
    test = test_gen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size, batch_size=batch_size,
        class_mode="binary", seed=SEED, shuffle=False
    )
    return train, val, test


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.7):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return tf.reduce_mean(alpha * tf.pow(1 - p_t, gamma) * bce)
    return loss


def build_cnn(img_size):
    inp = keras.Input(shape=(*img_size, 3))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out, name="CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.Precision(name="precision"),]
    )
    return model


# ─────────────────────────────────────────────
# TRENING
# ─────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(monitor="val_auc", patience=7, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
        ModelCheckpoint(
            filepath=os.path.join(RESULTS_DIR, "best_cnn.keras"),
            monitor="val_auc", save_best_only=True, mode="max"
        ),
    ]


def main():
    print("=" * 60)
    print("  CNN – Klasyfikacja zapalenia płuc")
    print("=" * 60)

    print("\n[1/3] Ładowanie danych...")
    train_gen, val_gen, test_gen = build_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"  Trening:   {train_gen.samples} obrazów")
    print(f"  Walidacja: {val_gen.samples} obrazów")
    print(f"  Test:      {test_gen.samples} obrazów")

    print("\n[2/3] Trening CNN...")
    model = build_cnn(IMG_SIZE)
    model.summary()

    t0 = time.time()

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_gen.classes
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"  Wagi klas: NORMAL={class_weights[0]:.2f}, PNEUMONIA={class_weights[1]:.2f}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        class_weight=class_weight_dict,
        verbose=1
    )
    elapsed = time.time() - t0

    print("\n[3/3] Ewaluacja na zbiorze testowym...")
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_true = test_gen.classes
    y_pred = (y_prob > 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=LABELS))

    # Zapis wyników do pliku (do użycia przez skrypt porównania)
    results = {
        "y_true":  y_true,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "history": history.history,
        "report":  report,
        "time":    elapsed,
        "epochs":  len(history.history["loss"]),
    }
    out_path = os.path.join(RESULTS_DIR, "cnn_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ Wyniki CNN zapisane: {out_path}")
    print(f"  Czas treningu: {elapsed/60:.1f} min  |  Epoki: {results['epochs']}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Sensitivity (PNEUMONIA recall): {report['PNEUMONIA']['recall']:.4f}")
    print(f"  Specificity (NORMAL recall):    {report['NORMAL']['recall']:.4f}")


if __name__ == "__main__":
    main()