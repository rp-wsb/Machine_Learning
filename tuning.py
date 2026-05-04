"""
Fine-tuning CapsNet z Focal Loss
POPRAWKA: zamiast load_model (problem z Lambda + safe_mode),
          odbudowujemy architekture i wczytujemy same wagi przez load_weights.
"""

import os
import time
import warnings
import numpy as np
import pickle
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import classification_report, roc_curve

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
DATA_DIR     = r"C:\Users\troch\PycharmProjects\Pneumonia\chest_xray"
IMG_SIZE     = (150, 150)
BATCH_SIZE   = 32
EPOCHS_FT    = 20
SEED         = 42
RESULTS_DIR  = "./wyniki"
MODEL_PATH   = os.path.join(RESULTS_DIR, "best_capsnet.keras")

ALPHA        = 0.75
GAMMA        = 2.0
LR_FINETUNE  = 2e-4

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

LABELS = ["NORMAL", "PNEUMONIA"]


# ─────────────────────────────────────────────
# CUSTOM LAYERS (identyczne jak w capsnet_fixed.py)
# ─────────────────────────────────────────────
class Squash(layers.Layer):
    def call(self, inputs):
        norm_sq = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        norm    = tf.sqrt(norm_sq + keras.backend.epsilon())
        return (norm_sq / (1.0 + norm_sq)) * (inputs / norm)


class PrimaryCaps(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, kernel_size=9, strides=2, **kw):
        super().__init__(**kw)
        self.num_capsules = num_capsules
        self.capsule_dim  = capsule_dim
        self.kernel_size  = kernel_size
        self.strides      = strides
        self.conv = layers.Conv2D(
            num_capsules * capsule_dim,
            kernel_size, strides=strides,
            padding="valid", activation="relu",
            kernel_initializer="he_normal"
        )
        self.bn     = layers.BatchNormalization()
        self.squash = Squash()

    def call(self, inputs, training=False):
        x     = self.conv(inputs)
        x     = self.bn(x, training=training)
        shape = tf.shape(x)
        h, w  = shape[1], shape[2]
        x = tf.reshape(x, [-1, h * w * self.num_capsules, self.capsule_dim])
        return self.squash(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "capsule_dim":  self.capsule_dim,
            "kernel_size":  self.kernel_size,
            "strides":      self.strides,
        })
        return config


class DigitCaps(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, num_routing=3, **kw):
        super().__init__(**kw)
        self.num_capsules = num_capsules
        self.capsule_dim  = capsule_dim
        self.num_routing  = num_routing
        self.squash       = Squash()

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_capsule_dim  = input_shape[2]
        self.W = self.add_weight(
            name="routing_weights",
            shape=(1, self.input_num_capsules, self.num_capsules,
                   self.capsule_dim, self.input_capsule_dim),
            initializer=tf.initializers.TruncatedNormal(stddev=0.1),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch           = tf.shape(inputs)[0]
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 2), 4)
        inputs_tiled    = tf.tile(inputs_expanded, [1, 1, self.num_capsules, 1, 1])
        W_tiled         = tf.tile(self.W, [batch, 1, 1, 1, 1])
        u_hat           = tf.squeeze(tf.matmul(W_tiled, inputs_tiled), axis=4)

        b = tf.zeros([batch, self.input_num_capsules, self.num_capsules, 1])
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            v = self.squash(s)
            if i < self.num_routing - 1:
                b = b + tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)

        return tf.squeeze(v, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "capsule_dim":  self.capsule_dim,
            "num_routing":  self.num_routing,
        })
        return config


# ─────────────────────────────────────────────
# ARCHITEKTURA (identyczna jak w capsnet_fixed.py)
# Musi byc dokladnie ta sama zeby load_weights zadzialalo
# ─────────────────────────────────────────────
def build_capsnet(img_size, num_routing=3, learning_rate=1e-3):
    inp = keras.Input(shape=(*img_size, 3))

    x = layers.Conv2D(32, 3, activation="relu", padding="same",
                      kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    x = PrimaryCaps(num_capsules=8, capsule_dim=16,
                    kernel_size=5, strides=2)(x)

    digit = DigitCaps(num_capsules=2, capsule_dim=32,
                      num_routing=num_routing)(x)

    norm = layers.Lambda(
        lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1) + keras.backend.epsilon())
    )(digit)

    flat = layers.Dropout(0.3)(norm)
    out  = layers.Dense(1, activation="sigmoid",
                        kernel_initializer="glorot_uniform")(flat)

    model = Model(inp, out, name="CapsNet_v2")
    return model


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────
def make_focal_loss(alpha=0.75, gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(),
                                  1.0 - keras.backend.epsilon())
        bce_pos   = -tf.math.log(y_pred)
        bce_neg   = -tf.math.log(1.0 - y_pred)
        focal_pos = tf.pow(1.0 - y_pred, gamma) * bce_pos
        focal_neg = tf.pow(y_pred,       gamma) * bce_neg
        loss = (alpha * y_true * focal_pos
                + (1.0 - alpha) * (1.0 - y_true) * focal_neg)
        return tf.reduce_mean(loss)

    focal_loss.__name__ = f"focal_a{alpha}_g{gamma}"
    return focal_loss


# ─────────────────────────────────────────────
# DANE
# ─────────────────────────────────────────────
def build_generators(data_dir, img_size, batch_size):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15],
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
# EWALUACJA Z OPTYMALNYM PROGIEM
# ─────────────────────────────────────────────
def evaluate_with_threshold(model, val_gen, test_gen, label=""):
    val_gen.reset()
    val_prob = model.predict(val_gen, verbose=0).ravel()
    val_true = val_gen.classes

    fpr, tpr, thresholds = roc_curve(val_true, val_prob)
    idx_y = np.argmax(tpr - fpr)
    thr_y = float(thresholds[idx_y])
    idx_s = np.where(tpr >= 0.92)[0]
    thr_s = float(thresholds[idx_s[0]]) if len(idx_s) else thr_y

    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_true = test_gen.classes

    print(f"\n{'='*55}")
    print(f"  Ewaluacja: {label}")
    print(f"{'='*55}")

    results = {}
    for name, thr in [("Youden", thr_y), ("Sens>=0.92", thr_s)]:
        y_pred  = (y_prob > thr).astype(int)
        report  = classification_report(
            y_true, y_pred, target_names=LABELS, output_dict=True
        )
        print(f"\n-- Prog {name}: {thr:.4f} --")
        print(classification_report(y_true, y_pred, target_names=LABELS))
        results[name] = {"threshold": thr, "report": report, "y_pred": y_pred}

    return y_prob, y_true, results


# ─────────────────────────────────────────────
# CALLBACKI
# ─────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(
            monitor="val_auc", patience=6,
            restore_best_weights=True, mode="max"
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(RESULTS_DIR, "best_capsnet_focal.weights.h5"),
            monitor="val_auc", save_best_only=True, mode="max",
            save_weights_only=True,   # zapisujemy tylko wagi – bez problemu z Lambda
            verbose=1
        ),
    ]


# ─────────────────────────────────────────────
# GŁÓWNY SKRYPT
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CapsNet Fine-tuning z Focal Loss")
    print(f"  alpha={ALPHA}  gamma={GAMMA}  LR={LR_FINETUNE}")
    print("=" * 55)

    print("\n[1/4] Ladowanie danych...")
    train_gen, val_gen, test_gen = build_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"  Trening: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

    # ── Odbuduj architekture i wczytaj wagi ──
    print(f"\n[2/4] Odbudowywanie modelu i wczytywanie wag z: {MODEL_PATH}")
    model = build_capsnet(IMG_SIZE, num_routing=3)

    # Wywolaj raz zeby zbudowac wagi (potrzebne przed load_weights)
    dummy = tf.zeros((1, *IMG_SIZE, 3))
    model(dummy, training=False)

    # load_weights by_name=False – kolejnosc warstw musi sie zgadzac
    model.load_weights(MODEL_PATH)
    print("  Wagi wczytane pomyslnie.")
    print(f"  Parametry: {model.count_params():,}")

    focal_loss = make_focal_loss(alpha=ALPHA, gamma=GAMMA)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LR_FINETUNE,
            clipnorm=1.0
        ),
        loss=focal_loss,
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    print("\n[3/4] Baseline przed fine-tuningiem...")
    _, _, baseline_results = evaluate_with_threshold(
        model, val_gen, test_gen, label="Baseline (przed FT)"
    )

    print(f"\n[4/4] Fine-tuning z focal loss ({EPOCHS_FT} epok maks.)...")
    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FT,
        callbacks=get_callbacks(),
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n  Czas: {elapsed/60:.1f} min | Epoki: {len(history.history['loss'])}")

    # Wczytaj najlepsze wagi z checkpointu
    best_weights_path = os.path.join(RESULTS_DIR, "best_capsnet_focal.weights.h5")
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
        print("  Wczytano najlepsze wagi z checkpointu.")

    print("\nEwaluacja po fine-tuningu...")
    y_prob, y_true, ft_results = evaluate_with_threshold(
        model, val_gen, test_gen, label="Po fine-tuningu (Focal Loss)"
    )

    out = {
        "y_true":           y_true,
        "y_prob":           y_prob,
        "history":          history.history,
        "baseline_results": baseline_results,
        "ft_results":       ft_results,
        "alpha":            ALPHA,
        "gamma":            GAMMA,
        "time":             elapsed,
        "epochs":           len(history.history["loss"]),
    }
    out_path = os.path.join(RESULTS_DIR, "capsnet_focal_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    # Podsumowanie
    print("\n" + "=" * 65)
    print("  PODSUMOWANIE POROWNAWCZE")
    print("=" * 65)

    b_y = baseline_results.get("Youden", {}).get("report", {})
    f_y = ft_results.get("Youden", {}).get("report", {})
    f_s = ft_results.get("Sens>=0.92", {}).get("report", {})

    def fmt(r, key, subkey=None):
        try:
            return f"{r[key][subkey]:.4f}" if subkey else f"{r[key]:.4f}"
        except Exception:
            return "  N/A "

    print(f"\n{'Metryka':<30} {'Baseline':>10} {'FT Youden':>10} {'FT Sens>=0.92':>14}")
    print("-" * 66)
    rows = [
        ("Sensitivity (PNEUMONIA)", "PNEUMONIA", "recall"),
        ("Specificity (NORMAL)",    "NORMAL",    "recall"),
        ("F1 PNEUMONIA",            "PNEUMONIA", "f1-score"),
        ("Accuracy",                "accuracy",  None),
    ]
    for metric, key, subkey in rows:
        print(f"  {metric:<28} "
              f"{fmt(b_y, key, subkey):>10} "
              f"{fmt(f_y, key, subkey):>10} "
              f"{fmt(f_s, key, subkey):>14}")

    print(f"\n  Wyniki zapisane: {out_path}")


if __name__ == "__main__":
    main()