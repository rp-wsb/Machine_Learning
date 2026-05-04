"""
Model CapsNet – klasyfikacja zapalenia płuc
Dataset: chest_xray (Kaggle)
Wyniki zapisywane do: ./wyniki/capsnet_*
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
DATA_DIR    = r"C:\Users\troch\PycharmProjects\Pneumonia\chest_xray"
IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 10
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
# WARSTWY KAPSUŁOWE
# ─────────────────────────────────────────────
class Squash(layers.Layer):
    """Funkcja aktywacji squash dla kapsułek."""
    def call(self, inputs):
        norm_sq = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        norm    = tf.sqrt(norm_sq + keras.backend.epsilon())
        return (norm_sq / (1 + norm_sq)) * (inputs / norm)


class PrimaryCaps(layers.Layer):
    """Warstwa kapsułek pierwotnych."""
    def __init__(self, num_capsules, capsule_dim, kernel_size=9, strides=2, **kw):
        super().__init__(**kw)
        self.num_capsules = num_capsules
        self.capsule_dim  = capsule_dim
        self.conv = layers.Conv2D(
            num_capsules * capsule_dim,
            kernel_size, strides=strides,
            padding="valid", activation="relu"
        )
        self.squash = Squash()

    def call(self, inputs):
        x     = self.conv(inputs)
        shape = tf.shape(x)
        h, w  = shape[1], shape[2]
        x = tf.reshape(x, [-1, h * w * self.num_capsules, self.capsule_dim])
        return self.squash(x)


class DigitCaps(layers.Layer):
    """Warstwa kapsułek cyfrowych z routingiem dynamicznym."""
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
            initializer="glorot_uniform", trainable=True
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


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_capsnet(img_size, num_routing=3):
    inp = keras.Input(shape=(*img_size, 3))

    x = layers.Conv2D(64, 9, activation="relu", padding="valid")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = PrimaryCaps(num_capsules=8, capsule_dim=8, kernel_size=9, strides=2)(x)
    digit = DigitCaps(num_capsules=2, capsule_dim=16, num_routing=num_routing)(x)

    norm = layers.Lambda(lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1)))(digit)
    out  = layers.Lambda(lambda z: z[:, 1:2])(norm)
    out  = layers.Activation("sigmoid")(out)

    model = Model(inp, out, name="CapsNet")
    model.compile(
        optimizer=keras.optimizers.Adam(5e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


# ─────────────────────────────────────────────
# TRENING
# ─────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            filepath=os.path.join(RESULTS_DIR, "best_capsnet.keras"),
            monitor="val_auc", save_best_only=True, mode="max"
        ),
    ]


def main():
    print("=" * 60)
    print("  CapsNet – Klasyfikacja zapalenia płuc")
    print("=" * 60)

    print("\n[1/3] Ładowanie danych...")
    train_gen, val_gen, test_gen = build_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"  Trening:   {train_gen.samples} obrazów")
    print(f"  Walidacja: {val_gen.samples} obrazów")
    print(f"  Test:      {test_gen.samples} obrazów")

    print("\n[2/3] Trening CapsNet...")
    model = build_capsnet(IMG_SIZE)
    model.summary()

    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
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
    out_path = os.path.join(RESULTS_DIR, "capsnet_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ Wyniki CapsNet zapisane: {out_path}")
    print(f"  Czas treningu: {elapsed/60:.1f} min  |  Epoki: {results['epochs']}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Sensitivity (PNEUMONIA recall): {report['PNEUMONIA']['recall']:.4f}")
    print(f"  Specificity (NORMAL recall):    {report['NORMAL']['recall']:.4f}")


if __name__ == "__main__":
    main()