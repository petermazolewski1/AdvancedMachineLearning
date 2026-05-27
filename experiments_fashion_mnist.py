import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
UPPER_BODY_CLASSES = [0, 2, 3, 4, 6]
NON_UPPER_CLASSES = [1, 5, 7, 8, 9]


def prep_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return x_train, y_train, x_test, y_test


def make_backbone(include_extra_conv=False, augmentation=None, out_dim=10):
    m = models.Sequential()
    m.add(layers.Input(shape=(28, 28, 1)))
    if augmentation is not None:
        m.add(augmentation)
    m.add(layers.Conv2D(32, (3, 3), activation="relu"))
    m.add(layers.MaxPooling2D((2, 2)))
    if include_extra_conv:
        m.add(layers.Conv2D(64, (3, 3), activation="relu"))
        m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dense(out_dim, activation="softmax"))
    return m


def train_sparse(model, x_train, y_train, epochs=6, batch_size=128, val_split=0.1, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[es],
        verbose=0,
    )
    return model


def train_onehot(model, x_train, y_train_onehot, loss_obj, epochs=6, batch_size=128, val_split=0.1, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_obj,
        metrics=["accuracy"],
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model.fit(
        x_train,
        y_train_onehot,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[es],
        verbose=0,
    )
    return model


def evaluate_model(model, x_test, y_test, name):
    probs = model.predict(x_test, verbose=0)
    pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    rep = classification_report(y_test, pred, output_dict=True, target_names=CLASS_NAMES)
    shirt_f1 = rep["Shirt"]["f1-score"]
    shirt_recall = rep["Shirt"]["recall"]
    upper_confusion = sum(
        rep[cname]["support"] - (rep[cname]["recall"] * rep[cname]["support"])
        for cname in ["T-shirt/top", "Pullover", "Dress", "Coat", "Shirt"]
    )
    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "shirt_f1": shirt_f1,
        "shirt_recall": shirt_recall,
        "upper_body_errors_est": upper_confusion,
        "y_pred": pred,
    }


def targeted_augmented_dataset(x_train, y_train):
    # Add extra augmented samples only for upper-body classes.
    upper_idx = np.where(np.isin(y_train, UPPER_BODY_CLASSES))[0]
    rng = np.random.default_rng(SEED)
    sample_n = min(8000, len(upper_idx))
    chosen = rng.choice(upper_idx, size=sample_n, replace=False)
    x_sel = x_train[chosen]
    y_sel = y_train[chosen]

    x_aug = tf.convert_to_tensor(x_sel)
    x_aug = tf.image.random_flip_left_right(x_aug, seed=SEED)
    x_aug = tf.image.random_brightness(x_aug, max_delta=0.08, seed=SEED)
    x_aug = tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEED)(x_aug, training=True)
    x_aug = tf.keras.layers.RandomZoom(0.1, 0.1, seed=SEED)(x_aug, training=True)
    x_aug = tf.clip_by_value(x_aug, 0.0, 1.0).numpy()

    x_comb = np.concatenate([x_train, x_aug], axis=0)
    y_comb = np.concatenate([y_train, y_sel], axis=0)
    return x_comb, y_comb


def two_stage_predict(stage_a, stage_b, fallback_10cls, x_test):
    # Stage A: upper-body vs other
    upper_probs = stage_a.predict(x_test, verbose=0)
    is_upper = np.argmax(upper_probs, axis=1) == 1

    # Stage B: fine-grained upper-body class among [0,2,3,4,6]
    upper_idx = np.where(is_upper)[0]
    preds = np.argmax(fallback_10cls.predict(x_test, verbose=0), axis=1)

    if len(upper_idx) > 0:
        upper_probs_5 = stage_b.predict(x_test[upper_idx], verbose=0)
        upper_pred_5 = np.argmax(upper_probs_5, axis=1)
        mapped = np.array([UPPER_BODY_CLASSES[i] for i in upper_pred_5], dtype=np.int32)
        preds[upper_idx] = mapped
    return preds


def main():
    x_train, y_train, x_test, y_test = prep_data()
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=10)

    results = []

    # 0) Baseline
    baseline = make_backbone(include_extra_conv=False, augmentation=None, out_dim=10)
    baseline = train_sparse(baseline, x_train, y_train)
    r0 = evaluate_model(baseline, x_test, y_test, "Baseline CNN")
    results.append(r0)
    print("Finished: Baseline CNN")

    # 1) Focal loss
    focal_model = make_backbone(include_extra_conv=False, augmentation=None, out_dim=10)
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)
    focal_model = train_onehot(focal_model, x_train, y_train_onehot, focal_loss)
    r1 = evaluate_model(focal_model, x_test, y_test, "Focal Loss CNN")
    results.append(r1)
    print("Finished: Focal Loss CNN")

    # 2) Targeted augmentation for upper-body classes
    x_targ, y_targ = targeted_augmented_dataset(x_train, y_train)
    targ_model = make_backbone(include_extra_conv=False, augmentation=None, out_dim=10)
    targ_model = train_sparse(targ_model, x_targ, y_targ)
    r2 = evaluate_model(targ_model, x_test, y_test, "Targeted Augmentation CNN")
    results.append(r2)
    print("Finished: Targeted Augmentation CNN")

    # 3) Two-stage model
    # Stage A binary labels: 1 if upper-body else 0
    y_a = np.isin(y_train, UPPER_BODY_CLASSES).astype(np.int32)
    stage_a = make_backbone(include_extra_conv=False, augmentation=None, out_dim=2)
    stage_a = train_sparse(stage_a, x_train, y_a)

    # Stage B upper-body fine-grained labels in [0..4]
    upper_train_idx = np.where(np.isin(y_train, UPPER_BODY_CLASSES))[0]
    x_b = x_train[upper_train_idx]
    y_b_orig = y_train[upper_train_idx]
    cls_to_5 = {c: i for i, c in enumerate(UPPER_BODY_CLASSES)}
    y_b = np.array([cls_to_5[c] for c in y_b_orig], dtype=np.int32)
    stage_b = make_backbone(include_extra_conv=False, augmentation=None, out_dim=5)
    stage_b = train_sparse(stage_b, x_b, y_b)

    pred_two_stage = two_stage_predict(stage_a, stage_b, baseline, x_test)
    acc_ts = accuracy_score(y_test, pred_two_stage)
    macro_f1_ts = f1_score(y_test, pred_two_stage, average="macro")
    rep_ts = classification_report(y_test, pred_two_stage, output_dict=True, target_names=CLASS_NAMES)
    r3 = {
        "model": "Two-Stage (Upper-body + Fallback)",
        "accuracy": acc_ts,
        "macro_f1": macro_f1_ts,
        "shirt_f1": rep_ts["Shirt"]["f1-score"],
        "shirt_recall": rep_ts["Shirt"]["recall"],
        "upper_body_errors_est": sum(
            rep_ts[cname]["support"] - (rep_ts[cname]["recall"] * rep_ts[cname]["support"])
            for cname in ["T-shirt/top", "Pullover", "Dress", "Coat", "Shirt"]
        ),
        "y_pred": pred_two_stage,
    }
    results.append(r3)
    print("Finished: Two-Stage")

    # 4) Label smoothing
    smooth_model = make_backbone(include_extra_conv=False, augmentation=None, out_dim=10)
    smooth_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    smooth_model = train_onehot(smooth_model, x_train, y_train_onehot, smooth_loss)
    r4 = evaluate_model(smooth_model, x_test, y_test, "Label Smoothing CNN")
    results.append(r4)
    print("Finished: Label Smoothing CNN")

    # 5) Hard-example mining (focus on upper-body mistakes)
    hem_model = make_backbone(include_extra_conv=False, augmentation=None, out_dim=10)
    hem_model = train_sparse(hem_model, x_train, y_train, epochs=5)
    train_pred = np.argmax(hem_model.predict(x_train, verbose=0), axis=1)
    hard_idx = np.where(
        (train_pred != y_train) & np.isin(y_train, UPPER_BODY_CLASSES)
    )[0]
    if len(hard_idx) > 0:
        # Duplicate hard examples to upweight them.
        x_h = np.concatenate([x_train, x_train[hard_idx], x_train[hard_idx]], axis=0)
        y_h = np.concatenate([y_train, y_train[hard_idx], y_train[hard_idx]], axis=0)
        hem_model = train_sparse(hem_model, x_h, y_h, epochs=3, lr=3e-4)
    r5 = evaluate_model(hem_model, x_test, y_test, "Hard-Example Mining CNN")
    results.append(r5)
    print("Finished: Hard-Example Mining CNN")

    # Print comparison
    keys = ["model", "accuracy", "macro_f1", "shirt_f1", "shirt_recall", "upper_body_errors_est"]
    print("\n=== Comparison (higher is better except upper_body_errors_est) ===")
    for row in results:
        print(
            f"{row['model']:<38} "
            f"acc={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} "
            f"shirt_f1={row['shirt_f1']:.4f} "
            f"shirt_recall={row['shirt_recall']:.4f} "
            f"upper_err={row['upper_body_errors_est']:.0f}"
        )


if __name__ == "__main__":
    main()
