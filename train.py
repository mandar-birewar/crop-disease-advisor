"""Train a MobileNetV2 transfer-learning classifier on the PlantVillage dataset.

Example:
    python train.py --data path/to/plantvillage/color
    python train.py --data path/to/plantvillage/color --max-per-class 300   # fast, laptop-friendly

Outputs:
    model/crop_disease_model.keras   trained model
    model/class_names.json           class index -> name
    results/training_curves.png      accuracy / loss curves
    results/confusion_matrix.png     normalised confusion matrix on the test set
    results/classification_report.txt  precision / recall / F1 per class
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

keras = tf.keras
layers = keras.layers

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_args():
    p = argparse.ArgumentParser(description="Train the crop disease classifier")
    p.add_argument("--data", required=True, help="folder with one sub-folder per class")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5, help="epochs with the base model frozen")
    p.add_argument("--finetune-epochs", type=int, default=3, help="epochs after unfreezing (0 = skip)")
    p.add_argument("--unfreeze", type=int, default=30, help="number of top base layers to fine-tune")
    p.add_argument("--max-per-class", type=int, default=0, help="cap images per class (0 = use all)")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (smoke tests only)")
    p.add_argument("--out-dir", default="model")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def list_images(data_dir, max_per_class=0):
    rng = np.random.RandomState(SEED)
    class_names, paths, labels = [], [], []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
        if not files:  # skip empty or wrapper folders (e.g. a nested PlantVillage/ copy)
            continue
        if max_per_class and len(files) > max_per_class:
            files = list(rng.choice(files, max_per_class, replace=False))
        idx = len(class_names)
        class_names.append(name)
        paths += [os.path.join(folder, f) for f in files]
        labels += [idx] * len(files)
    if not class_names:
        raise SystemExit(f"No class folders with images found in {data_dir}")
    return np.array(paths), np.array(labels, dtype="int32"), class_names


def split_data(paths, labels):
    """Stratified 70 / 15 / 15 split into train, validation and test."""
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    p_val, p_te, y_val, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    return (p_tr, y_tr), (p_val, y_val), (p_te, y_te)


AUGMENT = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentation",
)


def make_ds(paths, labels, img_size, batch, training):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths.astype(str), labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    if training:
        ds = ds.map(lambda x, y: (AUGMENT(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------- model
def build_model(num_classes, img_size, pretrained=True):
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(inputs)  # 0..255 -> -1..1
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="crop_disease_net"), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]


# ------------------------------------------------------------------------- reports
def plot_curves(histories, out_path):
    def joined(key):
        return sum((h.history[key] for h in histories), [])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(joined(key), label="train")
        ax.plot(joined("val_" + key), label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(model, test_ds, class_names, results_dir):
    y_true, y_pred = [], []
    for x, y in test_ds:  # labels and predictions are collected in the same pass
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))

    idx = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=idx, target_names=class_names, digits=4, zero_division=0
    )
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n{report}\n\nClass index map:\n")
        f.writelines(f"{i:2d}  {n}\n" for i, n in enumerate(class_names))

    cm = confusion_matrix(y_true, y_pred, labels=idx).astype(float)
    cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(idx)
    ax.set_yticks(idx)
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title("Normalised confusion matrix (test set)")
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    return acc


# -------------------------------------------------------------------------- main
def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    paths, labels, class_names = list_images(args.data, args.max_per_class)
    print(f"{len(paths)} images, {len(class_names)} classes")
    (p_tr, y_tr), (p_val, y_val), (p_te, y_te) = split_data(paths, labels)
    print(f"train {len(p_tr)} | val {len(p_val)} | test {len(p_te)}")

    train_ds = make_ds(p_tr, y_tr, args.img_size, args.batch, training=True)
    val_ds = make_ds(p_val, y_val, args.img_size, args.batch, training=False)
    test_ds = make_ds(p_te, y_te, args.img_size, args.batch, training=False)

    model, base = build_model(len(class_names), args.img_size, pretrained=not args.no_pretrained)

    # Stage 1: train only the new classifier head
    compile_model(model, 1e-3)
    histories = [
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(), shuffle=False)
    ]

    # Stage 2: fine-tune the top layers of MobileNetV2 with a small learning rate
    if args.finetune_epochs > 0:
        base.trainable = True
        for layer in base.layers[: len(base.layers) - args.unfreeze]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics fixed
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, 1e-5)
        histories.append(
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.finetune_epochs,
                callbacks=callbacks(),
                shuffle=False,
            )
        )

    plot_curves(histories, os.path.join(args.results_dir, "training_curves.png"))
    acc = evaluate(model, test_ds, class_names, args.results_dir)
    print(f"Test accuracy: {acc:.4f}")

    model.save(os.path.join(args.out_dir, "crop_disease_model.keras"))
    with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved model and class names to {args.out_dir}/, reports to {args.results_dir}/")


if __name__ == "__main__":
    main()
