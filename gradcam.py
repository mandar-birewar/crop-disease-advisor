"""Grad-CAM: highlights the leaf regions that drove the model's prediction."""
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow import keras


def _split_model(model):
    """Return (layers before the backbone, backbone, layers after the backbone)."""
    all_layers = model.layers
    base_idx = next(i for i, l in enumerate(all_layers) if isinstance(l, keras.Model))
    pre = [l for l in all_layers[:base_idx] if not isinstance(l, keras.layers.InputLayer)]
    return pre, all_layers[base_idx], all_layers[base_idx + 1:]


def make_gradcam(model, img_batch, class_index=None):
    """Compute a Grad-CAM map for one image.

    img_batch: float array of shape (1, H, W, 3) with pixel values 0..255.
    Returns (cam, probs): cam is an (h, w) array in 0..1, probs the class probabilities.
    """
    pre, base, post = _split_model(model)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    for layer in pre:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv_out = base(x, training=False)  # last conv feature map of MobileNetV2
        tape.watch(conv_out)
        y = conv_out
        for layer in post:
            y = layer(y)
        if class_index is None:
            class_index = int(tf.argmax(y[0]))
        score = y[:, class_index]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # importance of each feature channel
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), y[0].numpy()


def overlay_heatmap(pil_img, cam, alpha=0.45):
    """Blend a coloured Grad-CAM heatmap on top of the original image."""
    img = pil_img.convert("RGB")
    heat = Image.fromarray(np.uint8(255 * cam)).resize(img.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * (np.asarray(img) / 255.0) + alpha * colored
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
