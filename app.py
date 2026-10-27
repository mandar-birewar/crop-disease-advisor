"""Gradio web app: upload a leaf photo -> disease, confidence, heatmap and advice."""
import glob
import json
import os

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow import keras

from advisor import get_advice, pretty_label
from gradcam import make_gradcam, overlay_heatmap

MODEL_PATH = os.environ.get("MODEL_PATH", "model/crop_disease_model.keras")
NAMES_PATH = os.environ.get("NAMES_PATH", "model/class_names.json")
LOW_CONFIDENCE = 0.60

model = keras.models.load_model(MODEL_PATH)
with open(NAMES_PATH) as f:
    class_names = json.load(f)
IMG_SIZE = model.input_shape[1]


def render_advice(class_name, confidence):
    info = get_advice(class_name)
    healthy = info["kind"] == "Healthy"
    md = ""
    if confidence < LOW_CONFIDENCE:
        md += (
            f"> **Low confidence ({confidence:.0%}).** Retake the photo: one leaf, "
            "good daylight, plain background.\n\n"
        )
    icon = "✅" if healthy else "🌿"
    md += f"### {icon} {info['crop']} - {info['disease']}\n"
    md += f"**Type:** {info['kind']}  \n**Confidence:** {confidence:.1%}\n\n"
    md += f"**What to look for:** {info['symptoms']}\n\n"
    md += f"**Recommended action:** {info['treatment']}\n\n"
    md += f"**Prevention:** {info['prevention']}\n\n"
    md += (
        "---\n*AI screening aid only. Confirm with your local agriculture officer or "
        "Krishi Vigyan Kendra (KVK) before spraying anything.*"
    )
    return md


def diagnose(pil_image):
    if pil_image is None:
        raise gr.Error("Please upload a leaf photo first.")
    img = pil_image.convert("RGB")
    arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)[None]

    cam, probs = make_gradcam(model, arr)
    top = np.argsort(probs)[::-1][:3]
    scores = {pretty_label(class_names[i]): float(probs[i]) for i in top}

    shown = img.copy()
    shown.thumbnail((512, 512))
    heatmap = overlay_heatmap(shown, cam)
    return scores, heatmap, render_advice(class_names[top[0]], float(probs[top[0]]))


CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#title h1 {margin-bottom: 0;}
"""
THEME = gr.themes.Soft(primary_hue="green")

# Gradio 6 moved theme/css from gr.Blocks() to launch(); support both.
GRADIO_6_PLUS = int(gr.__version__.split(".")[0]) >= 6
BLOCKS_STYLE = {} if GRADIO_6_PLUS else {"theme": THEME, "css": CSS}
LAUNCH_STYLE = {"theme": THEME, "css": CSS} if GRADIO_6_PLUS else {}

with gr.Blocks(title="Crop Disease Advisor", **BLOCKS_STYLE) as demo:
    gr.Markdown(
        "# 🌱 Crop Disease Advisor\n"
        "Upload a photo of a single leaf. The deep-learning model identifies the disease, "
        "shows *where* it looked (Grad-CAM) and suggests what to do next.",
        elem_id="title",
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Leaf photo", sources=["upload"], height=360)
            run_btn = gr.Button("Diagnose", variant="primary")
            example_files = sorted(glob.glob("examples/*.jpg") + glob.glob("examples/*.png"))
            if example_files:
                gr.Examples(examples=example_files, inputs=image_in, label="Try a sample leaf")
        with gr.Column():
            label_out = gr.Label(num_top_classes=3, label="Top predictions")
            heat_out = gr.Image(label="Grad-CAM: regions the model focused on", height=300)
    advice_out = gr.Markdown()

    run_btn.click(diagnose, inputs=image_in, outputs=[label_out, heat_out, advice_out])

if __name__ == "__main__":
    demo.launch(**LAUNCH_STYLE)
