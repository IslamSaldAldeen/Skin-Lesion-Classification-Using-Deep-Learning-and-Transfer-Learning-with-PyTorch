from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.config import CLASS_FULL_NAMES, CLASS_NAMES, MODELS_DIR
from src.gradcam import make_gradcam_overlay
from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import get_device, load_checkpoint


st.set_page_config(
    page_title="HAM10000 Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide"
)

st.title("HAM10000 Skin Lesion Classification Demo")

st.warning(
    "Academic decision-support prototype only. "
    "This app is not clinically validated and must not replace a dermatologist or professional diagnosis."
)


# Binary medical category for each HAM10000 class
# Benign = non-cancerous lesion category
# Malignant = cancerous or potentially cancer-related lesion category
LESION_CATEGORY = {
    "akiec": "Malignant",
    "bcc": "Malignant",
    "bkl": "Benign",
    "df": "Benign",
    "mel": "Malignant",
    "nv": "Benign",
    "vasc": "Benign",
}


# Short educational description for each predicted class
LESION_DESCRIPTION = {
    "akiec": (
        "Actinic keratoses / intraepithelial carcinoma are sun-related lesions. "
        "They may be considered precancerous or early cancer-related lesions and should be checked by a specialist."
    ),
    "bcc": (
        "Basal cell carcinoma is a common type of skin cancer. "
        "It usually grows slowly and is often linked to long-term sun exposure."
    ),
    "bkl": (
        "Benign keratosis-like lesions are non-cancerous skin growths. "
        "They can sometimes look similar to other pigmented skin lesions."
    ),
    "df": (
        "Dermatofibroma is a common benign skin lesion. "
        "It often appears as a small firm bump on the skin."
    ),
    "mel": (
        "Melanoma is a serious type of skin cancer that starts from melanocytes, "
        "the pigment-producing cells of the skin."
    ),
    "nv": (
        "Melanocytic nevi are common moles. "
        "They are usually benign and are made of pigment-producing cells called melanocytes."
    ),
    "vasc": (
        "Vascular lesions are usually benign lesions related to blood vessels. "
        "They may appear red, purple, or blue."
    ),
}


MODEL_OPTIONS = sorted(MODELS_DIR.glob("best_*.pth"))

model_path = st.sidebar.selectbox(
    "Choose trained model",
    options=MODEL_OPTIONS if MODEL_OPTIONS else [Path("models/best_exp2_resnet50_weighted_head.pth")],
    format_func=lambda p: str(p),
)

architecture = st.sidebar.selectbox(
    "Architecture",
    ["resnet50", "efficientnet_b0"]
)

show_gradcam = st.sidebar.checkbox(
    "Show Grad-CAM heatmap",
    value=True
)


@st.cache_resource
def load_model(model_path_str: str, architecture: str):
    device = get_device()

    model = build_model(
        architecture=architecture,
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze="none"
    ).to(device)

    checkpoint_path = Path(model_path_str)

    if not checkpoint_path.exists():
        return (
            None,
            device,
            f"Model file not found: {checkpoint_path}. "
            "Train first using python -m src.train --weighted-loss"
        )

    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    return model, device, None


model, device, error = load_model(str(model_path), architecture)

if error:
    st.error(error)


uploaded = st.file_uploader(
    "Upload a dermoscopic skin lesion image",
    type=["jpg", "jpeg", "png"]
)


if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    if model is not None:
        transform = get_eval_transforms()
        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu()

        top_probs, top_indices = torch.topk(probs, k=3)

        top3 = pd.DataFrame({
            "Class": [
                CLASS_NAMES[i] for i in top_indices.tolist()
            ],
            "Medical name": [
                CLASS_FULL_NAMES[CLASS_NAMES[i]] for i in top_indices.tolist()
            ],
            "Category": [
                LESION_CATEGORY.get(CLASS_NAMES[i], "Unknown") for i in top_indices.tolist()
            ],
            "Confidence": [
                f"{p.item():.2%}" for p in top_probs
            ],
        })

        pred_idx = int(top_indices[0])
        pred_class = CLASS_NAMES[pred_idx]
        pred_category = LESION_CATEGORY.get(pred_class, "Unknown")
        pred_description = LESION_DESCRIPTION.get(
            pred_class,
            "No description available for this class."
        )

        with col2:
            st.subheader("Prediction Details")

            st.metric(
                "Predicted class",
                f"{pred_class} — {pred_category}"
            )

            st.write(f"**Medical name:** {CLASS_FULL_NAMES[pred_class]}")
            st.write(f"**Category:** {pred_category}")
            st.write(f"**Confidence:** {top_probs[0].item():.2%}")

            st.info(f"**About this class:** {pred_description}")

            st.subheader("Top-3 predictions")
            st.dataframe(top3, use_container_width=True)

        if show_gradcam:
            st.subheader("Grad-CAM explanation")

            try:
                overlay, cam_class, cam_conf = make_gradcam_overlay(
                    model,
                    image,
                    device
                )

                cam_category = LESION_CATEGORY.get(cam_class, "Unknown")

                st.image(
                    overlay,
                    caption=f"Regions influencing prediction: {cam_class} — {cam_category} ({cam_conf:.2%})",
                    use_container_width=False
                )

            except Exception as exc:
                st.info(
                    f"Grad-CAM could not be generated for this model/image: {exc}"
                )

else:
    st.info(
        "Upload an image to see the predicted class, confidence score, "
        "top-3 predictions, class description, and optional Grad-CAM."
    )