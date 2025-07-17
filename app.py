# app.py
"""
Streamlit Image Caption Generator
Author: chinna
Description: Upload an image and get a natural‑language caption using the Salesforce BLIP model.
Run: streamlit run app.py
"""

import io
from typing import Tuple

import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


@st.cache_resource(show_spinner="Loading BLIP model…")
def load_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration, str]:
    """Lazy‑load BLIP processor & model, move to GPU if available."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def generate_caption(
    image: Image.Image, processor: BlipProcessor, model: BlipForConditionalGeneration, device: str
) -> str:
    """Generate a caption for the given PIL image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=30, num_beams=5)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.capitalize()


def main() -> None:
    st.set_page_config(page_title="Image Caption Generator", page_icon="📷")
    st.title("📷 Image Caption Generator")
    st.markdown(
        "Upload an image and the model will describe it in natural language.\n\n"
        "Model: **Salesforce/blip-image-captioning-base** (Hugging Face Transformers)"
    )

    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Generate Caption"):
            processor, model, device = load_model()
            caption = generate_caption(image, processor, model, device)
            st.success(caption)

    st.sidebar.header("About")
    st.sidebar.markdown(
        "Built with **Streamlit**, **PyTorch**, and **Hugging Face Transformers**."\
        "\nAuthor: chinna"
    )


if __name__ == "__main__":
    main()
