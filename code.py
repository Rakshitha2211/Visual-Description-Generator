import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Captioning parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate captions
def generate_caption(image, model, feature_extractor, tokenizer):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    # Preprocess the image
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate captions
    model.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

# Streamlit App
def main():
    st.title("Visual Description Generator")
    st.write("Upload an image to generate a caption.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Generating caption...")
        model, feature_extractor, tokenizer = load_model()
        caption = generate_caption(image, model, feature_extractor, tokenizer)

        # Display caption with custom color
        caption_html = f"<p style='color:red; font-size:18px; font-weight:bold;'>{caption}</p>"
        st.markdown(caption_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
