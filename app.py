import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image

# Model Configuration
model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Streamlit UI
st.image("PragyanAI_Transperent.png")
st.title("Image Generation with Stable Diffusion")
st.write("Enter a text prompt and generate an image.")

prompt = st.text_area("Enter your prompt here:")

if st.button("Generate Image") and prompt:
    with st.spinner("Generating image..."):
        image = generate_image(prompt)
        
    st.image(image, caption="Generated Image", use_column_width=True)
