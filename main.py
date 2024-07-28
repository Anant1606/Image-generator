import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Ensure CUDA is available
print(torch.cuda.is_available())
print(torch.version.cuda)

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Load the model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
    revision="fp16" if CFG.device == "cuda" else None,
    use_auth_token='your_hugging_face_auth_token',
    guidance_scale=CFG.image_gen_guidance_scale
)

# Move the model to the appropriate device
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image

# Streamlit app
st.title("Image Generation with Stable Diffusion")
st.write("Enter a prompt to generate an image:")

prompt = st.text_input("Prompt")

if st.button("Generate Image"):
    if prompt:
        st.write("Generating image...")
        image = generate_image(prompt, image_gen_model)
        
        # Display the image
        st.image(image, caption='Generated Image', use_column_width=True)
    else:
        st.write("Please enter a prompt.")

