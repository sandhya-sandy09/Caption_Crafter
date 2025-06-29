import streamlit as st
from PIL import Image
from blip import generate_image_caption
from llm import generate_caption_with_memory

st.title("Caption Crafter")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
scene = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    display_image = image.resize((250, 250))
    st.image(display_image, caption="Uploaded Image", use_container_width=False)
    scene = generate_image_caption(image)
else:
    scene = st.text_area("🖊️ Describe your scene manually (if no image uploaded):", placeholder="e.g., A cozy café table with coffee and book")

mood = st.selectbox("Mood:",["Aesthetic", "Cool", "Romantic", "Sassy", "Funny", "Professional", "Melancholic", "Inspiring", "Minimalist"])
style = st.selectbox("Style:",["Poetic", "Witty", "One-liner", "Rhyming", "Classic Instagram", "Short & Punchy", "Deep & Thoughtful"]
)
num_captions = st.slider("How many captions do you want?", min_value=1, max_value=5, value=3)
user_input = st.text_input("💬 Want to add your thoughts for your caption?")


if st.button("✨ Generate Caption"):
    if scene and scene.strip():
        with st.spinner("Crafting your caption..."):
            captions = generate_caption_with_memory(
                mood=mood,
                style=style,
                scene=scene,
                user_thoughts=user_input,
                num=num_captions
            )

            st.markdown("### 🎯 Your Captions")
            for i, caption in enumerate(captions, 1):
                st.success(f"{i}. {caption}")
    else:
        st.warning("Please upload an image or enter a scene description.")







        





