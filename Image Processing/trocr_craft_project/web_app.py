import streamlit as st
from PIL import Image
import tempfile
from app import hybrid_ocr as run_ocr


st.set_page_config(page_title="Text Extraction from an Image", layout="wide")


st.markdown("""
<style>
    .block-container { max-width: 1200px; padding-top: 2rem; }
    .card {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 6px #eee;
    }
    .stButton>button {
        background-color: #FF385C;
        color: white;
        font-weight: bold;
        border-radius: 999px;
        padding: 10px 24px;
    }
    .arrow {
        font-size: 48px;
        color: #FF385C;
        text-align: center;
        margin-top: 140px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Text Extraction from an Image</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload → Analyze → Download</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([4,1,4])


with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload your Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350)
    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown("<div class='arrow'>➡️</div>", unsafe_allow_html=True)


with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Extracted Text")

    if uploaded:
        if st.button("Extract Text"):
            with st.spinner("Processing the image..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp.name)
                    extracted_text = run_ocr(tmp.name)

            st.success("✅ Text extraction completed!")
            st.text_area("Result", extracted_text, height=280)

            st.download_button("⬇️ Download result.txt", extracted_text, "result.txt", "text/plain")
    else:
        st.info("Please upload an image to proceed.")

    st.markdown("</div>", unsafe_allow_html=True)
