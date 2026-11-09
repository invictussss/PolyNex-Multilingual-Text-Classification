
import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PolyNex",
    layout="centered"
)

MODEL_PATH = Path("models/multilingual_text_classifier.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Initialize session state
if "headline" not in st.session_state:
    st.session_state.headline = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"


# Custom CSS
st.markdown("""
<style>

.block-container {
    max-width: 950px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #A8A8B3;
    margin-bottom: 35px;
}

.section-heading {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    font-weight: 600;
    transition: 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    border-color: #8B9CFF;
    color: #AAB7FF;
}

.result-card {
    background: rgba(70, 110, 220, 0.13);
    border: 1px solid rgba(110, 140, 255, 0.55);
    border-radius: 18px;
    padding: 35px 20px;
    text-align: center;
    margin-top: 20px;
}

.result-category {
    font-size: 36px;
    font-weight: 800;
    margin: 12px 0;
}

.result-label {
    color: #AAB7FF;
    font-size: 14px;
    letter-spacing: 2px;
}

.footer-text {
    text-align: center;
    color: #999;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">PolyNex</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multilingual Text Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# Language selection
st.markdown(
    '<div class="section-heading">'
    '01 &nbsp; Language Selection'
    '</div>',
    unsafe_allow_html=True
)

language = st.selectbox(
    "Choose your preferred language:",
    ["English", "Hindi", "Bengali", "Tamil"]
)


# Sample headlines
samples = {
    "English": "India won the cricket match against Australia.",
    "Hindi": "भारतीय टीम ने क्रिकेट मैच में शानदार जीत हासिल की।",
    "Bengali": "ভারত ক্রিকেট ম্যাচে অসাধারণ জয় অর্জন করেছে।",
    "Tamil": "இந்திய அணி கிரிக்கெட் போட்டியில் வெற்றி பெற்றது."
}


# Sample and Reset buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Load Sample Text"):
        st.session_state.headline = samples[language]
        st.session_state.prediction = None
        st.rerun()

with col2:
    if st.button("Reset / Clear"):
        st.session_state.headline = ""
        st.session_state.prediction = None
        st.rerun()


st.divider()


# Text input
st.markdown(
    '<div class="section-heading">'
    '02 &nbsp; Enter Your Text'
    '</div>',
    unsafe_allow_html=True
)

with st.form("classification_form"):

    headline = st.text_input(
        "News Headline",
        key="headline",
        placeholder="Type your headline and press Enter..."
    )

    classify = st.form_submit_button(
        "Classify Text",
        use_container_width=True
    )


# Classification
if classify:

    if not headline.strip():

        st.warning("Please enter some text first.")
        st.session_state.prediction = None

    else:

        model_input = language + " " + headline.strip()

        prediction = model.predict([model_input])[0]

        st.session_state.prediction = prediction
        st.session_state.selected_language = language


# Display result
if st.session_state.prediction:

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '03 &nbsp; Classification Result'
        '</div>',
        unsafe_allow_html=True
    )

    prediction = st.session_state.prediction

    category_icons = {
        "Sports": "🏆",
        "Technology": "💻",
        "Entertainment": "🎬"
    }

    icon = category_icons.get(prediction, "")

    result_html = f"""
    <div class="result-card">
    <div class="result-label">PREDICTED CATEGORY</div>
    <div class="result-category">{icon} {prediction}</div>
    <div style="color:#B0B0B0; margin-top:10px;">
        Language: {st.session_state.selected_language}
    </div>
    <div style="color:#8ED6A8; margin-top:15px;">
        ✓ Classification Successful
    </div>
    </div>
    """

    st.markdown(result_html, unsafe_allow_html=True)


# Footer
st.markdown(
    '<div class="footer-text">'
    'PolyNex | Multilingual Text Classification'
    '</div>',
    unsafe_allow_html=True
)

