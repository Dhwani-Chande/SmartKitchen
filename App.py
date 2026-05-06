import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# CONFIG & PATHS  (all relative — no hardcoded paths)
# ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR      = os.path.join(BASE_DIR, "upload_images")
DATASET_CSV     = os.path.join(BASE_DIR, "IndianFoodDatasetCSV.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "FV.h5")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LAZY-LOAD MODEL & DATASET  (cached so they load once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    from keras.models import load_model as keras_load
    return keras_load(MODEL_PATH)

@st.cache_data(show_spinner="Loading recipe dataset…")
def load_recipes():
    df = pd.read_csv(DATASET_CSV)
    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()
    return df

# ──────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────
LABELS = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper',
    4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower',
    8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant',
    12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno',
    16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear',
    24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato',
    28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
    32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon',
}

FRUITS = {
    'apple', 'banana', 'bell pepper', 'chilli pepper', 'grapes',
    'jalepeno', 'kiwi', 'lemon', 'mango', 'orange', 'paprika',
    'pear', 'pineapple', 'pomegranate', 'watermelon',
}

VEGETABLES = {
    'beetroot', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'lettuce',
    'onion', 'peas', 'potato', 'raddish', 'soy beans', 'spinach',
    'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
}

# ──────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────
def apply_styles():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L2lzOTc1Ny1pbWFnZS1rd3Z5ZHlvdy5qcGc.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background-color: #001F3F;
        border-radius: 10px;
        opacity: 0.95;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CALORIE FETCH  (with better CSS selector fallback + timeout)
# ──────────────────────────────────────────────
# Known approximate calories per 100g as a fallback if scraping fails
CALORIE_FALLBACK = {
    'apple': 52, 'banana': 89, 'beetroot': 43, 'bell pepper': 31,
    'cabbage': 25, 'capsicum': 20, 'carrot': 41, 'cauliflower': 25,
    'chilli pepper': 40, 'corn': 86, 'cucumber': 16, 'eggplant': 25,
    'garlic': 149, 'ginger': 80, 'grapes': 67, 'jalepeno': 29,
    'kiwi': 61, 'lemon': 29, 'lettuce': 15, 'mango': 60,
    'onion': 40, 'orange': 47, 'paprika': 282, 'pear': 57,
    'peas': 81, 'pineapple': 50, 'pomegranate': 83, 'potato': 77,
    'raddish': 16, 'soy beans': 147, 'spinach': 23, 'sweetcorn': 86,
    'sweetpotato': 86, 'tomato': 18, 'turnip': 28, 'watermelon': 30,
}

def fetch_calories(prediction: str) -> str:
    """Try to scrape calories from Google; fall back to static table."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.google.com/search?q=calories+in+{prediction}+per+100g"
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try multiple known Google result div classes (they change over time)
        for cls in ["BNeawe iBp4i AP7Wnd", "BNeawe", "IZ6rdc"]:
            tag = soup.find("div", class_=cls)
            if tag and "cal" in tag.text.lower():
                return tag.text
    except Exception:
        pass

    # Static fallback
    kcal = CALORIE_FALLBACK.get(prediction.lower())
    if kcal:
        return f"~{kcal} kcal per 100g (estimated)"
    return "Calorie data unavailable"

# ──────────────────────────────────────────────
# IMAGE CLASSIFICATION
# ──────────────────────────────────────────────
def classify_image(img_path: str) -> str:
    """Load image, preprocess, run model, return predicted label."""
    model = load_model()
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0          # normalize
    arr = np.expand_dims(arr, axis=0)                       # add batch dim
    preds = model.predict(arr, verbose=0)
    label = LABELS[preds.argmax(axis=-1)[0]]
    return label.capitalize()

# ──────────────────────────────────────────────
# RECIPE RECOMMENDATION  (improved: ingredient-level matching)
# ──────────────────────────────────────────────
def recommend_recipes(prediction: str, recipes_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Match prediction against recipe names AND ingredients columns,
    deduplicate, return top n.
    """
    pred_lower = prediction.lower()
    mask_name  = recipes_df["TranslatedRecipeName"].str.lower().str.contains(
                    pred_lower, na=False)
    mask_ingr  = recipes_df["TranslatedIngredients"].str.lower().str.contains(
                    pred_lower, na=False)
    matched = recipes_df[mask_name | mask_ingr].drop_duplicates(
                subset="TranslatedRecipeName")
    return matched.head(n)

# ──────────────────────────────────────────────
# DISPLAY HELPERS
# ──────────────────────────────────────────────
DIVIDER = "<hr style='border:1px solid #001F3F;'/>"

def h(tag, text, color="#001F3F"):
    return f"<{tag} style='color:{color};'>{text}</{tag}>"

def display_result(result: str):
    category = "Fruit 🍎" if result.lower() in FRUITS else "Vegetable 🥦"
    st.info(f"**Category:** {category}")
    st.success(f"**Predicted:** {result}")
    cal = fetch_calories(result)
    st.warning(f"**Calories:** {cal}")

def display_recommendations(recipes: pd.DataFrame):
    if recipes.empty:
        st.info("No recipes found for this ingredient. Try browsing the Recipes page!")
        return
    st.markdown(h("h2", "Recommended Recipes 🍳"), unsafe_allow_html=True)
    for _, row in recipes.iterrows():
        st.markdown(f"#### {row.get('TranslatedRecipeName', 'Recipe')}")
        with st.expander("View ingredients & instructions"):
            st.markdown(f"**Ingredients:** {row.get('TranslatedIngredients', 'N/A')}")
            st.markdown(f"**Instructions:** {row.get('TranslatedInstructions', 'N/A')}")
            url = row.get("URL", "")
            if url:
                st.markdown(f"[🔗 Full Recipe]({url})")
        st.markdown(DIVIDER, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────
def page_home():
    st.markdown(h("h1", "🍽️ Smart Kitchen"), unsafe_allow_html=True)
    st.markdown(h("h3", "About"), unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px; color:#001F3F;'>
    Long work hours leave professionals exhausted, making healthy meals a daunting task.
    SmartKitchen helps you identify the ingredients you have on hand and instantly suggests
    nutritious Indian recipes — reducing food waste and decision fatigue.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(h("h3", "How It Works"), unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📷\n**Upload or snap** a photo of your fruit or vegetable.")
    with col2:
        st.markdown("### 🧠\n**AI identifies** the ingredient using MobileNet.")
    with col3:
        st.markdown("### 🍛\n**Recipes & calories** are shown instantly.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(h("h3", "Contact"), unsafe_allow_html=True)
    st.markdown("<p style='color:#001F3F;'>📧 smartref@gmail.com</p>",
                unsafe_allow_html=True)


def page_recipes():
    recipes_df = load_recipes()
    st.markdown(h("h1", "📖 Browse Recipes"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        diets = ["All"] + sorted(recipes_df["Diet"].dropna().unique().tolist())
        diet_sel = st.selectbox("Diet Preference", diets, key="diet_sel")
    with col2:
        courses = ["All"] + sorted(recipes_df["Course"].dropna().unique().tolist())
        course_sel = st.selectbox("Course", courses, key="course_sel")

    search_q = st.text_input("🔍 Search by ingredient or recipe name", key="recipe_search")

    filtered = recipes_df.copy()
    if diet_sel != "All":
        filtered = filtered[filtered["Diet"] == diet_sel]
    if course_sel != "All":
        filtered = filtered[filtered["Course"] == course_sel]
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = filtered[
            filtered["TranslatedRecipeName"].str.lower().str.contains(q, na=False) |
            filtered["TranslatedIngredients"].str.lower().str.contains(q, na=False)
        ]

    st.markdown(f"<p style='color:#001F3F;'><b>{len(filtered)} recipes found</b></p>",
                unsafe_allow_html=True)

    sample = filtered.sample(min(10, len(filtered)), random_state=42) if len(filtered) > 0 else filtered
    for _, row in sample.iterrows():
        st.markdown(f"#### {row.get('TranslatedRecipeName', '')}")
        with st.expander("View Recipe"):
            st.markdown(f"**Ingredients:** {row.get('TranslatedIngredients', 'N/A')}")
            st.markdown(f"**Instructions:** {row.get('TranslatedInstructions', 'N/A')}")
            url = row.get("URL", "")
            if url:
                st.markdown(f"[🔗 Full Recipe]({url})")
        st.markdown(DIVIDER, unsafe_allow_html=True)


def page_ingredients():
    recipes_df = load_recipes()
    st.markdown(h("h1", "🥦 Identify Ingredients"), unsafe_allow_html=True)
    st.markdown(h("h3", "Fruit & Vegetable Classification"), unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded).resize((300, 300))
        st.image(img, caption="Uploaded Image", use_column_width=False)

        save_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Classifying…"):
            result = classify_image(save_path)

        display_result(result)
        recs = recommend_recipes(result, recipes_df)
        display_recommendations(recs)


def page_camera():
    recipes_df = load_recipes()
    st.markdown(h("h1", "📷 Live Camera"), unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#001F3F;'>Use the camera input below to take a photo and identify your ingredient.</p>",
        unsafe_allow_html=True,
    )

    # Use Streamlit's built-in camera_input (no OpenCV VideoCapture loop needed)
    cam_image = st.camera_input("Take a photo")

    if cam_image is not None:
        save_path = os.path.join(UPLOAD_DIR, "camera_capture.jpg")
        with open(save_path, "wb") as f:
            f.write(cam_image.getbuffer())

        with st.spinner("Classifying…"):
            result = classify_image(save_path)

        display_result(result)
        recs = recommend_recipes(result, recipes_df)
        display_recommendations(recs)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SmartKitchen",
        page_icon="🍽️",
        layout="wide",
    )
    apply_styles()

    pages = {
        "🏠 Home":        page_home,
        "📖 Recipes":     page_recipes,
        "🥦 Ingredients": page_ingredients,
        "📷 Camera":      page_camera,
    }

    st.sidebar.title("🍽️ SmartKitchen")
    st.sidebar.markdown("---")
    selection = st.sidebar.radio("Navigate", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size:12px;'>Built with TensorFlow · MobileNet · Streamlit</p>",
        unsafe_allow_html=True,
    )

    pages[selection]()


if __name__ == "__main__":
    main()
