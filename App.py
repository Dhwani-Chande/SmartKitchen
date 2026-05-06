import os
import re
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "upload_images")
CSV_PATH   = os.path.join(BASE_DIR, "IndianFoodDatasetCSV.csv")
MODEL_PATH = os.path.join(BASE_DIR, "FV.h5")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Cached resources ───────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(show_spinner="Loading recipes…")
def load_recipes():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    return df

# ── Data ───────────────────────────────────────
LABELS = {
    0:'apple', 1:'banana', 2:'beetroot', 3:'bell pepper', 4:'cabbage',
    5:'capsicum', 6:'carrot', 7:'cauliflower', 8:'chilli pepper', 9:'corn',
    10:'cucumber', 11:'eggplant', 12:'garlic', 13:'ginger', 14:'grapes',
    15:'jalepeno', 16:'kiwi', 17:'lemon', 18:'lettuce', 19:'mango',
    20:'onion', 21:'orange', 22:'paprika', 23:'pear', 24:'peas',
    25:'pineapple', 26:'pomegranate', 27:'potato', 28:'raddish',
    29:'soy beans', 30:'spinach', 31:'sweetcorn', 32:'sweetpotato',
    33:'tomato', 34:'turnip', 35:'watermelon',
}
FRUITS = {
    'apple','banana','bell pepper','chilli pepper','grapes','jalepeno',
    'kiwi','lemon','mango','orange','paprika','pear','pineapple',
    'pomegranate','watermelon',
}
EMOJI = {
    'apple':'🍎','banana':'🍌','bell pepper':'🫑','chilli pepper':'🌶️',
    'grapes':'🍇','jalepeno':'🌶️','kiwi':'🥝','lemon':'🍋','mango':'🥭',
    'orange':'🍊','paprika':'🫑','pear':'🍐','pineapple':'🍍',
    'pomegranate':'🍎','watermelon':'🍉','beetroot':'🫚','cabbage':'🥬',
    'capsicum':'🫑','carrot':'🥕','cauliflower':'🥦','corn':'🌽',
    'cucumber':'🥒','eggplant':'🍆','garlic':'🧄','ginger':'🫚',
    'lettuce':'🥬','onion':'🧅','peas':'🫛','potato':'🥔',
    'raddish':'🌱','soy beans':'🫘','spinach':'🥬','sweetcorn':'🌽',
    'sweetpotato':'🍠','tomato':'🍅','turnip':'🫚',
}
CALORIES = {
    'apple':52,'banana':89,'beetroot':43,'bell pepper':31,'cabbage':25,
    'capsicum':20,'carrot':41,'cauliflower':25,'chilli pepper':40,
    'corn':86,'cucumber':16,'eggplant':25,'garlic':149,'ginger':80,
    'grapes':67,'jalepeno':29,'kiwi':61,'lemon':29,'lettuce':15,
    'mango':60,'onion':40,'orange':47,'paprika':282,'pear':57,
    'peas':81,'pineapple':50,'pomegranate':83,'potato':77,'raddish':16,
    'soy beans':147,'spinach':23,'sweetcorn':86,'sweetpotato':86,
    'tomato':18,'turnip':28,'watermelon':30,
}
NUTRITION = {
    'apple':     {'Carbs':'14g','Protein':'0.3g','Fat':'0.2g','Fiber':'2.4g','Vitamins':'C, B6'},
    'banana':    {'Carbs':'23g','Protein':'1.1g','Fat':'0.3g','Fiber':'2.6g','Vitamins':'B6, C'},
    'carrot':    {'Carbs':'10g','Protein':'0.9g','Fat':'0.2g','Fiber':'2.8g','Vitamins':'A, K'},
    'tomato':    {'Carbs':'3.9g','Protein':'0.9g','Fat':'0.2g','Fiber':'1.2g','Vitamins':'C, K'},
    'spinach':   {'Carbs':'3.6g','Protein':'2.9g','Fat':'0.4g','Fiber':'2.2g','Vitamins':'K, A, C'},
    'potato':    {'Carbs':'17g','Protein':'2g','Fat':'0.1g','Fiber':'2.2g','Vitamins':'C, B6'},
    'mango':     {'Carbs':'15g','Protein':'0.8g','Fat':'0.4g','Fiber':'1.6g','Vitamins':'C, A'},
    'orange':    {'Carbs':'12g','Protein':'0.9g','Fat':'0.1g','Fiber':'2.4g','Vitamins':'C, B1'},
    'cucumber':  {'Carbs':'3.6g','Protein':'0.7g','Fat':'0.1g','Fiber':'0.5g','Vitamins':'K, C'},
    'onion':     {'Carbs':'9g','Protein':'1.1g','Fat':'0.1g','Fiber':'1.7g','Vitamins':'C, B6'},
    'grapes':    {'Carbs':'18g','Protein':'0.7g','Fat':'0.2g','Fiber':'0.9g','Vitamins':'C, K'},
    'pineapple': {'Carbs':'13g','Protein':'0.5g','Fat':'0.1g','Fiber':'1.4g','Vitamins':'C, B6'},
    'watermelon':{'Carbs':'8g','Protein':'0.6g','Fat':'0.2g','Fiber':'0.4g','Vitamins':'C, A'},
    'peas':      {'Carbs':'14g','Protein':'5g','Fat':'0.4g','Fiber':'5g','Vitamins':'C, K, B1'},
    'garlic':    {'Carbs':'33g','Protein':'6.4g','Fat':'0.5g','Fiber':'2.1g','Vitamins':'C, B6'},
    'cauliflower':{'Carbs':'5g','Protein':'1.9g','Fat':'0.3g','Fiber':'2g','Vitamins':'C, K'},
    'corn':      {'Carbs':'19g','Protein':'3.3g','Fat':'1.4g','Fiber':'2.4g','Vitamins':'B5, B9'},
    'lemon':     {'Carbs':'9g','Protein':'1.1g','Fat':'0.3g','Fiber':'2.8g','Vitamins':'C, B6'},
    'eggplant':  {'Carbs':'6g','Protein':'1g','Fat':'0.2g','Fiber':'3g','Vitamins':'B1, B6'},
    'pear':      {'Carbs':'15g','Protein':'0.4g','Fat':'0.1g','Fiber':'3.1g','Vitamins':'C, K'},
    'kiwi':      {'Carbs':'15g','Protein':'1.1g','Fat':'0.5g','Fiber':'3g','Vitamins':'C, K'},
}

# ── Core logic ─────────────────────────────────
def classify_image(img_path):
    model = load_model()
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
    return LABELS[model.predict(arr, verbose=0).argmax(axis=-1)[0]].capitalize()

def recommend_recipes(pred, recipes_df, n=8):
    p = pred.lower()
    mask = (
        recipes_df["TranslatedRecipeName"].str.lower().str.contains(p, na=False) |
        recipes_df["TranslatedIngredients"].str.lower().str.contains(p, na=False)
    )
    return recipes_df[mask].drop_duplicates("TranslatedRecipeName").head(n)

def format_ingredients(raw):
    if not raw or raw.strip() == "":
        return []
    return [i.strip() for i in raw.split(",") if i.strip()]

def format_instructions(raw):
    if not raw or raw.strip() == "":
        return []
    steps = [s.strip() for s in re.split(r"(?<=[.!?]) +(?=[A-Z])", raw) if s.strip()]
    return steps if len(steps) > 1 else [raw.strip()]

# ── Shared UI ──────────────────────────────────
def show_result(result):
    label = result.lower()
    emoji = EMOJI.get(label, "🌿")
    cat   = "Fruit" if label in FRUITS else "Vegetable"
    kcal  = CALORIES.get(label, "—")

    col1, col2, col3 = st.columns(3)
    col1.metric("Identified", f"{emoji} {result}")
    col2.metric("Category", cat)
    col3.metric("Calories (per 100g)", f"{kcal} kcal")

    nut = NUTRITION.get(label)
    if nut:
        st.write("**Nutrition per 100g**")
        nc = st.columns(len(nut))
        for col, (k, v) in zip(nc, nut.items()):
            col.metric(k, v)

def show_recipes(result, recipes_df):
    recs = recommend_recipes(result, recipes_df)
    st.write(f"**Recipes using {result}** — {len(recs)} found")
    if recs.empty:
        st.info("No recipes found. Try Browse Recipes to search manually.")
        return
    for _, row in recs.iterrows():
        name   = row.get("TranslatedRecipeName", "Recipe")
        diet   = row.get("Diet", "")
        course = row.get("Course", "")
        url    = row.get("URL", "")
        tag    = " · ".join(filter(None, [diet, course]))
        with st.expander(f"🍳 {name}" + (f"  —  {tag}" if tag else "")):
            ingredients = format_ingredients(row.get("TranslatedIngredients", ""))
            steps = format_instructions(row.get("TranslatedInstructions", ""))
            if ingredients:
                st.write("**Ingredients**")
                for item in ingredients:
                    st.write(f"• {item}")
            if steps:
                st.write("**Instructions**")
                for i, step in enumerate(steps, 1):
                    st.write(f"{i}. {step}")
            if url:
                st.link_button("View full recipe", url)

# ── Pages ──────────────────────────────────────
def page_home():
    st.title("🍽️ SmartKitchen")
    st.write("Point your camera at any fruit or vegetable — get instant AI identification, calories, nutrition, and Indian recipe suggestions.")
    st.divider()

    st.subheader("How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📷 1. Capture**\n\nUpload a photo or use the live camera.")
    with c2:
        st.info("**🧠 2. Identify**\n\nMobileNet AI classifies the ingredient instantly.")
    with c3:
        st.info("**🍛 3. Cook**\n\nGet calories, nutrition, and matching recipes.")

    st.divider()
    st.subheader("Coverage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ingredients", "36")
    c2.metric("Fruits", "15")
    c3.metric("Vegetables", "21")
    c4.metric("Recipes", "1,000+")

    st.divider()
    st.subheader("Supported Ingredients")
    cols = st.columns(6)
    for i, label in enumerate(sorted(LABELS.values())):
        cols[i % 6].write(f"{EMOJI.get(label,'🌿')} {label.title()}")


def page_identify():
    recipes_df = load_recipes()
    st.title("🥦 Identify Ingredient")
    st.write("Upload a photo of a fruit or vegetable to identify it and get recipe suggestions.")
    st.divider()

    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if not uploaded:
        return

    col_img, col_res = st.columns([1, 1.5])
    with col_img:
        st.image(Image.open(uploaded), use_column_width=True)

    save_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with col_res:
        with st.spinner("Analysing image…"):
            result = classify_image(save_path)
        show_result(result)

    st.divider()
    show_recipes(result, recipes_df)


def page_camera():
    recipes_df = load_recipes()
    st.title("📷 Camera")
    st.write("Take a photo to identify your ingredient instantly.")
    st.divider()

    img = st.camera_input("Point at a fruit or vegetable")
    if not img:
        return

    save_path = os.path.join(UPLOAD_DIR, "cam.jpg")
    with open(save_path, "wb") as f:
        f.write(img.getbuffer())

    with st.spinner("Analysing image…"):
        result = classify_image(save_path)

    show_result(result)
    st.divider()
    show_recipes(result, recipes_df)


def page_recipes():
    recipes_df = load_recipes()
    st.title("📖 Browse Recipes")
    st.write("Filter and search over 1,000 Indian recipes.")
    st.divider()

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        diets = ["All"] + sorted(recipes_df["Diet"].dropna().unique().tolist())
        diet_sel = st.selectbox("Diet", diets)
    with c2:
        courses = ["All"] + sorted(recipes_df["Course"].dropna().unique().tolist())
        course_sel = st.selectbox("Course", courses)
    with c3:
        search_q = st.text_input("Search by name or ingredient", placeholder="e.g. paneer, tomato…")

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

    total = len(filtered)
    st.caption(f"Showing 12 of {total} recipes")

    if total == 0:
        st.info("No recipes match. Try adjusting your filters.")
        return

    sample = filtered.sample(min(12, total), random_state=42)
    for _, row in sample.iterrows():
        name   = row.get("TranslatedRecipeName", "Recipe")
        diet   = row.get("Diet", "")
        course = row.get("Course", "")
        url    = row.get("URL", "")
        tag    = " · ".join(filter(None, [diet, course]))
        with st.expander(f"🍳 {name}" + (f"  —  {tag}" if tag else "")):
            ingredients = format_ingredients(row.get("TranslatedIngredients", ""))
            steps = format_instructions(row.get("TranslatedInstructions", ""))
            if ingredients:
                st.write("**Ingredients**")
                for item in ingredients:
                    st.write(f"• {item}")
            if steps:
                st.write("**Instructions**")
                for i, step in enumerate(steps, 1):
                    st.write(f"{i}. {step}")
            if url:
                st.link_button("View full recipe", url)

    st.caption("SmartKitchen · MobileNet · TensorFlow · Streamlit · By Dhwani Chande")


# ── Main ───────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SmartKitchen",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    PAGES = {
        "🏠 Home":       page_home,
        "🥦 Identify":   page_identify,
        "📷 Camera":     page_camera,
        "📖 Recipes":    page_recipes,
    }

    with st.sidebar:
        st.title("🍽️ SmartKitchen")
        st.caption("AI Recipe Assistant")
        st.divider()
        sel = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("TensorFlow · MobileNet · Streamlit")

    PAGES[sel]()


if __name__ == "__main__":
    main()
