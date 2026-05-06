import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "upload_images")
CSV_PATH   = os.path.join(BASE_DIR, "IndianFoodDatasetCSV.csv")
MODEL_PATH = os.path.join(BASE_DIR, "FV.h5")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(show_spinner="Loading recipes…")
def load_recipes():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    return df

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
FRUIT_EMOJI = {
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
CALORIE_DATA = {
    'apple':52,'banana':89,'beetroot':43,'bell pepper':31,'cabbage':25,
    'capsicum':20,'carrot':41,'cauliflower':25,'chilli pepper':40,
    'corn':86,'cucumber':16,'eggplant':25,'garlic':149,'ginger':80,
    'grapes':67,'jalepeno':29,'kiwi':61,'lemon':29,'lettuce':15,
    'mango':60,'onion':40,'orange':47,'paprika':282,'pear':57,
    'peas':81,'pineapple':50,'pomegranate':83,'potato':77,'raddish':16,
    'soy beans':147,'spinach':23,'sweetcorn':86,'sweetpotato':86,
    'tomato':18,'turnip':28,'watermelon':30,
}
NUTRITION_INFO = {
    'apple':     {'carbs':'14g','protein':'0.3g','fat':'0.2g','fiber':'2.4g','vitamin':'C, B6'},
    'banana':    {'carbs':'23g','protein':'1.1g','fat':'0.3g','fiber':'2.6g','vitamin':'B6, C'},
    'carrot':    {'carbs':'10g','protein':'0.9g','fat':'0.2g','fiber':'2.8g','vitamin':'A, K'},
    'tomato':    {'carbs':'3.9g','protein':'0.9g','fat':'0.2g','fiber':'1.2g','vitamin':'C, K'},
    'spinach':   {'carbs':'3.6g','protein':'2.9g','fat':'0.4g','fiber':'2.2g','vitamin':'K, A, C'},
    'potato':    {'carbs':'17g','protein':'2g','fat':'0.1g','fiber':'2.2g','vitamin':'C, B6'},
    'mango':     {'carbs':'15g','protein':'0.8g','fat':'0.4g','fiber':'1.6g','vitamin':'C, A'},
    'orange':    {'carbs':'12g','protein':'0.9g','fat':'0.1g','fiber':'2.4g','vitamin':'C, B1'},
    'cucumber':  {'carbs':'3.6g','protein':'0.7g','fat':'0.1g','fiber':'0.5g','vitamin':'K, C'},
    'onion':     {'carbs':'9g','protein':'1.1g','fat':'0.1g','fiber':'1.7g','vitamin':'C, B6'},
    'mango':     {'carbs':'15g','protein':'0.8g','fat':'0.4g','fiber':'1.6g','vitamin':'C, A'},
    'grapes':    {'carbs':'18g','protein':'0.7g','fat':'0.2g','fiber':'0.9g','vitamin':'C, K'},
    'pineapple': {'carbs':'13g','protein':'0.5g','fat':'0.1g','fiber':'1.4g','vitamin':'C, B6'},
    'watermelon':{'carbs':'8g','protein':'0.6g','fat':'0.2g','fiber':'0.4g','vitamin':'C, A'},
    'peas':      {'carbs':'14g','protein':'5g','fat':'0.4g','fiber':'5g','vitamin':'C, K, B1'},
    'garlic':    {'carbs':'33g','protein':'6.4g','fat':'0.5g','fiber':'2.1g','vitamin':'C, B6'},
    'cauliflower':{'carbs':'5g','protein':'1.9g','fat':'0.3g','fiber':'2g','vitamin':'C, K'},
    'corn':      {'carbs':'19g','protein':'3.3g','fat':'1.4g','fiber':'2.4g','vitamin':'B5, B9'},
    'lemon':     {'carbs':'9g','protein':'1.1g','fat':'0.3g','fiber':'2.8g','vitamin':'C, B6'},
    'eggplant':  {'carbs':'6g','protein':'1g','fat':'0.2g','fiber':'3g','vitamin':'B1, B6'},
}

def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif !important; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        min-height: 100vh;
    }
    [data-testid="stHeader"] { background: transparent; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1100px; }

    .sk-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .sk-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }

    .sk-hero {
        background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,183,77,0.15));
        border: 1px solid rgba(255,150,50,0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sk-hero h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #FF6B35, #FFB74D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }
    .sk-hero p { color: #b0bec5 !important; font-size: 1.1rem !important; }

    .sk-step {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        min-height: 160px;
    }
    .sk-step .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .sk-step h3 { color: #FFB74D !important; font-size: 1rem !important; margin-bottom: 0.4rem !important; }
    .sk-step p  { color: #90a4ae !important; font-size: 0.88rem !important; }

    .sk-result {
        background: linear-gradient(135deg, rgba(76,175,80,0.15), rgba(129,199,132,0.1));
        border: 1px solid rgba(76,175,80,0.4);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .sk-result-label { color: #81c784 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }
    .sk-result-value { color: #ffffff !important; font-size: 1.6rem !important; font-weight: 600 !important; }

    .sk-calorie {
        background: linear-gradient(135deg, rgba(255,152,0,0.15), rgba(255,183,77,0.1));
        border: 1px solid rgba(255,152,0,0.4);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
    }

    .sk-nut-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 0.6rem; }
    .sk-nut-item { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 0.5rem 0.8rem; text-align: center; }
    .sk-nut-label { color: #78909c !important; font-size: 0.7rem !important; }
    .sk-nut-val   { color: #e0e0e0 !important; font-size: 0.95rem !important; font-weight: 600 !important; }

    .sk-recipe-title { color: #ffffff !important; font-size: 1.05rem !important; font-weight: 600 !important; margin: 0 !important; }
    .sk-recipe-meta  { color: #78909c !important; font-size: 0.8rem !important; }

    .sk-section-title {
        color: #FFB74D !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        border-left: 4px solid #FF6B35;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0 !important;
        display: block;
    }

    .stTextInput input { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 10px !important; color: white !important; }
    .stTextInput label, .stSelectbox label, .stFileUploader label { color: #b0bec5 !important; font-weight: 500 !important; }
    [data-testid="stExpander"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; }
    [data-testid="stExpander"] summary { color: #b0bec5 !important; }
    [data-testid="stExpander"] p, [data-testid="stExpander"] span, [data-testid="stExpander"] div { color: #cfd8dc !important; }
    [data-testid="stExpander"] a { color: #FFB74D !important; }

    .sk-badge { display: inline-block; background: linear-gradient(135deg, #FF6B35, #FFB74D); color: white !important; border-radius: 20px; padding: 2px 14px; font-size: 0.82rem; font-weight: 600; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,0.03) !important; border: 2px dashed rgba(255,183,77,0.4) !important; border-radius: 14px !important; }
    .sk-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,183,77,0.3), transparent); margin: 1.2rem 0; }
    .sk-footer { text-align: center; color: #546e7a !important; font-size: 0.78rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 3rem; }
    </style>
    """, unsafe_allow_html=True)

def fetch_calories(prediction):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.google.com/search?q=calories+in+{prediction}+per+100g"
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        for cls in ["BNeawe iBp4i AP7Wnd", "BNeawe", "IZ6rdc"]:
            tag = soup.find("div", class_=cls)
            if tag and "cal" in tag.text.lower():
                return tag.text
    except Exception:
        pass
    kcal = CALORIE_DATA.get(prediction.lower())
    return f"~{kcal} kcal per 100g" if kcal else "Data unavailable"

def classify_image(img_path):
    model = load_model()
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    return LABELS[preds.argmax(axis=-1)[0]].capitalize()

def recommend_recipes(prediction, recipes_df, n=6):
    pred_lower = prediction.lower()
    mask = (
        recipes_df["TranslatedRecipeName"].str.lower().str.contains(pred_lower, na=False) |
        recipes_df["TranslatedIngredients"].str.lower().str.contains(pred_lower, na=False)
    )
    return recipes_df[mask].drop_duplicates(subset="TranslatedRecipeName").head(n)

def show_classification_result(result):
    label_lower = result.lower()
    emoji = FRUIT_EMOJI.get(label_lower, "🌿")
    category = "Fruit" if label_lower in FRUITS else "Vegetable"
    kcal = CALORIE_DATA.get(label_lower, "?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sk-result">
            <div class="sk-result-label">Identified as</div>
            <div class="sk-result-value">{emoji} {result}</div>
            <div style="color:#81c784;font-size:0.85rem;margin-top:4px;">Category: {category}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="sk-calorie">
            <div class="sk-result-label" style="color:#ffcc80 !important;">Calories</div>
            <div style="color:#FFB74D;font-size:2rem;font-weight:700;">{kcal}</div>
            <div style="color:#ffb74d;font-size:0.8rem;">kcal per 100g</div>
        </div>""", unsafe_allow_html=True)

    nut = NUTRITION_INFO.get(label_lower)
    if nut:
        st.markdown('<span class="sk-section-title">Nutrition (per 100g)</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sk-card">
            <div class="sk-nut-grid">
                <div class="sk-nut-item"><div class="sk-nut-label">Carbs</div><div class="sk-nut-val">{nut['carbs']}</div></div>
                <div class="sk-nut-item"><div class="sk-nut-label">Protein</div><div class="sk-nut-val">{nut['protein']}</div></div>
                <div class="sk-nut-item"><div class="sk-nut-label">Fat</div><div class="sk-nut-val">{nut['fat']}</div></div>
                <div class="sk-nut-item"><div class="sk-nut-label">Fiber</div><div class="sk-nut-val">{nut['fiber']}</div></div>
                <div class="sk-nut-item" style="grid-column:span 2"><div class="sk-nut-label">Key Vitamins</div><div class="sk-nut-val">{nut['vitamin']}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

def show_recipes(result, recipes_df):
    recs = recommend_recipes(result, recipes_df)
    st.markdown(f'<span class="sk-section-title">Recipes using {result} &nbsp;<span class="sk-badge">{len(recs)} found</span></span>', unsafe_allow_html=True)
    if recs.empty:
        st.markdown('<div class="sk-card"><p style="color:#78909c;">No recipes found. Try the Browse Recipes page.</p></div>', unsafe_allow_html=True)
        return
    for _, row in recs.iterrows():
        name   = row.get("TranslatedRecipeName", "Recipe")
        diet   = row.get("Diet", "")
        course = row.get("Course", "")
        url    = row.get("URL", "")
        meta   = " · ".join(filter(None, [diet, course]))
        st.markdown(f'<div class="sk-card"><div class="sk-recipe-title">🍳 {name}</div><div class="sk-recipe-meta">{meta}</div></div>', unsafe_allow_html=True)
        with st.expander("View recipe"):
            st.markdown(f"**Ingredients:** {row.get('TranslatedIngredients','N/A')}")
            st.markdown(f"**Instructions:** {row.get('TranslatedInstructions','N/A')}")
            if url:
                st.markdown(f"[🔗 Full Recipe →]({url})")

def page_home():
    st.markdown("""
    <div class="sk-hero">
        <h1>🍽️ SmartKitchen</h1>
        <p>Point your camera at any fruit or vegetable — get instant AI identification,<br>
        calorie info, nutrition data, and personalised Indian recipe suggestions.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<span class="sk-section-title">How It Works</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in zip([c1,c2,c3],
        ["📷","🧠","🍛"],
        ["Capture","AI Identifies","Cook!"],
        ["Upload a photo or use your webcam to snap a picture of your ingredient.",
         "Our MobileNet model instantly classifies your fruit or vegetable.",
         "Get calories, nutrition info, and matching Indian recipes in seconds."]):
        with col:
            st.markdown(f'<div class="sk-step"><div class="icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown('<span class="sk-section-title">What We Support</span>', unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    for col, num, label in zip([s1,s2,s3,s4], ["36","15","21","1000+"], ["Ingredients","Fruits","Vegetables","Recipes"]):
        with col:
            st.markdown(f'<div class="sk-card" style="text-align:center;padding:1rem;"><div style="font-size:2rem;font-weight:700;color:#FFB74D;">{num}</div><div style="color:#78909c;font-size:0.85rem;">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<span class="sk-section-title">Supported Ingredients</span>', unsafe_allow_html=True)
    all_labels = sorted(LABELS.values())
    cols = st.columns(6)
    for i, label in enumerate(all_labels):
        emoji = FRUIT_EMOJI.get(label, "🌿")
        with cols[i % 6]:
            st.markdown(f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:6px;text-align:center;margin-bottom:6px;font-size:0.78rem;color:#b0bec5;">{emoji} {label.title()}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sk-footer">Built with TensorFlow · MobileNet · Streamlit &nbsp;|&nbsp; By Dhwani Chande</div>', unsafe_allow_html=True)

def page_identify():
    recipes_df = load_recipes()
    st.markdown('<h2 style="color:white;">🥦 Identify Ingredient</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#78909c;margin-bottom:1.5rem;">Upload a photo of a fruit or vegetable to identify it and get recipe suggestions.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop an image here or click to browse", type=["jpg","jpeg","png"])
    if uploaded:
        col_img, col_res = st.columns([1, 1.5])
        with col_img:
            st.image(Image.open(uploaded), use_column_width=True, caption="Your image")
        save_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with col_res:
            with st.spinner("Analysing image…"):
                result = classify_image(save_path)
            show_classification_result(result)
        st.markdown('<div class="sk-divider"></div>', unsafe_allow_html=True)
        show_recipes(result, recipes_df)

def page_camera():
    recipes_df = load_recipes()
    st.markdown('<h2 style="color:white;">📷 Camera Capture</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#78909c;margin-bottom:1.5rem;">Take a live photo to identify your ingredient instantly.</p>', unsafe_allow_html=True)
    cam_image = st.camera_input("Point at a fruit or vegetable and capture")
    if cam_image:
        save_path = os.path.join(UPLOAD_DIR, "camera_capture.jpg")
        with open(save_path, "wb") as f:
            f.write(cam_image.getbuffer())
        with st.spinner("Analysing image…"):
            result = classify_image(save_path)
        show_classification_result(result)
        st.markdown('<div class="sk-divider"></div>', unsafe_allow_html=True)
        show_recipes(result, recipes_df)

def page_recipes():
    recipes_df = load_recipes()
    st.markdown('<h2 style="color:white;">📖 Browse Recipes</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        diets = ["All"] + sorted(recipes_df["Diet"].dropna().unique().tolist())
        diet_sel = st.selectbox("Diet", diets)
    with c2:
        courses = ["All"] + sorted(recipes_df["Course"].dropna().unique().tolist())
        course_sel = st.selectbox("Course", courses)
    with c3:
        search_q = st.text_input("Search recipes or ingredients")

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
    st.markdown(f'<p style="color:#78909c;margin-bottom:1rem;">Showing <span class="sk-badge">{min(12,total)}</span> of {total} recipes</p>', unsafe_allow_html=True)

    if total == 0:
        st.markdown('<div class="sk-card"><p style="color:#78909c;text-align:center;">No recipes match. Try adjusting filters.</p></div>', unsafe_allow_html=True)
        return

    sample = filtered.sample(min(12, total), random_state=42)
    for _, row in sample.iterrows():
        name   = row.get("TranslatedRecipeName","Recipe")
        diet   = row.get("Diet","")
        course = row.get("Course","")
        url    = row.get("URL","")
        meta   = " · ".join(filter(None,[diet,course]))
        st.markdown(f'<div class="sk-card"><div class="sk-recipe-title">🍳 {name}</div><div class="sk-recipe-meta">{meta}</div></div>', unsafe_allow_html=True)
        with st.expander("View recipe"):
            st.markdown(f"**Ingredients:** {row.get('TranslatedIngredients','N/A')}")
            st.markdown(f"**Instructions:** {row.get('TranslatedInstructions','N/A')}")
            if url:
                st.markdown(f"[🔗 Full Recipe →]({url})")

    st.markdown('<div class="sk-footer">Built with TensorFlow · MobileNet · Streamlit | By Dhwani Chande</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="SmartKitchen", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")
    apply_styles()

    pages = {
        "🏠  Home":           page_home,
        "🥦  Identify":       page_identify,
        "📷  Camera":         page_camera,
        "📖  Browse Recipes": page_recipes,
    }

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2.5rem;">🍽️</div>
            <div style="font-size:1.3rem;font-weight:700;color:#FFB74D;">SmartKitchen</div>
            <div style="font-size:0.75rem;color:#546e7a;margin-top:4px;">AI-Powered Recipe Assistant</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.08);margin-bottom:1rem;">
        """, unsafe_allow_html=True)
        selection = st.radio("", list(pages.keys()), label_visibility="collapsed")
        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.08);margin-top:2rem;">
        <div style="font-size:0.72rem;color:#37474f;text-align:center;padding-top:0.5rem;">
            MobileNet · TensorFlow · Streamlit<br>
            <span style="color:#4caf50;">● Live</span>
        </div>
        """, unsafe_allow_html=True)

    pages[selection]()

if __name__ == "__main__":
    main()
