import os
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
    0:'apple',1:'banana',2:'beetroot',3:'bell pepper',4:'cabbage',
    5:'capsicum',6:'carrot',7:'cauliflower',8:'chilli pepper',9:'corn',
    10:'cucumber',11:'eggplant',12:'garlic',13:'ginger',14:'grapes',
    15:'jalepeno',16:'kiwi',17:'lemon',18:'lettuce',19:'mango',
    20:'onion',21:'orange',22:'paprika',23:'pear',24:'peas',
    25:'pineapple',26:'pomegranate',27:'potato',28:'raddish',
    29:'soy beans',30:'spinach',31:'sweetcorn',32:'sweetpotato',
    33:'tomato',34:'turnip',35:'watermelon',
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
    'pomegranate':'🍎','watermelon':'🍉','beetroot':'❤️','cabbage':'🥬',
    'capsicum':'🫑','carrot':'🥕','cauliflower':'🥦','corn':'🌽',
    'cucumber':'🥒','eggplant':'🍆','garlic':'🧄','ginger':'🟡',
    'lettuce':'🥬','onion':'🧅','peas':'🫛','potato':'🥔',
    'raddish':'🌱','soy beans':'🫘','spinach':'🥬','sweetcorn':'🌽',
    'sweetpotato':'🍠','tomato':'🍅','turnip':'⚪',
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

# ── Styles ─────────────────────────────────────
def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* App background */
    .stApp, [data-testid="stAppViewContainer"] {
        background: #0a0f1e !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1e2a3a !important;
    }
    [data-testid="stSidebar"] * { color: #8b949e !important; }
    [data-testid="stSidebar"] .stRadio > label { color: #e6edf3 !important; }

    /* Main container */
    .block-container {
        padding: 2rem 3rem 4rem !important;
        max-width: 1000px !important;
    }

    /* Page titles */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 0.25rem;
    }
    .page-sub {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Metric cards using native st.metric */
    [data-testid="stMetric"] {
        background: #161b27 !important;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.5rem !important; font-weight: 600 !important; }

    /* Result boxes */
    .result-box {
        background: #0d2818;
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-label { color: #3fb950; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 4px; }
    .result-value { color: #e6edf3; font-size: 1.8rem; font-weight: 700; }
    .result-sub   { color: #8b949e; font-size: 0.82rem; margin-top: 2px; }

    .calorie-box {
        background: #1a1200;
        border: 1px solid #9e6a03;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .calorie-label { color: #d29922; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 4px; }
    .calorie-value { color: #f0883e; font-size: 2.2rem; font-weight: 700; }
    .calorie-sub   { color: #8b949e; font-size: 0.82rem; margin-top: 2px; }

    /* Section heading */
    .section-heading {
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 0;
        border-bottom: 1px solid #21262d;
        margin: 1.5rem 0 1rem 0;
    }

    /* Nutrition grid */
    .nut-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .nut-cell {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
    }
    .nut-label { color: #8b949e; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .nut-value { color: #e6edf3; font-size: 0.95rem; font-weight: 600; margin-top: 2px; }

    /* Recipe cards — pure native, no HTML wrapper needed */
    [data-testid="stExpander"] {
        background: #161b27 !important;
        border: 1px solid #21262d !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #388bfd !important;
    }
    /* ── Expander: nuke the icon text, keep the label ── */
    [data-testid="stExpander"] summary {
        color: #e6edf3 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        padding: 0.85rem 1rem !important;
        list-style: none !important;
    }
    [data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
    /* Target every known Streamlit icon class and zero its font-size */
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpander"] .eyeqlp53,
    [data-testid="stExpander"] span[class*="expanderIcon"],
    [data-testid="stExpander"] span.material-symbols-rounded,
    [data-testid="stExpander"] span[class*="material"] {
        font-size: 0 !important;
        width: 1.2rem !important;
        height: 1.2rem !important;
        overflow: hidden !important;
    }
    /* Content inside */
    [data-testid="stExpander"] p     { color: #8b949e !important; font-size: 0.88rem !important; line-height: 1.6 !important; }
    [data-testid="stExpander"] strong { color: #c9d1d9 !important; }
    [data-testid="stExpander"] a      { color: #388bfd !important; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0d1f12 0%, #0a1628 100%);
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3fb950, #58a6ff, #f0883e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero-sub { color: #8b949e; font-size: 1rem; line-height: 1.7; }

    /* Step card */
    .step-card {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
    .step-icon { font-size: 2.2rem; margin-bottom: 0.7rem; }
    .step-title { color: #58a6ff; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.4rem; }
    .step-desc  { color: #8b949e; font-size: 0.82rem; line-height: 1.5; }

    /* Stat card */
    .stat-card {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-num   { color: #f0883e; font-size: 2rem; font-weight: 700; }
    .stat-label { color: #8b949e; font-size: 0.78rem; margin-top: 2px; }

    /* Ingredient pill */
    .ing-pill {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 5px 8px;
        text-align: center;
        font-size: 0.75rem;
        color: #8b949e;
        margin-bottom: 5px;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: #388bfd22;
        border: 1px solid #388bfd66;
        color: #58a6ff;
        border-radius: 20px;
        padding: 1px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Upload zone */
    [data-testid="stFileUploaderDropzone"] {
        background: #161b27 !important;
        border: 2px dashed #21262d !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #388bfd !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #161b27 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    .stSelectbox label, .stTextInput label { color: #8b949e !important; font-size: 0.82rem !important; }

    /* Text input */
    .stTextInput input {
        background: #161b27 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    .stTextInput input:focus { border-color: #388bfd !important; }

    /* Divider */
    .divider { height: 1px; background: #21262d; margin: 1.5rem 0; }

    /* Footer */
    .footer {
        text-align: center;
        color: #484f58;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #21262d;
        margin-top: 3rem;
    }

    /* Camera widget */
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img { border-radius: 12px !important; }

    /* Hide duplicate upload button text */
    [data-testid="stFileUploaderDropzone"] button { visibility: hidden !important; position: relative; }
    [data-testid="stFileUploaderDropzone"] button::after {
        visibility: visible;
        content: "Browse Files";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        color: #58a6ff;
        font-size: 0.85rem;
        font-weight: 500;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }

    /* ── Pre-load Material Symbols font so it never falls back to text ── */
    @font-face {
        font-family: 'Material Symbols Rounded';
        font-style: normal;
        src: url(https://fonts.gstatic.com/s/materialsymbolsrounded/v235/syl0-zNym6-2r347v-CVetCt3H16e50-_R20YOdDRFt4.woff2) format('woff2');
    }

    /* Suppress icon text ONLY when font hasn't loaded yet */
    span.material-symbols-rounded,
    span[class*="material-symbols"],
    span[class*="material-icons"] {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 1.1rem !important;
        overflow: hidden !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 1.2rem !important;
        height: 1.2rem !important;
        vertical-align: middle !important;
    }

    /* Sidebar collapse arrow - hide completely, we don't need it */
    [data-testid="collapsedControl"] span.material-symbols-rounded,
    [data-testid="baseButton-headerNoPadding"] span.material-symbols-rounded {
        font-size: 0 !important;
    }

    /* Spinner */
    .stSpinner { color: #58a6ff !important; }

    /* Info/success/warning boxes */
    .stAlert { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Logic ──────────────────────────────────────
def fetch_calories(prediction):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(f"https://www.google.com/search?q=calories+in+{prediction}+per+100g", headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        for cls in ["BNeawe iBp4i AP7Wnd", "BNeawe", "IZ6rdc"]:
            tag = soup.find("div", class_=cls)
            if tag and "cal" in tag.text.lower():
                return tag.text
    except Exception:
        pass
    return None

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

# ── Shared UI pieces ───────────────────────────
def show_result(result):
    label = result.lower()
    emoji = EMOJI.get(label, "🌿")
    cat   = "Fruit 🍎" if label in FRUITS else "Vegetable 🥦"
    kcal  = CALORIES.get(label)
    live_cal = fetch_calories(result)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Identified</div>
            <div class="result-value">{emoji} {result}</div>
            <div class="result-sub">{cat}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        display_cal = live_cal if live_cal else (f"{kcal} kcal / 100g" if kcal else "—")
        st.markdown(f"""
        <div class="calorie-box">
            <div class="calorie-label">Calories</div>
            <div class="calorie-value">{kcal if kcal else "—"}</div>
            <div class="calorie-sub">{display_cal}</div>
        </div>""", unsafe_allow_html=True)

    nut = NUTRITION.get(label)
    if nut:
        st.markdown('<div class="section-heading">Nutrition per 100g</div>', unsafe_allow_html=True)
        cells = "".join(
            f'<div class="nut-cell"><div class="nut-label">{k}</div><div class="nut-value">{v}</div></div>'
            for k, v in nut.items()
        )
        st.markdown(f'<div class="nut-row">{cells}</div>', unsafe_allow_html=True)

def recipe_card(key, name, tag, ing_raw, ins_raw, url, idx):
    """Render a custom collapsible recipe card without st.expander."""
    import re
    state_key = f"recipe_open_{key}_{idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    # Header row — clickable button styled as card header
    is_open = st.session_state[state_key]
    label_text = f"🍳  {name}" + (f"  ·  {tag}" if tag else "")

    st.markdown(f"""
    <div style="background:#161b27;border:1px solid {'#388bfd' if is_open else '#21262d'};
         border-radius:10px;margin-bottom:0.4rem;overflow:hidden;">
        <div id="card-header-{key}-{idx}" style="padding:0;margin:0;"></div>
    </div>""", unsafe_allow_html=True)

    if st.button(label_text, key=state_key + "_btn",
                 use_container_width=True,
                 type="secondary"):
        st.session_state[state_key] = not st.session_state[state_key]
        st.rerun()

    if st.session_state[state_key]:
        with st.container():
            st.markdown('<div style="background:#161b27;border:1px solid #388bfd;border-top:none;border-radius:0 0 10px 10px;padding:1rem 1.2rem;margin-top:-0.4rem;">', unsafe_allow_html=True)
            # Format ingredients
            if ing_raw and ing_raw != "N/A":
                items = [i.strip() for i in ing_raw.split(",") if i.strip()]
                ing_md = "\n".join(f"- {i}" for i in items)
            else:
                ing_md = "_Not available_"
            # Format instructions
            if ins_raw and ins_raw != "N/A":
                sentences = [s.strip() for s in re.split("(?<=[.!?]) +(?=[A-Z])", ins_raw) if s.strip()]
                ins_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences)) if len(sentences) > 1 else ins_raw
            else:
                ins_md = "_Not available_"
            st.markdown("**🥗 Ingredients**")
            st.markdown(ing_md)
            st.markdown("---")
            st.markdown("**📋 Instructions**")
            st.markdown(ins_md)
            if url:
                st.markdown(f"[🔗 View full recipe →]({url})")
            st.markdown('</div>', unsafe_allow_html=True)


def show_recipes(result, recipes_df):
    recs = recommend_recipes(result, recipes_df)
    count = len(recs)
    st.markdown(
        f'<div class="section-heading">Recipes using {result} &nbsp;<span class="badge">{count} found</span></div>',
        unsafe_allow_html=True
    )
    if recs.empty:
        st.info("No recipes found — try Browse Recipes to search manually.")
        return
    for idx, (_, row) in enumerate(recs.iterrows()):
        name = row.get("TranslatedRecipeName", "Recipe")
        diet = row.get("Diet", "")
        course = row.get("Course", "")
        url  = row.get("URL", "")
        tag  = " · ".join(filter(None, [diet, course]))
        recipe_card("show", name, tag, row.get("TranslatedIngredients",""), row.get("TranslatedInstructions",""), url, idx)


def page_home():
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🍽️ SmartKitchen</div>
        <div class="hero-sub">
            Point your camera at any fruit or vegetable.<br>
            Get instant AI identification, calories, nutrition, and Indian recipe suggestions.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">How It Works</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, icon, title, desc in zip(cols,
        ["📷", "🧠", "🍛"],
        ["1. Capture", "2. Identify", "3. Cook"],
        [
            "Upload a photo or use the live camera to snap your fruit or vegetable.",
            "MobileNet AI classifies the ingredient in under a second.",
            "Get calories, full nutrition data, and matching Indian recipes instantly.",
        ]):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Coverage</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, n, lbl in zip([c1,c2,c3,c4], ["36","15","21","1,000+"], ["Ingredients","Fruits","Vegetables","Recipes"]):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{n}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Supported Ingredients</div>', unsafe_allow_html=True)
    cols6 = st.columns(6)
    for i, lbl in enumerate(sorted(LABELS.values())):
        with cols6[i % 6]:
            st.markdown(f'<div class="ing-pill">{EMOJI.get(lbl,"🌿")} {lbl.title()}</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">SmartKitchen · MobileNet · TensorFlow · Streamlit · By Dhwani Chande</div>', unsafe_allow_html=True)


def page_identify():
    recipes_df = load_recipes()
    st.markdown('<div class="page-title">🥦 Identify Ingredient</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a photo to identify your fruit or vegetable and get recipe suggestions.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if not uploaded:
        st.markdown("""
        <div style="background:#161b27;border:2px dashed #21262d;border-radius:12px;
                    padding:3rem;text-align:center;color:#484f58;font-size:0.9rem;">
            📂 &nbsp; Drop an image here or click Browse Files above
        </div>""", unsafe_allow_html=True)
        return

    col_img, col_res = st.columns([1, 1.4])
    with col_img:
        st.image(Image.open(uploaded), use_column_width=True, caption="")
    
    save_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with col_res:
        with st.spinner("Analysing…"):
            result = classify_image(save_path)
        show_result(result)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    show_recipes(result, recipes_df)


def page_camera():
    recipes_df = load_recipes()
    st.markdown('<div class="page-title">📷 Camera</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Take a photo to identify your ingredient instantly.</div>', unsafe_allow_html=True)

    img = st.camera_input("")
    if not img:
        return

    save_path = os.path.join(UPLOAD_DIR, "cam.jpg")
    with open(save_path, "wb") as f:
        f.write(img.getbuffer())

    with st.spinner("Analysing…"):
        result = classify_image(save_path)

    show_result(result)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    show_recipes(result, recipes_df)


def page_recipes():
    recipes_df = load_recipes()
    st.markdown('<div class="page-title">📖 Browse Recipes</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Filter and search over 1,000 Indian recipes.</div>', unsafe_allow_html=True)

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
    st.markdown(
        f'<p style="color:#484f58;font-size:0.82rem;margin:0.5rem 0 1rem;">Showing <span class="badge">{min(12,total)}</span> of {total} recipes</p>',
        unsafe_allow_html=True
    )

    if total == 0:
        st.info("No recipes match. Try adjusting your filters.")
        return

    sample = filtered.sample(min(12, total), random_state=42).reset_index(drop=True)
    for idx, (_, row) in enumerate(sample.iterrows()):
        name = row.get("TranslatedRecipeName","Recipe")
        diet = row.get("Diet","")
        course = row.get("Course","")
        url  = row.get("URL","")
        tag  = " · ".join(filter(None,[diet,course]))
        recipe_card("browse", name, tag, row.get("TranslatedIngredients",""), row.get("TranslatedInstructions",""), url, idx)

    st.markdown('<div class="footer">SmartKitchen · By Dhwani Chande</div>', unsafe_allow_html=True)


# ── Main ───────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SmartKitchen",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    PAGES = {
        "🏠  Home":        page_home,
        "🥦  Identify":    page_identify,
        "📷  Camera":      page_camera,
        "📖  Recipes":     page_recipes,
    }

    with st.sidebar:
        st.markdown("""
        <div style="padding:1.5rem 0 1rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:6px;">🍽️</div>
            <div style="color:#e6edf3;font-weight:700;font-size:1.1rem;">SmartKitchen</div>
            <div style="color:#484f58;font-size:0.72rem;margin-top:3px;">AI Recipe Assistant</div>
        </div>
        <hr style="border-color:#21262d;margin:0 0 1rem;">
        """, unsafe_allow_html=True)

        sel = st.radio("", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown("""
        <hr style="border-color:#21262d;margin-top:auto;">
        <div style="font-size:0.7rem;color:#484f58;text-align:center;padding:0.8rem 0;">
            TensorFlow · MobileNet · Streamlit<br>
            <span style="color:#3fb950;">● Live</span>
        </div>
        """, unsafe_allow_html=True)

    PAGES[sel]()


if __name__ == "__main__":
    main()
