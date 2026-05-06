import os
import re
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from supabase import create_client

# ── Config ─────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "upload_images")
CSV_PATH   = os.path.join(BASE_DIR, "IndianFoodDatasetCSV.csv")
MODEL_PATH = os.path.join(BASE_DIR, "FV.h5")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPABASE_URL = "https://bxiynhnvauqilagtvleg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4aXluaG52YXVxaWxhZ3R2bGVnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgwOTA5MTYsImV4cCI6MjA5MzY2NjkxNn0.sUmX2XUOp0U17V-9gIsffdxJnj9SMiY11OAvoXBpILo"

# ── Supabase ───────────────────────────────────
@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def db_get_or_create_profile(auth_user) -> dict:
    """Get or create a profile row for an authenticated Supabase user."""
    sb = get_supabase()
    uid = auth_user.id
    res = sb.table("users").select("*").eq("id", uid).execute()
    if res.data:
        return res.data[0]
    # First login — create profile
    name = (auth_user.user_metadata or {}).get("full_name") or auth_user.email.split("@")[0]
    new = sb.table("users").insert({
        "id": uid,
        "name": name,
        "diet_preference": "All"
    }).execute()
    return new.data[0]

def db_update_diet(user_id: str, diet: str):
    get_supabase().table("users").update({"diet_preference": diet}).eq("id", user_id).execute()

def db_save_favourite(user_id: str, row: dict):
    sb = get_supabase()
    # avoid duplicates
    exists = sb.table("favourites").select("id").eq("user_id", user_id).eq("recipe_name", row.get("TranslatedRecipeName","")).execute()
    if not exists.data:
        sb.table("favourites").insert({
            "user_id": user_id,
            "recipe_name": row.get("TranslatedRecipeName", ""),
            "diet":        row.get("Diet", ""),
            "course":      row.get("Course", ""),
            "ingredients": row.get("TranslatedIngredients", ""),
            "instructions":row.get("TranslatedInstructions", ""),
            "url":         row.get("URL", ""),
        }).execute()
        return True
    return False

def db_remove_favourite(user_id: str, recipe_name: str):
    get_supabase().table("favourites").delete().eq("user_id", user_id).eq("recipe_name", recipe_name).execute()

def db_get_favourites(user_id: str) -> list:
    res = get_supabase().table("favourites").select("*").eq("user_id", user_id).order("saved_at", desc=True).execute()
    return res.data or []

def db_log_search(user_id: str, ingredient: str):
    get_supabase().table("search_history").insert({"user_id": user_id, "ingredient": ingredient}).execute()

def db_get_history(user_id: str) -> list:
    res = get_supabase().table("search_history").select("*").eq("user_id", user_id).order("searched_at", desc=True).limit(20).execute()
    return res.data or []

# ── ML ─────────────────────────────────────────
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
    'apple':{'Carbs':'14g','Protein':'0.3g','Fat':'0.2g','Fiber':'2.4g','Vitamins':'C, B6'},
    'banana':{'Carbs':'23g','Protein':'1.1g','Fat':'0.3g','Fiber':'2.6g','Vitamins':'B6, C'},
    'carrot':{'Carbs':'10g','Protein':'0.9g','Fat':'0.2g','Fiber':'2.8g','Vitamins':'A, K'},
    'tomato':{'Carbs':'3.9g','Protein':'0.9g','Fat':'0.2g','Fiber':'1.2g','Vitamins':'C, K'},
    'spinach':{'Carbs':'3.6g','Protein':'2.9g','Fat':'0.4g','Fiber':'2.2g','Vitamins':'K, A, C'},
    'potato':{'Carbs':'17g','Protein':'2g','Fat':'0.1g','Fiber':'2.2g','Vitamins':'C, B6'},
    'mango':{'Carbs':'15g','Protein':'0.8g','Fat':'0.4g','Fiber':'1.6g','Vitamins':'C, A'},
    'orange':{'Carbs':'12g','Protein':'0.9g','Fat':'0.1g','Fiber':'2.4g','Vitamins':'C, B1'},
    'cucumber':{'Carbs':'3.6g','Protein':'0.7g','Fat':'0.1g','Fiber':'0.5g','Vitamins':'K, C'},
    'onion':{'Carbs':'9g','Protein':'1.1g','Fat':'0.1g','Fiber':'1.7g','Vitamins':'C, B6'},
    'grapes':{'Carbs':'18g','Protein':'0.7g','Fat':'0.2g','Fiber':'0.9g','Vitamins':'C, K'},
    'pineapple':{'Carbs':'13g','Protein':'0.5g','Fat':'0.1g','Fiber':'1.4g','Vitamins':'C, B6'},
    'watermelon':{'Carbs':'8g','Protein':'0.6g','Fat':'0.2g','Fiber':'0.4g','Vitamins':'C, A'},
    'peas':{'Carbs':'14g','Protein':'5g','Fat':'0.4g','Fiber':'5g','Vitamins':'C, K, B1'},
    'garlic':{'Carbs':'33g','Protein':'6.4g','Fat':'0.5g','Fiber':'2.1g','Vitamins':'C, B6'},
    'cauliflower':{'Carbs':'5g','Protein':'1.9g','Fat':'0.3g','Fiber':'2g','Vitamins':'C, K'},
    'corn':{'Carbs':'19g','Protein':'3.3g','Fat':'1.4g','Fiber':'2.4g','Vitamins':'B5, B9'},
    'lemon':{'Carbs':'9g','Protein':'1.1g','Fat':'0.3g','Fiber':'2.8g','Vitamins':'C, B6'},
    'eggplant':{'Carbs':'6g','Protein':'1g','Fat':'0.2g','Fiber':'3g','Vitamins':'B1, B6'},
    'pear':{'Carbs':'15g','Protein':'0.4g','Fat':'0.1g','Fiber':'3.1g','Vitamins':'C, K'},
    'kiwi':{'Carbs':'15g','Protein':'1.1g','Fat':'0.5g','Fiber':'3g','Vitamins':'C, K'},
    'cabbage':{'Carbs':'6g','Protein':'1.3g','Fat':'0.1g','Fiber':'2.5g','Vitamins':'C, K'},
    'beetroot':{'Carbs':'10g','Protein':'1.6g','Fat':'0.2g','Fiber':'2.8g','Vitamins':'B9, C'},
    'bell pepper':{'Carbs':'6g','Protein':'1g','Fat':'0.3g','Fiber':'2.1g','Vitamins':'C, B6, A'},
    'lettuce':{'Carbs':'2.9g','Protein':'1.4g','Fat':'0.2g','Fiber':'1.3g','Vitamins':'K, A, C'},
    'sweetpotato':{'Carbs':'20g','Protein':'1.6g','Fat':'0.1g','Fiber':'3g','Vitamins':'A, C, B6'},
    'ginger':{'Carbs':'18g','Protein':'1.8g','Fat':'0.8g','Fiber':'2g','Vitamins':'B6, C'},
    'pomegranate':{'Carbs':'19g','Protein':'1.7g','Fat':'1.2g','Fiber':'4g','Vitamins':'C, K, B9'},
}
DIET_OPTIONS = ["All","Vegetarian","Non Vegetarian","Vegan","Eggetarian",
                "High Protein Vegetarian","High Protein Non Vegetarian",
                "Diabetic Friendly","No Onion No Garlic (Sattvic)"]

# ── Helpers ────────────────────────────────────
def classify_image(img_path: str) -> tuple:
    model = load_model()
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = preds.argmax()
    confidence = float(preds[top_idx]) * 100
    label = LABELS[top_idx]
    return label.capitalize(), confidence

def recommend_recipes(ingredients: list, recipes_df: pd.DataFrame, diet: str = "All", n: int = 12) -> pd.DataFrame:
    """
    Score-based recommendation:
    - Each recipe gets a score = number of basket ingredients it contains
    - Sort by score descending
    - Return top n with match info attached
    - Never returns empty — always gives best available matches
    """
    if not ingredients:
        return pd.DataFrame()

    df = recipes_df.copy()

    # Apply diet filter first
    if diet and diet != "All":
        df = df[df["Diet"] == diet]

    if df.empty:
        # Relax diet filter if nothing found
        df = recipes_df.copy()

    # Score each recipe
    def score_recipe(row):
        text = (
            str(row.get("TranslatedRecipeName", "")) + " " +
            str(row.get("TranslatedIngredients", ""))
        ).lower()
        return sum(1 for ing in ingredients if ing.lower() in text)

    df = df.copy()
    df["_match_score"] = df.apply(score_recipe, axis=1)
    df["_match_count"] = df["_match_score"]  # same for display
    df["_total_ingredients"] = len(ingredients)

    # Only keep recipes that match at least 1 ingredient
    df = df[df["_match_score"] > 0]

    if df.empty:
        return pd.DataFrame()

    # Sort by score descending, deduplicate
    df = df.sort_values("_match_score", ascending=False)
    df = df.drop_duplicates("TranslatedRecipeName")

    return df.head(n)

def format_ingredients(raw: str) -> list:
    return [i.strip() for i in raw.split(",") if i.strip()] if raw else []

def format_instructions(raw: str) -> list:
    if not raw:
        return []
    steps = [s.strip() for s in re.split(r"(?<=[.!?]) +(?=[A-Z])", raw) if s.strip()]
    return steps if len(steps) > 1 else [raw.strip()]

def show_recipe_expander(row: dict, user_id: str = None, key_prefix: str = "r", match_label: str = ""):
    import hashlib
    name   = row.get("TranslatedRecipeName") or row.get("recipe_name", "Recipe")
    diet   = row.get("Diet") or row.get("diet", "")
    course = row.get("Course") or row.get("course", "")
    url    = row.get("URL") or row.get("url", "")
    ing    = row.get("TranslatedIngredients") or row.get("ingredients", "")
    ins    = row.get("TranslatedInstructions") or row.get("instructions", "")
    tag    = " · ".join(filter(None, [diet, course]))
    match_part = f"  ✅ {match_label}" if match_label else ""
    title  = f"🍳 {name}" + (f"  —  {tag}" if tag else "") + match_part
    # unique key based on full name hash to avoid duplicate widget keys
    uid = hashlib.md5(f"{key_prefix}_{name}".encode()).hexdigest()[:10]

    with st.expander(title):
        c1, c2 = st.columns([3, 1])
        with c1:
            if ing:
                st.write("**🥗 Ingredients**")
                for item in format_ingredients(ing):
                    st.write(f"• {item}")
            if ins:
                st.write("**📋 Instructions**")
                for i, step in enumerate(format_instructions(ins), 1):
                    st.write(f"{i}. {step}")
        with c2:
            if url:
                st.link_button("🔗 Full Recipe", url, key=f"link_{uid}", use_container_width=True)
            if user_id:
                if st.button("❤️ Save", key=f"fav_{uid}", use_container_width=True):
                    saved = db_save_favourite(user_id, row if isinstance(row, dict) else row.to_dict())
                    st.toast("Saved to favourites! ❤️" if saved else "Already in favourites")

# ── Session helpers ────────────────────────────
def get_user():
    return st.session_state.get("user")

def auth_page():
    """Full login / signup page using Supabase Auth."""
    sb = get_supabase()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem;">
            <div style="font-size:3.5rem;">🍽️</div>
            <div style="font-size:2rem;font-weight:800;color:#2C1A0E;margin-bottom:0.3rem;font-family:Georgia,serif;">SmartKitchen</div>
            <div style="color:#8D6E63;font-size:0.95rem;">AI-powered ingredient identification & Indian recipes</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        st.subheader("Welcome back!")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In", type="primary", use_container_width=True, key="login_btn"):
            if not email or not password:
                st.warning("Please fill in both fields.")
            else:
                try:
                    res = sb.auth.sign_in_with_password({"email": email, "password": password})
                    profile = db_get_or_create_profile(res.user)
                    st.session_state["user"] = profile
                    st.session_state["auth_user"] = res.user
                    st.rerun()
                except Exception as e:
                    st.error(f"Sign in failed — check your email and password.")

    with tab_signup:
        st.subheader("Join SmartKitchen")
        name = st.text_input("Your name", key="signup_name", placeholder="e.g. Dhwani")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password (min 6 chars)", type="password", key="signup_pass")
        password2 = st.text_input("Confirm password", type="password", key="signup_pass2")
        if st.button("Create Account", type="primary", use_container_width=True, key="signup_btn"):
                if not name or not email or not password:
                    st.warning("Please fill in all fields.")
                elif password != password2:
                    st.error("Passwords don't match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        res = sb.auth.sign_up({
                            "email": email,
                            "password": password,
                            "options": {"data": {"full_name": name}}
                        })
                        profile = db_get_or_create_profile(res.user)
                        st.session_state["user"] = profile
                        st.session_state["auth_user"] = res.user
                        st.success(f"Welcome, {name}! 🎉")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Signup failed: {e}")

def require_auth() -> bool:
    """Returns True if user is logged in, otherwise shows auth page."""
    if not get_user():
        auth_page()
        return False
    return True

# ── Pages ──────────────────────────────────────
def page_home():
    user = get_user()

    st.title("🍽️ SmartKitchen")
    if user:
        st.write(f"Welcome back, **{user['name']}**! 👋")
    else:
        st.write("AI-powered ingredient identification & Indian recipe suggestions.")
    st.divider()

    st.subheader("How It Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**📷 Capture**\n\nUpload a photo or use camera.")
    with c2:
        st.info("**🧠 Identify**\n\nAI classifies your ingredient instantly.")
    with c3:
        st.info("**🍛 Cook**\n\nGet matching Indian recipes.")
    with c4:
        st.info("**❤️ Save**\n\nFavourite recipes saved to your profile.")

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

    st.divider()
    st.caption("Built with TensorFlow · MobileNet · Streamlit · Supabase  |  By Dhwani Chande")


def page_identify():
    user = get_user()
    recipes_df = load_recipes()

    st.title("🥦 Identify Ingredients")
    st.write("Upload photos or pick manually — then find recipes using all your ingredients.")
    st.divider()

    # ── Ingredient basket ──
    if "basket" not in st.session_state:
        st.session_state["basket"] = []

    tab1, tab2, tab3 = st.tabs(["📤 Upload Photo", "📷 Camera", "☑️ Pick Manually"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload one or more images", type=["jpg","jpeg","png"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("Identify All", type="primary"):
            for f in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                with st.spinner(f"Identifying {f.name}…"):
                    label, conf = classify_image(save_path)
                if label.lower() not in [b.lower() for b in st.session_state["basket"]]:
                    st.session_state["basket"].append(label)
                    db_log_search(user["id"], label)
                st.success(f"**{label}** ({conf:.0f}% confident)")

    with tab2:
        cam = st.camera_input("Take a photo")
        if cam and st.button("Identify Photo", type="primary"):
            save_path = os.path.join(UPLOAD_DIR, "cam.jpg")
            with open(save_path, "wb") as out:
                out.write(cam.getbuffer())
            with st.spinner("Identifying…"):
                label, conf = classify_image(save_path)
            if label.lower() not in [b.lower() for b in st.session_state["basket"]]:
                st.session_state["basket"].append(label)
                db_log_search(user["id"], label)
            st.success(f"**{label}** ({conf:.0f}% confident)")

    with tab3:
        st.write("Pick ingredients you have:")
        all_labels = sorted(LABELS.values())
        cols = st.columns(4)
        selected_manual = []
        for i, label in enumerate(all_labels):
            with cols[i % 4]:
                if st.checkbox(f"{EMOJI.get(label,'🌿')} {label.title()}", key=f"chk_{label}"):
                    selected_manual.append(label.capitalize())
        if selected_manual and st.button("Add to basket", type="primary"):
            for item in selected_manual:
                if item.lower() not in [b.lower() for b in st.session_state["basket"]]:
                    st.session_state["basket"].append(item)
            st.rerun()

    # ── Basket ──
    st.divider()
    st.subheader("🧺 Your Ingredient Basket")

    if not st.session_state["basket"]:
        st.info("No ingredients yet — upload a photo or pick manually above.")
        return

    # Show basket with remove buttons
    basket_cols = st.columns(min(len(st.session_state["basket"]), 6))
    to_remove = None
    for i, item in enumerate(st.session_state["basket"]):
        with basket_cols[i % 6]:
            st.write(f"{EMOJI.get(item.lower(),'🌿')} **{item}**")
            kcal = CALORIES.get(item.lower(), "?")
            st.caption(f"{kcal} kcal/100g")
            if st.button("✕", key=f"rm_{item}"):
                to_remove = item

    if to_remove:
        st.session_state["basket"].remove(to_remove)
        st.rerun()

    if st.button("Clear all", key="clear_basket"):
        st.session_state["basket"] = []
        st.rerun()

    # Show nutrition for first identified item
    first = st.session_state["basket"][0].lower() if st.session_state["basket"] else None
    nut = NUTRITION.get(first) if first else None
    if nut:
        st.divider()
        st.write(f"**Nutrition for {first.title()} (per 100g)**")
        ncols = st.columns(len(nut))
        for col, (k, v) in zip(ncols, nut.items()):
            col.metric(k, v)

    # ── Recipe results ──
    st.divider()
    st.subheader("🍛 Recipes With Your Ingredients")

    diet_filter = st.selectbox(
        "Filter by diet",
        ["All"] + DIET_OPTIONS[1:],
        index=DIET_OPTIONS.index(user.get("diet_preference","All")) if user.get("diet_preference") in DIET_OPTIONS else 0,
        key="identify_diet"
        # ⚠️ Session-only filter — does not save to profile
    )

    recs = recommend_recipes(st.session_state["basket"], recipes_df, diet_filter)

    if recs.empty:
        st.warning("No recipes found. Try a different diet filter or add more ingredients.")
        return

    # Show match summary
    total_ings = len(st.session_state["basket"])
    perfect = len(recs[recs["_match_count"] == total_ings])
    st.caption(f"Found **{len(recs)} recipes** · {perfect} use all your ingredients")

    for _, row in recs.iterrows():
        match = int(row.get("_match_count", 0))
        total = int(row.get("_total_ingredients", total_ings))
        # Add match badge to key_prefix so it shows in expander
        match_label = f"{match}/{total} ingredients" if total > 1 else ""
        show_recipe_expander(row.to_dict(), user["id"],
                             key_prefix=f"identify_{match}",
                             match_label=match_label)


def page_recipes():
    user = get_user()
    recipes_df = load_recipes()

    st.title("📖 Browse Recipes")
    st.write("Filter and search over 1,000 Indian recipes.")
    st.divider()

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        saved_diet = user.get("diet_preference", "All")
        diet_idx = DIET_OPTIONS.index(saved_diet) if saved_diet in DIET_OPTIONS else 0
        diet_sel = st.selectbox("Diet", DIET_OPTIONS, index=diet_idx)
        # ⚠️ Do NOT auto-save here — diet preference is only saved in My Profile
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
        show_recipe_expander(row.to_dict(), user["id"], key_prefix="browse")

    st.caption("SmartKitchen · By Dhwani Chande")


def page_profile():
    user = get_user()
    st.title(f"👤 {user['name']}'s Profile")
    st.divider()

    tab1, tab2 = st.tabs(["❤️ Saved Recipes", "🕐 Search History"])

    with tab1:
        favs = db_get_favourites(user["id"])
        if not favs:
            st.info("No saved recipes yet. Hit ❤️ Save on any recipe to add it here!")
        else:
            st.write(f"**{len(favs)} saved recipes**")
            for row in favs:
                col1, col2 = st.columns([5, 1])
                with col1:
                    show_recipe_expander(row, user_id=None, key_prefix="profile")
                with col2:
                    if st.button("🗑️", key=f"del_{row['id']}", help="Remove"):
                        db_remove_favourite(user["id"], row["recipe_name"])
                        st.rerun()

    with tab2:
        history = db_get_history(user["id"])
        if not history:
            st.info("No search history yet.")
        else:
            st.write(f"**Last {len(history)} searches**")
            cols = st.columns(4)
            for i, h in enumerate(history):
                ing = h["ingredient"]
                cols[i % 4].write(f"{EMOJI.get(ing.lower(),'🌿')} {ing}  \n_{h['searched_at'][:10]}_")

    st.divider()
    st.subheader("⚙️ Preferences")
    saved_diet = user.get("diet_preference", "All")
    diet_idx = DIET_OPTIONS.index(saved_diet) if saved_diet in DIET_OPTIONS else 0
    new_diet = st.selectbox("Default diet filter", DIET_OPTIONS, index=diet_idx)
    if st.button("Save preference", type="primary"):
        db_update_diet(user["id"], new_diet)
        st.session_state["user"]["diet_preference"] = new_diet
        st.success("Saved!")

    st.divider()
    if st.button("Sign out", type="secondary"):
        try:
            get_supabase().auth.sign_out()
        except Exception:
            pass
        for key in ["user", "auth_user", "basket"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Main ───────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SmartKitchen",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Global styles ─────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Lato', sans-serif !important; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

    /* Warm parchment background */
    .stApp { background-color: #FFF8EF !important; }

    /* Main block */
    .block-container { padding-top: 2rem !important; max-width: 1050px !important; }

    /* Sidebar — dark walnut */
    [data-testid="stSidebar"] {
        background-color: #2C1A0E !important;
        border-right: 1px solid #4a2c1a !important;
    }
    [data-testid="stSidebar"] * { color: #F5DEB3 !important; }
    [data-testid="stSidebar"] h1 { color: #FFDAA0 !important; font-family: 'Playfair Display', serif !important; }
    [data-testid="stSidebar"] p { color: #F5DEB3 !important; }
    [data-testid="stSidebar"] span { color: #F5DEB3 !important; }
    [data-testid="stSidebar"] label { color: #F5DEB3 !important; }
    [data-testid="stSidebar"] .stCaption p { color: #C8A882 !important; }
    [data-testid="stSidebar"] small { color: #C8A882 !important; }
    [data-testid="stSidebar"] a { color: #FFDAA0 !important; }
    /* Radio options */
    [data-testid="stSidebar"] .stRadio label p { color: #F5DEB3 !important; font-size: 0.95rem !important; }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #F5DEB3 !important; }

    /* Headings */
    h1 { color: #2C1A0E !important; font-weight: 700 !important; }
    h2 { color: #3E2723 !important; font-weight: 600 !important; }
    h3 { color: #5D4037 !important; }
    p, .stMarkdown p { color: #5D4037 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFAF3 !important;
        border: 1px solid #E8D5B7 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 8px rgba(160,82,45,0.08) !important;
    }
    [data-testid="stMetricLabel"] { color: #8D6E63 !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #2C1A0E !important; font-weight: 700 !important; }

    /* Expanders — recipe cards */
    [data-testid="stExpander"] {
        background: #FFFAF3 !important;
        border: 1px solid #E8D5B7 !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 2px 6px rgba(160,82,45,0.06) !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #A0522D !important;
        box-shadow: 0 4px 12px rgba(160,82,45,0.15) !important;
    }
    [data-testid="stExpander"] summary {
        color: #2C1A0E !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stExpander"] p { color: #6D4C41 !important; font-size: 0.88rem !important; line-height: 1.7 !important; }
    [data-testid="stExpander"] strong { color: #A0522D !important; }
    [data-testid="stExpander"] a { color: #A0522D !important; }

    /* Primary buttons — cinnamon */
    .stButton > button[kind="primary"] {
        background: #A0522D !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: background 0.2s, transform 0.15s, box-shadow 0.15s !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #7B3F22 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(160,82,45,0.35) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #FFF5E6 !important;
        border: 1px solid #D7B896 !important;
        border-radius: 8px !important;
        color: #8D6E63 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #FFEEDD !important;
        border-color: #A0522D !important;
        color: #A0522D !important;
    }

    /* Text inputs */
    .stTextInput input {
        background: #FFFAF3 !important;
        border: 1px solid #D7B896 !important;
        border-radius: 8px !important;
        color: #2C1A0E !important;
    }
    .stTextInput input:focus {
        border-color: #A0522D !important;
        box-shadow: 0 0 0 3px rgba(160,82,45,0.12) !important;
    }
    .stTextInput label { color: #8D6E63 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #FFFAF3 !important;
        border: 1px solid #D7B896 !important;
        border-radius: 8px !important;
        color: #2C1A0E !important;
    }
    .stSelectbox label { color: #8D6E63 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: #FFF5E6 !important;
        border: 2px dashed #D7B896 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #A0522D !important;
        background: #FFEEDD !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #F5E6D3 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid #E8D5B7 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: #8D6E63 !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #A0522D !important;
        color: #ffffff !important;
    }

    /* Alerts */
    .stAlert { border-radius: 10px !important; }

    /* Divider */
    hr { border-color: #E8D5B7 !important; }

    /* Caption */
    .stCaptionContainer p { color: #A0856A !important; }

    /* Checkbox */
    .stCheckbox label { color: #5D4037 !important; }

    /* Radio in sidebar */
    [data-testid="stSidebar"] .stRadio label { color: #F5DEB3 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #FFF8EF; }
    ::-webkit-scrollbar-thumb { background: #D7B896; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #A0522D; }
    </style>
    """, unsafe_allow_html=True)

    # ── Gate: show auth page if not logged in ──
    if not get_user():
        # Hide sidebar entirely on auth page
        st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        </style>
        """, unsafe_allow_html=True)
        auth_page()
        return

    # ── Logged in: show full app ───────────────
    PAGES = {
        "🏠 Home":       page_home,
        "🥦 Identify":   page_identify,
        "📖 Recipes":    page_recipes,
        "👤 My Profile": page_profile,
    }

    user = get_user()
    with st.sidebar:
        st.title("🍽️ SmartKitchen")
        st.caption("AI Recipe Assistant")
        st.divider()
        st.write(f"👤 **{user['name']}**")
        auth_user = st.session_state.get("auth_user")
        if auth_user:
            st.caption(f"✉️ {auth_user.email}")
        diet = user.get("diet_preference","All")
        if diet != "All":
            st.caption(f"🥗 {diet}")
        st.divider()
        sel = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("TensorFlow · MobileNet · Streamlit · Supabase")

    PAGES[sel]()


if __name__ == "__main__":
    main()
