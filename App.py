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

def db_get_or_create_user(name: str) -> dict:
    sb = get_supabase()
    res = sb.table("users").select("*").eq("name", name).execute()
    if res.data:
        return res.data[0]
    new = sb.table("users").insert({"name": name, "diet_preference": "All"}).execute()
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

def recommend_recipes(ingredients: list, recipes_df: pd.DataFrame, diet: str = "All", n: int = 10) -> pd.DataFrame:
    if not ingredients:
        return pd.DataFrame()
    mask = pd.Series([True] * len(recipes_df))
    for ing in ingredients:
        p = ing.lower()
        mask &= (
            recipes_df["TranslatedRecipeName"].str.lower().str.contains(p, na=False) |
            recipes_df["TranslatedIngredients"].str.lower().str.contains(p, na=False)
        )
    if diet and diet != "All":
        mask &= recipes_df["Diet"] == diet
    return recipes_df[mask].drop_duplicates("TranslatedRecipeName").head(n)

def format_ingredients(raw: str) -> list:
    return [i.strip() for i in raw.split(",") if i.strip()] if raw else []

def format_instructions(raw: str) -> list:
    if not raw:
        return []
    steps = [s.strip() for s in re.split(r"(?<=[.!?]) +(?=[A-Z])", raw) if s.strip()]
    return steps if len(steps) > 1 else [raw.strip()]

def show_recipe_expander(row: dict, user_id: str = None):
    name   = row.get("TranslatedRecipeName") or row.get("recipe_name", "Recipe")
    diet   = row.get("Diet") or row.get("diet", "")
    course = row.get("Course") or row.get("course", "")
    url    = row.get("URL") or row.get("url", "")
    ing    = row.get("TranslatedIngredients") or row.get("ingredients", "")
    ins    = row.get("TranslatedInstructions") or row.get("instructions", "")
    tag    = " · ".join(filter(None, [diet, course]))
    title  = f"🍳 {name}" + (f"  —  {tag}" if tag else "")

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
                st.link_button("🔗 Full Recipe", url, use_container_width=True)
            if user_id:
                fav_key = f"fav_{name[:30]}"
                if st.button("❤️ Save", key=fav_key, use_container_width=True):
                    saved = db_save_favourite(user_id, row if isinstance(row, dict) else row.to_dict())
                    st.toast("Saved to favourites! ❤️" if saved else "Already in favourites")

# ── Session helpers ────────────────────────────
def get_user():
    return st.session_state.get("user")

def require_name() -> bool:
    if not get_user():
        st.info("👋 Enter your name to get started — no password needed.")
        name = st.text_input("Your name", placeholder="e.g. Dhwani")
        if st.button("Let's go →", type="primary"):
            if name.strip():
                user = db_get_or_create_user(name.strip())
                st.session_state["user"] = user
                st.rerun()
            else:
                st.warning("Please enter a name.")
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
    if not require_name():
        return

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
    )

    recs = recommend_recipes(st.session_state["basket"], recipes_df, diet_filter)

    if recs.empty:
        st.warning("No recipes found for this combination. Try removing an ingredient or changing diet filter.")
        # Fallback: show recipes for first ingredient only
        if len(st.session_state["basket"]) > 1:
            st.write(f"Showing recipes for **{st.session_state['basket'][0]}** only:")
            recs = recommend_recipes([st.session_state["basket"][0]], recipes_df, diet_filter)
    
    st.caption(f"{len(recs)} recipes found")
    for _, row in recs.iterrows():
        show_recipe_expander(row.to_dict(), user["id"])


def page_recipes():
    if not require_name():
        return

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
        if diet_sel != saved_diet:
            db_update_diet(user["id"], diet_sel)
            st.session_state["user"]["diet_preference"] = diet_sel
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
        show_recipe_expander(row.to_dict(), user["id"])

    st.caption("SmartKitchen · By Dhwani Chande")


def page_profile():
    if not require_name():
        return

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
                    show_recipe_expander(row, user_id=None)
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
    if st.button("Sign out"):
        del st.session_state["user"]
        if "basket" in st.session_state:
            del st.session_state["basket"]
        st.rerun()


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
        "📖 Recipes":    page_recipes,
        "👤 My Profile": page_profile,
    }

    with st.sidebar:
        st.title("🍽️ SmartKitchen")
        st.caption("AI Recipe Assistant")
        user = get_user()
        if user:
            st.caption(f"👤 {user['name']}")
            diet = user.get("diet_preference","All")
            if diet != "All":
                st.caption(f"🥗 {diet}")
        st.divider()
        sel = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("TensorFlow · MobileNet · Streamlit · Supabase")

    PAGES[sel]()


if __name__ == "__main__":
    main()
