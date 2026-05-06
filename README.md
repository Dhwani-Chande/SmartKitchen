<div align="center">

# 🍽️ SmartKitchen

**AI-powered ingredient identification & Indian recipe recommendations**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-smartkitchen.streamlit.app-FF4B4B?style=for-the-badge)](https://smartkitchen.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)

*Point your camera at any fruit or vegetable — get instant AI identification, calories, nutrition data, and personalised Indian recipe suggestions.*

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 **AI Identification** | Upload a photo or use live camera — MobileNet classifies 36 fruits & vegetables with confidence score |
| 🧺 **Multi-Ingredient Basket** | Add multiple ingredients via photo, camera, or manual selection — recipes scored against all of them |
| 🍛 **Smart Recommendations** | Score-based matching — recipes ranked by how many of your ingredients they use, with match % shown |
| 🔢 **Nutrition Data** | Calories + full macros (carbs, protein, fat, fiber, vitamins) for every identified ingredient |
| 📖 **Recipe Browser** | 6,800+ Indian recipes filterable by diet, course, and keyword search |
| 👤 **User Accounts** | Email + password auth via Supabase — sign up, save favourites, track search history |
| ❤️ **Favourites** | Save any recipe — persists across sessions and devices |
| 🥗 **Diet Preferences** | Set your diet once (Vegan, Vegetarian, Diabetic Friendly, etc.) — auto-applied everywhere |

---

## 🎯 How It Works

```
📷 Upload Photo  ──►  🧠 MobileNet CNN  ──►  🏷️ Label + Confidence
                                                      │
                                               🧺 Ingredient Basket
                                                      │
                                        ┌─────────────┴─────────────┐
                                        │   Score-Based Matching     │
                                        │   recipe_score = Σ(ing ∈  │
                                        │   recipe_ingredients)      │
                                        └─────────────┬─────────────┘
                                                      │
                                        🍛 Ranked Recipe Results
                                           with match % badges
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | MobileNet V1 (transfer learning) |
| Input | 224 × 224 × 3 RGB |
| Output | 36 classes (15 fruits + 21 vegetables) |
| Framework | TensorFlow 2.15 / Keras |
| Training | Custom dataset, fine-tuned final layers |
| Format | `.h5` (Keras SavedModel) |

Training notebook: [`Fruit_Veg_Classification_Mobilenet.ipynb`](./Fruit_Veg_Classification_Mobilenet.ipynb)

---

## 🗂️ Project Structure

```
SmartKitchen/
├── App.py                                    # Main Streamlit app (auth, UI, ML inference)
├── FV.h5                                     # Pre-trained MobileNet model weights
├── Fruit_Veg_Classification_Mobilenet.ipynb  # Model training & evaluation notebook
├── IndianFoodDatasetCSV.csv                  # 6,800+ Indian recipes dataset
├── dataset/                                  # Training image data (36 classes)
├── upload_images/                            # Temp storage for uploaded/captured images
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone

```bash
git clone https://github.com/Dhwani-Chande/SmartKitchen.git
cd SmartKitchen
```

### 2. Virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Supabase

Create a `.env` file or set these directly in `App.py`:

```
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_anon_key
```

Create these tables in your Supabase project:

```sql
create table users (
  id uuid primary key,
  name text not null,
  diet_preference text default 'All',
  created_at timestamp default now()
);

create table favourites (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references users(id) on delete cascade,
  recipe_name text not null,
  diet text, course text,
  ingredients text, instructions text, url text,
  saved_at timestamp default now()
);

create table search_history (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references users(id) on delete cascade,
  ingredient text not null,
  searched_at timestamp default now()
);
```

### 5. Run

```bash
streamlit run App.py
```

---

## 📊 Dataset

### 🥦 Image Dataset (Model Training)
[![Kaggle](https://img.shields.io/badge/Kaggle-Fruit_&_Vegetable_Recognition-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

Training images sourced from the **Fruit and Vegetable Image Recognition** dataset on Kaggle by [kritikseth](https://www.kaggle.com/kritikseth). Organised into 36 class folders — one per fruit/vegetable.

To download:
```bash
kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition
```
Or download directly from the [Kaggle page](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition).

### 🍛 Recipe Dataset
**`IndianFoodDatasetCSV.csv`** — 6,800+ Indian recipes with translated names, ingredients, instructions, diet tags, course tags, and source URLs.

**Supported ingredients:**

🍎 Apple · 🍌 Banana · 🫚 Beetroot · 🫑 Bell Pepper · 🥬 Cabbage · 🫑 Capsicum · 🥕 Carrot · 🥦 Cauliflower · 🌶️ Chilli Pepper · 🌽 Corn · 🥒 Cucumber · 🍆 Eggplant · 🧄 Garlic · 🫚 Ginger · 🍇 Grapes · 🌶️ Jalepeno · 🥝 Kiwi · 🍋 Lemon · 🥬 Lettuce · 🥭 Mango · 🧅 Onion · 🍊 Orange · 🫑 Paprika · 🍐 Pear · 🫛 Peas · 🍍 Pineapple · 🍎 Pomegranate · 🥔 Potato · 🌱 Raddish · 🫘 Soy Beans · 🥬 Spinach · 🌽 Sweetcorn · 🍠 Sweetpotato · 🍅 Tomato · 🫚 Turnip · 🍉 Watermelon

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **ML Model** | TensorFlow / Keras / MobileNet |
| **Database** | Supabase (PostgreSQL) |
| **Auth** | Supabase Auth (email + password) |
| **Hosting** | Streamlit Community Cloud |
| **Language** | Python 3.11 |

---

## 🌱 Roadmap

- [x] AI fruit & vegetable classification
- [x] Multi-ingredient basket
- [x] Score-based recipe matching
- [x] User authentication
- [x] Favourites & search history
- [x] Nutrition data
- [ ] Google OAuth login
- [ ] Password reset flow
- [ ] Expand to 100+ ingredient classes
- [ ] Nutrition API (replace static table)
- [ ] Mobile PWA

---

## 👩‍💻 Author

**Dhwani Chande**

[![GitHub](https://img.shields.io/badge/GitHub-Dhwani--Chande-181717?style=flat&logo=github)](https://github.com/Dhwani-Chande)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-dhwani--chande29-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/dhwani-chande29/)

---

## 📄 License

This project is open for educational and research use. Free to use with attribution.
