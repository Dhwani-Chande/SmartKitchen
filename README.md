# 🍽️ SmartKitchen

SmartKitchen is a machine-learning–powered web app that **identifies fruits and vegetables from photos** and instantly recommends **Indian recipes** based on what you have on hand.

Built with TensorFlow/Keras (MobileNet transfer learning) and deployed as a Streamlit application.

---

## 🚀 Features

- 📷 **Image classification** — upload a photo or use your webcam to identify a fruit or vegetable (36 classes)
- 🧠 **MobileNet transfer learning** — fast, accurate CNN inference from a pre-trained `.h5` model
- 🍛 **Recipe recommendations** — matched by ingredient name AND ingredient list from an Indian food dataset
- 📊 **Recipe browser** — filter by diet type and course, or search by keyword
- 🔢 **Calorie lookup** — live fetch with static fallback table so it never silently fails
- 📱 **Responsive UI** — Streamlit layout with sidebar navigation

---

## 📁 Project Structure

```
SmartKitchen/
├── App.py                                    # Main Streamlit application
├── FV.h5                                     # Pre-trained MobileNet model
├── Fruit_Veg_Classification_Mobilenet.ipynb  # Training notebook
├── IndianFoodDatasetCSV.csv                  # Indian recipe dataset
├── dataset/                                  # Training image data
├── upload_images/                            # Temp folder for uploaded/captured images
├── requirements.txt                          # Minimal dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhwani-Chande/SmartKitchen.git
cd SmartKitchen
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon users:** replace `tensorflow` with `tensorflow-macos` in requirements.txt

### 4. Run the app

```bash
streamlit run App.py
```

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Architecture | MobileNet (transfer learning) |
| Input size | 224 × 224 × 3 |
| Output classes | 36 (fruits + vegetables) |
| Framework | TensorFlow / Keras |
| Saved format | `.h5` |

Training code and data preprocessing are in `Fruit_Veg_Classification_Mobilenet.ipynb`.

---

## 📊 Dataset

- **Image data**: stored in `dataset/` directory, organised by class
- **Recipe data**: `IndianFoodDatasetCSV.csv` — columns used:
  - `TranslatedRecipeName`
  - `TranslatedIngredients`
  - `TranslatedInstructions`
  - `Diet`
  - `Course`
  - `URL`

---

## 🌱 Roadmap

- [ ] Nutritional API integration (replace Google scraping)
- [ ] Multi-ingredient detection (detect several items in one photo)
- [ ] Expand to 100+ ingredient classes
- [ ] User pantry tracker (remember what you have)
- [ ] Mobile-optimised UI / PWA

---

## 👩‍💻 Author

**Dhwani Chande**  
GitHub: [Dhwani-Chande](https://github.com/Dhwani-Chande)  
LinkedIn: [dhwani-chande29](https://www.linkedin.com/in/dhwani-chande29/)

---

## 📄 License

Educational and research use. Free to use with attribution.
