# SmartKitchen 🍽️

SmartKitchen is a machine learning–based application that focuses on **food recognition and classification**, with a primary emphasis on **fruit and vegetable identification** using deep learning. The project leverages a **MobileNet-based CNN model** to classify food items and can be extended for smart kitchen, nutrition, or food-management use cases.

---

## 🚀 Project Overview

The goal of SmartKitchen is to:

* Automatically **classify fruits and vegetables from images**
* Demonstrate the use of **transfer learning (MobileNet)** for image classification
* Provide a foundation for smart kitchen applications such as food tracking, inventory management, or nutrition analysis

The repository contains model training code, datasets, and an application script to run predictions using a trained model.

---

## 🧠 Key Features

* 📷 Image-based fruit & vegetable classification
* 🧩 Transfer learning using **MobileNet**
* 💾 Pre-trained model saved as `.h5`
* 📊 Dataset support via CSV and image folders
* 🐍 Python-based implementation

---

## 📁 Repository Structure

```text
SmartKitchen/
│
├── App.py                                # Main application script for inference
├── FV.h5                                 # Trained MobileNet model
├── Fruit_Veg_Classification_Mobilenet.ipynb  # Model training & experimentation notebook
├── IndianFoodDatasetCSV.csv              # Food-related dataset (CSV format)
├── dataset/                              # Image dataset directory
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation
```

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MobileNet (Transfer Learning)**
* **NumPy & Pandas**
* **Jupyter Notebook**

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/Dhwani-Chande/SmartKitchen.git
cd SmartKitchen
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option 1: Run the Application

```bash
python App.py
```

This will load the trained model (`FV.h5`) and perform predictions based on the input logic defined in `App.py`.

### Option 2: Explore / Train the Model

Open the Jupyter Notebook:

```bash
jupyter notebook Fruit_Veg_Classification_Mobilenet.ipynb
```

Use this notebook to:

* Understand data preprocessing
* Train or fine-tune the MobileNet model
* Evaluate performance

---

## 📊 Dataset

* The project uses image data stored in the `dataset/` directory
* Additional structured food data is available in `IndianFoodDatasetCSV.csv`

You can replace or expand the dataset to improve accuracy or support more food categories.

---

## 🌱 Future Improvements

* Add real-time image capture (camera integration)
* Expand classification beyond fruits & vegetables
* Integrate nutritional information per item
* Build a web or mobile interface
* Improve model accuracy with more training data

---

## 👩‍💻 Author

**Dhwani Chande**
📌 GitHub: [https://github.com/Dhwani-Chande](https://github.com/Dhwani-Chande)
📌 LinkedIn: [https://www.linkedin.com/in/dhwani-chande29/](https://www.linkedin.com/in/dhwani-chande29/)

---

## 📄 License

This project is intended for **educational and research purposes**. You are free to use and modify it with proper attribution.

---

⭐ If you found this project helpful, feel free to star the repository!
