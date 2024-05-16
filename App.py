import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.models import load_model
import cv2
import pandas as pd
import os

large_font_style = "style='font-size:24px;'"

page_by_img = """
<style>
[data-testid="stAppViewContainer"]{
    background-image: url("https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L2lzOTc1Ny1pbWFnZS1rd3Z5ZHlvdy5qcGc.jpg");
    background-size: cover;
}

[data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: #001F3F; 
    border-radius: 10px;
    opacity: 0.9;

[data-testid="stVerticalBlock"] {
        font-size: 18px;
        color: #001F3F;
    }
}

</style>
"""
st.markdown(page_by_img, unsafe_allow_html=True)

# Load dataset and model
recipes_df = pd.read_csv('/Users/dhwanichande/Desktop/Project/Recipe Dataset/IndianFoodDatasetCSV.csv')
model = load_model('FV.h5')

# Define labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

# Fruits and Vegetables
fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Color ranges for masking
red_lower = (17, 15, 100)
red_upper = (180, 255, 255)

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't fetch the Calories")
        print(e)
        return None

def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize to match the input size of your model
    img_array = np.array(img)
    img_array = cv2.cvtColor(np.uint8(img_array), cv2.COLOR_RGB2BGR)  # Convert PIL image to BGR format
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    answer = model.predict(img_array)
    y_class = answer.argmax(axis=-1)
    prediction = labels[y_class[0]]
    
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(np.uint8(img_array[0]), cv2.COLOR_BGR2HSV)

    # Create a mask for the target color range
    mask = cv2.inRange(hsv_image, red_lower, red_upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected fruits (optional)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_array[0], (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the final image with detected fruits (optional)
    #cv2.imwrite("detected_fruits.jpg", img_array[0])
    #st.image(hsv_image, use_column_width=False)

    return prediction.capitalize()

def recommend_recipes(prediction):
    predicted_recipes = recipes_df[recipes_df['TranslatedRecipeName'].str.contains(prediction, case=False)]
    return predicted_recipes.head(5)  # Get top 5 recommended recipes

def display_recommendations(recipes):
    css1 = """
    <style>
    [data-testid="stVerticalBlock"] {
        font-size: 18px;
        color: #001F3F;
    }
    </style>
    """
    st.markdown(css1, unsafe_allow_html=True)
    st.markdown("<h1 style='color: #001F3F;'>Recommended Recipes</h1>", unsafe_allow_html=True)
    #st.subheader("Recommended Recipes")
    for index, recipe in recipes.iterrows():
        st.write(f"**{recipe['TranslatedRecipeName']}**")
        st.write(f"**Ingredients: {recipe['TranslatedIngredients']}**")
        st.write(f"**Instructions: {recipe['TranslatedInstructions']}**")
        st.write(f"**Link: {recipe['URL']}**")
        st.markdown("<p style='font-size: 18px; color: #001F3F;'><b>______________________________________________________________________________</b></p>", unsafe_allow_html=True)

def camera():
    cam = cv2.VideoCapture(0)  # 0 is the default camera

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from the webcam.")
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(img, channels="RGB")

        img_counter = 2
        save_folder = '/Users/dhwanichande/Desktop/Project/upload_images'

        try:
            if st.button("Capture", key='webcam_capture_button'):            
                save_image_path = os.path.join(save_folder, f"webcam_capture_{img_counter}.png")
                cv2.imwrite(save_image_path, frame)
                st.success(f"Image captured and saved as {save_image_path}")
                img_counter += 1

                result = prepare_image(save_image_path)
                prediction = result.lower().capitalize()

                recommended_recipes = recommend_recipes(prediction)
                display_recommendations(recommended_recipes)

                if result:
                    category = "Vegetable" if result in vegetables else "Fruit"
                    st.info(f'**Category : {category}**')
                    st.success(f"**Predicted : {result}**")
                    cal = fetch_calories(result)
                    if cal:
                        st.warning(f'**{cal} (100 grams)**')

            # Check if the "Stop" button is clicked
            if st.button("Stop", key='webcam_stop_button'):           
                break

        except AttributeError as e:
            if "DuplicateWidgetID" in str(e):
                st.warning("A widget with the same ID already exists. Skipping...")
            else:
                st.error(f"An error occurred: {e}")

    cam.release()
    cv2.destroyAllWindows()


def main():
    pages = {
        "Home": home,
        "Recipes": recipes,
        "Ingredients": ingredients,
        "Camera": camera
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()


def home():
    st.markdown("<p style='font-size:80px; color: #001F3F;'><b>Smart Refrigerator</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:24px; color: #001F3F;'><b>About Us</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; color: #001F3F;'><b>Long work hours leave professionals exhausted, making healthy meals a daunting task. They lack a clear picture of what ingredients they have on hand, leading to wasted food or unhealthy take-out choices due to indecisiveness about what to cook. This cycle of exhaustion and unhealthy eating creates a vicious loop, further draining their energy and motivation to cook nutritious meals. </b></p>", unsafe_allow_html=True)
    st.write("\n\n")
    st.image("Smart.png", use_column_width=False,width=300)  
    st.write("\n\n")
    st.write("\n\n")
    st.markdown("<footer style='font-size:24px; color: #001F3F;'><b>Contact Us</b></footer>", unsafe_allow_html=True)
    st.markdown("<footer style='font-size:15px; color: #001F3F;'><b>Mobile No : 7922918326</b></footer>", unsafe_allow_html=True)
    st.markdown("<footer style='font-size:15px; color: #001F3F;'><b>Mail ID : smartref@gmail.com</b></footer>", unsafe_allow_html=True)


def recipes():
    css = """
    <style>
    [data-testid="stVerticalBlock"] {
        font-size: 18px;
        color: #001F3F;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.markdown("<h1 style='color: #001F3F;'>Recipes</h1>", unsafe_allow_html=True)
    # Extract unique values from the "Diet" column
    all_diet_preferences = recipes_df['Diet'].unique()
    all_courses = recipes_df['Course'].unique()

    # Add widget for user preferences
    st.markdown("<p style='font-size: 18px; color: #001F3F;'><b>Select Your Diet Preferences:</b></p>", unsafe_allow_html=True)
    preferences = st.selectbox('', [''] + list(all_diet_preferences), key="diet_selectbox")
    st.markdown("<p style='font-size: 18px; color: #001F3F;'><b>Select Your Course Preferences:</b></p>", unsafe_allow_html=True)
    courses = st.selectbox("", [''] + list(all_courses), key="course_selectbox")

    # Display TranslatedRecipeName, TranslatedIngredients, and TranslatedInstructions
    filtered_recipes = recipes_df.copy()
    if preferences:
        filtered_recipes = filtered_recipes[filtered_recipes['Diet'] == preferences]
    
    # Apply filters based on user course preference
    if courses:
        filtered_recipes = filtered_recipes[filtered_recipes['Course'] == courses]

    # Randomly select 10 rows from the filtered DataFrame
    random_recipes = filtered_recipes.sample(n=10, random_state=1)
    for index, row in random_recipes.iterrows():
        st.write(f"**{row['TranslatedRecipeName']}**")
        st.write(f"**Ingredients: {row['TranslatedIngredients']}**")
        st.write(f"**Instructions: {row['TranslatedInstructions']}**")
        st.write(f"**Link: {row['URL']}**")
        st.markdown("<p style='font-size: 18px; color: #001F3F;'><b>______________________________________________________________________________</b></p>", unsafe_allow_html=True)

def ingredients():

    st.markdown("<h1 style='color: #001F3F;'>Ingredients</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #001F3F;'>Fruits🍍-Vegetable🍅 Classification</h2>", unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = prepare_image(save_image_path)
        prediction = result.lower().capitalize()

        if result:
            category = "Vegetable" if result in vegetables else "Fruit"
            st.info(f'**Category : {category}**')
            st.success(f"**Predicted : {result}**")
            cal = fetch_calories(result)
            if cal:
                st.warning(f'**{cal} (100 grams)**')
            recommended_recipes = recommend_recipes(prediction)
            display_recommendations(recommended_recipes)

if __name__ == "__main__":
    main()
