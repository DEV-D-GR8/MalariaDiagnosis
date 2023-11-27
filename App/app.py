import streamlit as st
import glob
import random
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

model_path = 'App/test9576.h5'
model = load_model(model_path)

parasitized_folder = 'App/CellImages/Parasitized'
uninfected_folder = 'App/CellImages/Uninfected'

def main():
    st.title("Malaria Cell Classification")
    st.sidebar.title("Cell Classification")

    option = st.sidebar.radio("Select an option:", ("Browse File", "Random"))

    if option == "Browse File":
        uploaded_file = st.sidebar.file_uploader("Choose a cell image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            img_array = np.array(image.resize((224, 224)))
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = "Parasitized" if prediction[0][0] < 0.5 else "Uninfected"

            st.sidebar.header("Cell Classification Result")
            st.sidebar.subheader(f"Predicted Class: {result}")
            st.sidebar.subheader("Prediction Probability:")
            st.sidebar.write(f"{result}: {prediction[0][0]:.4f}")

    elif option == "Random":
        parasitized_images = glob.glob(f"{parasitized_folder}/*.png")
        uninfected_images = glob.glob(f"{uninfected_folder}/*.png")
        image_files = parasitized_images + uninfected_images

        if not image_files:
            st.sidebar.warning("No PNG files found in the specified folders.")
            return

        if st.sidebar.button("Get Random Image"):
            random_image_path = random.choice(image_files)

            random_image = Image.open(random_image_path)
            st.image(random_image, caption="Randomly Chosen Image", use_column_width=True)

            expected_class = "Parasitized" if random_image_path in parasitized_images else "Uninfected"

            st.sidebar.header("Expected Class")
            st.sidebar.write(expected_class)

            img_array = np.array(random_image.resize((224, 224)))
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = "Parasitized" if prediction[0][0] < 0.5 else "Uninfected"

            st.sidebar.header("Cell Classification Result")
            st.sidebar.subheader(f"Predicted Class: {result}")
            st.sidebar.subheader("Prediction Probability:")
            st.sidebar.write(f"Certainity: {((100 - (prediction[0][0]) * 100)) if prediction[0][0] < 0.5 else prediction[0][0]*100:.4f}%")

if __name__ == "__main__":
    main()
