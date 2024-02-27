from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pickle
import os
import time

st.title('Clothing Recommender Engine')

# Define model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPool2D()])
model.summary()

# Load stored features and images
file_img = pickle.load(open(r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Clothing-Ver-2\\images.pkl', 'rb'))
feature_list = pickle.load(open(r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Clothing-Ver-2\\fetaures.pkl', 'rb'))

# Load and filter products_df
products_df = pd.read_csv('C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-small\\myntradataset\\styles.csv', error_bad_lines=False)
valid_ids = set(products_df['id'])

# Define function to save uploaded image
def Save_img(upload_img):
    try:
        with open(os.path.join('uploads', upload_img.name), 'wb') as f:
            f.write(upload_img.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False

# Define function to extract features from an image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    expanded_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(expanded_img_arr)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Define function for product recommendation
def prod_recom(features, feature_list):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    nbrs.fit(feature_list)
    distances, indices = nbrs.kneighbors([features])
    return indices

upload_img = st.file_uploader("Choose an image")

if upload_img is not None:
    if Save_img(upload_img):
        st.image(Image.open(upload_img))
        st.header("File uploaded successfully!")
        features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        progress_text = "Please wait! Analysing Data and Generating Recommendations."
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)
        ind = prod_recom(features, feature_list)
        
        # Display recommended products with images and details.
        for i in range(10):
            with st.container():
                image_path = file_img[ind[0][i]]
                st.image(Image.open(image_path))
                product_id = os.path.basename(image_path).split('.')[0]
                product_info = products_df.loc[products_df['id'] == int(product_id)].iloc[0]
                st.write(f"{product_info['productDisplayName']} - {product_info['baseColour']}")
    else:
        st.header("Some error occurred")
