
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

st.set_page_config(page_title="Clothing Product Recommendation", layout="wide")

# Custom CSS to improve UI
def load_custom_css():
    custom_css = """
    <style>
        .stContainer {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        /* Style for product names to ensure uniform height */
        .productName {
            height: 3em; /* Adjust based on your font size to fit two lines */
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2; /* Number of lines to show */
            -webkit-box-orient: vertical;
        }
        /* You can add more styles here */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

load_custom_css()

st.title("Clothing Product Recommendation System (image)")

# Define model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPool2D()])


# Load stored features and images
file_img = pickle.load(open(r'images.pkl', 'rb'))
feature_list = pickle.load(open(r'features.pkl', 'rb'))

# Load and filter products_df
products_df = pd.read_csv('C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv', error_bad_lines=False)
valid_ids = set(products_df['id'])

# Function to save uploaded image
def Save_img(upload_img):
    try:
        with open(os.path.join('uploads', upload_img.name), 'wb') as f:
            f.write(upload_img.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

# Function to extract features from an image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    expanded_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(expanded_img_arr)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function for product recommendation
def prod_recom(features, feature_list):
    nbrs = NearestNeighbors(n_neighbors=12, algorithm='brute', metric='euclidean')
    nbrs.fit(feature_list)
    distances, indices = nbrs.kneighbors([features])
    return indices

upload_img = st.file_uploader("## Upload the product image", type=['png', 'jpg', 'jpeg'])

if upload_img is not None:
    if Save_img(upload_img):
        # Generate a horizontal line
        st.markdown("---")
        st.image(Image.open(upload_img), width=224)
        st.header("Product Image acquired successfully!")
        features = feature_extraction(os.path.join("uploads", upload_img.name), model)

        if st.button("Generate Recommendation 💡"):
            progress_text = "Please wait! Analysing Image Data and Generating Recommendations."
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1, text=progress_text)
        
            ind = prod_recom(features, feature_list)
        
            # Assuming the recommended products info is ready
            recommended_products = [products_df.loc[products_df['id'] == int(os.path.basename(file_img[i]).split('.')[0])] for i in ind[0]]
        
            num_columns = 4
            rows = len(recommended_products) // num_columns + (1 if len(recommended_products) % num_columns > 0 else 0)

            # Generate a horizontal line
            st.markdown("---")

            st.write("## Recommended Products")
            for i in range(rows):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    index = i * num_columns + j
                    if index < len(ind[0]):
                        with cols[j]:
                            image_path = file_img[ind[0][index]]
                            st.image(Image.open(image_path), width=150)
                            product_id = os.path.basename(image_path).split('.')[0]
                            product_info = products_df.loc[products_df['id'] == int(product_id)].iloc[0]
                            # Wrap the product name in a div with class productName and apply custom CSS
                            st.markdown(f"<div class='productName'><b>{product_info['productDisplayName']}</b></div>", unsafe_allow_html=True)
                            st.caption(f"{product_info['baseColour']}, {product_info['articleType']}")
                            # Use expanders for more info
                            with st.expander("See more"):
                                st.write(f"Season: {product_info['season']}")
                                st.write(f"Usage: {product_info['usage']}")
                                st.write(f"Gender: {product_info['gender']}")
                                # st.write(f"Material: {product_info['material']}")
                                # st.write(f"Price: {product_info['price']}")
                                # st.write(f"Discounted Price: {product_info['discountedPrice']}")
                                # st.write(f"Year: {str(product_info['year'])}")
                                st.write(f"Year: {int(product_info['year'])}")
                                # 'gender', 'usage', 'season', 'year', 'material', 'price', 'discountedPrice'
    else:
        st.header("Some error occurred")
