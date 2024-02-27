# from PIL import Image
# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.layers import GlobalMaxPool2D
# from tensorflow.keras.preprocessing import image
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
# import pickle
# import os
# import json
# import time

# # Function to improve UI with custom CSS
# def load_custom_css():
#     custom_css = """
#     <style>
#         .stContainer {
#             border: 2px solid #4CAF50;
#             border-radius: 10px;
#             padding: 10px;
#             margin-bottom: 20px;
#         }
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# load_custom_css()
# st.title('Clothing Recommender Engine')

# # Load the model
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
# model = tf.keras.Sequential([model, GlobalMaxPool2D()])

# # Load stored features, images, and pre-generated JSON features
# file_img = pickle.load(open('images.pkl', 'rb'))
# feature_list = pickle.load(open('features.pkl', 'rb'))
# pre_generated_json_features = pickle.load(open('json_features.pkl', 'rb'))

# # Assuming the path to the JSON files for test images
# test_json_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'  # Update this path

# # Function to dynamically extract JSON features for an uploaded image
# def extract_json_features_for_image(image_id, json_path):
#     json_file_path = os.path.join(json_path, f"{image_id}.json")
#     if os.path.exists(json_file_path):
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)['data']
#             return pd.Series(data)
#     else:
#         return pd.Series()

# # Function to save uploaded image
# def save_img(upload_img):
#     try:
#         with open(os.path.join('uploads', upload_img.name), 'wb') as f:
#             f.write(upload_img.getbuffer())
#         return True
#     except Exception as e:
#         st.error(f"Error saving image: {e}")
#         return False

# # Function to extract features from an image
# def feature_extraction(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_arr = image.img_to_array(img)
#     expanded_img_arr = np.expand_dims(img_arr, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_arr)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#     return normalized_result

# # Recommendation function with attribute-based logic
# def prod_recom(features, feature_list, json_features, attribute, uploaded_img_id):
#     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
#     nbrs.fit(np.array(feature_list))
#     distances, indices = nbrs.kneighbors([features])

#     recommended_products = []
#     for idx in indices[0][:10]:  # Limit to top 10 recommendations
#         # Assuming file_img names contain the product IDs that match those in json_features['id']
#         product_id = int(os.path.basename(file_img[idx]).split('.')[0])
#         product_info = json_features.loc[json_features['id'] == product_id]
        
#         if not product_info.empty and attribute in product_info.columns:
#             attr_value = product_info.iloc[0][attribute]
#             image_path = file_img[idx]
#             recommended_products.append((image_path, attr_value))
#         else:
#             # Handle the case where no matching product info is found or attribute is missing
#             continue

#     return recommended_products

# # Streamlit UI for attribute selection
# attribute = st.selectbox(
#     'Select an attribute for recommendations:',
#     ['price', 'discountedPrice', 'gender', 'baseColour', 'articleType', 'season', 'material', 'usage', 'year']
# )

# upload_img = st.file_uploader("Choose an image")

# if upload_img is not None:
#     if save_img(upload_img):
#         st.image(Image.open(upload_img), width=224)
#         st.header("File uploaded successfully!")
#         features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        
#         uploaded_img_id = upload_img.name.split('.')[0]
        
#         if int(uploaded_img_id) in pre_generated_json_features['id'].values:
#             uploaded_img_json_features = pre_generated_json_features
#         else:
#             uploaded_img_json_features = extract_json_features_for_image(uploaded_img_id, test_json_path)
#             if uploaded_img_json_features.empty:
#                 st.error("JSON features for the uploaded image could not be found or extracted.")
#                 st.stop()
#             uploaded_img_json_features = pd.DataFrame([uploaded_img_json_features])

#         recommended_products = prod_recom(features, feature_list, uploaded_img_json_features, attribute, uploaded_img_id)
        
#         st.write("Recommended Products:")
#         for img_path, attr_value in recommended_products:
#             caption = f"{attribute}: {attr_value}"
#             st.image(Image.open(img_path), width=100, caption=caption)
#     else:
#         st.header("Some error occurred")


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
import json
import time

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
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

load_custom_css()
st.title('Clothing Recommender Engine')

# Load the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPool2D()])

# Load stored features, images, and pre-generated JSON features
file_img = pickle.load(open('images.pkl', 'rb'))
feature_list = pickle.load(open('features.pkl', 'rb'))
pre_generated_json_features = pickle.load(open('json_features.pkl', 'rb'))

# Define path to the JSON files for test images
test_json_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'  # Update this path

# Function to dynamically extract JSON features for an uploaded image
def extract_json_features_for_image(image_id, json_path):
    json_file_path = os.path.join(json_path, f"{image_id}.json")
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)['data']
            return pd.Series(data)
    else:
        return pd.Series()

# Function to save uploaded image
def save_img(upload_img):
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

def prod_recom(features, feature_list, json_features, attribute, uploaded_img_id):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    nbrs.fit(np.array(feature_list))
    distances, indices = nbrs.kneighbors([features])

    recommended_products = []
    for idx in indices[0][:10]:  # Limit to top 10 recommendations
        # Extract product ID from the image filename
        product_id = int(os.path.basename(file_img[idx]).split('.')[0])
        # Find matching product info in json_features
        product_info = json_features[json_features['id'] == product_id]
        if not product_info.empty:
            attr_value = product_info.iloc[0][attribute] if attribute in product_info.columns else "N/A"
            recommended_products.append((file_img[idx], attr_value))
    return recommended_products

# Streamlit UI for attribute selection
attribute = st.selectbox(
    'Select an attribute for recommendations:',
    ['price', 'discountedPrice', 'gender', 'baseColour', 'brandName', 'season', 'material', 'usage', 'year']
)

upload_img = st.file_uploader("Choose an image")

if upload_img is not None:
    if save_img(upload_img):
        st.image(Image.open(upload_img), width=224)
        st.header("File uploaded successfully!")
        features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        
        uploaded_img_id = upload_img.name.split('.')[0]
        
        # Attempt to use pre-generated JSON features; if not found, try dynamic extraction
        uploaded_img_json_features = extract_json_features_for_image(uploaded_img_id, test_json_path)
        if uploaded_img_json_features.empty:
            st.error("JSON features for the uploaded image could not be found or extracted.")
            st.stop()
        
        recommended_products = prod_recom(features, feature_list, pre_generated_json_features, attribute, uploaded_img_id)
        
        # Display recommended products with attribute values
        st.write("Recommended Products:")
        for img_path, attr_value in recommended_products:
            caption = f"{attribute}: {attr_value}"
            st.image(Image.open(img_path), width=100, caption=caption)
    else:
        st.header("Some error occurred")
