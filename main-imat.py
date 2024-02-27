# main-imat.py

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import json
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# Configuration: Update these paths according to your directory structure

CSV_PATH = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
TRAIN_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles'
# TEST_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'
TRAIN_IMAGE_FOLDER = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images'


# Configuration for better aesthetics
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

def load_json_files(directory_path):
    data_list = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                extracted_data = {
                    'id': int(data['data']['id']),
                    'description': data['data'].get('productDescriptors', {}).get('description', {}).get('value', ''),
                    'visualTag': data['data'].get('visualTag', ''),
                    'articleAttributes': ' '.join([f"{k}:{v}" for k, v in data['data'].get('articleAttributes', {}).items()]),
                    'price': data['data'].get('price'),
                    'discountedPrice': data['data'].get('discountedPrice')
                }
                data_list.append(extracted_data)
    return pd.DataFrame(data_list)

def preprocess_and_combine_data(csv_path, json_directory_path):
    styles_df = pd.read_csv(csv_path, error_bad_lines=False)
    json_df = load_json_files(json_directory_path)
    combined_df = pd.merge(styles_df, json_df, on='id', how='left')
    return combined_df

def combine_features(combined_df):
    text_columns = ['description', 'visualTag', 'articleAttributes']
    combined_text = combined_df[text_columns].fillna('').agg(' '.join, axis=1)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    text_features = tfidf_vectorizer.fit_transform(combined_text)
    
    categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    onehot_encoder = OneHotEncoder(sparse=True)
    categorical_features = onehot_encoder.fit_transform(combined_df[categorical_columns].fillna('Unknown'))
    
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(combined_df[['price', 'discountedPrice']].fillna(0))
    
    combined_features = hstack([text_features, categorical_features, csr_matrix(numerical_features)])
    return combined_features

# Function to display product attributes
def display_product_attributes(product_id, combined_df):
    product_data = combined_df[combined_df['id'] == product_id]
    if not product_data.empty:
        product_data = product_data.iloc[0]
        attributes = ['productDisplayName', 'baseColour', 'season', 'usage', 'price', 'discountedPrice']
        for attr in attributes:
            st.text(f"{attr}: {product_data.get(attr, 'N/A')}")
    else:
        st.error("Product data not found.")

def generate_recommendations(product_id, combined_features, combined_df):
    product_idx = combined_df.index[combined_df['id'] == product_id].tolist()[0]
    similarity_matrix = cosine_similarity(combined_features, combined_features)
    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:13]  # Get top 12 recommendations
    product_indices = [i[0] for i in sim_scores]
    return combined_df.iloc[product_indices]['id'].tolist()

def find_image_path(product_id, image_folder):
    for ext in ['.jpg', '.jpeg', '.png']:
        image_path = os.path.join(image_folder, f"{product_id}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

def load_product_attributes(product_id, json_directory):
    """
    Load attributes for a specific product ID from its JSON file in the given directory.
    """
    json_path = os.path.join(json_directory, f"{product_id}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)['data']
            # Extract relevant attributes
            attributes = {
                'productDisplayName': data.get('productDisplayName', 'No name available'),
                'baseColour': data.get('baseColour', 'No colour data available'),
                'season': data.get('season', 'No season data available'),
                'usage': data.get('usage', 'No usage data available'),
                'price': data.get('price', 'No price available'),
                'discountedPrice': data.get('discountedPrice', 'No discounted price available')
            }
            return attributes
    return {'Error': 'Product data not available'}

def display_image_with_attributes(image, product_data, caption="", is_uploaded_product=True, image_width=300):
    """
    image: PIL Image object or path to the image file.
    product_data: Dictionary containing product attributes.
    """
    if is_uploaded_product:
        # Configuration for the uploaded product
        col1, col2 = st.columns([1, 2])
    else:
        # Configuration for recommended products
        col1, col2 = st.columns([2, 3])

    with col1:
        st.image(image, caption=caption, width=image_width)  # Directly display the PIL Image object
    with col2:
        st.write("### Product Attributes")
        for key, value in product_data.items():
            st.markdown(f"**{key.capitalize()}**: {value}")


# Streamlit UI

st.title("Clothing Product Recommendation System (image and attributes)")

uploaded_file = st.file_uploader("## Upload the product image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:

    uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])
    # Directly use the uploaded file for the image
    uploaded_img = Image.open(uploaded_file)
    uploaded_product_data = load_product_attributes(uploaded_img_id, TRAIN_JSON_DIRECTORY)  # Correctly loading from test data

    # Display uploaded product attributes
    combined_df = preprocess_and_combine_data(CSV_PATH, TRAIN_JSON_DIRECTORY)  # Use train directory for recommendations
    combined_features = combine_features(combined_df)

    # Generate a horizontal line
    st.markdown("---")
    
    display_image_with_attributes(uploaded_img, uploaded_product_data, is_uploaded_product=True, image_width=224)
    st.header("Product Image and Attributes acquired successfully!")
    if st.button("Generate Recommendation 💡"):
        recommended_ids = generate_recommendations(uploaded_img_id, combined_features, combined_df)

        progress_text = "Please wait! Analysing Image and Attributes Data and Generating Recommendations."
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)

        num_columns = 4
        recommended_count = len(recommended_ids)
        rows = recommended_count // num_columns + (1 if recommended_count % num_columns > 0 else 0)

         # Generate a horizontal line
        st.markdown("---")

        st.write("## Recommended Products")

        for i in range(rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < recommended_count:  # Ensure index is within the range of recommended_ids
                    with cols[j]:
                        rec_id = recommended_ids[index]
                        image_path = find_image_path(rec_id, TRAIN_IMAGE_FOLDER)
                        product_info = load_product_attributes(rec_id, TRAIN_JSON_DIRECTORY)
                    
                        if image_path:
                            st.image(Image.open(image_path), width=150)
                            st.markdown(f"<div class='productName'><b>{product_info.get('productDisplayName', 'No Name')}</b></div>", unsafe_allow_html=True)
                            st.caption(f"{product_info['baseColour']}")
                            with st.expander("See more"):
                                for key, value in product_info.items():
                                    if key not in ['productDisplayName', 'baseColour']:  # Skip already displayed info
                                        st.markdown(f"**{key.capitalize()}**: {value}")
                else:
                    with cols[j]:
                        st.empty()  # Use st.empty() to handle empty slots gracefully