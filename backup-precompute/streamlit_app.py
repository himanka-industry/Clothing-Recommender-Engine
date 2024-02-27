# # streamlit_app.py

# import streamlit as st
# from shared_utils import preprocess_and_combine_data, get_image_path, load_json_files, combine_features
# import numpy as np
# import pandas as pd
# import os
# import json

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

# # Configuration: Update these paths according to your directory structure
# # SIMILARITY_MATRIX_PATH = 'similarity_matrix.npy'
# # CSV_PATH = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
# # TRAIN_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
# # TEST_IMAGE_FOLDER = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_te'
# # TRAIN_IMAGE_FOLDER = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'

# # Configuration: Update these paths according to your directory structure
# SIMILARITY_MATRIX_PATH = 'similarity_matrix.npy'
# CSV_PATH = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
# TRAIN_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
# TEST_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'
# TRAIN_IMAGE_FOLDER = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'

# # Load precomputed similarity matrix and training data
# # similarity_matrix = np.load(SIMILARITY_MATRIX_PATH)
# # train_df = preprocess_and_combine_data(CSV_PATH, TRAIN_JSON_DIRECTORY)
# # train_features = combine_features(train_df)
# # train_df.set_index('id', inplace=True)

# # Load precomputed similarity matrix and training data
# # similarity_matrix = np.load(SIMILARITY_MATRIX_PATH)
# # train_df = pd.read_csv(CSV_PATH)
# # train_df.set_index('id', inplace=True)

# # Load precomputed similarity matrix and product details
# similarity_matrix = np.load(SIMILARITY_MATRIX_PATH)
# products_df = pd.read_csv(CSV_PATH, error_bad_lines=False)
# products_df.set_index('id', inplace=True)


# def load_product_json_details(product_id, json_directory):
#     json_path = os.path.join(json_directory, f"{product_id}.json")
#     if os.path.exists(json_path):
#         with open(json_path, 'r') as file:
#             product_data = json.load(file)
#             # Filter for required attributes
#             required_attributes = ['visualTag', 'gender', 'baseColour', 'season', 'usage', 'price', 'discountedPrice']
#             filtered_data = {attr: product_data['data'].get(attr, 'Not Available') for attr in required_attributes}
#             return filtered_data
#     return {}


# # # Function to generate recommendations
# # def generate_recommendations(selected_id, num_recommendations=5):
# #     if selected_id not in train_df.index:
# #         return []
# #     selected_index = train_df.index.get_loc(selected_id)
# #     similarities = similarity_matrix[selected_index]
# #     sorted_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]  # Exclude self
# #     return train_df.iloc[sorted_indices].index.tolist()

# # # Display product attributes (assuming attributes are in train_df for simplicity)
# # def display_product_info(product_id):
# #     if product_id in train_df.index:
# #         product_info = train_df.loc[product_id]
# #         st.write(product_info)
# #     else:
# #         st.error("Product information not available.")

# def display_product_info(product_data):
#     if product_data:
#         # Display each attribute and its value
#         for key, value in product_data.items():
#             st.text(f"{key}: {value}")
#     else:
#         st.error("Product information not available.")


# # def generate_recommendations(selected_id, num_recommendations=5):
# #     if selected_id not in similarity_matrix.shape[0]:
# #         return []
# #     similarities = similarity_matrix[selected_id]
# #     sorted_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]
# #     recommended_ids = products_df.iloc[sorted_indices].index
# #     return recommended_ids
        
# # def get_image_path(product_id, image_folder):
# #     # Assuming images are named as "<product_id>.jpg"
# #     image_path = os.path.join(image_folder, f"{product_id}.jpg")
# #     if os.path.exists(image_path):
# #         return image_path
# #     return None

# def get_image_path(product_id, image_folder):
#     image_path = os.path.join(image_folder, f"{product_id}.jpg")  # Adjust if your images have a different extension
#     print(f"Looking for image at: {image_path}")  # Debug print
#     if os.path.exists(image_path):
#         return image_path
#     print(f"Image not found for product ID: {product_id}")  # Debug print
#     return None

        
# def generate_recommendations(selected_id, products_df, similarity_matrix, num_recommendations=5):
#     # Ensure product_id is an integer
#     selected_id = int(selected_id)
    
#     # Find the index of the selected_id in the products dataframe
#     if selected_id in products_df.index:
#         product_index = products_df.index.get_loc(selected_id)
#     else:
#         st.error("Selected product ID not found in the similarity matrix.")
#         return []
    
#     # Check if product_index is valid for the similarity_matrix
#     if product_index >= 0 and product_index < len(similarity_matrix):
#         similarities = similarity_matrix[product_index]
#         sorted_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]  # Exclude the selected product itself
#         recommended_ids = products_df.iloc[sorted_indices].index.tolist()
#         return recommended_ids
#     else:
#         st.error("Selected product index is out of bounds.")
#         return []


# def main():
#     st.title("Fashion Product Recommendation System")
    
#     # uploaded_file = st.file_uploader("Upload a product image:", type=['png', 'jpg', 'jpeg'])
#     uploaded_file = st.file_uploader("Upload a product image:", type=['png', 'jpg', 'jpeg'])
    
    
#     # if uploaded_file is not None:
#     #     # Assuming the filename without extension is the product ID
#     #     product_id = os.path.splitext(uploaded_file.name)[0]
        
#     #     # Fetch and display the product details from the JSON file
#     #     product_data = load_product_json_details(product_id, TEST_JSON_DIRECTORY)
#     #     display_product_info(product_data)

#     # if uploaded_file is not None:
#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Product", use_column_width=True)
#         # Logic for handling uploaded file
#         product_id = os.path.splitext(uploaded_file.name)[0]
#         product_data = load_product_json_details(product_id, TEST_JSON_DIRECTORY)
#         display_product_info(product_data)
        
#         # if st.button("Generate Recommendations"):
#         #     recommended_ids = generate_recommendations(product_id, products_df, similarity_matrix, num_recommendations=5)
            
#         #     for rec_id in recommended_ids:
#         #         # Display each recommended product's image and details
#         #         rec_image_path = os.path.join(TRAIN_IMAGE_FOLDER, f"{rec_id}.jpg")  # Adjust format if necessary
#         #         if os.path.exists(rec_image_path):
#         #             st.image(rec_image_path, caption=f"Recommended Product ID: {rec_id}", width=300)
#         #             rec_product_data = load_product_json_details(str(rec_id), TRAIN_JSON_DIRECTORY)
#         #             display_product_info(rec_product_data)
#         #         else:
#         #             st.write(f"Image not available for Product ID: {rec_id}")

#         # if st.button("Generate Recommendations"):
#         #     recommended_ids = generate_recommendations(product_id, products_df, similarity_matrix, num_recommendations=5)
    
#         #     for rec_id in recommended_ids:
#         #     # Use get_image_path to find the image path
#         #         rec_image_path = get_image_path(rec_id, TRAIN_IMAGE_FOLDER)
#         #         if rec_image_path:  # If the image exists
#         #             st.image(rec_image_path, caption=f"Recommended Product ID: {rec_id}", width=300)
#         #             rec_product_data = load_product_json_details(str(rec_id), TRAIN_JSON_DIRECTORY)
#         #             display_product_info(rec_product_data)
#         #         else:
#         #             st.write(f"Image not available for Product ID: {rec_id}")

#         if st.button("Generate Recommendations"):
#             recommended_ids = generate_recommendations(product_id, products_df, similarity_matrix, num_recommendations=5)
    
#             for rec_id in recommended_ids:
#                 rec_image_path = get_image_path(rec_id, TRAIN_IMAGE_FOLDER)
#                 if rec_image_path:  # If the image exists
#                     st.image(rec_image_path, caption=f"Recommended Product ID: {rec_id}", width=300)
#                     rec_product_data = load_product_json_details(str(rec_id), TRAIN_JSON_DIRECTORY)
#                     display_product_info(rec_product_data)
#                 else:
#                     st.error(f"Image not available for Product ID: {rec_id}")


# if __name__ == "__main__":
#     main()


# streamlit_app.py

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
SIMILARITY_MATRIX_PATH = 'similarity_matrix.npy'
CSV_PATH = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
TRAIN_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
TEST_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'
TRAIN_IMAGE_FOLDER = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'
TEST_IMAGE_FOLDER = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_te'

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

# Load the precomputed similarity matrix and product details
# similarity_matrix = np.load(SIMILARITY_MATRIX_PATH)
# products_df = pd.read_csv(CSV_PATH, error_bad_lines=False)
# products_df.set_index('id', inplace=True)

# def load_product_json_details(product_id, json_directory):
#     json_path = os.path.join(json_directory, f"{product_id}.json")
#     if os.path.exists(json_path):
#         with open(json_path, 'r') as file:
#             product_data = json.load(file)['data']
#             # Filter for required attributes
#             required_attributes = ['visualTag', 'gender', 'baseColour', 'season', 'usage', 'price', 'discountedPrice']
#             filtered_data = {attr: product_data.get(attr, 'Not Available') for attr in required_attributes}
#             return filtered_data
#     return {}

# def display_product_info(product_data):
#     if product_data:
#         for key, value in product_data.items():
#             st.text(f"{key.capitalize()}: {value}")
#     else:
#         st.error("Product information not available.")

# def generate_recommendations(product_id, similarity_matrix, products_df, num_recommendations=5):
#     if product_id in products_df.index:
#         product_idx = products_df.index.get_loc(product_id)
#         similarities = similarity_matrix[product_idx]
#         sorted_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]
#         recommended_ids = products_df.iloc[sorted_indices].index
#         return recommended_ids
#     else:
#         return []

# st.title("Fashion Product Recommendation System")

# uploaded_file = st.file_uploader("Upload a product image:", type=['png', 'jpg', 'jpeg'])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Product", use_column_width=True)
#     uploaded_img_id = os.path.splitext(uploaded_file.name)[0]
    
#     # Fetch and display the uploaded product details from the JSON file
#     product_data = load_product_json_details(uploaded_img_id, TEST_JSON_DIRECTORY)
#     display_product_info(product_data)

#     if st.button("Generate Recommendation"):
#         recommended_ids = generate_recommendations(int(uploaded_img_id), similarity_matrix, products_df, num_recommendations=5)
        
#         for rec_id in recommended_ids:
#             rec_data = load_product_json_details(str(rec_id), TRAIN_JSON_DIRECTORY)
#             rec_image_path = os.path.join(TRAIN_IMAGE_FOLDER, f"{rec_id}.jpg")
#             if os.path.exists(rec_image_path):
#                 st.image(rec_image_path, caption=f"Recommended Product ID: {rec_id}", width=300)
#                 display_product_info(rec_data)
#             else:
#                 st.error(f"Image not available for Product ID: {rec_id}")

# Load and preprocess CSV and JSON data
# def load_and_preprocess_data(csv_path, json_directory_path):
#     styles_df = pd.read_csv(csv_path, error_bad_lines=False)
#     json_df = load_json_files(json_directory_path)
#     combined_df = pd.merge(styles_df, json_df, on='id', how='left')
#     return combined_df

# # Function to load JSON files from a directory
# def load_json_files(directory_path):
#     data_list = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.json'):
#             with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
#                 data = json.load(file)['data']
#                 data_list.append(data)
#     return pd.DataFrame(data_list)

# # Combine features for TF-IDF and one-hot encoding
# def combine_features(combined_df):
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
#     text_features = tfidf_vectorizer.fit_transform(combined_df['description'].fillna(''))
    
#     onehot_encoder = OneHotEncoder(sparse=True)
#     categorical_features = onehot_encoder.fit_transform(combined_df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']].fillna('Unknown'))
    
#     combined_features = hstack([text_features, categorical_features])
#     return combined_features

# # Generate recommendations based on cosine similarity
# def generate_recommendations(product_id, combined_df, combined_features):
#     product_idx = combined_df.index[combined_df['id'] == product_id].tolist()[0]
#     similarity_matrix = cosine_similarity(combined_features, combined_features)
#     sim_scores = list(enumerate(similarity_matrix[product_idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     product_indices = [i[0] for i in sim_scores[1:6]]  # Top 5 recommendations
#     recommended_ids = combined_df.iloc[product_indices]['id'].tolist()
#     return recommended_ids

# st.title("Fashion Product Recommendation System")

# uploaded_file = st.file_uploader("Upload a product image:", type=['png', 'jpg', 'jpeg'])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Product", use_column_width=True)
#     uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])
    
#     # Process and combine data for training
#     train_df = load_and_preprocess_data(CSV_PATH, TRAIN_JSON_DIRECTORY)
#     combined_features = combine_features(train_df)
    
#     if st.button("Generate Recommendation"):
#         recommended_ids = generate_recommendations(uploaded_img_id, train_df, combined_features)
        
#         for rec_id in recommended_ids:
#             rec_data = train_df[train_df['id'] == rec_id].iloc[0]
#             st.write(f"Recommended Product ID: {rec_id}")
#             rec_image_path = os.path.join(TRAIN_IMAGE_FOLDER, f"{rec_id}.jpg")
#             if os.path.exists(rec_image_path):
#                 st.image(rec_image_path, caption=f"Product ID: {rec_id}", width=300)
#             else:
#                 st.error(f"Image not available for Product ID: {rec_id}")

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
                # 'productDisplayName': data.get('productDisplayName', 'No name available'),
                # 'description': data.get('productDescriptors', {}).get('description', {}).get('value', 'No description available'),
                # 'visualTag': data.get('visualTag', 'No visual tag'),
                # Add or modify attributes as needed
                'price': data.get('price', 'No price available'),
                'discountedPrice': data.get('discountedPrice', 'No discounted price available')
                
            }
            return attributes
    return {'Error': 'Product data not available'}

# def display_image_with_attributes(image_path, product_data, caption="", is_uploaded_product=True):
#     """
#     Displays an image alongside its attributes, with an option to adjust the size based on whether
#     it's an uploaded product or a recommended product.
#     """
#     if is_uploaded_product:
#         # Larger display for uploaded product
#         col1, col2 = st.columns([1, 2])
#     else:
#         # Smaller display for recommended products
#         col1, col2 = st.columns([2, 3])

#     with col1:
#         if image_path:
#             st.image(image_path, caption=caption, use_column_width=True)
#         else:
#             st.error("Image not available.")
#     with col2:
#         st.write("### Product Attributes")
#         for key, value in product_data.items():
#             st.markdown(f"**{key.capitalize()}**: {value}")

# def display_image_with_attributes(image_path, product_data, caption="", is_uploaded_product=True):
#     """
#     Displays an image alongside its attributes, with an option to adjust the size based on whether
#     it's an uploaded product or a recommended product.
#     Uses PIL to load and display the image to ensure compatibility.
#     """
#     if is_uploaded_product:
#         # Larger display for uploaded product
#         col1, col2 = st.columns([1, 2])
#     else:
#         # Smaller display for recommended products
#         col1, col2 = st.columns([2, 3])

#     with col1:
#         if image_path:
#             image = Image.open(image_path)
#             st.image(image, caption=caption, use_column_width=True)
#         else:
#             st.error("Image not available.")
#     with col2:
#         st.write("### Product Attributes")
#         for key, value in product_data.items():
#             st.markdown(f"**{key.capitalize()}**: {value}")

def display_image_with_attributes(image_path, product_data, caption="", is_uploaded_product=True, image_width=300):
    """
    Displays an image alongside its attributes, with options to adjust the size and specify if
    it's an uploaded product or a recommended product.
    Uses PIL to load and display the image to ensure compatibility.
    """
    if is_uploaded_product:
        # Configuration for the uploaded product
        col1, col2 = st.columns([1, 2])
    else:
        # Configuration for recommended products
        col1, col2 = st.columns([2, 3])

    with col1:
        if image_path:
            image = Image.open(image_path)
            st.image(image, caption=caption, use_column_width=False, width=image_width)
        else:
            st.error("Image not available.")
    with col2:
        st.write("### Product Attributes")
        for key, value in product_data.items():
            st.markdown(f"**{key.capitalize()}**: {value}")

# Streamlit UI

st.title("Clothing Product Recommendation System (image and attributes)")

uploaded_file = st.file_uploader("## Upload the product image", type=['png', 'jpg', 'jpeg'])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Product", use_column_width=True)
#     uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])

#     combined_df = preprocess_and_combine_data(CSV_PATH, TRAIN_JSON_DIRECTORY)
#     combined_features = combine_features(combined_df)

#     if st.button("Generate Recommendation"):
#         recommended_ids = generate_recommendations(uploaded_img_id, combined_features, combined_df)
    
#         for rec_id in recommended_ids:
#             rec_image_path = find_image_path(rec_id, TRAIN_IMAGE_FOLDER)
#             if rec_image_path:
#                 st.image(rec_image_path, caption=f"Recommended Product ID: {rec_id}", width=300)
#             else:
#                 st.error(f"Image not available for Product ID: {rec_id}")

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Selected Product", use_column_width=True, width=224)
#     # st.image(Image.open(upload_img), )
#     uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])

#     # Display uploaded product attributes
#     combined_df = preprocess_and_combine_data(CSV_PATH, TEST_JSON_DIRECTORY)  # Use test directory for uploaded product
#     display_product_attributes(uploaded_img_id, combined_df)

#     combined_df = preprocess_and_combine_data(CSV_PATH, TRAIN_JSON_DIRECTORY)  # Use train directory for recommendations
#     combined_features = combine_features(combined_df)

#     if st.button("Generate Recommendation"):
#         recommended_ids = generate_recommendations(uploaded_img_id, combined_features, combined_df)
        
#         for rec_id in recommended_ids:
#             # rec_image_path = os.path.join(TRAIN_IMAGE_FOLDER, f"{rec_id}.jpg")
#             rec_image_path = find_image_path(rec_id, TRAIN_IMAGE_FOLDER)
#             if rec_image_path:
#                 st.image(rec_image_path, caption=f"Recommended Product ID: {rec_id}", width=300)
#                 display_product_attributes(rec_id, combined_df)
#             else:
#                 st.error(f"Image not available for Product ID: {rec_id}")


# uploaded_file = st.file_uploader("Upload a product image:", type=['png', 'jpg', 'jpeg'], help="Choose a fashion item image to upload.")

if uploaded_file is not None:
    uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])
    uploaded_img_path = find_image_path(uploaded_img_id, TEST_IMAGE_FOLDER)  # Function to find the image path in test folder
    uploaded_product_data = load_product_attributes(uploaded_img_id, TEST_JSON_DIRECTORY)  # Load attributes from JSON

    # Display uploaded product attributes
    combined_df = preprocess_and_combine_data(CSV_PATH, TEST_JSON_DIRECTORY)  # Use test directory for uploaded product
    # display_product_attributes(uploaded_img_id, combined_df)

    combined_df = preprocess_and_combine_data(CSV_PATH, TRAIN_JSON_DIRECTORY)  # Use train directory for recommendations f"Recommended Product ID: {rec_id}",
    combined_features = combine_features(combined_df)

    # Generate a horizontal line
    st.markdown("---")
    
    # st.write("## Uploaded Product")
    display_image_with_attributes(uploaded_img_path, uploaded_product_data, is_uploaded_product=True, image_width=224)
    st.header("Product Image and Attributes acquired successfully!")
    if st.button("Generate Recommendation 💡"):
        recommended_ids = generate_recommendations(uploaded_img_id, combined_features, combined_df)

        progress_text = "Please wait! Analysing Image and Attributes Data and Generating Recommendations."
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Assuming recommended_ids contains the list of recommended product IDs
        num_columns = 4
        rows = len(recommended_ids) // num_columns + (1 if len(recommended_ids) % num_columns > 0 else 0)
        
        # Generate a horizontal line
        st.markdown("---")

        st.write("## Recommended Products")
        # for rec_id in recommended_ids:
        #     rec_image_path = find_image_path(rec_id, TRAIN_IMAGE_FOLDER)
        #     rec_product_data = load_product_attributes(rec_id, TRAIN_JSON_DIRECTORY)
        #     display_image_with_attributes(rec_image_path, rec_product_data, is_uploaded_product=False, image_width=200)

        for i in range(rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < len(recommended_ids):
                    with cols[j]:
                        rec_id = recommended_ids[index]
                        image_path = find_image_path(rec_id, TRAIN_IMAGE_FOLDER)
                        product_info = load_product_attributes(rec_id, TRAIN_JSON_DIRECTORY)
                        
                        if image_path:
                            st.image(Image.open(image_path), width=150)
                            # Display product name and other attributes
                            st.markdown(f"<div class='productName'><b>{product_info.get('productDisplayName', 'No Name')}</b></div>", unsafe_allow_html=True)
                            # st.caption(f"{product_info.get('baseColour', 'N/A')}, {product_info.get('articleType', 'N/A')}")
                            st.caption(f"{product_info['baseColour']}")
                            # Expanders for more detailed info
                            with st.expander("See more"):
                                # st.write(f"Season: {product_info['season']}")
                                # st.write(f"Usage: {product_info['usage']}")
                                # st.write(f"Gender: {product_info['gender']}")
                                # st.write(f"Material: {product_info['material']}")
                                # st.write(f"Price: {product_info['price']}")
                                # st.write(f"Discounted Price: {product_info['discountedPrice']}")
                                # st.write(f"Year: {str(product_info['year'])}")
                                # for key, value in product_info.items():
                                #     st.markdown(f"**{key.capitalize()}**: {value}")
                                for key, value in list(product_info.items())[1:]:
                                    st.markdown(f"**{key.capitalize()}**: {value}")
                                # st.write(f"Season: {product_info.get('season', 'N/A')}")
                                # st.write(f"Usage: {product_info.get('usage', 'N/A')}")
                                # st.write(f"Gender: {product_info.get('gender', 'N/A')}")
                                # st.write(f"Price: {product_info.get('price', 'N/A')}")
                                # st.write(f"Discounted Price: {product_info.get('discountedPrice', 'N/A')}")