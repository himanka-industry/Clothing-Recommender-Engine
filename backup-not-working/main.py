import streamlit as st
from PIL import Image
import numpy as np
import json
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors

# Load the necessary pre-trained models and PCA components
with open('pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('pca_features.pkl', 'rb') as f:
    training_features = pickle.load(f)

# Initialize the ResNet50 model with global max pooling
model = Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalMaxPool2D()
])

# Initialize NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
nn_model.fit(training_features)

# Define the directory path for JSON files
json_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'  # Adjust this to the path where your JSON files are stored
image_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_te'

def extract_image_features(img):
    """Extract image features using ResNet50"""
    img = img.resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img).flatten()
    return features

def process_metadata(json_path):
    """Process the metadata from a JSON file"""
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file).get('data', {})
    meta_features = [
        json_data.get(attr, 'Unknown') if attr not in ['price', 'discountedPrice', 'year']
        else float(json_data.get(attr, '0') if json_data.get(attr, '0') not in ['', 'Unknown', None] else 0)
        for attr in ['brandName', 'baseColour', 'gender', 'usage', 'season', 'year', 'material', 'price', 'discountedPrice', 'productDisplayName']
    ]
    return meta_features

def combine_and_transform_features(image_features, meta_features):
    """Combine and transform image and metadata features"""
    encoded_meta = encoder.transform([meta_features[:6]])
    vectorized_description = vectorizer.transform([meta_features[-1]]).toarray()
    combined_features = np.hstack((image_features, encoded_meta.flatten(), vectorized_description.flatten()))
    pca_transformed_features = pca_model.transform([combined_features])
    return pca_transformed_features

# Streamlit UI
st.title("Product Recommendation System")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Construct the path to the JSON file based on uploaded image filename
    json_filename = os.path.splitext(uploaded_file.name)[0] + '.json'
    json_path = os.path.join(json_dir, json_filename)
    
    if os.path.exists(json_path):
        # Process the actual JSON metadata
        image_features = extract_image_features(image)
        meta_features = process_metadata(json_path)
        combined_features = combine_and_transform_features(image_features, meta_features)
        
        # Calculate similarity and find top matches
        # similarities = cosine_similarity(combined_features, training_features).flatten()
        # top_indices = np.argsort(-similarities)[:10]

        # Use NearestNeighbors to find top matches
        distances, indices = nn_model.kneighbors(combined_features)
        
        # st.write("Top 10 Recommendations:")
        st.write("Top 10 Recommendations:")
        for idx in indices[0]:
            st.write(f"Product ID: {idx}")  # Extend this with actual product info retrieval
    else:
        st.error("Metadata JSON file not found for the uploaded image.")