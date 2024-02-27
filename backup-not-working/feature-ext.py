import json
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPool2D

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable=False
model=tf.keras.Sequential([model,GlobalMaxPool2D()])
model.summary()

def extract_image_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img).flatten()
    return features

def process_metadata(json_path, attributes):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file).get('data', {})
    return [
        data.get(attr, 'Unknown') if attr not in ['price', 'discountedPrice', 'year']
        else float(data.get(attr, 0) if data.get(attr, 0) not in ['', 'Unknown', None] else 0)
        for attr in attributes
    ]

# Directories
image_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'
json_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'

# Attributes to process
attributes = ['brandName', 'baseColour', 'gender', 'usage', 'season', 'year', 'material', 'price', 'discountedPrice', 'productDisplayName']

# # Feature extraction
image_features = []
metadata = []

total_images = len(os.listdir(image_dir))
print(f"Total images to process: {total_images}")

for i, img_filename in enumerate(os.listdir(image_dir), start=1):
    if img_filename.endswith('.jpg'):
        print(f"Processing image {i} of {total_images}: {img_filename}")  # Print current loop info
        
        img_path = os.path.join(image_dir, img_filename)
        json_path = os.path.join(json_dir, img_filename.replace('.jpg', '.json'))
        
        try:
            img_features = extract_image_features(img_path, model)
            meta_features = process_metadata(json_path, attributes)
            
            image_features.append(img_features)
            metadata.append(meta_features)
        except Exception as e:
            print(f"Error processing {img_filename}: {e}")

# Convert lists to numpy arrays
image_features = np.array(image_features)
metadata = np.array(metadata, dtype='object')

# Encode categorical metadata
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_meta = encoder.fit_transform(metadata[:, :6])  # Adjust based on categorical data

# Vectorize 'productDisplayName'
vectorizer = TfidfVectorizer(max_features=100)
vectorized_descriptions = vectorizer.fit_transform(metadata[:, -1]).toarray()

# Combine all features
combined_features = np.hstack((image_features, encoded_meta, vectorized_descriptions))

# Standardize features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(combined_features)

# Apply PCA
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(standardized_features)

# Save components
with open('pca_features.pkl', 'wb') as f:
    pickle.dump(pca_features, f)
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"PCA completed. Shape of PCA features: {pca_features.shape}. Explained variance: {np.sum(pca.explained_variance_ratio_)}")