import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# Define paths to your data
csv_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
train_json_directory = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
test_json_directory = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'
test_image_folder = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_te'
train_image_folder = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'

def load_json_files(directory_path):
    data_list = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                extracted_data = {
                    'id': data['data']['id'],
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
    return combined_features, combined_df

def generate_recommendations(product_id, similarity_matrix, data):
    product_id = int(product_id)  # Ensure this matches the data type in your DataFrame
    matches = data.index[data['id'] == product_id].tolist()
    if not matches:
        print(f"Product ID {product_id} not found in the dataset.")
        return []
    product_idx = matches[0]
    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations
    product_indices = [i[0] for i in sim_scores]
    return data.iloc[product_indices]['id'].tolist()


def get_image_path(product_id, image_folder):
    for filename in os.listdir(image_folder):
        if filename.startswith(str(product_id)):
            return os.path.join(image_folder, filename)
    return None

def main():
    st.title("Product Recommendation System")

    test_images = [f for f in os.listdir(test_image_folder) if os.path.isfile(os.path.join(test_image_folder, f))]
    selected_image = st.selectbox("Choose a product image:", test_images)
    st.image(os.path.join(test_image_folder, selected_image), caption="Selected Product", use_column_width=True)

    # product_id = os.path.splitext(selected_image)[0]  # Extract product ID from filename
    product_id = int(os.path.splitext(selected_filename)[0])


    if st.button("Generate Recommendations"):
        train_df = preprocess_and_combine_data(csv_path, train_json_directory)
        combined_features, _ = combine_features(train_df)
        similarity_matrix = cosine_similarity(combined_features, combined_features)

        recommendations = generate_recommendations(product_id, similarity_matrix, train_df)

        for rec_id in recommendations:
            rec_image_path = get_image_path(rec_id, train_image_folder)
            if rec_image_path:
                st.image(rec_image_path, caption=f"Recommended Product: {rec_id}", use_column_width=True)
            else:
                st.write(f"Image not found for Product ID: {rec_id}")

if __name__ == "__main__":
    main()
