# shared_utils.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack
import json
import os

def load_json_files(directory_path):
    data_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Ensure that all expected keys exist in the data, even if empty
                defaults = {'id': None, 'description': '', 'visualTag': '', 'articleAttributes': {}, 'price': 0.0, 'discountedPrice': 0.0}
                for key, default in defaults.items():
                    data['data'].setdefault(key, default)
                data_list.append(data['data'])
    return pd.DataFrame(data_list)

def preprocess_and_combine_data(csv_path, json_directory_path):
    # Load the styles data from the CSV file
    styles_df = pd.read_csv(csv_path, error_bad_lines=False)
    
    # Load the product details from the JSON files in the specified directory
    json_df = load_json_files(json_directory_path)
    
    # Merge the CSV data with the JSON data on the 'id' column
    # Use an inner join to ensure only records present in both datasets are retained
    combined_df = pd.merge(styles_df, json_df, on='id', how='inner')
    
    return combined_df


def combine_features(combined_df):
    # Handle dictionary data in text columns by converting them to strings
    if 'articleAttributes' in combined_df.columns:
        combined_df['articleAttributes'] = combined_df['articleAttributes'].apply(
            lambda x: ' '.join([f"{k}:{v}" for k, v in x.items()]) if isinstance(x, dict) else x
        )

    # Ensure all text columns are treated as strings
    text_columns = ['description', 'visualTag', 'articleAttributes']
    for column in text_columns:
        combined_df[column] = combined_df[column].astype(str)
    
    combined_text = combined_df[text_columns].fillna('').agg(' '.join, axis=1)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    text_features = tfidf_vectorizer.fit_transform(combined_text)

    #attributes: ['visualTag', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'price', 'discountedPrice'] 

    # Ensure all expected categorical columns exist, fill missing ones with 'Unknown'
    categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    for column in categorical_columns:
        if column not in combined_df.columns:
            combined_df[column] = 'Unknown'

    onehot_encoder = OneHotEncoder(sparse=True)
    categorical_features = onehot_encoder.fit_transform(combined_df[categorical_columns].fillna('Unknown'))

    # Handle numerical features, ensuring they exist and filling missing ones with 0
    numerical_columns = ['price', 'discountedPrice']
    for column in numerical_columns:
        if column not in combined_df.columns:
            combined_df[column] = 0.0
    
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(combined_df[numerical_columns].fillna(0))

    # Combine all features into a single feature matrix
    combined_features = hstack([text_features, categorical_features, numerical_features])
    return combined_features


def get_image_path(product_id, image_folder):
    expected_path = os.path.join(image_folder, f"{product_id}.jpg")  # Adjust the extension as necessary
    if os.path.exists(expected_path):
        return expected_path
    else:
        print(f"Image file not found for product ID: {product_id}")
        return None
