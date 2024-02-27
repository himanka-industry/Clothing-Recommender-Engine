# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
# from scipy.sparse import hstack, csr_matrix
# import pickle
# import os
# import json

# # Assuming you have defined paths to your JSON directory
# # json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'

# # Load JSON data into a DataFrame
# # def load_and_combine_json_features(json_dir_path):
# #     data_list = []
# #     for filename in os.listdir(json_dir_path):
# #         if filename.endswith('.json'):
# #             file_path = os.path.join(json_dir_path, filename)
# #             with open(file_path, 'r', encoding='utf-8') as file:  # Specify the encoding here
# #                 data = json.load(file)['data']
# #                 data_list.append({
# #                     'id': data['id'],
# #                     'description': data.get('productDescriptors', {}).get('description', {}).get('value', ''),
# #                     'visualTag': data.get('visualTag', ''),
# #                     'articleAttributes': ' '.join([f"{k}:{v}" for k, v in data.get('articleAttributes', {}).items()]),
# #                     # Add other attributes as needed
# #                     'gender': data.get('gender', 'Unknown'),
# #                     'masterCategory': data.get('masterCategory', 'Unknown'),
# #                     'subCategory': data.get('subCategory', 'Unknown'),
# #                     'articleType': data.get('articleType', 'Unknown'),
# #                     'baseColour': data.get('baseColour', 'Unknown'),
# #                     'season': data.get('season', 'Unknown'),
# #                     'usage': data.get('usage', 'Unknown'),
# #                     'price': data.get('price', 0),
# #                     'discountedPrice': data.get('discountedPrice', 0)
# #                 })
# #     return pd.DataFrame(data_list)

# # def combine_features(combined_df):
# #     # Text features
# #     text_columns = ['description', 'visualTag', 'articleAttributes']
# #     combined_text = combined_df[text_columns].fillna('').agg(' '.join, axis=1)
# #     tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
# #     text_features = tfidf_vectorizer.fit_transform(combined_text)
    
# #     # Categorical features
# #     categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
# #     # Ensure no columns contain dictionaries
# #     for col in categorical_columns:
# #         combined_df[col] = combined_df[col].astype(str)  # Convert all to string
# #     onehot_encoder = OneHotEncoder(sparse=True)
# #     categorical_features = onehot_encoder.fit_transform(combined_df[categorical_columns].fillna('Unknown'))
    
# #     # Numerical features
# #     scaler = MinMaxScaler()
# #     numerical_features = scaler.fit_transform(combined_df[['price', 'discountedPrice']].fillna(0))
    
# #     # Combine all features
# #     combined_features = hstack([text_features, categorical_features, csr_matrix(numerical_features)])
# #     return combined_features, tfidf_vectorizer, onehot_encoder, scaler

# # # Example usage for training data
# # train_json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
# # train_data_df = load_and_combine_json_features(train_json_dir_path)
# # train_combined_features, train_tfidf, train_onehot, train_scaler = combine_features(train_data_df)

# # # Save training features and models
# # pickle.dump(train_combined_features, open('train_combined_features.pkl', 'wb'))
# # pickle.dump(train_tfidf, open('train_tfidf_vectorizer.pkl', 'wb'))
# # pickle.dump(train_onehot, open('train_onehot_encoder.pkl', 'wb'))
# # pickle.dump(train_scaler, open('train_scaler.pkl', 'wb'))

# # # Repeat the process for testing data
# # test_json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'
# # test_data_df = load_and_combine_json_features(test_json_dir_path)
# # test_combined_features, test_tfidf, test_onehot, test_scaler = combine_features(test_data_df)

# # # Save testing features
# # pickle.dump(test_combined_features, open('test_combined_features.pkl', 'wb'))
# # # No need to save the models again if you're using the same transformations
# # print("Test JSON data has been preprocessed and saved.")

# # # Path to your image directory
# # image_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'

# # # Generate list of image paths
# # images = [os.path.join(image_dir_path, file_name) for file_name in os.listdir(image_dir_path) if file_name.endswith(('.jpg', '.jpeg', '.png'))]

# # # Save the image paths to a .pkl file
# # pickle.dump(images, open('images-large-reco.pkl', 'wb'))

# # print(f"Saved {len(images)} image paths to images-large-reco.pkl")

# def load_and_save_json_attributes(json_dir_path, output_file_name):
#     data_list = []
#     for filename in os.listdir(json_dir_path):
#         if filename.endswith('.json'):
#             file_path = os.path.join(json_dir_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)['data']
#                 data_list.append({
#                     'id': data['id'],
#                     'description': data.get('productDescriptors', {}).get('description', {}).get('value', ''),
#                     'visualTag': data.get('visualTag', ''),
#                     'articleAttributes': ' '.join([f"{k}:{v}" for k, v in data.get('articleAttributes', {}).items()]),
#                     # Include other attributes as needed
#                     'gender': data.get('gender', 'Unknown'),
#                     'masterCategory': data.get('masterCategory', 'Unknown'),
#                     'subCategory': data.get('subCategory', 'Unknown'),
#                     'articleType': data.get('articleType', 'Unknown'),
#                     'baseColour': data.get('baseColour', 'Unknown'),
#                     'season': data.get('season', 'Unknown'),
#                     'usage': data.get('usage', 'Unknown'),
#                     'price': data.get('price', 0),
#                     'discountedPrice': data.get('discountedPrice', 0),
#                 })
#     json_attributes_df = pd.DataFrame(data_list)
#     # Save the DataFrame to a .pkl file
#     json_attributes_df.to_pickle(output_file_name)

# # Paths to JSON directories
# train_json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
# test_json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'

# # Generate and save both training and testing JSON data as DataFrames in .pkl files
# load_and_save_json_attributes(train_json_dir_path, 'train_json_data.pkl')
# load_and_save_json_attributes(test_json_dir_path, 'test_json_data.pkl')

# print("Generation of train and test .json .pkl files if finished!")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import pickle
import json
import os

def load_json_data(json_dir_path):
    """Loads JSON files from the specified directory into a pandas DataFrame."""
    data_list = []
    for filename in os.listdir(json_dir_path):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)['data']
                data_list.append({
                    'id': data['id'],
                    'description': data.get('productDescriptors', {}).get('description', {}).get('value', ''),
                    'visualTag': data.get('visualTag', ''),
                    'articleAttributes': ' '.join([f"{k}:{v}" for k, v in data.get('articleAttributes', {}).items()]),
                    'gender': data.get('gender', 'Unknown'),
                    'masterCategory': data.get('masterCategory', 'Unknown'),
                    'subCategory': data.get('subCategory', 'Unknown'),
                    'articleType': data.get('articleType', 'Unknown'),
                    'baseColour': data.get('baseColour', 'Unknown'),
                    'season': data.get('season', 'Unknown'),
                    'usage': data.get('usage', 'Unknown'),
                    'price': data.get('price', 0),
                    'discountedPrice': data.get('discountedPrice', 0),
                })
    return pd.DataFrame(data_list)

def preprocess_data(json_data):
    """Preprocesses the JSON data to generate a combined feature matrix."""
    # Combine text columns for TF-IDF
    text_data = json_data['description'] + ' ' + json_data['visualTag'] + ' ' + json_data['articleAttributes']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    text_features = tfidf_vectorizer.fit_transform(text_data)
    
    # Prepare and encode categorical data
    categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    categorical_data = json_data[categorical_columns].applymap(str).fillna('Unknown')
    onehot_encoder = OneHotEncoder(sparse=True)
    categorical_features = onehot_encoder.fit_transform(categorical_data)
    
    # Scale numerical features
    numerical_features = MinMaxScaler().fit_transform(json_data[['price', 'discountedPrice']].fillna(0).astype(np.float64))
    
    # Combine all features into a single matrix
    combined_features = hstack([text_features, categorical_features, csr_matrix(numerical_features)])
    return combined_features, tfidf_vectorizer, onehot_encoder

# Load JSON data from both training and testing directories
# train_json_dir_path = 'path/to/train/json'
# test_json_dir_path = 'path/to/test/json'
train_json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
test_json_dir_path = r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'
train_data = load_json_data(train_json_dir_path)
test_data = load_json_data(test_json_dir_path)

# Preprocess both datasets
train_combined_features, train_tfidf_vectorizer, train_onehot_encoder = preprocess_data(train_data)
test_combined_features, _, _ = preprocess_data(test_data)  # Reuse train vectorizer and encoder for test data

# Save preprocessed features and models
pickle.dump(train_combined_features, open('train_combined_features.pkl', 'wb'))
pickle.dump(test_combined_features, open('test_combined_features.pkl', 'wb'))
pickle.dump(train_tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(train_onehot_encoder, open('onehot_encoder.pkl', 'wb'))

print("Preprocessing complete and data saved.")
