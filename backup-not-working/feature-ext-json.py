# import json
# import pandas as pd
# import os
# import pickle

# # Assuming you have a different path for the JSON files
# json_files_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'  # Update this to the correct path

# # def extract_json_features(json_files_path):
# #     features = []
# #     for json_file in os.listdir(json_files_path):
# #         if json_file.endswith('.json'):
# #             file_path = os.path.join(json_files_path, json_file)
# #             with open(file_path, 'r') as file:
# #                 data = json.load(file)['data']
# #                 # Extract relevant features, for example:
# #                 feature_dict = {
# #                     'id': data.get('id'),
# #                     'price': data.get('price'),
# #                     'discountedPrice': data.get('discountedPrice'),
# #                     'gender': data.get('gender'),
# #                     'baseColour': data.get('baseColour'),
# #                     'articleType': data.get('articleType'),
# #                     'season': data.get('season'),
# #                     'material': data.get('material'),
# #                     'usage': data.get('usage'),  
# #                     'year': data.get('year'),  # Add or remove features as needed
# #                     # 'gender', 'usage', 'season', 'year', 'material', 'price', 'discountedPrice'
# #                 }
# #                 features.append(feature_dict)
# #     return pd.DataFrame(features)

# def extract_json_features(json_files_path):
#     features = []
#     for json_file in os.listdir(json_files_path):
#         if json_file.endswith('.json'):
#             file_path = os.path.join(json_files_path, json_file)
#             with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding here
#                 data = json.load(file)['data']
#                 # Extract relevant features, for example:
#                 feature_dict = {
#                     'id': data.get('id'),
#                     'price': data.get('price'),
#                     'discountedPrice': data.get('discountedPrice'),
#                     'gender': data.get('gender'),
#                     'baseColour': data.get('baseColour'),
#                     'brandName': data.get('brandName'),
#                     'season': data.get('season'),
#                     'material': data.get('material'),
#                     'usage': data.get('usage'),  
#                     'year': data.get('year'),
#                 }
#                 features.append(feature_dict)
#     return pd.DataFrame(features)


# # Extract and save JSON features
# json_features_df = extract_json_features(json_files_path)
# pickle.dump(json_features_df, open('json_features.pkl', 'wb'))

import json
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

json_files_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'  # Update this to your JSON files directory

def extract_and_process_json_features(json_files_path):
    # Initialize containers for features
    numerical_features = []
    categorical_features = []
    textual_features = []
    
    # Load JSON files and extract features
    for json_file in os.listdir(json_files_path):
        if json_file.endswith('.json'):
            file_path = os.path.join(json_files_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)['data']
                # Example numerical feature
                numerical_features.append([data.get('price', 0), data.get('discountedPrice', 0), data.get('year', 0)])
                # Example categorical feature
                categorical_features.append([data.get('gender', ''), data.get('baseColour', ''), data.get('season', '')])
                # Example textual feature
                textual_features.append(data.get('productDisplayName', ''))
    
    # Process numerical features
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    
    # Process categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features_encoded = encoder.fit_transform(categorical_features).toarray()
    
    # Process textual features
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    textual_features_tfidf = tfidf_vectorizer.fit_transform(textual_features).toarray()
    
    # Combine all features
    combined_features = np.hstack((numerical_features_scaled, categorical_features_encoded, textual_features_tfidf))
    
    return combined_features

# Extract features
combined_features = extract_and_process_json_features(json_files_path)

# Optionally, save the features for later use
pickle.dump(combined_features, open('combined_features.pkl', 'wb'))
