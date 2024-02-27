# import pandas as pd
# import numpy as np
# import json
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import hstack

# # Updated function to load JSON files from a directory
# def load_json_files(directory_path):
#     """
#     Load and process JSON files from a directory, returning a consolidated DataFrame.
    
#     :param directory_path: Path to the directory containing JSON files.
#     :return: A pandas DataFrame with the consolidated data.
#     """
#     data_list = []  # Initialize a list to store data from each file
    
#     # Iterate over every file in the directory
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
        
#         # Ensure it's a file and ends with '.json'
#         if os.path.isfile(file_path) and file_path.endswith('.json'):
#             with open(file_path, 'r', encoding='utf-8') as file:  # Specify the encoding here
#                 data = json.load(file)
#                 # Assuming 'data' contains the relevant information needed for your DataFrame
#                 # Adjust the following line to extract and structure the data as needed
#                 extracted_data = {'id': data['data']['id'], 'other_attribute': data['data'].get('other_attribute')}
#                 data_list.append(extracted_data)
    
#     # Convert the list of data into a DataFrame
#     df = pd.DataFrame(data_list)
#     return df


# # Function to load and preprocess CSV and JSON data
# def load_and_preprocess_data(csv_path, json_directory_path):
#     # Load CSV
#     styles_df = pd.read_csv(csv_path, error_bad_lines=False)
    
#     # Load and process JSON files from the directory
#     json_df = load_json_files(json_directory_path)
    
#     # Merge CSV and JSON data
#     combined_df = pd.merge(styles_df, json_df, on='id', how='left')
    
#     return combined_df

# # Function to encode categorical attributes
# def encode_categorical_features(data, categorical_columns):
#     onehot_encoder = OneHotEncoder(sparse=True)
#     encoded_data = onehot_encoder.fit_transform(data[categorical_columns])
#     return encoded_data, onehot_encoder

# # Function to vectorize text attributes
# def vectorize_text_features(data, text_columns):
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
#     tfidf_matrix = tfidf_vectorizer.fit_transform(data[text_columns].fillna(''))
#     return tfidf_matrix, tfidf_vectorizer

# # Function to calculate similarity
# def calculate_similarity(tfidf_matrix, encoded_categorical_data):
#     combined_features = hstack([tfidf_matrix, encoded_categorical_data])
#     similarity_matrix = cosine_similarity(combined_features, combined_features)
#     return similarity_matrix

# # Function to generate recommendations
# def generate_recommendations(product_id, similarity_matrix, data, top_n=10):
#     product_idx = data.index[data['id'] == product_id].tolist()[0]
#     sim_scores = list(enumerate(similarity_matrix[product_idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:top_n + 1]
#     product_indices = [i[0] for i in sim_scores]
#     return data['id'].iloc[product_indices].tolist()

# # Main function to run the recommender system
# def run_recommender_system(csv_path, json_paths, product_id):
#     combined_df = load_and_preprocess_data(csv_path, json_paths)
    
#     categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour']
#     text_columns = 'productDisplayName'  # Assuming a single text column for simplicity
    
#     encoded_categorical_data, _ = encode_categorical_features(combined_df, categorical_columns)
#     tfidf_matrix, _ = vectorize_text_features(combined_df, text_columns)
    
#     similarity_matrix = calculate_similarity(tfidf_matrix, encoded_categorical_data)
#     recommendations = generate_recommendations(product_id, similarity_matrix, combined_df)
    
#     return recommendations

# def evaluate_category_consistency(recommended_items, combined_df):
#     """
#     Calculate the consistency of item categories within recommended items.
    
#     :param recommended_items: List of recommended item IDs.
#     :param combined_df: DataFrame containing item IDs and their corresponding categories.
#     :return: Consistency score (float).
#     """
#     # Filter the dataset to include only recommended items
#     recommended_df = combined_df[combined_df['id'].isin(recommended_items)]
    
#     # Calculate the most common category among recommended items
#     main_category = recommended_df['masterCategory'].mode()[0]
    
#     # Calculate consistency as the proportion of recommended items in the main category
#     consistency = (recommended_df['masterCategory'] == main_category).mean()
#     return consistency

# # Example usage
# csv_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
# json_directory_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles'
# product_id = 1525  # Example product ID

# # Generate recommendations
# recommendations = run_recommender_system(csv_path, json_directory_path, product_id)
# print("Recommended product IDs:", recommendations)

# # Evaluate the category consistency of the recommendations
# # consistency_score = evaluate_category_consistency(recommendations, combined_df)
# # print(f"Category Consistency Score for product {product_id}: {consistency_score}")


# import pandas as pd
# import numpy as np
# import json
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import hstack

# # Function to load JSON files from a directory, adjusted for encoding
# def load_json_files(directory_path):
#     data_list = []
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
#         if os.path.isfile(file_path) and file_path.endswith('.json'):
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#                 # Extract necessary information here
#                 extracted_data = {'id': data['data']['id'], 'other_attribute': data['data'].get('other_attribute')}
#                 data_list.append(extracted_data)
#     return pd.DataFrame(data_list)

# # Load and preprocess CSV and JSON data
# def load_and_preprocess_data(csv_path, json_directory_path):
#     styles_df = pd.read_csv(csv_path, error_bad_lines=False)
#     json_df = load_json_files(json_directory_path)
#     combined_df = pd.merge(styles_df, json_df, on='id', how='left')
#     return combined_df

# # Encoding categorical attributes
# def encode_categorical_features(data, categorical_columns):
#     onehot_encoder = OneHotEncoder(sparse=True)
#     encoded_data = onehot_encoder.fit_transform(data[categorical_columns])
#     return encoded_data, onehot_encoder

# # Vectorizing text attributes
# def vectorize_text_features(data, text_columns):
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
#     tfidf_matrix = tfidf_vectorizer.fit_transform(data[text_columns].fillna(''))
#     return tfidf_matrix, tfidf_vectorizer

# # Calculate similarity
# def calculate_similarity(tfidf_matrix, encoded_categorical_data):
#     combined_features = hstack([tfidf_matrix, encoded_categorical_data])
#     similarity_matrix = cosine_similarity(combined_features, combined_features)
#     return similarity_matrix

# # Generate recommendations
# def generate_recommendations(product_id, similarity_matrix, data, top_n=10):
#     try:
#         # Attempt to find the index of the product ID
#         product_idx = data.index[data['id'] == product_id].tolist()[0]
#     except IndexError:
#         # If the product ID is not found, print a message and return an empty list
#         print(f"Product ID {product_id} not found in the dataset.")
#         return []

#     sim_scores = list(enumerate(similarity_matrix[product_idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:top_n + 1]
#     product_indices = [i[0] for i in sim_scores]
#     return data['id'].iloc[product_indices].tolist()

# # Evaluate category consistency
# def evaluate_category_consistency(recommended_items, combined_df):
#     recommended_df = combined_df[combined_df['id'].isin(recommended_items)]
#     main_category = recommended_df['masterCategory'].mode()[0]
#     consistency = (recommended_df['masterCategory'] == main_category).mean()
#     return consistency

# # Function to calculate diversity
# def calculate_diversity(recommended_items, combined_df):
#     categories = combined_df[combined_df['id'].isin(recommended_items)]['masterCategory']
#     unique_categories = categories.unique()
#     diversity_score = len(unique_categories) / len(recommended_items)
#     return diversity_score

# # Function to calculate novelty (simplified version)
# def calculate_novelty(recommended_items, combined_df):
#     # Assuming lower frequency of recommendation indicates higher novelty
#     # This is a simplified approach; for real applications, consider using more sophisticated methods
#     all_recommendations = combined_df['id'].value_counts()
#     novelty_scores = all_recommendations.loc[recommended_items].values
#     novelty_score = np.mean(1 / (1 + novelty_scores))  # Inverse frequency as a proxy for novelty
#     return novelty_score

# # Main function to run the recommender system and evaluate it
# def run_recommender_system_and_evaluate(csv_path, json_directory_path, test_product_ids):
#     combined_df = load_and_preprocess_data(csv_path, json_directory_path)
    
#     categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour']
#     text_columns = 'productDisplayName'
    
#     encoded_categorical_data, _ = encode_categorical_features(combined_df, categorical_columns)
#     tfidf_matrix, _ = vectorize_text_features(combined_df, text_columns)
    
#     similarity_matrix = calculate_similarity(tfidf_matrix, encoded_categorical_data)
    
#     evaluation_scores = []
#     diversity_scores = []
#     novelty_scores = []

#     for product_id in test_product_ids:
#         recommendations = generate_recommendations(product_id, similarity_matrix, combined_df)
#         print(f"Product ID {product_id} recommendations: {recommendations}")
        
#         diversity_score = calculate_diversity(recommendations, combined_df)
#         diversity_scores.append(diversity_score)
        
#         novelty_score = calculate_novelty(recommendations, combined_df)
#         novelty_scores.append(novelty_score)
        
#         print(f"Product ID {product_id}: Diversity Score = {diversity_score}, Novelty Score = {novelty_score}")

#     average_diversity_score = np.mean(diversity_scores)
#     average_novelty_score = np.mean(novelty_scores)
#     print(f"Average Diversity Score: {average_diversity_score}, Average Novelty Score: {average_novelty_score}")

#     for product_id in test_product_ids:
#         recommendations = generate_recommendations(product_id, similarity_matrix, combined_df)
#         score = evaluate_category_consistency(recommendations, combined_df)
#         evaluation_scores.append(score)
#         print(f"Product ID {product_id}: Category Consistency Score = {score}")

#     average_score = np.mean(evaluation_scores)
#     print(f"Average Category Consistency Score: {average_score}")

# # Example usage
# csv_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
# json_directory_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles'
# test_product_ids = [1525, 13679, 18883, 22476, 25632, 28286, 32743, 36024, 40082]  # Replace with actual product IDs for testing

# # Run the recommender system and evaluate
# run_recommender_system_and_evaluate(csv_path, json_directory_path, test_product_ids)

import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

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
                    # Assuming articleAttributes is a dictionary; convert it to a string for TF-IDF vectorization
                    'articleAttributes': ' '.join([f"{k}:{v}" for k, v in data['data'].get('articleAttributes', {}).items()]),
                    'price': data['data'].get('price'),
                    'discountedPrice': data['data'].get('discountedPrice')
                }
                data_list.append(extracted_data)
    return pd.DataFrame(data_list)

def load_and_preprocess_data(csv_path, json_directory_path):
    styles_df = pd.read_csv(csv_path, error_bad_lines=False)
    json_df = load_json_files(json_directory_path)
    combined_df = pd.merge(styles_df, json_df, on='id', how='left')
    return combined_df

def combine_features(combined_df):
    # Vectorize textual attributes
    text_columns = ['productDisplayName', 'description', 'visualTag', 'articleAttributes']
    combined_text = combined_df[text_columns].fillna('').agg(' '.join, axis=1)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    text_features = tfidf_vectorizer.fit_transform(combined_text)
    
    # Encode categorical attributes
    # categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'brandName', 'ageGroup']
    categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    onehot_encoder = OneHotEncoder(sparse=True)
    categorical_features = onehot_encoder.fit_transform(combined_df[categorical_columns].fillna('Unknown'))
    
    # Normalize numerical attributes (price and discountedPrice)
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(combined_df[['price', 'discountedPrice', 'year']].fillna(0))
    
    # Combine all features
    combined_features = hstack([text_features, categorical_features, csr_matrix(numerical_features)])
    return combined_features, combined_df

def generate_recommendations(product_id, similarity_matrix, data, top_n=10):
    try:
        product_idx = data.index[data['id'] == product_id].tolist()[0]
    except IndexError:
        print(f"Product ID {product_id} not found in the dataset.")
        return []

    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]
    return data.iloc[product_indices]['id'].tolist()

# Evaluate category consistency
def evaluate_category_consistency(recommended_items, combined_df):
    recommended_df = combined_df[combined_df['id'].isin(recommended_items)]
    main_category = recommended_df['masterCategory'].mode()[0]
    consistency = (recommended_df['masterCategory'] == main_category).mean()
    return consistency

def print_product_attributes(product_ids, combined_df):
    """
    Print selected attributes for a list of product IDs.
    """
    attributes = ['productDisplayName', 'masterCategory', 'subCategory', 'articleType', 
                  'baseColour', 'season', 'usage', 'price', 'discountedPrice']
    for product_id in product_ids:
        product_data = combined_df[combined_df['id'] == product_id][attributes]
        print(f"\nAttributes for Product ID {product_id}:")
        print(product_data.to_string(index=False))

def run_recommender_system_and_evaluate(csv_path, json_directory_path, test_product_ids):
    combined_df = load_and_preprocess_data(csv_path, json_directory_path)
    combined_features, _ = combine_features(combined_df)
    similarity_matrix = cosine_similarity(combined_features, combined_features)
    
    evaluation_scores = []
    for product_id in test_product_ids:
        recommendations = generate_recommendations(product_id, similarity_matrix, combined_df)
        score = evaluate_category_consistency(recommendations, combined_df)
        evaluation_scores.append(score)
        
        # Print attributes for the test product ID
        print_product_attributes([product_id], combined_df)
        # Print attributes for the recommended product IDs
        print_product_attributes(recommendations, combined_df)
        
        print(f"Category Consistency Score for Product ID {product_id}: {score}\n")

    average_score = np.mean(evaluation_scores)
    print(f"Average Category Consistency Score: {average_score}")

# Example usage
csv_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
json_directory_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles'
test_product_ids = [1525, 13679, 18883, 22476, 25632, 28286, 32743, 36024, 40082]  # Replace with actual product IDs for testing

# Run the recommender system and evaluate
run_recommender_system_and_evaluate(csv_path, json_directory_path, test_product_ids)