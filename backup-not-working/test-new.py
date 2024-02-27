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

def generate_recommendations(product_id, similarity_matrix, data):
    try:
        product_idx = data.index[data['id'] == product_id].tolist()[0]
    except IndexError:
        print(f"Product ID {product_id} not found in the dataset.")
        return []

    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations
    product_indices = [i[0] for i in sim_scores]
    return data.iloc[product_indices]['id'].tolist()

def print_product_attributes(product_ids, combined_df):
    attributes = ['id', 'productDisplayName', 'masterCategory', 'subCategory', 'articleType', 
                  'baseColour', 'season', 'usage', 'price', 'discountedPrice']
    for product_id in product_ids:
        product_data = combined_df.loc[combined_df['id'] == product_id, attributes]
        print(f"\nAttributes for Product ID {product_id}:")
        if not product_data.empty:
            print(product_data.to_string(index=False))
        else:
            print("Product data not found.")

def evaluate_category_consistency(recommended_items, combined_df):
    recommended_df = combined_df[combined_df['id'].isin(recommended_items)]
    main_category = recommended_df['masterCategory'].mode()[0]
    consistency = (recommended_df['masterCategory'] == main_category).mean()
    return consistency

def calculate_diversity(recommended_items, combined_df):
    recommended_df = combined_df[combined_df['id'].isin(recommended_items)]
    unique_categories = recommended_df['subCategory'].nunique()
    diversity_score = unique_categories / len(recommended_items) if recommended_items else 0
    return diversity_score

# def re_rank_for_diversity(recommendations, combined_df, top_n=5):
#     # Calculate initial diversity score
#     initial_diversity_score = calculate_diversity(recommendations, combined_df)
#     print(f"Initial Diversity Score: {initial_diversity_score}")

#     # Get category for each recommendation
#     recommended_categories = combined_df[combined_df['id'].isin(recommendations)]['subCategory']
    
#     # Re-rank by promoting items from underrepresented categories
#     unique_categories = recommended_categories.drop_duplicates().tolist()
#     re_ranked = []
#     for category in unique_categories:
#         for rec_id in recommendations:
#             if combined_df.loc[combined_df['id'] == rec_id, 'subCategory'].values[0] == category:
#                 re_ranked.append(rec_id)
#                 if len(re_ranked) == top_n:
#                     break
#         if len(re_ranked) == top_n:
#             break
    
#     # Calculate new diversity score
#     new_diversity_score = calculate_diversity(re_ranked, combined_df)
#     print(f"New Diversity Score: {new_diversity_score}")
    
#     return re_ranked

# def re_rank_for_diversity(recommendations, combined_df, top_n=5):
#     unique_categories = combined_df[combined_df['id'].isin(recommendations)]['masterCategory'].unique()
#     re_ranked = []
#     category_counts = {}

#     for rec_id in recommendations:
#         category = combined_df.loc[combined_df['id'] == rec_id, 'masterCategory'].values[0]
#         if category_counts.get(category, 0) < top_n / len(unique_categories):
#             re_ranked.append(rec_id)
#             category_counts[category] = category_counts.get(category, 0) + 1
#         if len(re_ranked) >= top_n:
#             break

#     # If not enough items to fill re_ranked, fill with the remaining recommendations
#     if len(re_ranked) < top_n:
#         for rec_id in recommendations:
#             if rec_id not in re_ranked:
#                 re_ranked.append(rec_id)
#             if len(re_ranked) >= top_n:
#                 break

#     return re_ranked


def run_recommender_system_and_evaluate(csv_path, train_json_directory, test_json_directory):
    train_df = preprocess_and_combine_data(csv_path, train_json_directory)
    test_df = preprocess_and_combine_data(csv_path, test_json_directory)
    
    train_features = combine_features(train_df)
    test_features = combine_features(test_df)
    
    # Adjust the similarity calculation as needed
    similarity_matrix = cosine_similarity(train_features, train_features)
    
    # For qualitative evaluation
    qualitative_test_ids = test_df['id'].sample(n=5, random_state=42).tolist()
    for product_id in qualitative_test_ids:
        recommendations = generate_recommendations(product_id, similarity_matrix, train_df)
        print(f"\nProduct ID {product_id} recommendations: {recommendations}")
        print_product_attributes([product_id], test_df)
        print_product_attributes(recommendations, train_df)
        
        category_consistency_score = evaluate_category_consistency(recommendations, train_df)
        print(f"Category Consistency Score: {category_consistency_score}")
        
        diversity_score = calculate_diversity(recommendations, train_df)
        print(f"Diversity Score: {diversity_score}")

        # Re-rank recommendations for enhanced diversity
        # re_ranked_recommendations = re_rank_for_diversity(recommendations, train_df)

        # enhanced_recommendations = re_rank_for_diversity(recommendations, train_df, top_n=5)

        # print(f"\nProduct ID {product_id} initial recommendations: {recommendations}")
        # print(f"Re-ranked recommendations: {enhanced_recommendations}")
        
        # print_product_attributes([product_id], test_df)  # Attributes of the test product
        # print_product_attributes(enhanced_recommendations, train_df)  # Attributes of the re-ranked recommendations
        
        # # Evaluate and print scores
        # new_category_consistency_score = evaluate_category_consistency(enhanced_recommendations, train_df)
        # print(f"Re Ranked Category Consistency Score: {new_category_consistency_score}")
        
        # re_ranked_diversity_score = calculate_diversity(enhanced_recommendations, train_df)
        # print(f"Re Ranked Diversity Score: {re_ranked_diversity_score}")
        
# Example usage
csv_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
train_json_directory = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
test_json_directory = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'

run_recommender_system_and_evaluate(csv_path, train_json_directory, test_json_directory)