# # precompute_similarity.py
# from shared_utils import preprocess_and_combine_data, combine_features
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Specify your paths
# csv_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
# json_directory_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'  # Assuming training data

# # Preprocess data and combine features
# combined_df = preprocess_and_combine_data(csv_path, json_directory_path)
# combined_features = combine_features(combined_df)

# # Compute similarity matrix
# similarity_matrix = cosine_similarity(combined_features)

# # Save the similarity matrix
# np.save('similarity_matrix.npy', similarity_matrix)

# precompute_similarity.py
from shared_utils import load_json_files, combine_features
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

CSV_PATH = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
TRAIN_JSON_DIRECTORY = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'

# train_df = pd.read_csv(CSV_PATH)
train_df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
json_df = load_json_files(TRAIN_JSON_DIRECTORY)
combined_df = pd.merge(train_df, json_df, on='id', how='inner')

combined_features = combine_features(combined_df)
similarity_matrix = cosine_similarity(combined_features)

np.save('similarity_matrix.npy', similarity_matrix)

