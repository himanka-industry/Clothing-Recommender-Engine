import pandas as pd
import json
import os

################################################################################################
################# Analyse the .json and .csv files for data understanding ######################
################################################################################################


# Replace 'path/to/styles.csv' with the actual path to your CSV file
styles_df = pd.read_csv('C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv', error_bad_lines=False)
json_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles'

all_keys = []
# all_categories = []

# Analyze product types
product_types = styles_df['articleType'].unique()
num_product_types = len(product_types)

# Find the product type with the highest number of products
product_counts = styles_df['articleType'].value_counts()
highest_number_product_type = product_counts.head(10)
# highest_number = product_counts.max()

# for json_file in os.listdir(json_dir):
#     if json_file.endswith('.json'):
#         # with open(os.path.join(json_dir, json_file), 'r') as f:
#         with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             all_keys.append(set(data.keys()))

# for json_file in os.listdir(json_dir):
#     if json_file.endswith('.json'):
#         try:
#             with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:  # Specify encoding
#                 data = json.load(f)
#                 # Assuming 'categories' are listed within 'data' tag
#                 if 'data' in data and 'categories' in data['data']:
#                     categories = data['data']['categories']
#                     all_categories.append(set(categories))
#         except UnicodeDecodeError as e:
#             print(f"Error reading {json_file}: {e}")

def process_json_attributes(data):
    attributes = {}
    for key, value in data.items():
        # Collect all keys
        attributes[key] = True
    return attributes.keys()

for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        file_path = os.path.join(json_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'data' in data:  # Assuming the relevant information is within a 'data' tag
                    keys = process_json_attributes(data['data'])
                    all_keys.append(set(keys))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_file}: {e}")
        except UnicodeDecodeError as e:
            print(f"Encoding error in {json_file}: {e}")

# Calculate the intersection of all key sets to find common keys
common_keys = set.intersection(*all_keys) if all_keys else set()

# print(f"Common categories across all items: {common_categories}")

# Find common keys across all JSON files
# common_keys = set.intersection(*all_keys)
            
# Find common categories across all JSON files
# common_categories = set.intersection(*all_categories) if all_categories else set()

# Select the top 10 product types with the highest counts
# top_10_product_types = product_counts.head(10)

# print("Top 10 unique product types with the highest number of products:")
# print(top_10_product_types)

# print(f"Total number of unique product types: {total_product_types}") 
# ({highest_number} products)

print(f"\nNumber of unique product types: {num_product_types}\n")
print(f"First few product types: {product_types[:10]}\n")
print(f"Top ten product types with the highest number of products:\n\n{highest_number_product_type}\n")
print(f"Common attributes across all product types:\n\n{common_keys}")
# print(f"Common categories across all items:\n\n{common_categories}")