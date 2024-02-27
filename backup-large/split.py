

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

################################################################################################
###################### ONE TIME EXECUTION FOR SPLITTING THE DATASET ############################
################################################################################################

# Paths setup
image_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images'
json_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles'
train_image_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'
test_image_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_te'
train_json_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_tr'
test_json_dir = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'

styles_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv'
train_styles_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\train_styles.csv'
test_styles_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\test_styles.csv'

# Make sure destination directories exist
for path in [train_image_dir, test_image_dir, train_json_dir, test_json_dir]:
    os.makedirs(path, exist_ok=True)

# Load the styles.csv, skipping bad lines
styles_df = pd.read_csv(styles_path, error_bad_lines=False)
styles_df['id'] = styles_df['id'].astype(str)

# Check for the existence of both image and .json files before deciding to include an ID
valid_ids = [
    str(id_) for id_ in styles_df['id']
    if os.path.exists(os.path.join(image_dir, f"{id_}.jpg")) and os.path.exists(os.path.join(json_dir, f"{id_}.json"))
]

# Split IDs into training and testing sets
train_ids, test_ids = train_test_split(valid_ids, test_size=0.2, random_state=42)

def copy_files(ids, source_dir, dest_dir, extension):
    copy_log = []
    for id_ in ids:
        source_path = os.path.join(source_dir, f"{id_}{extension}")
        dest_path = os.path.join(dest_dir, f"{id_}{extension}")
        if os.path.exists(source_path):
            try:
                shutil.copy(source_path, dest_path)
                copy_log.append(f"Successfully copied: {source_path} to {dest_path}")
            except Exception as e:
                copy_log.append(f"Error copying {source_path} to {dest_path}: {e}")
        else:
            copy_log.append(f"File not found: {source_path}")
    return copy_log

# Execute file copying
image_copy_log = copy_files(train_ids, image_dir, train_image_dir, '.jpg')
image_copy_log.extend(copy_files(test_ids, image_dir, test_image_dir, '.jpg'))
json_copy_log = copy_files(train_ids, json_dir, train_json_dir, '.json')
json_copy_log.extend(copy_files(test_ids, json_dir, test_json_dir, '.json'))

# Filter styles_df for train and test based on IDs and save
train_styles_df = styles_df[styles_df['id'].isin(train_ids)]
test_styles_df = styles_df[styles_df['id'].isin(test_ids)]
train_styles_df.to_csv(train_styles_path, index=False)
test_styles_df.to_csv(test_styles_path, index=False)

# Optionally, print or log the copy_log contents for review
for log_entry in image_copy_log + json_copy_log:
    print(log_entry[:20])

print(f"\nTransfer of files to training and testing folder complete!")
