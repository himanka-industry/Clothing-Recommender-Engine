# import streamlit as st
# from PIL import Image
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# # Update these paths according to your directory structure
# TRAIN_COMBINED_FEATURES_PATH = 'train_combined_features.pkl'
# TEST_COMBINED_FEATURES_PATH = 'test_combined_features.pkl'
# TRAIN_JSON_DATA_PATH = 'train_json_data.pkl'
# TEST_JSON_DATA_PATH = 'test_json_data.pkl'
# IMAGES_PATH = 'images-large-reco.pkl'

# # Load preprocessed data and models
# def load_preprocessed_data():
#     with open(TRAIN_COMBINED_FEATURES_PATH, 'rb') as f:
#         train_combined_features = pickle.load(f)
#     with open(TEST_COMBINED_FEATURES_PATH, 'rb') as f:
#         test_combined_features = pickle.load(f)
#     with open(TRAIN_JSON_DATA_PATH, 'rb') as f:
#         train_json_data = pd.read_pickle(f)
#     with open(TEST_JSON_DATA_PATH, 'rb') as f:
#         test_json_data = pd.read_pickle(f)
#     with open(IMAGES_PATH, 'rb') as f:
#         images_paths = pickle.load(f)
#     return train_combined_features, test_combined_features, train_json_data, test_json_data, images_paths

# train_combined_features, test_combined_features, train_json_data, test_json_data, images_paths = load_preprocessed_data()

# # Function to find image path based on product ID
# def get_image_path(product_id, images_paths):
#     for path in images_paths:
#         if str(product_id) in path:
#             return path
#     return None

# # Function to display product attributes and image
# def display_product_info(product_id, json_data, images_paths):
#     product_info = json_data.loc[json_data['id'] == product_id].iloc[0].to_dict()
#     image_path = get_image_path(product_id, images_paths)
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         if image_path:
#             st.image(Image.open(image_path), width=300)
#         else:
#             st.error("Image not available.")
#     with col2:
#         st.write("### Product Attributes")
#         for key, value in product_info.items():
#             st.markdown(f"**{key.capitalize()}**: {value}")

# # Streamlit UI
# st.title("Clothing Product Recommendation System")

# uploaded_file = st.file_uploader("Upload the product image", type=['png', 'jpg', 'jpeg'])

# if uploaded_file is not None:
#     uploaded_img = Image.open(uploaded_file)
#     uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])

#     # Display uploaded/test product attributes using test_json_data
#     if uploaded_img_id in test_json_data['id'].values:
#         st.image(uploaded_img, caption="Uploaded Product", width=300)
#         display_product_info(uploaded_img_id, test_json_data, images_paths)

#         if st.button("Generate Recommendation 💡"):
#             # Use uploaded_img_id to fetch the corresponding test feature
#             test_feature_idx = np.where(test_json_data['id'] == uploaded_img_id)[0][0]
#             test_feature = test_combined_features[test_feature_idx:test_feature_idx+1]

#             # Calculate similarity scores
#             similarity_scores = cosine_similarity(test_feature, train_combined_features).flatten()
#             top_indices = np.argsort(similarity_scores)[-12:][::-1]  # Get top 12 indices
#             recommended_ids = train_json_data.iloc[top_indices]['id'].tolist()

#             st.write("## Recommended Products")
#             for rec_id in recommended_ids:
#                 display_product_info(rec_id, train_json_data, images_paths)
#     else:
#         st.error("Product ID not found in test data.")


# import streamlit as st
# from PIL import Image
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# # Update these paths according to your directory structure
# TRAIN_FEATURES_PATH = 'train_combined_features.pkl'
# TEST_FEATURES_PATH = 'test_combined_features.pkl'
# TRAIN_JSON_DATA_PATH = 'train_json_data.pkl'
# TEST_JSON_DATA_PATH = 'test_json_data.pkl'
# IMAGES_PATH = 'images-large-reco.pkl'

# # Load preprocessed data and models
# with open(TRAIN_FEATURES_PATH, 'rb') as f:
#     train_features = pickle.load(f)
# with open(TEST_FEATURES_PATH, 'rb') as f:
#     test_features = pickle.load(f)
# with open(TRAIN_JSON_DATA_PATH, 'rb') as f:
#     train_json_data = pd.read_pickle(f)
# with open(TEST_JSON_DATA_PATH, 'rb') as f:
#     test_json_data = pd.read_pickle(f)
# with open(IMAGES_PATH, 'rb') as f:
#     images_paths = pickle.load(f)

# # Streamlit UI configuration and custom CSS
# st.set_page_config(page_title="Clothing Product Recommendation", layout="wide")
# def load_custom_css():
#     custom_css = """
#     <style>
#         .stContainer {
#             border: 2px solid #4CAF50;
#             border-radius: 10px;
#             padding: 10px;
#             margin-bottom: 20px;
#         }
#         .productName {
#             height: 3em;
#             overflow: hidden;
#             text-overflow: ellipsis;
#             display: -webkit-box;
#             -webkit-line-clamp: 2;
#             -webkit-box-orient: vertical;
#         }
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)
# load_custom_css()

# # Function to find image path based on product ID
# def get_image_path(product_id, images_paths):
#     for path in images_paths:
#         if str(product_id) in path:
#             return path
#     return None

# # Function to display product attributes and image
# def display_product_info(product_id, json_data, images_paths):
#     product_info = json_data.loc[json_data['id'] == product_id].iloc[0].to_dict()
#     image_path = get_image_path(product_id, images_paths)
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         if image_path:
#             st.image(Image.open(image_path), width=300)
#         else:
#             st.error("Image not available.")
#     with col2:
#         st.write("### Product Attributes")
#         for key, value in product_info.items():
#             st.markdown(f"**{key.capitalize()}**: {value}")

# # Function to generate recommendations
# def generate_recommendations(uploaded_img_id, test_features, train_features, test_json_data):
#     # Correctly find the index of the uploaded/test image's features within the test dataset
#     test_idx = np.where(test_json_data['id'] == uploaded_img_id)[0][0]
#     # Extract the specific feature vector for the uploaded image
#     test_feature_vector = test_features[test_idx:test_idx+1]
    
#     # Calculate similarity scores between the test feature vector and all train features
#     similarity_scores = cosine_similarity(test_feature_vector, train_features).flatten()
#     top_indices = np.argsort(similarity_scores)[-12:][::-1]  # Get top 12 indices, excluding the highest one (itself)
#     recommended_ids = train_json_data['id'].iloc[top_indices].tolist()

#     return recommended_ids

# # UI for uploading and displaying
# st.title("Clothing Product Recommendation System")

# uploaded_file = st.file_uploader("Upload the product image", type=['png', 'jpg', 'jpeg'])

# # UI for uploading and displaying
# if uploaded_file is not None:
#     uploaded_img = Image.open(uploaded_file)
#     uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])
    
#     # Display uploaded product attributes using test_json_data
#     if uploaded_img_id in test_json_data['id'].values:
#         display_product_info(uploaded_img_id, test_json_data, images_paths)

#         if st.button("Generate Recommendation 💡"):
#             recommended_ids = generate_recommendations(uploaded_img_id, test_features, train_features, test_json_data)
#             st.write("## Recommended Products")
#             for rec_id in recommended_ids:
#                 display_product_info(rec_id, train_json_data, images_paths)
#     else:
#         st.error("Product ID not found in test data.")

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define paths to your .pkl files
TRAIN_FEATURES_PATH = 'path/to/train_combined_features.pkl'
TFIDF_VECTORIZER_PATH = 'path/to/tfidf_vectorizer.pkl'
ONEHOT_ENCODER_PATH = 'path/to/onehot_encoder.pkl'
TRAIN_JSON_DATA_PATH = 'path/to/train_json_data.pkl'
IMAGES_PATH = 'path/to/images_paths.pkl'

# Load the preprocessed training features and models
with open(TRAIN_FEATURES_PATH, 'rb') as file:
    train_features = pickle.load(file)
with open(TFIDF_VECTORIZER_PATH, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open(ONEHOT_ENCODER_PATH, 'rb') as file:
    onehot_encoder = pickle.load(file)
with open(TRAIN_JSON_DATA_PATH, 'rb') as file:
    train_json_data = pd.read_pickle(file)
with open(IMAGES_PATH, 'rb') as file:
    images_paths = pickle.load(file)

# Function to display product info
def display_product_info(product_id, json_data, images_paths):
    product_info = json_data.loc[json_data['id'] == product_id].iloc[0].to_dict()
    image_path = next((path for path in images_paths if str(product_id) in path), None)
    col1, col2 = st.columns([1, 2])
    with col1:
        if image_path:
            st.image(Image.open(image_path), width=300)
        else:
            st.error("Image not available.")
    with col2:
        st.write("### Product Attributes")
        for key, value in product_info.items():
            st.markdown(f"**{key.capitalize()}**: {value}")

# Streamlit UI
st.title("Clothing Product Recommendation System")

uploaded_file = st.file_uploader("Upload the product image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Process for generating recommendations goes here
    st.image(uploaded_file, caption="Uploaded Product", width=300)
    
    # This part is simplified for illustration. In practice, you'd extract features from the uploaded image,
    # similar to how test features would be extracted and preprocessed.
    uploaded_img_id = int(os.path.splitext(uploaded_file.name)[0])  # Example to tie back to known test data

    if uploaded_img_id in train_json_data['id'].values:
        # Displaying attributes for the uploaded image, assuming it's part of the training data for demonstration
        display_product_info(uploaded_img_id, train_json_data, images_paths)

        # Generating recommendations (simplified example, assuming direct feature availability)
        # Normally, you'd extract features from uploaded_file and compare against train_features
        similarity_scores = cosine_similarity([train_features[uploaded_img_id]], train_features).flatten()
        top_indices = np.argsort(similarity_scores)[-11:-1]  # Top 10 excluding itself
        recommended_ids = train_json_data.iloc[top_indices]['id'].tolist()

        st.write("## Recommended Products")
        for rec_id in recommended_ids:
            display_product_info(rec_id, train_json_data, images_paths)
    else:
        st.error("Product ID not found in training data.")
