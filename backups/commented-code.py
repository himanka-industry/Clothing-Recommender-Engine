# # from PIL import Image
# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # import tensorflow as tf
# # from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# # from tensorflow.keras.layers import GlobalMaxPool2D
# # from tensorflow.keras.preprocessing import image
# # from sklearn.neighbors import NearestNeighbors
# # from numpy.linalg import norm
# # import pickle
# # import os
# # import time

# # # Function to improve UI with custom CSS
# # def load_custom_css():
# #     custom_css = """
# #     <style>
# #         .stContainer {
# #             border: 2px solid #4CAF50;
# #             border-radius: 10px;
# #             padding: 10px;
# #             margin-bottom: 20px;
# #         }
# #         /* Additional custom CSS can be added here */
# #     </style>
# #     """
# #     st.markdown(custom_css, unsafe_allow_html=True)

# # load_custom_css()
# # st.title('Clothing Recommender Engine')

# # # Define the model
# # model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# # model.trainable = False
# # model = tf.keras.Sequential([model, GlobalMaxPool2D()])

# # # Load stored features, images, and JSON features
# # file_img = pickle.load(open('images.pkl', 'rb'))
# # feature_list = pickle.load(open('features.pkl', 'rb'))
# # json_features = pickle.load(open('json_features.pkl', 'rb'))

# # # Function to save uploaded image
# # def save_img(upload_img):
# #     try:
# #         with open(os.path.join('uploads', upload_img.name), 'wb') as f:
# #             f.write(upload_img.getbuffer())
# #         return True
# #     except Exception as e:
# #         st.error(f"Error saving image: {e}")
# #         return False

# # # Function to extract features from an image
# # def feature_extraction(img_path, model):
# #     img = image.load_img(img_path, target_size=(224, 224))
# #     img_arr = image.img_to_array(img)
# #     expanded_img_arr = np.expand_dims(img_arr, axis=0)
# #     preprocessed_img = preprocess_input(expanded_img_arr)
# #     result = model.predict(preprocessed_img).flatten()
# #     normalized_result = result / norm(result)
# #     return normalized_result

# # # # Function for product recommendation with attribute-based logic adjustment
# # # def prod_recom(features, feature_list, json_features, attribute, uploaded_img_id):
# # #     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
# # #     nbrs.fit(feature_list)
# # #     distances, indices = nbrs.kneighbors([features])

# # #     # Retrieve the attribute value for the uploaded image
# # #     uploaded_img_attr_value = json_features.loc[json_features['id'] == int(uploaded_img_id), attribute].values[0]

# # #     # Re-ranking based on the selected attribute
# # #     # Simplified for demonstration; expand for your use case
# # #     if attribute in ['price', 'discountedPrice', 'year']:
# # #         attr_diff = json_features.iloc[indices[0]][attribute].apply(lambda x: abs(x - uploaded_img_attr_value))
# # #         sorted_indices = attr_diff.sort_values().index.to_list()
# # #     else:
# # #         exact_matches = json_features.iloc[indices[0]][attribute] == uploaded_img_attr_value
# # #         sorted_indices = exact_matches.sort_values(ascending=False).index.to_list()

# # #     adjusted_indices = [json_features.index.get_loc(idx) for idx in sorted_indices]
    
# # #     return [adjusted_indices[:10]]  # Limit to top 10 recommendations

# # # Function for product recommendation with attribute-based logic adjustment
# # def prod_recom(features, feature_list, json_features, attribute, uploaded_img_name):
# #     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
# #     nbrs.fit(feature_list)
# #     distances, indices = nbrs.kneighbors([features])

# #     uploaded_img_id = int(uploaded_img_name.split('.')[0])  # Ensure this matches the 'id' format in json_features

# #     # Check if the uploaded image's ID exists in json_features and the attribute exists
# #     if uploaded_img_id in json_features['id'].values and attribute in json_features.columns:
# #         uploaded_img_attr_value = json_features.loc[json_features['id'] == uploaded_img_id, attribute].values
# #         if len(uploaded_img_attr_value) > 0:
# #             uploaded_img_attr_value = uploaded_img_attr_value[0]
# #         else:
# #             st.error("Attribute value for the uploaded image not found.")
# #             return []
# #     else:
# #         st.error("Uploaded image ID or selected attribute not found in JSON features.")
# #         return []

# #     # Simplified re-ranking logic (as before, with error handling included)
# #     if attribute in ['price', 'discountedPrice', 'year']:
# #         attr_diff = json_features.iloc[indices[0]][attribute].apply(lambda x: abs(x - uploaded_img_attr_value))
# #         sorted_indices = attr_diff.sort_values().index.to_list()
# #     else:
# #         exact_matches = json_features.iloc[indices[0]][attribute] == uploaded_img_attr_value
# #         sorted_indices = exact_matches.sort_values(ascending=False).index.to_list()

# #     adjusted_indices = [json_features.index.get_loc(idx) for idx in sorted_indices]

# #     return [adjusted_indices[:10]]  # Limit to top 10 recommendations


# # # Streamlit UI for attribute selection
# # attribute = st.selectbox(
# #     'Select an attribute for recommendations:',
# #     ('price', 'discountedPrice', 'gender', 'baseColour', 'articleType', 'season', 'material', 'usage', 'year')
# # )

# # upload_img = st.file_uploader("Choose an image")

# # if upload_img is not None:
# #     if save_img(upload_img):
# #         st.image(Image.open(upload_img), width=224)
# #         st.header("File uploaded successfully!")
# #         features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        
# #         progress_text = "Please wait! Analyzing Data and Generating Recommendations."
# #         my_bar = st.progress(0)
# #         for percent_complete in range(100):
# #             time.sleep(0.02)
# #             my_bar.progress(percent_complete + 1)
        
# #         ind = prod_recom(features, feature_list, json_features, attribute, upload_img.name.split('.')[0])
        
# #         # Display recommended products with images and attributes
# #         for i in ind[0]:
# #             if i < len(file_img):
# #                 product_id = os.path.basename(file_img[i]).split('.')[0]
# #                 product_info = json_features.loc[json_features['id'] == int(product_id)]
# #                 caption = f"{attribute}: {product_info[attribute].values[0]}"
# #                 st.image(Image.open(file_img[i]), width=150, caption=caption)
# #     else:
# #         st.header("Some error occurred")


# # from PIL import Image
# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # import tensorflow as tf
# # from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# # from tensorflow.keras.layers import GlobalMaxPool2D
# # from tensorflow.keras.preprocessing import image
# # from sklearn.neighbors import NearestNeighbors
# # from numpy.linalg import norm
# # import pickle
# # import os
# # import json
# # import time

# # # Function to improve UI with custom CSS
# # def load_custom_css():
# #     custom_css = """
# #     <style>
# #         .stContainer {
# #             border: 2px solid #4CAF50;
# #             border-radius: 10px;
# #             padding: 10px;
# #             margin-bottom: 20px;
# #         }
# #     </style>
# #     """
# #     st.markdown(custom_css, unsafe_allow_html=True)

# # load_custom_css()
# # st.title('Clothing Recommender Engine')

# # # Define the model
# # model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# # model.trainable = False
# # model = tf.keras.Sequential([model, GlobalMaxPool2D()])

# # # Load stored features, images, and pre-generated JSON features
# # file_img = pickle.load(open('images.pkl', 'rb'))
# # feature_list = pickle.load(open('features.pkl', 'rb'))
# # pre_generated_json_features = pickle.load(open('json_features.pkl', 'rb'))

# # test_json_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'  # Update this path

# # # Function to dynamically extract JSON features for an uploaded image
# # def extract_json_features_for_image(image_id, json_path):
# #     json_file_path = os.path.join(json_path, f"{image_id}.json")
# #     if os.path.exists(json_file_path):
# #         with open(json_file_path, 'r') as file:
# #             data = json.load(file)['data']
# #             return pd.Series(data)
# #     else:
# #         return pd.Series()

# # # Function to save uploaded image
# # def save_img(upload_img):
# #     try:
# #         with open(os.path.join('uploads', upload_img.name), 'wb') as f:
# #             f.write(upload_img.getbuffer())
# #         return True
# #     except Exception as e:
# #         st.error(f"Error saving image: {e}")
# #         return False

# # # Function to extract features from an image
# # def feature_extraction(img_path, model):
# #     img = image.load_img(img_path, target_size=(224, 224))
# #     img_arr = image.img_to_array(img)
# #     expanded_img_arr = np.expand_dims(img_arr, axis=0)
# #     preprocessed_img = preprocess_input(expanded_img_arr)
# #     result = model.predict(preprocessed_img).flatten()
# #     normalized_result = result / norm(result)
# #     return normalized_result

# # # Streamlit UI for attribute selection
# # attribute = st.selectbox(
# #     'Select an attribute for recommendations:',
# #     ['price', 'discountedPrice', 'gender', 'baseColour', 'articleType', 'season', 'material', 'usage', 'year']
# # )

# # upload_img = st.file_uploader("Choose an image")

# # if upload_img is not None:
# #     if save_img(upload_img):
# #         st.image(Image.open(upload_img), width=224)
# #         st.header("File uploaded successfully!")
# #         features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        
# #         uploaded_img_id = upload_img.name.split('.')[0]
        
# #         # Check if the uploaded image's features are available in the pre-generated `.json` features
# #         uploaded_img_json_features = pre_generated_json_features[pre_generated_json_features['id'] == int(uploaded_img_id)]
# #         if uploaded_img_json_features.empty:
# #             # Dynamically extract JSON features for the uploaded image
# #             uploaded_img_json_features = extract_json_features_for_image(uploaded_img_id, test_json_path)
# #             if uploaded_img_json_features.empty:
# #                 st.error("JSON features for the uploaded image could not be found or extracted.")
# #                 st.stop()
        
# #         uploaded_img_json_features = pd.DataFrame([uploaded_img_json_features]) if isinstance(uploaded_img_json_features, pd.Series) else uploaded_img_json_features

# #         # Use the extracted JSON features for recommendation
# #         # Example: Filter recommendations based on the selected attribute value of the uploaded image
# #         attr_value = uploaded_img_json_features[attribute].iloc[0]
# #         st.write(f"{attribute} value of the uploaded image: {attr_value}")

# #         # Implementing a simple recommendation logic based on attribute similarity
# #         # Filtering the dataset to match the attribute value (demonstration purpose)
# #         similar_items = pre_generated_json_features[pre_generated_json_features[attribute] == attr_value][:5]
        
# #         if not similar_items.empty:
# #             st.write(f"Products similar to the selected {attribute}:")
# #             for idx, row in similar_items.iterrows():
# #                 img_path = next((img for img in file_img if str(row['id']) in img), None)
# #                 if img_path:
# #                     st.image(Image.open(img_path), caption=f"ID: {row['id']}", width=100)
# #                 else:
# #                     st.write(f"Image for ID {row['id']} not found.")
# #         else:
# #             st.write("No similar products found based on the selected attribute.")

# #     else:
# #         st.header("Some error occurred")


# # from PIL import Image
# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # import tensorflow as tf
# # from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# # from tensorflow.keras.layers import GlobalMaxPool2D
# # from tensorflow.keras.preprocessing import image
# # from sklearn.neighbors import NearestNeighbors
# # from numpy.linalg import norm
# # import pickle
# # import os
# # import json
# # import time

# # # Function to improve UI with custom CSS
# # def load_custom_css():
# #     custom_css = """
# #     <style>
# #         .stContainer {
# #             border: 2px solid #4CAF50;
# #             border-radius: 10px;
# #             padding: 10px;
# #             margin-bottom: 20px;
# #         }
# #         /* Additional custom CSS can be added here */
# #     </style>
# #     """
# #     st.markdown(custom_css, unsafe_allow_html=True)

# # load_custom_css()
# # st.title('Clothing Recommender Engine')

# # # Define the model
# # model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# # model.trainable = False
# # model = tf.keras.Sequential([model, GlobalMaxPool2D()])

# # # Load stored features, images, and pre-generated JSON features
# # file_img = pickle.load(open('images.pkl', 'rb'))
# # feature_list = pickle.load(open('features.pkl', 'rb'))
# # pre_generated_json_features = pickle.load(open('json_features.pkl', 'rb'))

# # # Assuming the path to the JSON files for test images
# # test_json_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'  # Update this path

# # # Function to dynamically extract JSON features for an uploaded image
# # def extract_json_features_for_image(image_id, json_path):
# #     json_file_path = os.path.join(json_path, f"{image_id}.json")
# #     if os.path.exists(json_file_path):
# #         with open(json_file_path, 'r') as file:
# #             data = json.load(file)['data']
# #             return pd.Series(data)
# #     else:
# #         return pd.Series()

# # # Function to save uploaded image
# # def save_img(upload_img):
# #     try:
# #         with open(os.path.join('uploads', upload_img.name), 'wb') as f:
# #             f.write(upload_img.getbuffer())
# #         return True
# #     except Exception as e:
# #         st.error(f"Error saving image: {e}")
# #         return False

# # # Function to extract features from an image
# # def feature_extraction(img_path, model):
# #     img = image.load_img(img_path, target_size=(224, 224))
# #     img_arr = image.img_to_array(img)
# #     expanded_img_arr = np.expand_dims(img_arr, axis=0)
# #     preprocessed_img = preprocess_input(expanded_img_arr)
# #     result = model.predict(preprocessed_img).flatten()
# #     normalized_result = result / norm(result)
# #     return normalized_result

# # # Recommendation function with attribute-based logic
# # def prod_recom(features, feature_list, json_features, attribute, uploaded_img_id):
# #     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
# #     nbrs.fit(feature_list)
# #     distances, indices = nbrs.kneighbors([features])
    
# #     uploaded_img_id = int(uploaded_img_id)  # Ensure this matches the 'id' format in json_features
    
# #     # Check if the uploaded image's ID exists in json_features and the attribute exists
# #     if uploaded_img_id in json_features['id'].values and attribute in json_features.columns:
# #         uploaded_img_attr_value = json_features.loc[json_features['id'] == uploaded_img_id, attribute].values
# #         if len(uploaded_img_attr_value) > 0:
# #             uploaded_img_attr_value = uploaded_img_attr_value[0]
# #         else:
# #             st.error("Attribute value for the uploaded image not found.")
# #             return []
# #     else:
# #         st.error("Uploaded image ID or selected attribute not found in JSON features.")
# #         return []

# #     # Ensuring indices are within the bounds of json_features
# #     valid_indices = [i for i in indices[0] if i < len(json_features)]
    
# #     # Re-ranking logic based on the attribute
# #     if attribute in ['price', 'discountedPrice', 'year']:
# #         # Calculate the absolute difference in attribute value for valid indices only
# #         attr_diff = json_features.iloc[valid_indices][attribute].apply(lambda x: abs(x - uploaded_img_attr_value))
# #         sorted_indices = attr_diff.sort_values().index.to_list()
# #     else:
# #         # For categorical attributes, prioritize exact matches for valid indices only
# #         exact_matches = json_features.iloc[valid_indices][attribute] == uploaded_img_attr_value
# #         sorted_indices = exact_matches.sort_values(ascending=False).index.to_list()

# #     # Convert sorted indices back to original indices in feature_list
# #     adjusted_indices = [feature_list.index[i] for i in sorted_indices]

# #     return adjusted_indices[:10]  # Limit to top 10 recommendations


# # # Streamlit UI for attribute selection
# # attribute = st.selectbox(
# #     'Select an attribute for recommendations:',
# #     ['price', 'discountedPrice', 'gender', 'baseColour', 'articleType', 'season', 'material', 'usage', 'year']
# # )

# # upload_img = st.file_uploader("Choose an image")

# # if upload_img is not None:
# #     if save_img(upload_img):
# #         st.image(Image.open(upload_img), width=224)
# #         st.header("File uploaded successfully!")
# #         features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        
# #         uploaded_img_id = upload_img.name.split('.')[0]
        
# #         # Check if the uploaded image's features are available in the pre-generated `.json` features
# #         if int(uploaded_img_id) in pre_generated_json_features['id'].values:
# #             uploaded_img_json_features = pre_generated_json_features
# #         else:
# #             # Dynamically extract JSON features for the uploaded image
# #             uploaded_img_json_features = extract_json_features_for_image(uploaded_img_id, test_json_path)
# #             if uploaded_img_json_features.empty:
# #                 st.error("JSON features for the uploaded image could not be found or extracted.")
# #                 st.stop()
# #             uploaded_img_json_features = pd.DataFrame([uploaded_img_json_features])

# #         # Generate recommendations using the specified attribute
# #         recommended_indices = prod_recom(features, feature_list, uploaded_img_json_features, attribute, uploaded_img_id)

# #         # Display recommended products
# #         st.write("Recommended Products:")
# #         for idx in recommended_indices:
# #             st.image(Image.open(file_img[idx]), width=100, caption=f"Recommended Product {idx+1}")
        
# #     else:
# #         st.header("Some error occurred")

# from PIL import Image
# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.layers import GlobalMaxPool2D
# from tensorflow.keras.preprocessing import image
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
# import pickle
# import os
# import json
# import time

# # Custom CSS to improve UI
# def load_custom_css():
#     custom_css = """
#     <style>
#         .stContainer {
#             border: 2px solid #4CAF50;
#             border-radius: 10px;
#             padding: 10px;
#             margin-bottom: 20px;
#         }
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# load_custom_css()
# st.title('Clothing Recommender Engine')

# # Load the model
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
# model = tf.keras.Sequential([model, GlobalMaxPool2D()])

# # Load stored features and pre-generated JSON features
# file_img = pickle.load(open('images.pkl', 'rb'))
# feature_list = pickle.load(open('features.pkl', 'rb'))
# pre_generated_json_features = pickle.load(open('json_features.pkl', 'rb'))

# # Define path to the JSON files for test images
# test_json_path = 'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles_te'  # Update this path

# # Function to dynamically extract JSON features for an uploaded image
# def extract_json_features_for_image(image_id, json_path):
#     json_file_path = os.path.join(json_path, f"{image_id}.json")
#     if os.path.exists(json_file_path):
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)['data']
#             return pd.Series(data)
#     return pd.Series()

# # Function to save uploaded image
# def save_img(upload_img):
#     try:
#         with open(os.path.join('uploads', upload_img.name), 'wb') as f:
#             f.write(upload_img.getbuffer())
#         return True
#     except Exception as e:
#         st.error(f"Error saving image: {e}")
#         return False

# # Function to extract features from an image
# def feature_extraction(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_arr = image.img_to_array(img)
#     expanded_img_arr = np.expand_dims(img_arr, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_arr)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#     return normalized_result

# # Function for product recommendation with attribute-based logic
# # def prod_recom(features, feature_list, json_features, attribute, uploaded_img_id):
# #     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
# #     features_array = np.array(feature_list)  # Ensure feature_list is an array
# #     nbrs.fit(features_array)
# #     distances, indices = nbrs.kneighbors([features])

# #     # Filter json_features based on 'id' matching 'uploaded_img_id'
# #     match = json_features[json_features['id'] == int(uploaded_img_id)]

# #     if not match.empty and attribute in json_features:
# #         uploaded_img_attr_value = match[attribute].iloc[0]
# #         # Example logic to re-rank based on attribute; needs customization
# #         # Here, simply prioritize images with similar attribute values
# #         # Adjust this logic based on specific attribute handling and comparison
# #     else:
# #         st.error("Uploaded image ID or attribute not found in JSON features.")
# #         return []

# #     # Displaying images based on initial nearest neighbors (adjust as needed)
# #     recommended_images = [file_img[idx] for idx in indices[0]]

# #     return recommended_images[:10]  # Return top 10 recommendations

# def prod_recom(features, feature_list, json_features, attribute, uploaded_img_id):
#     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
#     features_array = np.array(feature_list)  # Convert list of features to a NumPy array for NearestNeighbors
#     nbrs.fit(features_array)
#     distances, indices = nbrs.kneighbors([features])

#     uploaded_img_attr_value = json_features.loc[json_features['id'] == int(uploaded_img_id), attribute].values[0]

#     # Create a DataFrame for easier manipulation
#     recommendations_df = json_features.iloc[indices[0]].copy()
    
#     if attribute in ['price', 'discountedPrice', 'year']:  # Numerical attributes
#         # Calculate the absolute difference and sort
#         recommendations_df['attr_diff'] = recommendations_df[attribute].apply(lambda x: abs(x - uploaded_img_attr_value))
#         recommendations_df.sort_values(by='attr_diff', ascending=True, inplace=True)
#     else:  # Categorical attributes
#         # Mark exact matches and sort (exact matches come first)
#         recommendations_df['attr_match'] = recommendations_df[attribute] == uploaded_img_attr_value
#         recommendations_df.sort_values(by='attr_match', ascending=False, inplace=True)

#     # Extract sorted indices after re-ranking
#     sorted_indices = recommendations_df.index.to_list()

#     # Limit to top 10 recommendations
#     top_recommendations = sorted_indices[:10]

#     # Retrieve corresponding image paths for the top recommendations
#     recommended_images = [file_img[i] for i in top_recommendations]

#     # Modify the return statement of prod_recom to include top_recommendations indices or IDs
#     return recommended_images, top_recommendations # Return top 10 recommendations


# # UI for attribute selection
# attribute = st.selectbox(
#     'Select an attribute for recommendations:',
#     ['price', 'discountedPrice', 'gender', 'baseColour', 'articleType', 'season', 'material', 'usage', 'year']
# )

# upload_img = st.file_uploader("Choose an image")

# if upload_img is not None:
#     if save_img(upload_img):
#         st.image(Image.open(upload_img), width=224)
#         st.header("File uploaded successfully!")
#         features = feature_extraction(os.path.join("uploads", upload_img.name), model)
        
#         uploaded_img_id = upload_img.name.split('.')[0]
        
#         # Attempt to use pre-generated JSON features; if not found, try dynamic extraction
#         if int(uploaded_img_id) in pre_generated_json_features['id'].values:
#             uploaded_img_json_features = pre_generated_json_features
#         else:
#             uploaded_img_json_features = extract_json_features_for_image(uploaded_img_id, test_json_path)
#             if uploaded_img_json_features.empty:
#                 st.error("JSON features for the uploaded image could not be found or extracted.")
#                 st.stop()
#             uploaded_img_json_features = pd.DataFrame([uploaded_img_json_features])

#         # Generate and display recommendations
#         # Generate and display recommendations with attribute values
#         recommended_images, recommended_indices = prod_recom(features, feature_list, pre_generated_json_features, attribute, uploaded_img_id)
        
#         # for img_path in recommended_images:
#         #     st.image(Image.open(img_path), width=100, caption="Recommended Product")
#         # Display recommended products with attribute values
#         st.write("Recommended Products:")
#         for img_path in recommended_images:
#             # Extract the product ID from the image filename/path
#             product_id = os.path.basename(img_path).split('.')[0]
    
#         # Fetch the attribute value for the recommended product safely
#         attr_values = pre_generated_json_features.loc[pre_generated_json_features['id'] == int(product_id), attribute]
    
#         # Check if attr_values is not empty
#         if not attr_values.empty:
#             attr_value = attr_values.iloc[0]  # Safely access the first element
#             caption = f"{attribute} = {attr_value}"
#         else:
#             caption = "Attribute value not found"
    
#         st.image(Image.open(img_path), width=100, caption=caption)
#     else:
#         st.header("Some error occurred")


# from PIL import Image
# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.layers import GlobalMaxPool2D
# from tensorflow.keras.preprocessing import image
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
# import pickle
# import os
# import time

# # Custom CSS to improve UI
# def load_custom_css():
#     custom_css = """
#     <style>
#         .stContainer {
#             border: 2px solid #4CAF50;
#             border-radius: 10px;
#             padding: 10px;
#             margin-bottom: 20px;
#         }
#         /* You can add more styles here */
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# load_custom_css()

# st.title('Clothing Recommender Engine')

# # Define model
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
# model = tf.keras.Sequential([model, GlobalMaxPool2D()])


# # Load stored features and images
# file_img = pickle.load(open(r'images.pkl', 'rb'))
# feature_list = pickle.load(open(r'features.pkl', 'rb'))

# # Load and filter products_df
# products_df = pd.read_csv('C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\styles.csv', error_bad_lines=False)
# valid_ids = set(products_df['id'])

# # Function to save uploaded image
# def Save_img(upload_img):
#     try:
#         with open(os.path.join('uploads', upload_img.name), 'wb') as f:
#             f.write(upload_img.getbuffer())
#         return True
#     except Exception as e:
#         st.error(f"Error saving image: {e}")
#         return False

# # Function to extract features from an image
# def feature_extraction(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_arr = image.img_to_array(img)
#     expanded_img_arr = np.expand_dims(img_arr, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_arr)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#     return normalized_result

# # Function for product recommendation
# def prod_recom(features, feature_list):
#     nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
#     nbrs.fit(feature_list)
#     distances, indices = nbrs.kneighbors([features])
#     return indices

# upload_img = st.file_uploader("Choose an image")

# if upload_img is not None:
#     if Save_img(upload_img):
#         st.image(Image.open(upload_img), width=224)
#         st.header("File uploaded successfully!")
#         features = feature_extraction(os.path.join("uploads", upload_img.name), model)
#         progress_text = "Please wait! Analysing Data and Generating Recommendations."
#         my_bar = st.progress(0)
#         for percent_complete in range(100):
#             time.sleep(0.02)
#             my_bar.progress(percent_complete + 1, text=progress_text)
        
#         ind = prod_recom(features, feature_list)
        
#         # Assuming the recommended products info is ready
#         recommended_products = [products_df.loc[products_df['id'] == int(os.path.basename(file_img[i]).split('.')[0])] for i in ind[0]]
        
#         num_columns = 4
#         rows = len(recommended_products) // num_columns + (1 if len(recommended_products) % num_columns > 0 else 0)
        
#         for i in range(rows):
#             cols = st.columns(num_columns)
#             for j in range(num_columns):
#                 index = i * num_columns + j
#                 if index < len(ind[0]):
#                     with cols[j]:
#                         image_path = file_img[ind[0][index]]
#                         st.image(Image.open(image_path), width=150)
#                         product_id = os.path.basename(image_path).split('.')[0]
#                         product_info = products_df.loc[products_df['id'] == int(product_id)].iloc[0]
#                         st.markdown(f"**{product_info['productDisplayName']}**")
#                         st.caption(f"{product_info['baseColour']}, {product_info['articleType']}")
#                         # Use expanders for more info
#                         with st.expander("See more"):
#                             st.write(f"Season: {product_info['season']}")
#                             st.write(f"Usage: {product_info['usage']}")
#     else:
#         st.header("Some error occurred")

