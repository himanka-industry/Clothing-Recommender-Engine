
### Importing required libraries####
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm import tqdm
import pickle
import os


############################ Defining Model##############################################
model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False
model=tf.keras.Sequential([model,GlobalMaxPool2D()])
model.summary()

############### One time Code: need to extract features of 44k images, U can run this  ######
def image_preprocess(path,model):
    img=image.load_img(path, target_size=(224,224))
    img_arr=image.img_to_array(img)
    ex_img_arr=np.expand_dims(img_arr,axis=0)
    pre_pr_img=preprocess_input(ex_img_arr)
    result=model.predict(pre_pr_img).flatten()
    normal_result=result/norm(result)
    return normal_result
# path for the small dataset
path=r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-large\\fashion-dataset\\images_tr'

# path for the big dataset
# path=r'C:\\Users\\Himankak\\Documents\\Recommender Models\\Data\\Clothing-dataset-small\\myntradataset\\images'

images=[os.path.join(path,files) for files in os.listdir(path)]

pickle.dump(images,open('images.pkl','wb'))
feature_list=[]
for file in tqdm(images):
    feature_list.append(image_preprocess(file, model))
pickle.dump(feature_list,open('features.pkl','wb'))
#####################end #########################################################