# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:31:57 2021

@author: User
"""

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
from keras.preprocessing import image                  
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from PIL import ImageFile                           
ImageFile.LOAD_TRUNCATED_IMAGES = True 

model = load_model('saved_models/weights.best.from_deepcnnwithDO.hdf5')


def trial_prediction(img_path):
    img = load_img(img_path, target_size=(64, 64), grayscale=True)
    x = img_to_array(img)
    tensor = x #np.expand_dims(x, axis=0)
    test_img = np.expand_dims(tensor, axis=0)
    prediction_idx = np.argmax(model.predict(test_img))
    alphbt = ['শুন্য','এক','দুই','তিন','চার','পাঁচ','ছয়','সাত','আট','নয়']
    
    output = alphbt[prediction_idx]
    return output


print("Enter the file path for image.\nPress Ctrl+C to exit.")

while(True):
    loc = input('Image path: ')
    out = trial_prediction(loc)
    print(out)

