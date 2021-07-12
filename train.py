# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 06:56:28 2021

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
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset(path):
    data = load_files(path)
    alphabet_files = np.array(data['filenames'])
    targets = np.array(data['target'])
    alphabet_targets = np_utils.to_categorical(targets,10)

    return alphabet_files, alphabet_targets, targets
#%%
# load train and test datasets
train_files, train_targets, raw_train_targets = load_dataset('Test')
test_files, test_targets, raw_test_targets = load_dataset('Test')
#%%
digits = [item[9:-1] for item in sorted(glob("Train/*/"))]


# print statistics about the dataset
print('There are %d total digits' % len(digits))
print('There are %s total alphabet images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training alphabet images.' % len(train_files))
print('There are %d test alphabet images.' % len(test_files))

# Function to display the distribution of data in the training and test sets by alphabet classes
def plot_dist(target_set):
    plt.figure(figsize=(6, 4))
    labels, values = zip(*target_set.items())
    # indexes = np.arange(len(labels))
    width = 0.8
    plt.bar(labels, values, width)
    plt.xlabel('Digts')
    plt.ylabel('Frequency')
    plt.show()
    
#Splitting the Training set into the Training set and Validation set
from sklearn.model_selection import train_test_split

train_files, valid_files, train_targets, valid_targets = train_test_split(train_files, train_targets, test_size=0.2,
                                                                          random_state=0, stratify=raw_train_targets)


# print statistics about the dataset post split
print('There are %d total digits' % len(digits))
print('There are %s total digits images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training digits images.' % len(train_files))
print('There are %d validation digits images.' % len(valid_files))
print('There are %d test digits images.\n' % len(test_files))

### Prepare the Training, Validation and Test Datasets
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(64, 64), grayscale=True)
    # convert PIL.Image.Image type to 3D tensor with shape (64, 64, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 64, 64, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

#%%
# pre-process the data for Keras. We rescale the images by dividing every pixel in every image by 255.
# So the scale is now 0-1 instead of 0-255.
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255




#%%

# print number of training, validation, and test images
print(train_tensors.shape[0], 'train samples')
print(valid_tensors.shape[0], 'valid samples')
print(test_tensors.shape[0], 'test samples')

#%%

from CNN_Model import  cnn_model

model = cnn_model(train_tensors)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# generate an empty weight
import h5py
hf = h5py.File('saved_models/weights.best.from_deepcnnwithDO.hdf5', 'w')
hf.close()

from keras.callbacks import ModelCheckpoint

epochs = 50
batch_size = 128

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_deepcnnwithDO.hdf5',
                               verbose=1, save_best_only=True)
#%%
print('Training without augmentation..................')
model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

model.load_weights('saved_models/weights.best.from_deepcnnwithDO.hdf5')

#%%


from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.2,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.2,  # randomly shift images vertically (10% of total height)
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')  # randomly rotate images by 15 degrees


# create and configure augmented image generator
datagen_valid = ImageDataGenerator(
    width_shift_range=0.2,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.2,  # randomly shift images vertically (10% of total height)
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')

# fit augmented image generator on data
datagen_train.fit(train_tensors)
datagen_valid.fit(valid_tensors)


from keras.callbacks import ModelCheckpoint

batch_size = 1024
epochs = 60

# train the model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.with_augmentation_new.hdf5', verbose=1,
                               save_best_only=True)

print('Training with augmentation..................')

model.fit_generator(datagen_train.flow(train_tensors, train_targets, batch_size=batch_size),
                   
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=datagen_valid.flow(valid_tensors, valid_targets, batch_size=batch_size),
                 )








