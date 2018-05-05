import cv2                  # resizing image
import numpy as np
import os
from random import shuffle
from tqdm import tqdm       # progress bar for CLI

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

BASE_DIR = 'X:/Machine_Learning/Data/dogs_vs_cats'
TRAIN_DIR = BASE_DIR + '/train'
TEST_DIR = BASE_DIR + '/test1'
IMG_SIZE = 50
LEARNING_RATE = 0.001

MODEL_NAME = 'dogsvscats-{}-{}'.format(LEARNING_RATE, '2conv-basic');

# Prepare and Process the Data
def label_img(img):
    # dog.93.png
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

def prepare_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def prepare_test_data():
    # 93.png
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

# train_data = create_train_data()
train_data = np.load('train_data.npy')