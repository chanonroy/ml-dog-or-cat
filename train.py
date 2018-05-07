import cv2                  # resizing image
import numpy as np
import os
from random import shuffle
from tqdm import tqdm       # progress bar for CLI

BASE_DIR = 'X:/Machine_Learning/Data/dogs_vs_cats'
TRAIN_DIR = BASE_DIR + '/train'
TEST_DIR = BASE_DIR + '/test1'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}'.format(LR, '6conv-basic')

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
    np.save('./data/train_data.npy', training_data)
    return training_data

def prepare_test_data():
    # 93.png
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('./data/test_data.npy', testing_data)
    return testing_data

training_data = []

if os.path.isfile('./data/train_data.npy'):
    print('Loading training data...')
    training_data = np.load('./data/train_data.npy')
else:
    print('Generating training data...')
    training_data = prepare_train_data()

# Tensorflow
import models

model = models.setup_model(IMG_SIZE, LR)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')

train = training_data[: -500]
test = training_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)