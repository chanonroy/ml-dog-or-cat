import os
import cv2
import numpy as np
import models

IMG_SIZE = 50
LR = 1e-3

PATH = os.getcwd()
IMAGE_PATH = PATH + '/images/340.jpg'
MODEL_NAME = 'dogsvscats-0.001-6conv-basic'

class PetClassifier():
    def __init__(self):
        self.model = models.setup_model(MODEL_NAME, IMG_SIZE, LR)

    def parse_img_from_path(self, path):
        print(path)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        return np.array(img).reshape(IMG_SIZE, IMG_SIZE, 1)

    def predict(self, path):
        # cat = [1,0]
        # dog = [0,1]
        img_matrix = self.parse_img_from_path(path)
        model_out = self.model.predict([img_matrix])
        print(model_out)
        result = ''
        if np.argmax(model_out) == 1:
            result = 'Dog'
            print('Dog')
        else:
            result = 'Cat'
            print('Cat')
        return result

model = PetClassifier()
model.predict(IMAGE_PATH)