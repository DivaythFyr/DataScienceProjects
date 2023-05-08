import time
import cv2
import numpy as np
from keras.applications.mobilenet import preprocess_input
from Kick_Punch_Classifier.model import DeepModel


class ImageClassifier:
    def __init__(self):
        self.all_skus = {}
        self.model = DeepModel()
        self.predict_time = 0
        self.time_search = 0
        self.count_frame = 0
        self.top_k = 5

    def extract_features_from_img(self, cur_img):
        cur_img = cv2.resize(cur_img, (224, 224))
        img = preprocess_input(cur_img)
        img = np.expand_dims(img, axis=0)
        print("Input shape:", img.shape)
        feature = self.model.extract_feature(img)
        return feature

    def predict(self, img):
        self.count_frame += 1
        before_time = time.time()
        target_features = self.extract_features_from_img(img)
        self.predict_time += time.time() - before_time
        max_distance = 0
        result_Class = 0

        for Class, features_all in self.all_skus.items():
            for features in features_all:
                cur_distance = self.model.cosine_distance(target_features, features)
                cur_distance = cur_distance[0][0]
                if cur_distance > max_distance:
                    max_distance = cur_distance
                    result_Class = Class

        return result_Class, max_distance

    def add_img(self, img_path, id_img):
        img = cv2.imread(img_path)
        cur_img = img
        feature = self.extract_features_from_img(cur_img)
        if id_img not in self.all_skus:
            self.all_skus[id_img] = []
        self.all_skus[id_img].append(feature)
        return feature










       