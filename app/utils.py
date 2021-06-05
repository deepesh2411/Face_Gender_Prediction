import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cv2
import pickle

# loading the models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))


def pipeline(path, filename, color='bgr'):
    # settings
    gender_pred = ['Male', 'Female']
    font = cv2.FONT_HERSHEY_SIMPLEX
    # step 1: read the image
    img = cv2.imread(path)
    # step 2: convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # step 3: find the faces
    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)  # drawing rectangle
        crop_img = gray[y:y+h, x:x+w]                      # cropping image
        # step 4:  normalization
        crop_img = crop_img/255.0
        # step 5: resize into (100,100)
        if crop_img.shape[0] >= 100:
            crop_img_resize = cv2.resize(crop_img, (100, 100), cv2.INTER_AREA)
        else:
            crop_img_resize = cv2.resize(crop_img, (100, 100), cv2.INTER_CUBIC)
        # step 6: flatten
        crop_img_reshape = crop_img_resize.reshape(1, -1)
        # step 7: substract with mean
        crop_img_mean = crop_img_reshape - mean
        # step 8: get eigen image
        eigen_img = model_pca.transform(crop_img_mean)
        # step 9: pass to ml model
        y_prob = model_svm.predict_proba(eigen_img)[0] 
        y_pred = model_svm.predict(eigen_img)[0]
        # step 10:
        predict = y_prob.argmax()  # 0 or 1
        score = y_prob[predict]
        # step 11:
        text = "%s : %.2f" % (gender_pred[y_pred], score)
        cv2.putText(img, text, (x+5, y-5), font, 1, (255, 255, 0), 2)
    cv2.imwrite('./app/static/predict/{}'.format(filename), img)
