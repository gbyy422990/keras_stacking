#coding:utf-8
#Bin  GAO

from __future__ import print_function
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input


#读取标签文件
df = pd.read_csv('labels.csv')
df.head()
n=len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))


#resize训练样本到299x299，便于inception的特征提取
height_299 = 299
width_299 = 299

#resize训练样本到224x224，便于inception的特征提取
height_224 = 224
width_224 = 224


X_299 = np.zeros((n, height_299, width_299, 3), dtype=np.uint8)
X_224 = np.zeros((n, height_224, width_224, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

for i in tqdm(range(n)):
    X_299[i] = cv2.resize(cv2.imread('train/%s.jpg' % df['id'][i]), (height_299, width_299))
    X_224[i] = cv2.resize(cv2.imread('train/%s.jpg' % df['id'][i]), (height_224, width_224))
    y[i][class_to_num[df['breed'][i]]] = 1

#导出特征
def get_features(MODEL, data, height, width):
    cnn_model = MODEL(include_top=False, input_shape=(height, width, 3), weights='imagenet')

    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    print('Feature shape is:',features.shape)
    return features

inception_features = get_features(InceptionV3, X_299, height_299, width_299)
xception_features = get_features(Xception, X_299, height_299, width_299)
resnet = get_features(ResNet50,X_224, height_224, width_224)
features = np.concatenate([inception_features, xception_features, resnet], axis=-1)

inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features, y, batch_size=128, epochs=200, validation_split=0.1)

df2 = pd.read_csv('sample_submission.csv')
n_test = len(df2)

def get_feature_test(height,width):
    X_test = np.zeros((n_test, height, width, 3), dtype=np.uint8)
    for i in tqdm(range(n_test)):
        X_test[i] = cv2.resize(cv2.imread('test/%s.jpg' % df2['id'][i]), (width, width))

    return X_test

X_test_299=get_feature_test(height_299,width_299)
inception_features = get_features(InceptionV3, X_test_299,height_299,width_299)
xception_features = get_features(Xception, X_test_299,height_299,width_299)

X_test_224 = get_feature_test(height_224,width_224)
resnet = get_features(ResNet50, X_test_224,height_224, width_224)
features_test = np.concatenate([inception_features, xception_features,resnet], axis=-1)

y_pred = model.predict(features_test, batch_size=128)
for b in breed:
    df2[b] = y_pred[:,class_to_num[b]]
df2.to_csv('pred.csv', index=None)
