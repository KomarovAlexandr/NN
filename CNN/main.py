import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from prettytable import PrettyTable
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds
import tensorflow as tf
import logging
import numpy as np
import time
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import additional_func as af


data = np.array(af.load_data())
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(data[i+1], cmap=plt.get_cmap('gray'))
plt.figure()
data = data.reshape(data.shape[0], 28, 28, 1).astype(float) / 255.0

for i in range(0, 10):
    scaleKoef = 1.0 / np.max(data[i])
    data[i] = data[i] * scaleKoef

Nine_index = [4, 22, 33, 43, 45, 48, 54, 57, 87, 110]
data_1 = []

# Загрузка база
(trainX, trainy), (testX, testy) = mnist.load_data()
# 4 22 33 43 45 48 54 57 87 110
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    #plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    data_1.append(trainX[Nine_index[i]])
    plt.imshow(data_1[i], cmap=plt.get_cmap('gray'))
data_1.append(trainX[Nine_index[9]])
# plt.subplot(330 + 1 + 0)
# plt.imshow(data[9], cmap=plt.get_cmap('gray'))
# plt.subplot(330 + 1 + 1)
# plt.imshow(data_1[9], cmap=plt.get_cmap('gray'))
plt.show()

# Вывод размерности данных
print("Train: X=%s, y=%s" % (trainX.shape, trainy.shape))
print("Test: X=%s, y=%s" % (testX.shape, testy.shape))
# Размер картинок
imageSize_X = trainX.shape[1]
imageSize_Y = trainX.shape[2]
# Размерность выходного слоя
classSize = 10
# Подготовка данных
trainX = trainX.reshape(trainX.shape[0], imageSize_X, imageSize_Y, 1)
trainX.astype('float32')
trainX = trainX / 255.0
testX = testX.reshape(testX.shape[0], imageSize_X, imageSize_Y, 1)
testX.astype('float32')
testX = testX / 255.0
trainy = keras.utils.to_categorical(trainy, classSize)
testy = keras.utils.to_categorical(testy, classSize)

data_1 = np.array(data_1)
data_1 = data_1.reshape(data_1.shape[0], imageSize_X, imageSize_Y, 1)
data_1.astype('float32')
data_1 = data_1 / 255.0


model = Sequential()
model.add(Conv2D(30, (5, 5), activation='relu', input_shape=(imageSize_X, imageSize_Y, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu', input_shape=(imageSize_X, imageSize_Y, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
print("Model Fit ...")
model.fit(trainX, trainy, epochs=10, batch_size=100,
					verbose=1, validation_data=(testX, testy))
print("Model Predict ...")
predict = model.predict(data)

t = PrettyTable(['Numb', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

for i in range(0, 10):
	pred = list()
	for j in range(0, 10):
		pred.append("%.2f%%" % (predict[i,j] * 100))
	t.add_row([i, pred[0],
					pred[1],
					pred[2],
					pred[3],
					pred[4],
					pred[5],
					pred[6],
					pred[7],
					pred[8],
					pred[9]])
print(t)