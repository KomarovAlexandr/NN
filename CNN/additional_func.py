import cv2
from os import listdir
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from matplotlib import pyplot as plt


def load_data(shape, oneImage = 1, showFlag = 0):
    data = []
    if oneImage:
        for filename in listdir("data"):
            img = cv2.imread("data/" + filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (112, 112))
            if showFlag:
                cv2.imshow("Image_1", img)
                cv2.waitKey(1)

            img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = np.ones((5, 5), 'uint8')
            img = cv2.dilate(img, kernel)
            img = cv2.blur(img, ksize=(3, 3))

            if showFlag:
                cv2.imshow("Image_2", img)
                cv2.waitKey(50)
            img = cv2.resize(img, (shape[0], shape[1]))
            data.append(img)
    else:
        for filename in listdir():
            if filename.endswith("jpg"):
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (112, 112))
                if showFlag:
                    cv2.imshow("Image_1", img)
                    cv2.waitKey(1)

                img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)[1]
                kernel = np.ones((5, 5), 'uint8')
                img = cv2.dilate(img, kernel)
                img = cv2.blur(img, ksize=(3, 3))

                if showFlag:
                    cv2.imshow("Image_2", img)
                    cv2.waitKey(50)
                img = cv2.resize(img, (shape[0], shape[1]))
                data.append(img)

    data = np.array(data)
    if showFlag and oneImage:
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(data[i+1], cmap=plt.get_cmap('gray'))
        plt.show()

    data = data.reshape(data.shape[0], shape[0], shape[1], shape[2]).astype(float) / 255.0

    for i in range(0, len(data)):
        scaleKoef = 1.0 / np.max(data[i])
        data[i] = data[i] * scaleKoef
    return data


def getModel(shape):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), activation='relu', input_shape=shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


