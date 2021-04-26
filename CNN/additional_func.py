import PIL.ImageOps
import cv2
from matplotlib import image
from os import listdir
from PIL import Image
import numpy as np


def load_data():
    data = []
    for filename in listdir("data"):
        img = cv2.imread("data/" + filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (112, 112))
        cv2.imshow("Image_1", img)
        cv2.waitKey(1)

        img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = np.ones((5, 5), 'uint8')
        img = cv2.dilate(img, kernel)
        img = cv2.blur(img, ksize=(3, 3))

        # scaleKoef = 1.0 / np.max(img)
        # img = img * (1 + scaleKoef)

        cv2.imshow("Image_2", img)
        cv2.waitKey(100)
        img = cv2.resize(img, (28, 28))
        data.append(img)
    return data

