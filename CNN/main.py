import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from prettytable import PrettyTable
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import additional_func as af
import tensorflow as tf

"""
Флаги
"""
# Загрузить модели
loadModelFlag = 1
# Сохранение модели
saveModelFlag = 0
# Тест на произвольном изображении - 0, Тест на базе из 10 изображений от 0 до 9
freeImageTestFlag = 0
# Флаг отображения загруженных данных и данных для теста
showFlag = 0

# Загрузка
(trainX, trainy), (testX, testy) = mnist.load_data()
# Вывод размерности данных
print("Train: X=%s, y=%s" % (trainX.shape, trainy.shape))
print("Test: X=%s, y=%s" % (testX.shape, testy.shape))
if showFlag:
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
	plt.show()

# Размер картинок
imageSize_X = trainX.shape[1]
imageSize_Y = trainX.shape[2]
shape = (imageSize_X, imageSize_Y, 1)
# Размерность выходного слоя
classSize = 10

testData = af.load_data(shape, oneImage=freeImageTestFlag, showFlag=showFlag)

# Подготовка данных
trainX = trainX.reshape(trainX.shape[0], shape[0], shape[1], shape[2]).astype(float) / 255.0
testX = testX.reshape(testX.shape[0], shape[0], shape[1], shape[2]).astype(float) / 255.0

trainy = keras.utils.to_categorical(trainy, classSize)
testy = keras.utils.to_categorical(testy, classSize)

if loadModelFlag:
	print("Load model ...")
	model = tf.keras.models.load_model('model')
	print(model.summary())
else:
	print("Create model ...")
	model = af.getModel(shape)
	print(model.summary())
	print("Model Fit ...")
	model.fit(trainX, trainy, epochs=10, batch_size=100, verbose=1, validation_data=(testX, testy))

print("Model Predict ...")
predict = model.predict(testData)

if freeImageTestFlag:
	table = PrettyTable(['Numb', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	for i in range(0, 10):
		pred = list()
		for j in range(0, 10):
			pred.append("%.2f%%" % (predict[i, j] * 100))
		table.add_row([i, pred[0], pred[1], pred[2], pred[3],
					   pred[4], pred[5], pred[6], pred[7], pred[8], pred[9]])
	print(table)
else:
	percent = 0.0
	predictNumber = 0
	for i in range(0, 10):
		if predict[0, i] > percent:
			percent = predict[0, i]
			predictNumber = i
	print("This is", predictNumber, "  ", percent * 100, "%")

if saveModelFlag:
	model.save('model')