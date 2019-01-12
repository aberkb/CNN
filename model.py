# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
from tensorflow import keras
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

data = pd.read_csv("hmnist_64_64_L.csv") 

Y = data["label"]
data.drop(["label"],axis=1, inplace=True)
X = data

img = X.iloc[75].as_matrix()
img = img.reshape(64,64)
X = X.values.reshape(-1,64,64,1)
Y = Y.values
Y = to_categorical(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=151)

model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (64,64,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))          
model.add(Dense(128,activation = "relu"))
model.add(Dense(128,activation = "relu"))
model.add(Dense(32,activation = "relu"))

model.add(Dense(9, activation = "softmax"))

model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
        rotation_range=0.5, 
        zoom_range = 0.5, 
        width_shift_range=0.5,  
        height_shift_range=0.5, 
        horizontal_flip=True, 
        vertical_flip=True)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train,y_train, batch_size=200),
                              epochs = 50, validation_data = (x_test,y_test), steps_per_epoch=200)

Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
print(classification_report(Y_true, Y_pred_classes))
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(18, 16))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="summer_r", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
