# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from skimage.data import imread 
import json
import bson
import io
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from keras.utils import np_utils

f = open('train_example.bson', 'rb')
bs = f.read() # reads bytes from the specified file
docs = bson.decode_all(bs)
docs = bson.decode_file_iter(open('train_example.bson', 'rb'))
data = pd.DataFrame.from_dict(docs)
data.head()

#Printing sample images for each category
for i in range(5):
    picture = imread(io.BytesIO(data.imgs[i][0]['picture']))
    plt.figure()
    plt.imshow(picture)

X = []
for i in range(data.shape[0]):
    X.append(imread(io.BytesIO(data.imgs[i][0]['picture'])))
    
X = np.array(X,dtype=np.float32)/255. 
X.shape

y = data.category_id.values
y.shape

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)
dummy_y.shape
num_classes = dummy_y.shape[1]
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,dummy_y,test_size=0.3)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

encoded_y
encoder.inverse_transform(encoded_y)

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

model = Sequential()
model.add(Conv2D(filters=5,input_shape=(180,180,3),kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=num_classes,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=16,epochs=5,validation_data=(X_test,Y_test))





submission = pd.read_csv('sample_submission.csv', index_col='_id')
submission['category_id']=1


Inp=Input(shape=(180,180,3))

x = Conv2D(32, kernel_size=(3, 3), activation='relu',name = 'Conv_01')(Inp)
x = Conv2D(64, (3, 3), activation='relu',name = 'Conv_02')(x)
x = MaxPooling2D(pool_size=(2, 2),name = 'MaxPool_01')(x)
x = Dropout(0.25,name = 'Dropout_01')(x)
x = Flatten(name = 'Flatten_01')(x)
x = Dense(128, activation='relu',name = 'Dense_01')(x)
x = Dropout(0.5,name = 'Dropout_02')(x)
output = Dense(36, activation='softmax',name = 'Dense_02')(x)
model = Model(Inp,output)
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 16
epochs = 10
hist = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks = None,
          validation_data=(X_test, Y_test))