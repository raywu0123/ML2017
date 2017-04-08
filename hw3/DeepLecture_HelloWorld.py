import numpy as np
import csv
import time
import  sys
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam


def read_file(file_name,stat):
    if stat=='train':
        number=28000
    elif stat=='test':
        number=7178
    X=np.array([[[0.0]*48]*48]*number,dtype='float32')
    Y=np.array([[0.0]]*number,dtype='float32')
    with open(file_name, newline='', encoding='Big5') as f:
        reader = csv.reader(f)
        count=0
        for row in reader:
            if count!=0:
                X[count-1]=np.array(row[1].split(),dtype='float32').reshape(48,48)
                Y[count-1]=np.array(row[0],dtype='float32')
            if count==number:
                break
            count+=1
    return X,Y

x_train,y_train=read_file('train.csv',stat='train')
x_train/=255
y_train = np_utils.to_categorical(y_train, 7)
x_train=np.expand_dims(x_train,axis=4)
print(x_train.shape)
input('read in files......pause')

model = Sequential()
model.add(Convolution2D(filters=25,kernel_size=5,input_shape=(48,48,1),data_format='channels_last'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(filters=50,kernel_size=5))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())

model.add(Dense(output_dim=500))
model.add(Activation('relu'))
model.add(Dense(output_dim=500))
model.add(Activation('relu'))
model.add(Dense(output_dim=500))
model.add(Activation('relu'))
model.add(Dense(output_dim=7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=40)
model.save('04081514.h5')


