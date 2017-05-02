import numpy as np
import csv
import time
import  sys
import random
from keras.models import Sequential,load_model
from keras.utils import np_utils,plot_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,normalization,advanced_activations,ZeroPadding2D
from keras.optimizers import SGD, Adam,rmsprop
from keras.regularizers import*
from keras.preprocessing.image import ImageDataGenerator

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
input('read in files......pause__hw3_DNN')

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,data_format='channels_last')

datagen.fit(x_train)

model = Sequential()
model.add(Flatten(input_shape=(48,48,1)))

model.add(Dense(output_dim=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(output_dim=7))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

Con=True
while(Con):
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),steps_per_epoch=len(x_train), epochs=1)
    a=input('Continue?(Y/n)')
    if a=='n':
        Con=False
model.save('04280113.h5')

