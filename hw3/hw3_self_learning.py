__author__ = 'ray'
import numpy as np
import csv
import time
import  sys
import random
from keras.models import Sequential,load_model
from keras.utils import np_utils,plot_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,normalization,advanced_activations,ZeroPadding2D
from keras.optimizers import SGD, Adam,rmsprop,adagrad,adadelta
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
input('read in files......pause')

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,data_format='channels_last')
datagen.fit(x_train)
#model=load_model('04222009.h5')

model = Sequential()

model.add(Convolution2D(filters=16,kernel_size=5,input_shape=(48,48,1),padding='same'))
model.add(advanced_activations.LeakyReLU())
model.add(MaxPooling2D((2,2)))##

model.add(Convolution2D(filters=64,kernel_size=5,padding='same'))
model.add(advanced_activations.LeakyReLU())
model.add(MaxPooling2D((2,2)))##

model.add(Convolution2D(filters=128,kernel_size=5,padding='same'))
model.add(advanced_activations.LeakyReLU())
model.add(MaxPooling2D((2,2)))##

model.add(Convolution2D(filters=128,kernel_size=5,padding='same'))
model.add(advanced_activations.LeakyReLU())
model.add(MaxPooling2D((2,2)))##

model.add(Flatten())

model.add(Dense(output_dim=2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=7))
model.add(Activation('softmax'))

print(model.summary())

opt=Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
Con=True

r=0.5
x_train_labeled=x_train[:int(28000*r),:,:,:]
y_train_labeled=y_train[:int(28000*r),:]

x_train_unlabeled=x_train[int(28000*r):,:,:,:]

print(x_train_labeled.shape,x_train_unlabeled.shape)
input()

model.fit_generator(datagen.flow(x_train_labeled, y_train_labeled, batch_size=1),steps_per_epoch=len(x_train_labeled)/100, epochs=1)
print('learned from labeled data')
y_train_unlabeled=model.predict(x_train_unlabeled)
print('predicted result for unlabeled data')
y_train_unlabeled=np_utils.to_categorical(np.expand_dims(np.argmax(y_train_unlabeled,1),1), 7)
print(y_train_unlabeled)

x_train_full=x_train
y_train_self=np.concatenate((y_train_labeled,y_train_unlabeled))
print(y_train.shape)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=1),steps_per_epoch=len(x_train), epochs=1)


model.save('self_train_05021600.h5')

