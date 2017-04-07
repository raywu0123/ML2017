import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


# categorical_crossentropy

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255
    print(x_train[0])
    print(y_train[0])
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
model.add(Dense(input_dim=28 * 28, output_dim=689))
model.add(Activation('relu'))
model.add(Dense(output_dim=689))
model.add(Activation('relu'))
model.add(Dense(output_dim=689))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10)

score = model.evaluate(x_test, y_test)
print('\nTest Acc:', score[1])
