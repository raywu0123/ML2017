__author__ = 'ray'
import csv
import os
import argparse
from keras.models import load_model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils,plot_model
from scipy import ndimage

def read_file(file_name,stat):
    if stat=='train':
        number=280#############################
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

def main():
    model_path = './models/04231500.h5'
    emotion_classifier = load_model(model_path)
    print('model loaded.')

    x_train,y_train=read_file('train.csv',stat='train')
    x_train/=255
    y_train = np_utils.to_categorical(y_train, 7)
    x_train=np.expand_dims(x_train,axis=4)
    print(x_train[0].shape)
    input('read in files......pause')
    private_pixels=x_train

    input_img = emotion_classifier.input
    img_ids = [20]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(np.array([private_pixels[idx]]))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])
        h=fn([np.array([private_pixels[idx]]),0])
        #heatmap = np.array([[0.6]*48]*48)
        heatmap=h[0].reshape(48,48)
        heatmap=np.abs((heatmap-np.mean(heatmap))/np.max(heatmap))
        heatmap=ndimage.gaussian_filter(heatmap,sigma=1.5)
        print(heatmap)
        thres = 0.20
        see = private_pixels[idx].reshape(48, 48)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('./problem4/saliency_original_'+str(idx), dpi=100)

        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('./problem4/sacliency_map_'+str(idx), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('./problem4/saliency_masked_'+str(idx), dpi=100)



if __name__ == "__main__":
    main()