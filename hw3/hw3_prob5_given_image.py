__author__ = 'ray'
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import csv
import numpy as np

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



def main():
    emotion_classifier = load_model('./models/04231500.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['leaky_re_lu_1']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


    x_train,y_train=read_file('train.csv',stat='train')
    x_train/=255
    x_train=np.expand_dims(x_train,axis=4)
    choose_id=10
    print('data loaded.')

    photo=np.array([x_train[choose_id]])
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/4, 4, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        fig.savefig('./problem5/given_10')


if __name__ == "__main__":
    main()