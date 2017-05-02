__author__ = 'ray'
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np


model_path='./models/04231500.h5'
nb_filter=16
num_step=320
NUM_STEPS=num_step
RECORD_FREQ=20


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func,filter_images,filter_idx):
    lr=0.1
    for i in range(num_step):
        func=iter_func([input_image_data])
        target=func[0]
        grads=np.array([func[1]])
        if i % RECORD_FREQ==0:
            filter_images[i//RECORD_FREQ].append([input_image_data[0].reshape(48,48),float(target)])
        input_image_data+=lr*np.array(grads)

    return filter_images

def main():
    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ['leaky_re_lu_1']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])


            filter_imgs = grad_ascent(num_step, input_img_data, iterate,filter_imgs,filter_idx)
            ###

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/4, 4, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                print(it,i)
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            fig.savefig('./problem5/'+str(it))

if __name__ == "__main__":
    main()
