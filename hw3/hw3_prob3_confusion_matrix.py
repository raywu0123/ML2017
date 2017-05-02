__author__ = 'ray'
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def read_file(file_name,stat):
    lower=0
    if stat=='train':
        upper=28000
    elif stat=='test':
        upper=7178
    elif stat=='valid':
        lower=28000
        upper=28709
    number=upper-lower

    X=np.array([[[0.0]*48]*48]*number,dtype='float32')
    Y=np.array([[0.0]]*number,dtype='float32')
    with open(file_name, newline='', encoding='Big5') as f:
        reader = csv.reader(f)
        count=0
        for row in reader:
            if count>lower:
                X[count-lower-1]=np.array(row[1].split(),dtype='float32').reshape(48,48)
                Y[count-lower-1]=np.array(row[0],dtype='float32')
            if count==upper:
                break
            count+=1
    return X,Y

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    x_train,y_train=read_file('train.csv',stat='train')
    x_train/=255
    y_train=(y_train.T)[0]
    x_train=np.expand_dims(x_train,axis=4)

    emotion_classifier = load_model('04231500.h5')
    input('model loaded.')
    np.set_printoptions(precision=2)
    dev_feats = x_train
    predictions = emotion_classifier.predict_classes(dev_feats)
    te_labels = y_train

    #print(te_labels,predictions)
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()

main()