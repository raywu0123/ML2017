__author__ = 'ray'
import numpy as np
import csv
import  sys
from keras.models import load_model

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

def write_file(filename,h_theta):
    with open(filename, 'w') as csvfile:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for start_point in range(h_theta.shape[0]):
            pm25_prediction=int(h_theta[start_point][0])
            print(str(start_point)+","+str(pm25_prediction))
            writer.writerow({'id': str(start_point), 'label': str(pm25_prediction)})


print(sys.argv)
test_filename=sys.argv[1]
prediction_filenamce=sys.argv[2]

x_test=read_file(test_filename,stat='test')[0]
x_test/=255
x_test=np.expand_dims(x_test,axis=4)
print(x_test.shape)


model=load_model('CNN_model.h5')
print('model loaded')

y_train=model.predict(x_test)
print(y_train)
y_train_noncat=np.expand_dims(np.argmax(y_train,1),1)
print(y_train_noncat)
write_file(prediction_filenamce,y_train_noncat)

print(model.summary())
#plot_model(model, to_file='model_DNN.png',show_shapes=True)

