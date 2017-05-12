__author__ = 'ray'
import numpy as np
from sklearn.decomposition import PCA
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import csv
import sys

def write_file(filename,h_theta):
    with open(filename, 'w') as csvfile:
        fieldnames = ['SetId', 'LogDim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for start_point in range(h_theta.shape[0]):
            pm25_prediction=h_theta[start_point]
            print(str(start_point)+","+str(pm25_prediction))
            writer.writerow({'SetId': str(start_point), 'LogDim': str(pm25_prediction)})
def print_stat(data):
    print('mean= ',np.mean(data))
    print('max= ',np.max(data))
    print('min= ',np.min(data))

d_i=[]
def train_model():
    train_data=np.load('dim_train_xy_pca.npz')
    train_x=train_data['arr_0']
    train_y=np.array([train_data['arr_1']]).T
    print(train_x.shape)
    print(train_y.shape)
    #train_y=np_utils.to_categorical(train_y-1,60)
    #print(train_y.shape)

    model = Sequential()

    model.add(Dense(input_shape=(100,),output_dim=2000))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=2000))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=2000))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])

    model.fit(train_x[:9000],train_y[:9000],batch_size=100,epochs=10)
    print(model.evaluate(train_x[9000:],train_y[9000:]))

    model.save('pca_dim_model.h5')

#train_model()

model=load_model('pca_dim_model.h5')
test_data=np.load(sys.argv[1])
print('test_data_loaded.')
num_data=len(test_data.files)
for i in range(num_data):
    test_x=PCA().fit(test_data[str(i)]).explained_variance_ratio_.cumsum()
    input(test_data[str(i)].shape)
    test_x=np.array([test_x])
    test_y=model.predict(test_x)
    print(i,test_y[0][0],np.log(test_y[0][0]))
    d_i.append(test_y[0][0])

d_i=np.array(d_i)
print_stat(d_i)
log_d_i=np.log(d_i).T


print('log_d_i=\n',log_d_i)
write_file(sys.argv[2],log_d_i)
test_data.close()
