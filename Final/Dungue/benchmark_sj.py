
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,GRU,Conv1D,Flatten
from keras.optimizers import Adam


# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')

LOOk_BACK=100

def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k',
                'reanalysis_avg_temp_k',
                'station_min_temp_c',
                'station_avg_temp_c',
                'station_max_temp_c',
                'reanalysis_precip_amt_kg_per_m2',
                'ndvi_nw',
                'precipitation_amt_mm',
                'station_precip_mm',
                'station_diur_temp_rng_c',]
    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq

def build_model_sj(num_feature,look_back=LOOk_BACK):
    model=Sequential()
    model.add(LSTM(200,input_shape=(look_back,num_feature),return_sequences=True,unit_forget_bias=True))
    model.add(LSTM(200,return_sequences=True,unit_forget_bias=True))
    model.add(LSTM(200,return_sequences=False,unit_forget_bias=True))
    model.add(Dense(1000,activation='linear'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(1,activation='linear'))
    print(model.summary())
    print('\nmodel built.')
    return model

def GEN(feature_label,batchsize=1,look_back=LOOk_BACK):
    batch_features = np.zeros((batchsize, look_back,feature_label.shape[1]))
    batch_labels = np.zeros((batchsize,1))
    while True:
        for i in range(batchsize):
            start_index=np.random.randint(0,feature_label.shape[0]-look_back-1)
            batch_features[i]=feature_label[start_index:start_index+look_back]
            batch_labels[i]=feature_label[start_index+look_back][-1]
            #print(batch_features)
            #print(batch_labels)
        yield batch_features,batch_labels

def get_prediction(test_data,model,look_back=LOOk_BACK):
    '''
    :param test_data: data with labels in the first "look back frames" + the rest without labels
    :param model: the trained model
    :return: labels for the last part
    '''
    num_unknown_labels=test_data.shape[0]-look_back
    for start_index in range(num_unknown_labels):
        test_data[start_index+look_back,-1]=model.predict(np.array([test_data[start_index:start_index+look_back]]))
    return test_data

def main(train=True):
    sj_train, iq_train = preprocess_data('dengue_features_train.csv',
                                    labels_path="dengue_labels_train.csv")##pandas Dataframe objects
    sj_scaler=MinMaxScaler().fit(sj_train)

    sj_train=sj_scaler.transform(sj_train)

    sj_train_subtrain=sj_train[:800]
    sj_train_subtest=sj_train[800:-1]
    print(sj_train_subtrain.shape)
    print(sj_train_subtest.shape)
    num_feature=sj_train.shape[1]

    sj_test,iq_test=preprocess_data('dengue_features_test.csv',
                                    labels_path=None)##pandas Dataframe objects
    print(sj_test.shape)
    sj_test=np.concatenate((sj_test,np.zeros((sj_test.shape[0],1))),axis=1)
    sj_test_scaler=MinMaxScaler().fit(sj_test)
    sj_test=sj_test_scaler.transform(sj_test)
    #test_data=np.concatenate((sj_train[-LOOk_BACK-1:-1],sj_test))
    test_data=sj_train
    print(sj_test.shape)
    print(test_data.shape)
    #################################MORE PREPROCESSING########################################
    if train:
        sj_model=build_model_sj(num_feature)

        subtrain_batchsize=10
        subtest_batchsize=10
        adam = Adam(lr=0.001,decay=0.01,clipvalue=0.1)
        sj_model.compile(optimizer=adam,loss='mean_absolute_error')

        earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(filepath='best_model.hdf5',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_loss',
                                     mode='min')


        print(test_data)## concated (look back part of subtrain) and (subtest)
        print(test_data.shape)

        history=sj_model.fit_generator(generator=GEN(sj_train_subtrain,subtrain_batchsize),
                               steps_per_epoch=sj_train_subtrain.shape[0]/subtrain_batchsize,
                               validation_data=GEN(test_data,subtest_batchsize),
                               validation_steps=test_data.shape[0]/subtest_batchsize,
                               callbacks=[earlystopping,checkpoint],
                               epochs=100)

    best_model=load_model('./models/sj_best_model_1.hdf5')


    x_axis=np.linspace(1,test_data.shape[0],num=test_data.shape[0])
    plt.plot(x_axis,test_data[:,-1],'r')

    sj_train_subtest_prediction=get_prediction(test_data,best_model)
    print(sj_train_subtest_prediction)
    print(sj_train_subtest_prediction.shape)

    plt.plot(x_axis,sj_train_subtest_prediction[:,-1],'b')
    plt.show()



if __name__=='__main__':
    main(train=False)
