__author__ = 'ray'

import pandas as pd
import numpy as np
from keras.layers import Embedding, Dense, Merge,Input,Add,Reshape,Dot,Concatenate,Flatten,Dropout
from keras.models import Sequential,Model,load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model():
    embedding_size=32
    movie_input = Input(shape=[1])
    movie_vec = Embedding(3952+1, embedding_size,embeddings_initializer='random_normal')(movie_input)
    movie_vec=Flatten()(movie_vec)

    user_input = Input(shape=[1])
    user_vec = Embedding(6040+1, embedding_size,embeddings_initializer='random_normal')(user_input)
    user_vec=Flatten()(user_vec)

    user_feat=Input(shape=[3])

    merge=Concatenate()([movie_vec,user_vec,user_feat])
    merge=Dense(500,activation='relu')(merge)
    merge = Dense(1, activation='relu')(merge)
    model = Model([movie_input, user_input,user_feat], merge)


    print(model.summary())
    return model

def get_user_feat(user_id,users):
    Length=user_id.shape[0]
    result=np.zeros((Length,4),dtype=int)
    users=users[:,:4]
    user_feat_index=0
    for index,user in enumerate(users):
        if user[1]=='M':
            users[index,1]=1
        else:
            users[index,1]=0
    users=np.array(users,dtype=int)
    for result_index,user in enumerate(user_id):
        while user!=users[user_feat_index][0]:
            user_feat_index+=1
        result[result_index]=users[user_feat_index]
    return result


train_data=pd.read_csv('train.csv',index_col=[0])
users=np.asarray(pd.read_csv('users.csv',sep="::"))
movies=pd.read_csv('movies.csv',sep="::")

user_id=np.asarray([train_data['UserID']]).T
user_feats=get_user_feat(user_id,users)
movie_id=np.asarray([train_data['MovieID']]).T
rate=np.asarray([train_data['Rating']]).T

'''
################ Rating Normalization
MEAN=np.mean(rate)
STD=np.std(rate)
rate=(rate-MEAN)/STD
####################################
'''
full=np.concatenate((movie_id,user_feats),axis=1)
full=np.concatenate((full,rate),axis=1)
full=np.asarray(full,dtype=float)
full[:,3]=(full[:,3]-np.mean(full[:,3]))/np.std(full[:,3])
full[:,4]=(full[:,4]-np.mean(full[:,4]))/np.std(full[:,4])
print(full)
np.random.shuffle(full)
print(full.shape)

model=build_model()
model.compile(optimizer='adam',loss='mean_squared_error')

earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best_model.hdf5',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_loss',
                                     mode='min')
history=model.fit([full[:,0],full[:,1],full[:,2:5]],full[:,5],validation_split=0.1,batch_size=1000,epochs=50,callbacks=[earlystopping,checkpoint],verbose=2)
best_model=load_model('best_model.hdf5')
############################################ Train Model

test_data=pd.read_csv('test.csv',index_col=[0])
sample=pd.read_csv('SampleSubmission.csv',index_col=[0])
test_user_id=np.asarray([test_data['UserID']]).T
test_movie_id=np.asarray([test_data['MovieID']]).T
test_user_feats=get_user_feat(test_user_id,users)
test_user_feats=np.asanyarray(test_user_feats,dtype=float)
test_user_feats[:,2]=(test_user_feats[:,2]-np.mean(test_user_feats[:,2]))/np.std(test_user_feats[:,2])
test_user_feats[:,3]=(test_user_feats[:,3]-np.mean(test_user_feats[:,3]))/np.std(test_user_feats[:,3])
print(test_user_feats)
prediction=best_model.predict([test_movie_id,test_user_feats[:,0],test_user_feats[:,1:]])
#prediction=prediction*STD+MEAN
prediction[prediction<0]=0
prediction[prediction>5]=5
#prediction=np.round(prediction)
print(prediction.shape)
#print(sample)
sample['Rating']=prediction.T[0]
sample.to_csv('prediction.csv')
print(prediction)