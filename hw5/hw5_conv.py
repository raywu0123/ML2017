author='b05901189_raywu'
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence,text
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding,LSTM,TimeDistributed,Bidirectional,Conv1D,Dropout,Flatten
from keras.initializers import constant
from keras.optimizers import Adam
import json

import keras.backend as K

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def readfile(path):
    file=open(path,"r",encoding='utf-8')
    i=0
    tags=[]
    texts=[]
    n_line=1
    for line in file:
        if n_line==1: n_line+=1
        else:
            line_split=line.split('"',2)
            tags.append(line_split[1].split(" "))
            texts.append(line_split[2])
            #print(len(tags),line_split[0])
    return tags,texts

def encode_tags(tags):
    encoded_tags=[]
    encoder=LabelEncoder()
    tag_ex=[]
    for tag in tags:
        for label in tag:
            if label not in tag_ex:
                tag_ex.append(label)
    #print(tag_ex,tag_ex.__len__())
    encoder.fit(tag_ex)
    for tag in tags:
        encoded_tags.append(np.sum(np_utils.to_categorical(encoder.transform(tag),len(tag_ex)),axis=0))
        #print(encoded_tags[-1])
    return np.array(encoded_tags),encoder

def pre_texts(texts):
    tokenizer=text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequence_texts=tokenizer.texts_to_sequences(texts)
    padded_sequence_texts=sequence.pad_sequences(sequence_texts)
    print(type(padded_sequence_texts))
    #padded_sequence_texts=np.reshape(padded_sequence_texts,(padded_sequence_texts.shape[0],1,padded_sequence_texts.shape[1]))
    return padded_sequence_texts,tokenizer

def rnn_model(tags,texts):
    num_features=np.max(texts)+1
    #print(num_features)
    num_categories=tags.shape[1]
    embedding_vecor_length = 512
    model = Sequential()
    model.add(Embedding(num_features, embedding_vecor_length, input_length=texts.shape[1]))
    model.add(Conv1D(64, 5, border_mode='same'))
    model.add(TimeDistributed(Dense(512,activation='linear')))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_categories,activation='hard_sigmoid'))

    print(model.summary())
    return model

if __name__=='__main__':
    tags,texts=readfile('train_data.csv')
    encoded_tags,encoder=encode_tags(tags)
    padded_sequence_texts,tokenizer=pre_texts(texts)
    print('tag shape:',encoded_tags.shape)
    print('text shape:',padded_sequence_texts.shape)

    ##train model
    model=rnn_model(encoded_tags,padded_sequence_texts)
    opt=Adam(lr=1e-3,clipvalue=1e-1)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy',f1_score])
    model.fit(padded_sequence_texts,encoded_tags,batch_size=17,epochs=5,validation_split=0.1)
    json_string=model.to_json()

    with open('conv_model.txt', 'w') as outfile:
        json.dump(json_string, outfile)
