from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence,text
import numpy as np
from keras.models import Sequential,load_model,model_from_json
from keras.layers import Dense, Embedding,LSTM,TimeDistributed,Bidirectional,Dropout,Flatten,Conv1D
from keras.initializers import constant
from keras.optimizers import Adam
import keras.backend as K
import json

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)
def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

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

def readfile_test(path):
    file=open(path,"r",encoding='utf-8')
    i=0
    tags=[]
    texts=[]
    n_line=1
    for line in file:
        if n_line==1: n_line+=1
        else:
            line_split=line.split(',',1)
            #print(line_split)
            texts.append(line_split[1])
            #print(len(tags),line_split[0])
    return  texts

def encode_tags(tags,given_enc=None):
    encoded_tags=[]
    encoder=LabelEncoder()
    tag_ex=[]
    for tag in tags:
        for label in tag:
            if label not in tag_ex:
                tag_ex.append(label)
    #print(tag_ex,tag_ex.__len__())
    if given_enc==None:
        encoder.fit(tag_ex)
    else:
        encoder=given_enc

    for tag in tags:
        encoded_tags.append(np.sum(np_utils.to_categorical(encoder.transform(tag),len(tag_ex)),axis=0))
        #print(encoded_tags[-1])
    return np.array(encoded_tags),encoder

def decode_tags(encoded_tags,encoder,thres=0.5):
    encoded_tags=np.floor(encoded_tags/thres)
    decoded_tags=[]
    for tag in encoded_tags:
        numb_tag=np.nonzero(tag)
        #print(numb_tag)
        label=encoder.inverse_transform(numb_tag)[0]
        #print(label)
        decoded_tags.append(label)
    return decoded_tags
def pre_texts(texts):
    num_words=10000
    max_len=310
    tokenizer=text.Tokenizer(nb_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequence_texts=tokenizer.texts_to_sequences(texts)
    padded_sequence_texts=sequence.pad_sequences(sequence_texts,maxlen=max_len)
    #padded_sequence_texts=np.reshape(padded_sequence_texts,(padded_sequence_texts.shape[0],1,padded_sequence_texts.shape[1]))
    return padded_sequence_texts,tokenizer

def pre_texts_test(texts,given_tok,length):
    tokenizer=given_tok
    sequence_texts=tokenizer.texts_to_sequences(texts)
    padded_sequence_texts=sequence.pad_sequences(sequence_texts,maxlen=length)
    #padded_sequence_texts=np.reshape(padded_sequence_texts,(padded_sequence_texts.shape[0],1,padded_sequence_texts.shape[1]))
    return padded_sequence_texts,tokenizer

def write_file(path,tags):
    with open(path,mode='w') as file:
        file.write('"id","tags"\n')
        for index,tag in enumerate(tags):
            file.write('"'+str(index)+'",')
            join_tag=' '.join(tag)
            file.write('"'+join_tag+'"\n')
    file.close()
if __name__=='__main__':
    train_tags,train_texts=readfile('train_data.csv')
    train_encoded_tags,encoder=encode_tags(train_tags)
    train_padded_sequence_texts,tokenizer=pre_texts(train_texts)

    print('train tag shape:',train_encoded_tags.shape)
    print('train text shape:',train_padded_sequence_texts.shape)

    test_texts=readfile_test('test_data.csv')
    test_padded_sequence_texts=pre_texts_test(test_texts,tokenizer,length=train_padded_sequence_texts.shape[1])[0]

    model=load_model('rnn_model_bitime.h5',custom_objects={'fmeasure':fmeasure})
    print('model loaded.')

    test_encoded_tags=model.predict(test_padded_sequence_texts)
    print('test tag shape:',test_encoded_tags.shape)
    print('test text shape:',test_padded_sequence_texts.shape)

    test_tags=decode_tags(test_encoded_tags,encoder,thres=0.5)
    write_file('rnn_bitime_pred.csv',test_tags)