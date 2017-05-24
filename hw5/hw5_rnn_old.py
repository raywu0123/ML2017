author='b05901189_raywu'
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence,text
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding,LSTM,TimeDistributed,Bidirectional,Flatten
from keras.initializers import constant
import sklearn.metrics.classification
from keras.optimizers import Adam
import keras.backend as K

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
    num_words=10000
    max_len=310
    tokenizer=text.Tokenizer(nb_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequence_texts=tokenizer.texts_to_sequences(texts)
    padded_sequence_texts=sequence.pad_sequences(sequence_texts,maxlen=max_len)
    #padded_sequence_texts=np.reshape(padded_sequence_texts,(padded_sequence_texts.shape[0],1,padded_sequence_texts.shape[1]))
    return padded_sequence_texts,tokenizer

def rnn_model(tags,texts,tokenizer):
    '''
    EMBEDDING_DIM=100
    embeddings_index = {}
    word_index = tokenizer.word_index
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    '''
    num_features=np.max(texts)+1
    #print(num_features)
    num_categories=tags.shape[1]


    model = Sequential()
    model.add(Embedding(num_features,output_dim=128,input_length=texts.shape[1]))
    '''
    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=texts.shape[1],
                            trainable=False)
    model.add(embedding_layer)
    '''
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True,bias_initializer=constant(10))))
    model.add(TimeDistributed(Dense(128,activation='linear')))

    model.add(Flatten())
    model.add(Dense(num_categories, activation='sigmoid'))
    print(model.summary())
    return model

if __name__=='__main__':
    tags,texts=readfile('train_data.csv')
    encoded_tags,encoder=encode_tags(tags)
    padded_sequence_texts,tokenizer=pre_texts(texts)
    print('tag shape:',encoded_tags.shape)
    print('text shape:',padded_sequence_texts.shape)
    tag_index=np.arange(0,38)

    for i in range(38):
        print(encoder.inverse_transform(tag_index)[i],np.sum(encoded_tags,axis=0)[i])
    print(np.mean(np.sum(encoded_tags,axis=1)))
    print(np.std(np.sum(encoded_tags,axis=1)))
    input()
    class_weight=10./np.sum(encoded_tags,axis=0)**2
    model=rnn_model(encoded_tags,padded_sequence_texts,tokenizer)
    opt=Adam(lr=0.001,decay=0.02,clipnorm=0.1)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',fmeasure])
    model.fit(padded_sequence_texts,encoded_tags,batch_size=5,epochs=30,validation_split=0.1,class_weight=class_weight)
    model.save('rnn_model.h5')