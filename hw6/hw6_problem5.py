__author__ = 'ray'

import pandas as pd
import numpy as np
from keras.layers import Embedding, Dense, Merge,Input,Add,Reshape,Dot,concatenate,Flatten
from keras.models import Sequential,Model,load_model
from matplotlib import pyplot as plt
from sklearn.manifold import*


def get_movie_genres(movies):
    Original_Array=list(movies['Genres'])
    Split_Array=[]
    for genres in Original_Array:
        Split_Array.append(genres.split('|'))
    Categories=[]
    for movie in Split_Array:
        for genres in movie:
            if genres not in Categories:
                Categories.append(genres)
    #print(Categories)
    Result=[]
    for movie in Split_Array:
        if 'Drama' in movie or 'Musical' in movie:
            Result.append('g')
        elif 'Comedy' in movie or 'Romance' in movie:
            Result.append('k')
        elif 'Thriller' in movie or 'Horror' in movie or 'Crime' in movie:
            Result.append('b')
        elif 'Documentary' in movie or 'Film-Noir' in movie or 'War' in movie:
            Result.append('r')
        elif 'Animation' in movie or "Children's" in movie or 'Adventure' in movie or 'Fantasy' in movie:
            Result.append('c')
        else:
            Result.append('w')
    return Result
def build_model():
    embedding_size=8
    movie_input = Input(shape=[1])
    movie_vec = Embedding(6040+1, embedding_size,embeddings_initializer='random_normal')(movie_input)
    movie_vec=Flatten()(movie_vec)
    movie_bias=Embedding(6040+1,1,embeddings_initializer='zeros')(movie_input)
    movie_bias=Flatten()(movie_bias)

    user_input = Input(shape=[1])
    user_vec = (Embedding(3952+1, embedding_size,embeddings_initializer='random_normal')(user_input))
    user_vec=Flatten()(user_vec)
    user_bias=Embedding(3952+1,1,embeddings_initializer='zeros')(user_input)
    user_bias=Flatten()(user_bias)



    merge=Dot(axes=1)([movie_vec,user_vec])
    merge=Add()([merge,movie_bias,user_bias])

    model = Model([movie_input, user_input], merge)


    print(model.summary())
    return model

def main():
    train_data=pd.read_csv('train.csv',index_col=[0])
    train_data = train_data.sample(frac=1., random_state=0)
    #print(train_data)
    users=pd.read_csv('users.csv',sep="::")
    movies=pd.read_csv('movies.csv',sep="::")

    movies_genres=get_movie_genres(movies)

    rate=np.asarray([train_data['Rating']]).T
    MEAN=np.mean(rate)
    STD=np.std(rate)

    best_model=load_model('./models/best_model_normalized.hdf5')
    movie_emb_full=np.array(best_model.layers[3].get_weights()).squeeze()
    movie_emb=[]
    movie_id=np.array(movies['movieID'])
    for id in movie_id:
        movie_emb.append(movie_emb_full[id])
    movie_emb=np.asarray(movie_emb)
    print('movie_emb.shape=',movie_emb.shape)
    #np.save('movie_emb.npy',movie_emb)
    transform=TSNE()
    tsne_movie_emb=transform.fit_transform(movie_emb[:1000,:])
    print(tsne_movie_emb.shape)
    vis_x=tsne_movie_emb[:,0]
    vis_y=tsne_movie_emb[:,1]

    sc=plt.scatter(vis_x,vis_y,c=movies_genres[:1000])
    plt.show()
if __name__=='__main__':
    main()