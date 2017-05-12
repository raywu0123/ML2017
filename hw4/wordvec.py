__author__ = 'ray'
import word2vec
import adjustText
import nltk
import sklearn.decomposition
from sklearn.manifold import TSNE
import string
import numpy as np
import matplotlib.pyplot as plt

def choose(vocabs,V):
    chosen_vocabs=[]
    vectors=[]
    types=["JJ","NNP","NNS","NN"]
    tys=[]
    for index,vocab in enumerate(vocabs):
        take=True
        for letter in vocab[0]:
            if letter not in string.ascii_letters:
                take=False
        if vocab[1] not in types:
            take=False
        if len(vocab[0])==1:
            take=False
        if take:
            chosen_vocabs.append(vocab[0])
            vectors.append(V[index])
            if vocab[1]=='JJ':
                tys.append('r')
            elif vocab[1]=='NNP':
                tys.append('g')
            elif vocab[1]=='NNS':
                tys.append('b')
            elif vocab[1]=='NN':
                tys.append('y')


    return chosen_vocabs,np.array(vectors),tys
def print_list(list):
    for item in list:
        print(item)
def pca(vectors):
    return sklearn.decomposition.PCA(n_components=2).fit_transform(vectors)
def tsne(vectors):
    return TSNE(n_components=2,random_state=0).fit_transform(vectors)

#word2vec.word2vec("all.txt",'all.bin', size=100, verbose=True)
model=word2vec.load('all.bin')

print(model.vocab.shape)
print(model.vectors.shape)
tagged_vocabs=nltk.pos_tag(model.vocab)
chosen_vocabs,vectors,t=choose(tagged_vocabs,model.vectors)
print_list(chosen_vocabs[:500])
print(vectors.shape)
#projected_vectors=tsne(vectors[:500])
projected_vectors=pca(vectors[:500])
print(projected_vectors.shape)

plt.figure(figsize=(9, 6))
plt.scatter(projected_vectors[:250,0], projected_vectors[:250,1], s=15, c=t[:250], edgecolors=(1,1,1,0))
texts = []
#print(t[:200])
for x, y, s in zip(projected_vectors[:250,0], projected_vectors[:250,1], chosen_vocabs[:250]):
    texts.append(plt.text(x, y, s, size=7))
plt.title(str(adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))))
plt.show()
