__author__ = 'ray'
import numpy as np
from scipy import ndimage,misc
import matplotlib.pyplot as plt
from PIL import Image

def read_images(file_path):
    images=[]
    for name in ['A','B','C','D','E','F','G','H','I','J']:
        for num in range(10):
            image=ndimage.imread(file_path+name+'0'+str(num)+'.bmp')
            image=image.reshape(64*64)
            image=image-np.mean(image)
            images.append(image)

    images=np.array(images)
    return images
def pca(images):
    S=np.cov(images.T)
    eigenvalues,eigenvectors=np.linalg.eigh(S)
    return eigenvalues,eigenvectors
def average_face(images):
    average_face=np.mean(images,axis=0)
    plt.imshow(average_face.reshape(64,64),cmap='gray')
    plt.show()
def plot_faces(faces,x,y,filename):
    fig = plt.figure(figsize=(64, 64))
    for i in range(x*y):
        ax = fig.add_subplot(x, y, i+1)
        ax.imshow(faces[i].reshape(64,64), cmap='gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()
    fig.savefig(filename)
def reconstruct(images,eigenfaces):
    recon_faces=[]
    coeffs=np.dot(images,eigenfaces.T)
    recon_faces=np.dot(coeffs,eigenfaces)
    return recon_faces
def RMSE(images,reconstruction):
    error=images-reconstruction
    RMSE=np.mean(np.square(error))**0.5
    return RMSE


def main():
    images=read_images('faceExpressionDatabase/')
    '''
    eig_val,eig_vec=pca(images)
    np.save('eigenvalue',eig_val)
    np.save('eigenvector',eig_vec)
    '''

    eig_val=np.load('eigenvalue.npy')
    eig_vec=np.load('eigenvector.npy')
    eig_vec=eig_vec.T
    eig_vec=eig_vec[::-1]




    recon_faces=reconstruct(images=images,eigenfaces=eig_vec[0:4020])
    '''
    plot_faces(images,10,10,'original_faces.png')
    plot_faces(recon_faces,10,10,'reconstruct_faces.png')
    '''
    print(RMSE(images,recon_faces)/256)


main()