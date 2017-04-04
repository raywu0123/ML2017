# coding = Big5
__author__ = 'ray'

import numpy as np
import csv
import time
import itertools as ite
import sys
def read_file(file_name):
    Bp=[]
    with open(file_name, newline='', encoding='Big5') as f:
        reader = csv.reader(f)
        for row in reader:
            Bp.append(row)
        B=np.array(Bp)
    return B
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i
def extract_data(A,bias_x,bias_y):
    return (A[bias_y:,:]).astype(float)
def load_theta(filename,feat_n):
    file_l=file_len(filename)
    file=open(filename,'r')
    l=np.array([0.0])
    l.resize((feat_n+1,1))
    index=0
    for line in file:
        #print(line,float(line))
        if index!=file_l:
            l[index][0]=float(line)
        else:
            l[-1][0]=float(line)
        index+=1
    return l
def sigmoid(matrix):
    return (np.exp(matrix))/(1.0+np.exp(matrix))
def normalize(data):
    temp=data.T
    temp[0]/=temp[0].mean()
    temp[1]/=temp[1].mean()
    temp[3]/=5000.0
    temp[4]/=1000.0
    temp[5]/=3.5*24
    return temp.T
def training(train_x,train_y):
    feat_n=train_x.shape[1]
    data_n=train_x.shape[0]
    N1=0    #number of label1
    N2=0    #number of label0
    mu_1=np.array([[0.0]*feat_n])
    mu_2=np.array([[0.0]*feat_n])
    Sigma_1=np.array([[0.0]*feat_n]*feat_n)
    Sigma_2=np.array([[0.0]*feat_n]*feat_n)
    for i in range(data_n):
        if train_y[i][0]==1:
            N1+=1
            mu_1+=train_x[i:i+1]
        elif train_y[i][0]==0:
            N2+=1
            mu_2+=train_x[i:i+1]
    mu_1/=float(N1)
    mu_2/=float(N2)
    #print(N1,N2,N1+N2)
    ##calculated mu

    for i in range(data_n):
        if train_y[i][0]==1:
            v=train_x[i:i+1]-mu_1
            Sigma_1+=np.dot(v.T,v)
        elif train_y[i][0]==0:
            v=train_x[i:i+1]-mu_2
            Sigma_2+=np.dot(v.T,v)
    Sigma_1*=1.0/float(N1)
    Sigma_2*=1.0/float(N2)
    Sigma=(Sigma_1*N1+Sigma_2*N2)/float(N1+N2)
    return Sigma,mu_1,mu_2,N1,N2


##extract data from file and convert to data points
A=read_file(sys.argv[3])
train_data=extract_data(A,bias_x=0,bias_y=1)
train_data=normalize(train_data)
print(train_data)
print(train_data.shape)
B=read_file(sys.argv[4])
y_hat=extract_data(B,bias_x=0,bias_y=0)




Sigma,mu_1,mu_2,N1,N2=training(train_x=train_data,train_y=y_hat)
'''
t=abs(mu_1-mu_2)
for i in range(len(t[0])):
    if t[0][i]>=0.15:
        print(i,(mu_1-mu_2)[0][i],mu_1[0][i],mu_2[0][i],A[0][i])
input('pause')
'''
def P(x):
    def Gaussion(mu,Sigma,x):
        v=x-mu
        Sigma_det=np.linalg.det(Sigma)
        Sigma_inv=np.linalg.inv(Sigma)
        #print(Sigma_det)
        #print(np.exp(-0.5*np.dot(np.dot(v,Sigma_inv),v.T)))
        return np.exp(-0.5*np.dot(np.dot(v,Sigma_inv),v.T))
    return Gaussion(mu_1,Sigma,x)*N1/(Gaussion(mu_1,Sigma,x)*N1+Gaussion(mu_2,Sigma,x)*N2)

C=read_file(sys.argv[5])
test_data=extract_data(C,bias_x=0,bias_y=1)
test_data=normalize(test_data)

h_theta=np.array([[0.0]]*test_data.shape[0])

for i in range(h_theta.shape[0]):
    h_theta[i][0]=P(test_data[i])[0][0]
    try:
        a=int(h_theta[i][0]*2)
    except:
        a=0
        h_theta[i][0]=0.0
    print(i+1,a)

with open(sys.argv[6], 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for start_point in range(h_theta.shape[0]):
        pm25_prediction=int(h_theta[start_point][0]*2)
        print(str(start_point+1)+","+str(pm25_prediction))
        writer.writerow({'id': str(start_point+1), 'label': str(pm25_prediction)})
