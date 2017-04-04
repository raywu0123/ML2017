# coding = Big5
__author__ = 'ray'
##plan: using only pm25 data as feature
##train with first order terms for 30 secs
##save theta to a file
##read theta from file and train with second order terms



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
def add_bias(matrix):
    n_row=matrix.shape[0]
    n_collumn=matrix.shape[1]
    b=np.array([[1.0]]*n_row)
    return np.concatenate((matrix,b),axis=1)
def extract_data(A,bias_x,bias_y):
    return (A[bias_y:,:]).astype(float)

def expand(data,dimension=2):
    L=[0,1,3,4,5]
    for i in L:
        data=np.concatenate((data,data[:,i:i+1]**2),1)
        #data=np.concatenate((data,data[:,i:i+1]**3),1)
    #print(data[0])
    data=np.concatenate((data,data[:,53:53+1]*data[:,65:65+1]),1)##Canada Husband
    data=np.concatenate((data,data[:,24:24+1]*data[:,65:65+1]),1)##Bachelor Husband
    return data
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
    #return (np.exp(matrix))/(1.0+np.exp(matrix))
    return np.clip(1.0/(1.0+np.exp(-matrix)),1e-15,1.0-1e-15)
def normalize(data):
    temp=data.T

    temp[0]/=temp[0].mean()
    temp[1]/=temp[1].mean()
    temp[3]/=5000.0
    temp[4]/=1000.0
    temp[5]/=3.5*24
    '''
    for i in range(106):
        if np.var(temp[i])!=0:
            temp[i]=(temp[i]-np.mean(temp[i]))/np.var(temp[i])
        else:
            temp[i]=temp[i]-np.mean(temp[i])
    '''
    return temp.T
def training(train_x,train_y,r_learn):
    feat_n=train_x.shape[1]
    theta=np.array([[0.0]])
    theta.resize((feat_n,1))
    reg_mask=np.copy(theta)
    for i in range(106,111):
        reg_mask[i]=1.0
    grad_square_sum=np.copy(theta)
    theta=np.random.rand(feat_n,1)
    theta=theta*2-1.0
    h_theta=sigmoid(np.dot(train_x,theta))
    error=train_y-h_theta
    error_sum=np.sum(error**2)**0.5
    print(error_sum)
    training_it=0
    training_time=float(input('training_time='))
    start_time=time.time()
    while(time.time()-start_time<=training_time):
        h_theta=sigmoid(np.dot(train_x,theta))
        error=y_hat-h_theta
        grad=np.dot(train_x.T,error)
        grad_square_sum+=np.square(grad)
        #print('errror=\n',error)
        #print(grad_square_sum)
        #print(grad)
        #input()
        theta+= grad* r_learn/(np.sqrt(grad_square_sum))#-1*theta*reg_mask*r_learn/(np.sqrt(grad_square_sum))###regulation
        error_sum=np.sqrt(np.sum(np.square(error)))
        error_sum2=np.sum(abs(error))/y_hat.shape[0]
        if training_it%100==0: print("time= ",int(time.time()-start_time)," , ",error_sum,1-error_sum2)
        training_it+=1
        '''
        r_learn*=1.01
        if error_sum_previous<error_sum:
            r_learn/=1.1
            #print('lower')

        error_sum_previous=error_sum
        '''
    output=open("theta_adagrad.txt","w")
    for term in theta:
        output.write(str(term[0])+'\n')
    output.close()
def write_file(filename,h_theta):
    with open(filename, 'w') as csvfile:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for start_point in range(h_theta.shape[0]):
            pm25_prediction=int(h_theta[start_point][0]*2)
            print(str(start_point+1)+","+str(pm25_prediction))
            writer.writerow({'id': str(start_point+1), 'label': str(pm25_prediction)})


##extract data from file and convert to data points
A=read_file('X_train')
train_data=extract_data(A,bias_x=0,bias_y=1)
train_data=normalize(train_data)
train_data=expand(train_data)
train_data=add_bias(train_data)

#print(train_data)
print(train_data.shape)
B=read_file('Y_train')
y_hat=extract_data(B,bias_x=0,bias_y=0)
training(train_x=train_data,train_y=y_hat,r_learn=2.5e0)
theta=load_theta('theta_adagrad.txt',feat_n=106+5+1+1)

'''
for i in range(theta.shape[0]):
    if abs(theta[i][0])>=3.0:
        print(i,theta[i][0],A[0][i])
'''
C=read_file('X_test')
test_data=extract_data(C,bias_x=0,bias_y=1)
test_data=normalize(test_data)
test_data=expand(test_data)
test_data=add_bias(test_data)
h_theta=sigmoid(np.dot(test_data,theta))
#print(h_theta)
write_file('submission_logistic.txt',h_theta)

