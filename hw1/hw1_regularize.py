# coding = Big5
__author__ = 'ray'
##average of the test data

import numpy as np
import csv
import sys


def read_file(file_name):
    Bp=[]
    with open(file_name, newline='', encoding='Big5') as f:
        reader = csv.reader(f)
        for row in reader:
            Bp.append(row)
        B=np.array(Bp)
    return B

###reading from training file, store in train_data
A=read_file('train.csv')
train_data=np.array([])
train_data.resize((18,24*240))
for i in range(240):
    for j in range(18):
        if j==10:
            pass
        else:
            for k in range(24):
                train_data[j][k+i*24]=float(A[1+18*i+j][3+k])

trails=24*240-10  ##number of data points, ie frame shifts
theta_collection=[]
r_learn=5.4e-10

##training step
##train_data_part stores data of a single data point
theta=np.array([0.0]*(9*18+1))  ##9 hours*18 featurs +bias
theta_new=np.copy(theta)

for training_it in range(100000):
    #print(theta)
    error_sum=0.0
    for start_point in range(trails):
        train_data_part=train_data[:,0+start_point:9+start_point].flatten()
        train_data_part.resize((18*9+1))
        train_data_part[-1]=1.0
        y_hat=train_data[9][9+start_point] ##label
        error=(y_hat-np.dot(theta.T,train_data_part))
        theta_new+=error*r_learn*train_data_part #gradient descent
        error_sum+=error**2
    if training_it%100==0: print(str(training_it)+" , "+str(error_sum**0.5))##print the error_sum
    theta=np.copy(theta_new)

print(theta)
###reading from test file, store in test_data
B=read_file('test_X.csv')
test_data=np.array([])
test_data.resize((18,9*240))
for i in range(240):
    for j in range(18):
        if j==10:
            pass
        else:
            for k in range(9):
                test_data[j][k+i*9]=float(B[18*i+j][2+k])


with open('submission.csv', 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for start_point in range(240):
        test_data_part=test_data[:,0+start_point*9:9+start_point*9].flatten()
        test_data_part.resize((18*9+1))
        test_data_part[-1]=1.0
        #print(test_data_part)
        pm25_prediction=int(round(np.dot(theta.T,test_data_part)))
        print('id_'+str(start_point)+","+str(pm25_prediction))
        writer.writerow({'id': 'id_'+str(start_point), 'value': str(pm25_prediction)})
