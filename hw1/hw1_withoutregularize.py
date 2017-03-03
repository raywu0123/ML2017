# coding = Big5
__author__ = 'ray'


import numpy as np
import csv
import time

def read_file(file_name):
    Bp=[]
    with open(file_name, newline='', encoding='Big5') as f:
        reader = csv.reader(f)
        for row in reader:
            Bp.append(row)
        B=np.array(Bp)
    return B

###reading from training file, store in train_data, shape of 18*5760
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
##forming data point array
train_data_points=np.array([[0.0]])
train_data_points.resize((24*240-10,9*18+1)) ##shape of 5750*163
y_hat=np.array([[0.0]])
y_hat.resize((24*240-10,1))##label of training_data

for start_point in range(trails):
    train_data_part=train_data[:,0+start_point:9+start_point].flatten()
    train_data_part.resize((18*9+1))
    train_data_part[-1]=1.0
    train_data_points[start_point]=np.copy(train_data_part)

    y_hat[start_point]=train_data[9][9+start_point]

r_learn=5.5e-10
##training step
theta=np.array([[0.0]])##9 hours*18 featurs + 1bias
theta.resize((9*18+1,1))
theta_new=np.copy(theta)
training_it=0
error_sum_previous=0.0
while(time.clock()<=60*10):
    h_theta=np.dot(train_data_points,theta)
    error=y_hat-h_theta
    theta+=np.dot(train_data_points.T,error) * r_learn
    error_sum=np.sum(error**2)**0.5
    if training_it%1000==0: print("time= ",time.clock()," , ",error_sum)
    training_it+=1


    if error_sum_previous-error_sum<=1e-2 and error_sum_previous>error_sum:
        r_learn*=1.1
        #print('raise')
    elif error_sum_previous<error_sum:
        r_learn/=1.1
        #print('lower')

    error_sum_previous=error_sum


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

test_data_points=np.array([[0.0]])
test_data_points.resize((240,9*18+1)) ##shape of 5750*163
for start_point in range(240):
    test_data_part=test_data[:,0+start_point*9:9+start_point*9].flatten()
    test_data_part.resize((18*9+1))
    test_data_part[-1]=1.0
    test_data_points[start_point]=np.copy(test_data_part)
#print(test_data_points.shape)
h_theta=np.dot(test_data_points,theta)

with open('submission.csv', 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for start_point in range(240):
        pm25_prediction=int(round(h_theta[start_point][0]))
        print('id_'+str(start_point)+","+str(pm25_prediction))
        writer.writerow({'id': 'id_'+str(start_point), 'value': str(pm25_prediction)})
