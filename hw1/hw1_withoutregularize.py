# coding = Big5
__author__ = 'ray'
##using gradient descent
##only first order terms

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
start_time=time.time()
###reading from training file, store in train_data, shape of 17*5760
A=read_file('train.csv')

def extract_data(A,hour,bias_x,bias_y):
    data=np.array([])
    data.resize((17,hour*240))
    for i in range(240):
        jp=0
        for j in range(18):
            if j==10:
                pass
            else:
                for k in range(hour):
                    data[jp][k+i*hour]=float(A[bias_y+18*i+j][bias_x+k])
                jp+=1
    return data
train_data=extract_data(A,24,3,1)


##forming data point array
def data_to_point(data,trails,diff_m):
    data_points=np.array([[0.0]])
    data_points.resize((trails,9*17+1)) ##shape of 5750*154
    for start_point in range(trails):
        data_part=data[:,0+start_point*diff_m:9+start_point*diff_m].flatten()
        data_part.resize((17*9+1))
        data_part[-1]=1.0
        data_points[start_point]=np.copy(data_part)
    return data_points

y_hat=np.array([[0.0]])
y_hat.resize((24*240-10,1))##label of training_data

train_data_points=data_to_point(train_data,24*240-10,1)
print(train_data_points)
for start_point in range(24*240-10):
    y_hat[start_point]=train_data[9][9+start_point]

r_learn=5e-3
##training step
theta=np.array([[0.0]])##9 hours*18 featurs + 1bias
theta.resize((9*17+1,1))
grad_square_sum=np.copy(theta)
training_it=0
error_sum_previous=0.0
training_time=float(input('training_time='))
while(time.time()-start_time<=training_time):
    h_theta=np.dot(train_data_points,theta)
    error=y_hat-h_theta
    grad=np.dot(train_data_points.T,error)
    grad_square_sum+=grad**2
    theta+= grad* r_learn/(grad_square_sum**0.5)
    error_sum=np.sum(error**2)**0.5
    if training_it%1000==0: print("time= ",time.time()-start_time," , ",error_sum)
    training_it+=1
    '''
    r_learn*=1.001

    if error_sum_previous<error_sum:
        r_learn/=1.005
        #print('lower')

    error_sum_previous=error_sum
    '''



###reading from test file, store in test_data
B=read_file('test_X.csv')
test_data=extract_data(B,9,2,0)
test_data_points=data_to_point(test_data,240,9)
h_theta=np.dot(test_data_points,theta)

with open('submission.csv', 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for start_point in range(240):
        pm25_prediction=int(round(h_theta[start_point][0]))
        print('id_'+str(start_point)+","+str(pm25_prediction))
        writer.writerow({'id': 'id_'+str(start_point), 'value': str(pm25_prediction)})
