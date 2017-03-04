# coding = Big5
__author__ = 'ray'
feat_num=int(9)


import numpy as np
import csv
import time
import itertools as ite
def read_file(file_name):
    Bp=[]
    with open(file_name, newline='', encoding='Big5') as f:
        reader = csv.reader(f)
        for row in reader:
            Bp.append(row)
        B=np.array(Bp)
    return B


def expand(feature,dimension):
    c=np.copy(feature)
    for i in range(dimension-1):
        c_n=np.array([0.0])
        c_n.resize()
        c_n=np.prod(np.asarray(list(ite.combinations_with_replacement(feature,i+2))),axis=1)
        c=np.concatenate((c,c_n))
    return c

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

def normalize(data_part):
    if data_part.var()==0.0:
        return data_part
    else:
        return (data_part.__sub__(data_part.mean())).__mul__((1.0/data_part.var())**0.5)

##forming data point array
def data_to_point(data,trails,diff_m):
    data_points=np.array([[0.0]])
    data_points.resize((trails,feat_num+1)) ##shape of
    for start_point in range(trails):
        data_part=data[9:10,0+start_point*diff_m:9+start_point*diff_m].flatten() ##only pm25 feature
        #data_part_norm=normalize(data_part)
        data_part_norm=data_part
        data_part_expand=expand(data_part_norm,dimension=1)
        data_part_expand.resize((feat_num+1))
        data_part_expand[-1]=1.0
        data_points[start_point]=np.copy(data_part_expand)

    return data_points


start_time=time.time()
##extract data from file and convert to data points
A=read_file('train.csv')
train_data=extract_data(A,24,3,1)
print(train_data[9])
train_data_points=data_to_point(train_data,24*240-10,1)

##extract labels from data
y_hat=np.array([[0.0]])
y_hat.resize((24*240-10,1))##label of training_data
for start_point in range(24*240-10):
    y_hat[start_point]=train_data[9][9+start_point]

r_learn=5e-20
##training step
theta=np.array([[0.0]])##featur_num + bias
theta.resize((feat_num+1,1))
theta_new=np.copy(theta)
training_it=0
error_sum_previous=0.0
training_time=float(input('training_time='))
#training_time=30
while(time.time()-start_time<=training_time):
    h_theta=np.dot(train_data_points,theta)
    error=y_hat-h_theta
    theta+=np.dot(train_data_points.T,error) * r_learn
    error_sum=np.sum(error**2)**0.5
    if training_it%50==0: print("time= ",int(time.time()-start_time)," , ",error_sum,r_learn)
    training_it+=1

    r_learn*=1.01

    if error_sum_previous<error_sum:
        r_learn/=1.1
        #print('lower')

    error_sum_previous=error_sum

print(theta)
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

