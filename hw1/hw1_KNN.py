
# coding = Big5
__author__ = 'ray'
##plan: using only pm25 data as feature
##regularize train data points
import numpy as np
import csv
import time
import itertools as ite
import matplotlib.pyplot as plt

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
        return np.array([0.0]*data_part.size)
    else:
        return (data_part.__sub__(data_part.mean())).__mul__((1.0/data_part.std()))
##forming data point array
def data_to_point(data,trails,diff_m,norm):
    feat_num=9
    data_points=np.array([[0.0]])
    data_points.resize((trails,feat_num)) ##shape of
    y_hat=np.array([[0.0]])
    y_hat.resize((24*240-10,1))##label of training_data
    for start_point in range(trails):
        data_part=data[9:10,0+start_point*diff_m:9+start_point*diff_m].flatten() ##only pm25 feature
        y_hat[start_point]=train_data[9][9+start_point]
        if norm == True:
            data_part_norm=normalize(data_part)
        else:
            data_part_norm=data_part
        if data_part.std() != 0:
            y_hat[start_point]=(y_hat[start_point]-data_part.mean())/(data_part.std())
        else:
            y_hat[start_point]=0.0
        data_points[start_point]=np.copy(data_part_norm)

    return data_points,y_hat

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

def training(dimension,feat_n,r_learn,theta_file_name=''):
    train_data_points=data_to_point(train_data,trails=24*240-10,diff_m=1,dim=dimension,feat_num=feat_n)
    ##training step
    theta=np.array([[0.0]])##featur_num + bias
    theta.resize((feat_n+1,1))
    grad_square_sum=np.copy(theta)
    if dimension>1:
        theta=load_theta(theta_file_name,feat_n=feat_n)
    print(theta)
    h_theta=np.dot(train_data_points,theta)
    error=y_hat-h_theta
    error_sum=np.sum(error**2)**0.5
    print(error_sum)
    training_it=0
    error_sum_previous=0.0
    training_time=float(input('training_time='))
    start_time=time.time()
    #training_time=30
    while(time.time()-start_time<=training_time):
        h_theta=np.dot(train_data_points,theta)
        error=y_hat-h_theta
        grad=np.dot(train_data_points.T,error)
        grad_square_sum+=np.square(grad)
        theta+= grad* r_learn/(np.sqrt(grad_square_sum))-1000*theta*r_learn/(np.sqrt(grad_square_sum))
        error_sum=np.sqrt(np.sum(np.square(error)))
        if training_it%1000==0: print("time= ",int(time.time()-start_time)," , ",error_sum)
        training_it+=1
        '''
        r_learn*=1.01
        if error_sum_previous<error_sum:
            r_learn/=1.1
            #print('lower')

        error_sum_previous=error_sum
        '''
    output=open("theta_"+str(dimension)+"_adagrad.txt","w")
    for term in theta:
        output.write(str(term[0])+'\n')
    output.close()

def distance(v1,v2):
    return np.sum(np.square(v1-v2))

##extract data from file and convert to data points
A=read_file('train.csv')
train_data=extract_data(A,24,3,1)
train_data_points,y_hat=data_to_point(train_data,24*240-10,1,norm=True)
print('train_data_points=\n',train_data_points)
print('y_hat=\n',y_hat)
###reading from test file, store in test_data
B=read_file('test_X.csv')
test_data=extract_data(B,9,2,0)
test_data_points=(data_to_point(test_data,trails=240,diff_m=9,norm=False))[0]
h_theta=np.array([0.0])
h_theta.resize((240,1))





for j in range(len(test_data_points)):
    mean=np.mean(test_data_points[j])
    std=np.std(test_data_points[j])
    test_data_point_norm=normalize(test_data_points[j])

    weights=np.array([0.0])
    weights.resize((24*240-10,1))
    for i in range(len(train_data_points)):
        train_data_point=train_data_points[i]
        d=distance(train_data_point,test_data_point_norm)

        weights[i][0]=d**-5

        #if d<1.555: print(i)

    #print(np.max(weights),np.min(weights))
    weight_sum=np.sum(weights)
    h_theta_norm=np.sum(y_hat*weights)/weight_sum
    h_theta[j][0]=h_theta_norm*std+mean
    if  h_theta[j][0]<0.0: h_theta[j][0]=0.0
    #print(j,h_theta[j][0],np.sum(y_hat*weights),weight_sum)
    print(j, np.max(weights),np.min(weights),np.mean(weights),h_theta[j][0]),
    #plt.plot(np.arange(0,24*240-10,1,float),weights)
    #plt.show()

x=np.arange(0,9,1,float)
plt.plot(x,train_data_points[4924],'r--',x,normalize(test_data_points[0]),'b')
plt.show()



with open('submission.csv', 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for start_point in range(240):
        pm25_prediction=int(round(h_theta[start_point][0]))
        print('id_'+str(start_point)+","+str(pm25_prediction))
        writer.writerow({'id': 'id_'+str(start_point), 'value': str(pm25_prediction)})
