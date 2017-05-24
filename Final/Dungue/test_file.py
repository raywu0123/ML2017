__author__ = 'ray'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A=pd.read_csv('prediction_1_1.csv',index_col=[0,1,2])
B=pd.read_csv('submission-5.csv',index_col=[0,1,2])
total_cases_A=A['total_cases']
total_cases_B=B['total_cases']
index=np.arange(total_cases_A.shape[0])
plt.plot(index,total_cases_A,'r')
plt.plot(index,total_cases_B,'b')
plt.show()
print(A)