import numpy as np
import sys

def read_file(file_name):
    file=open(file_name,"r")
    Bp=[]
    for line in file:
        Bp.append(line.split(","))
    B=np.array(Bp,float)
    file.close()
    return B

print sys.argv

A=read_file(sys.argv[1])
B=read_file(sys.argv[2])
C=np.dot(A,B)
C=[sorted(C[0])]
#print C
output=open('ans_one.txt',"w")

for num in C[0]:
    output.write(str(num)+"\n")
output.close()


