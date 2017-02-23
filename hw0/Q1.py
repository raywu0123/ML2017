import numpy as np

def read_file(file_name):
    file=open(file_name,"r")
    Bp=[]
    for line in file:
        Bp.append(line.split(","))
    B=np.array(Bp,float)
    file.close()
    return B

A=read_file("matrixA.txt")
B=read_file("matrixB.txt")
C=np.dot(A,B)
print C
output=open("ans_one.txt","w")

for num in C[0]:
    output.write(str(num)+"\n")
output.close()



