# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:09:59 2019

@author: CE216
"""
f2=open('dataset1.txt','r')
f3=open('dataset2.txt','r')
n2_dataset=[[None]*3 for i in range(2000)]
n3_dataset=[[None]*3 for i in range(2000)]
diff=[[None]*3 for i in range(2000)]

# read data
for i in range(2000):
    tmp=f2.readline()
    n2_dataset[i][0]=float(tmp.split()[0])
    n2_dataset[i][1]=float(tmp.split()[1])
    n2_dataset[i][2]=float(tmp.split()[2])
    tmp=f3.readline()
    n3_dataset[i][0]=float(tmp.split()[0])
    n3_dataset[i][1]=float(tmp.split()[1])
    n3_dataset[i][2]=float(tmp.split()[2])
    diff[i][0]=n2_dataset[i][0]
    diff[i][1]=n2_dataset[i][1]
    diff[i][2]=n2_dataset[i][2]-n3_dataset[i][2]
    
# Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(0, 0, 0, c='r', marker='^')
ax.scatter(0, 10, 1, c='r', marker='^')
ax.scatter(10, 0, 1, c='r', marker='^')
ax.scatter(10, 10, 0, c='r', marker='^')

X=[]
Y=[]
Z=[]
for i in range (len(n2_dataset)):
    X.append(diff[i][0])
    Y.append(diff[i][1])
    Z.append(diff[i][2])
    
# plot dots
#ax.scatter(X,Y,Z,c='b')
#plot surface
ax.plot_trisurf(X,Y,Z,cmap=plt.get_cmap('rainbow'))

ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')

sum=0
for i in range(len(diff)):
    sum+=diff[i][2]