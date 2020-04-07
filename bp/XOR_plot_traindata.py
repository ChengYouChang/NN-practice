# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:40:08 2019

@author: CE216
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random

fig = plt.figure()
ax = Axes3D(fig)

dataset=[[0,0,0],
         [0,1,1],
         [1,0,1],
         [1,1,0]]
X=[]
Y=[]
Z=[]
for i in range (len(dataset)):
    X.append(dataset[i][0])
    Y.append(dataset[i][1])
    Z.append(dataset[i][2])
ax.scatter(X,Y,Z,c='r',marker='^')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')

test_dataset=[[None]*3 for i in range(10)]
for i in range(10):
    test_dataset[i][0]=random()
    test_dataset[i][1]=random()
    test_dataset[i][2]=None
    
test_dataset=[[None]*3 for i in range(16)]
j=0
for i in range(16):
    test_dataset[i][0]=(j+1)*0.2
    test_dataset[i][1]=(i+1)*0.2
    test_dataset[i][2]=None
    j+=1
    if(j%4==0):
        j=0
