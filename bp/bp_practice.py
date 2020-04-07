# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:43:49 2019

@author: CE216
"""
import numpy as np
np.random.seed(5)

def sigmoid(x):
    return 1. /(1+np.exp(-x))
def cost(a,y):
    return (a-y)**2

# init a
a=[0,0,0,0]    
# random weight, bias
w=np.random.rand(3)
b=np.random.rand(3)


dataset=[0,1,0,1,0]

# forward propagation
a[0]=dataset[0]
for i in range (3):
    a[i+1]=sigmoid(a[i]*w[i]+b[i])
c = cost(a[3],dataset[0])
print('cost:'+str(c))

# 看 a[3] 的趨勢，堆於cost的影響
del_a=0.1
a_plus_del_a=a[3]+del_a
a_sub_del_a=a[3]-del_a

a_ideal=[0,0,0,0]
w_ideal=[0,0,0]
b_ideal=[0,0,0]
if(cost(a_plus_del_a,dataset[0])<c):
    a_ideal[3]=a[3]+del_a
elif (cost(a_sub_del_a,dataset[0])<c):
    a_ideal[3]=a[3]-del_a
    