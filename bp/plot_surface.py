# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:15:29 2019

@author: CE216
"""

fin = open('test_dataset.txt','r')
test_dataset=[[None]*3 for i in range(2000)]

# read data
for i in range(len(test_dataset)):
    tmp=fin.readline()
    test_dataset[i][0]=float(tmp.split()[0])
    test_dataset[i][1]=float(tmp.split()[1])
    test_dataset[i][2]=float(tmp.split()[2])
    
'''
# X, Y value
X = np.arange(-1.5, 2.5, 0.25)
Y = np.arange(-1.5, 2.5, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
'''