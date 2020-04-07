# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:46:11 2019

@author: CE216
"""

from math import exp
from random import seed
from random import random
import numpy as np

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]   # add bias
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]  # bias+input*weight
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))   #sigmoid
    #return np.maximum(0.0,activation)

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)     # input*weight+bias
			neuron['output'] = transfer(activation)              # do sigmoid
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)    # sigmoid derivative
'''
    if output>0:
        return 1
    else:
        return 0
'''


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)

dataset=[[0,0,0],
         [0,1,1],
         [1,0,1],
         [1,1,0]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 3, n_outputs)
train_network(network, dataset, 0.5, 5000, n_outputs)
for layer in network:
	print(layer)

# test the test data set, and the result will put at testdata[2]
def test(network,test_data):
    output=forward_propagate(network,test_data)
    test_data[2]=output[1]
    return test_data
'''
# make some test data (randomly)(0.0~1.0)
seed(2)
test_dataset=[[None]*3 for i in range(2000)]
for i in range(len(test_dataset)):
    test_dataset[i][0]=random()
    test_dataset[i][1]=random()
    test_dataset[i][2]=None
''' 
# make some test data (randomly)(-1.0~2.0)
import random
seed(2)
test_dataset=[[None]*3 for i in range(2000)]
for i in range(len(test_dataset)):
    test_dataset[i][0]=random.uniform(-1.5,2.5)
    test_dataset[i][1]=random.uniform(-1.5,2.5)
    test_dataset[i][2]=None

# run the network with the dataset we create
for i in range(len(test_dataset)):
    test(network,test_dataset[i])
    
# Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(0, 0, 0, c='k', marker='^')
ax.scatter(0, 1, 1, c='k', marker='^')
ax.scatter(1, 0, 1, c='k', marker='^')
ax.scatter(1, 1, 0, c='k', marker='^')

X=[]
Y=[]
Z=[]
for i in range (len(test_dataset)):
    X.append(test_dataset[i][0])
    Y.append(test_dataset[i][1])
    Z.append(test_dataset[i][2])
# plot dots
ax.scatter(X,Y,Z,c='b')
    
# plot surfaces
#ax.plot_trisurf(X,Y,Z,cmap=plt.get_cmap('rainbow'))

ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
'''
# save the test_dataset to file
fout = open('neuron3.txt','w')
for i in range(len(X)):
    fout.write(str(X[i])+' '+str(Y[i])+' '+str(Z[i])+'\n')
fout.close()
'''