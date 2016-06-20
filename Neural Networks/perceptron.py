# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:04:42 2016

@author: jsullivan
"""


# Perceptron in Python
# This is a binary classification problem, modeling the NAND function
# https://en.wikipedia.org/wiki/Perceptron#Learning_algorithm



import numpy as np
import matplotlib.pyplot as plt


#     Features
#     x0 x1 x2
x = [[1., 0., 0.],                                  # First training input vector
     [1., 0., 1.],                                  # Second training input vector
     [1., 1., 0.],                                  # Third training input vector
     [1., 1., 1.]]                                  # Fourth training input vector

# Desired Outputs      
#    z           
z = [1.,                                            # First desired output
     1.,                                            # Second desired output
     1.,                                            # Third desired output 
     0.]                                            # Fourth desired output 
           
t = 0.5                                             # Threshold
r = 0.1                                             # Learning rate



# Creating the perceptron function 
def perceptron(x, z, t, r):
    e = [1., 1., 1., 1.]                            # Initial errors 
    w = [0., 0., 0.]                                # Initial weights
    it = 0                                          # Iteration counter
    while 1. in e:                                  # While all the errors are not zero, keep going
        for i in xrange(0, len(x)):                 
            s = np.dot(x[i], w)                     # Dot product of the features and weights, this is the output 
            if s > t:                               # If the output is greater than the threshold, then...
                n = 1                               # n = 1...
            else:                                   # else...
                n = 0                               # n = 0...
            e[i] = z[i] - n                         # error check
            d = r * e[i]                            # correction term
        
            for j in xrange(0, len(w)):             # updating the weights
                w[j] = w[j] + x[i][j]*d
        it += 1                                     # updating the counter to track number of iterations 
    return it


                
# Plotting number of iterations to convergence as a function of learning rate
rr = np.linspace(0.05, 1., 10.)                     # creating array for learning rate

def learning_rate_sensitivity(x, z, t, rr):
    x_rr = []                                       # creating empty list for iterations
    for i in xrange(0, len(rr)):            
        a = rr[i]                                   # need to update learning rate each time
        x_rr.append(perceptron(x, z, t, a))         # append iteration list 
    return x_rr                 
    
x_rr = learning_rate_sensitivity(x, z, t, rr)

plt.plot(rr, x_rr)
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Perceptron Learning Rate Sensitivity')
        
    
    
    
