
from __future__ import print_function
from types import new_class
import numpy as np
import sys
import timeit
import os
from collections import defaultdict

from numpy.compat.py3k import contextlib_nullcontext
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        #self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        self.n_components = n_components
        self.mixture_probs = np.zeros(n_components)
        weights=np.zeros((n_components,dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        # Initializing p(kth_tree)
        randomWeights = np.random.rand(n_components, dataset.shape[0])
        for i in range(randomWeights.shape[0]):
            self.mixture_probs = randomWeights[i]/np.sum(randomWeights[i])
        # Instantiating a variable for storing Chow-Liu probability
        tempDataWeights = np.zeros((n_components, dataset.shape[0]))

        # Initializing one CLT for each value of the latent variable
        self.clt_list = [CLT() for i in range(n_components)]
        # Learning Chow-Liu Trees for each value of the hidden variable
        for k in range(n_components):
            self.clt_list[k].learn(dataset)
        
        secondLL = 0
        for itr in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            # Your code for E-step here

            for k in range(n_components):
                # getProb gives one value for each example
                
                for i, sample in enumerate(dataset):
                    tempDataWeights[k][i] = self.clt_list[k].getProb(sample)

                # DataWeights is of (2, 12k) dimensions
                # weights becomes (n_component x dataset.shape[0]) 
                weights[k] = np.multiply(self.mixture_probs[k], tempDataWeights[k])/np.sum(np.multiply(self.mixture_probs[k], tempDataWeights[k]))

                # M-step: Update the Chow-Liu Trees and the mixture probabilities
                # Your code for M-Step here
                self.clt_list[k].update(dataset, weights[k])

            if(itr == 0):
                secondLL = mix.computeLL(dataset)
                print("Completed the first iteration.")
                print(secondLL)
            else:
                print("Entered in the iteration ... ", itr)
                firstLL = mix.computeLL(dataset)
                print(firstLL)
                secondLL = firstLL
    '''
        Compute the log-likelihood score of the dataset
    '''
    
    def computeLL(self, dataset):
        
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        llTemp = 0
        for sample in range(dataset.shape[0]):
            for k in range(self.n_components):
                
                llTemp += np.multiply(self.mixture_probs[k], self.clt_list[k].getProb(dataset[sample]))
            ll += np.log(llTemp)
        return ll/dataset.shape[0]
    
'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''
# Making python file runtime ready
serialNumber = 0
index = defaultdict(list)
for i, file in enumerate(os.listdir("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset")):
    if(i % 3 == 0):
        serialNumber += 1
    index[serialNumber].append(file)

print("Serial number\t Dataset\n")
for key, values in index.items():
    print(key, "\t\t", values[1])

print("\nEnter the serial number for which dataset to run")
selection = input()
for key, values in index.items():
    if(key == int(selection)):
        dataset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\"+index[key][1])
        validateset = Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\"+index[key][2])
        testset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\"+index[key][0])
        break

mix = MIXTURE_CLT()

# VALIDATION - uncomment the next 10 lines for tuning the value of the hidden variable k
print("Please wait, while we learn mixture models for...")
print(index[int(selection)][1])
print("Printing log likelihood after each iteration ...")

#Latent variable can take values from [2, 5, 10, 20]
# for hiddenVariable in [20, 10]:
#     start = timeit.default_timer()
#     mix.learn(dataset, n_components=hiddenVariable, max_iter=3, epsilon=1e-5)
#     print("Running on the validation set when the hidden variable can take up to", hiddenVariable,"values")
#     print("Log likelihood = ", mix.computeLL(validateset))
#     stop = timeit.default_timer()
#     print("Runtime - ", stop-start)

# TESTING - refer the code starting below
# Below are the values of a hidden variable (for each dataset) we have gotten based on validation sets
kTest = [20, 20, 20, 20, 2, 2, 20, 20, 20, 20]
hiddenVariableList = defaultdict(int)
i = 0
for key, values in index.items():
    hiddenVariableList[key] = kTest[i]
    i += 1

for i in range(5):
    start = timeit.default_timer()
    print("NOTE - program is printing log likelihood after each iteration ...")
    mix.learn(dataset, n_components=hiddenVariableList[int(selection)], max_iter=1, epsilon=1e-5)
    print("Running on the testset when the hidden variable can take up to", hiddenVariableList[int(selection)],"values")
    print("Log likelihood = ", mix.computeLL(testset))
    stop = timeit.default_timer()
    print("Runtime - ", stop-start)