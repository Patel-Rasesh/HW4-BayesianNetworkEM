
from __future__ import print_function
import numpy as np
import sys
import time
from CLT_RandomForest import CLT_RandomForest
from Util import *
from CLT_class import CLT
import random

class RandomForest():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, hyperparameterR = 1000):
        # For each component and each data point, we have a weight
        self.n_components = n_components
        # For RandomForest, weigts can be uniform - keeping them 1
        weights=np.ones((n_components,dataset.shape[0]))
        self.mixture_probs = [1/n_components] * dataset.shape[0]

        self.clt_list = [CLT_RandomForest() for i in range(n_components)]

        for k in range(n_components):
            # Bootstrap samples before lerning a tree
            bootstrapSet = ()
            for i in range(dataset.shape[1]):
                randomSample = random.rand(0, dataset.shape[0])
                bootstrapSample = dataset[randomSample]
                bootstrapSet.append(bootstrapSample)
            self.clt_list[k].learn(bootstrapSet)
            self.clt_list[k].update(bootstrapSet, weights[k], hyperparameterR=1000)
            print(RandomForest.computeLL(bootstrapSet))
            
    '''
        Compute the log-likelihood score of the dataset
    '''
    
    def computeLL(self, dataset):
        
        ll = 0.0
        clt = CLT()
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        #clt.computeLL(dataset) / self.mixture_probs
        # I think we need to use one of the parameters - clt.xycount or something
        for k in range(self.n_components):
            ll += sum(self.clt_list[k].computeLL(dataset) * self.mixture_probs / dataset.shape[0])
            #print(self.mixture_probs[k])
        return ll  
    
dataset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.ts.data")
testset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.test.data")
forest = RandomForest()
forest.learn(dataset, n_components=2, max_iter=50, hyperparameterR = 1000)
#print(mix.computeLL(testset)/dataset.shape[0])

# Next steps - 
# 1 - check dimensions of self.mixture_probs
# 2 - check dimensions of bootstrapset and sample
# 3 - Run test file in MIXTURE_CLT