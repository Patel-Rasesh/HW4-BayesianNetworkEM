
from __future__ import print_function
import numpy as np
import sys
import timeit
import os
from collections import defaultdict
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        self.n_components = n_components
        weights=np.zeros((n_components,dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        # Initializing p(kth_tree)
        randomWeights = np.random.rand(n_components, dataset.shape[0])
        self.mixture_probs = randomWeights/np.sum(randomWeights)

        # Instantiating a variable for storing Chow-Liu probability
        tempDataWeights = np.zeros((n_components, dataset.shape[0]))

        # For storing the logliklihood from the previous iteration
        secondLL = 0

        # Initializing one CLT for each value of the latent variable
        self.clt_list = [CLT() for i in range(n_components)]
        # Learning each Chow-Liu Tree
        for k in range(n_components):
            self.clt_list[k].learn(dataset)
            
        for itr in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            # Your code for E-step here

            for k in range(n_components):
                # getProb gives one value for each example
                for i, sample in enumerate(dataset):
                    tempDataWeights[k][i] = self.clt_list[k].getProb(sample)
                #tempDataWeights[k] = self.clt_list[k].getProb(dataset)
                # DataWeights is of (2, 12k) dimensions
                dataWeights = tempDataWeights/np.sum(tempDataWeights)

                # weights becomes 2 x 12k 
            weights = np.multiply(self.mixture_probs, dataWeights)/np.sum(np.multiply(self.mixture_probs, dataWeights))

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            # Your code for M-Step here
            for k in range(n_components):
                self.clt_list[k].update(dataset, weights[k])
            
            # Compare two consecutive log liklihoods. And if the difference is less than Epsilon, break/converge.
            # Since logliklihood is only going to increase, we can take difference accordingly
            if(itr == 0):
                secondLL = mix.computeLL(dataset)
            else:
                print("Entered in the iteration ... ", itr)
                firstLL = mix.computeLL(dataset)
                print(firstLL)
                if(abs(firstLL - secondLL) < epsilon):
                    print("Exiting because the increase in log likelihood value is less than epsilon ... ")
                    break
                secondLL = firstLL
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
            ll += np.sum(np.multiply(self.clt_list[k].computeLL(dataset), self.mixture_probs[k])) / dataset.shape[0]
            #print(self.mixture_probs[k])
        return ll  
    
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

        dataset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.ts.data")
        validateset = Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.valid.data")
        testset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.test.data")
        break

# dataset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.ts.data")
# validateset = Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.valid.data")
# testset=Util.load_dataset("C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework4\\dataset\\accidents.test.data")
mix = MIXTURE_CLT()
print("Please wait, while we learn mixture models ...")
print("Printing log likelihood after each iteration ...")

# Latent variable can take values from [2, 5, 10, 20]
for hiddenVariable in [2, 5, 10, 20]:
    start = timeit.default_timer()
    mix.learn(dataset, n_components=hiddenVariable, max_iter=10, epsilon=1e-5)
    print("Running on the validation set when the hidden variable can take up to", hiddenVariable,"values")
    print("Log likelihood = ", mix.computeLL(validateset))
    stop = timeit.default_timer()
    print("Runtime - ", stop-start)