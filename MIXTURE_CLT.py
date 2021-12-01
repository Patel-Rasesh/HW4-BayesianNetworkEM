
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
        #self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        self.n_components = n_components
        self.mixture_probs = np.zeros((n_components,dataset.shape[0]))
        weights=np.zeros((n_components,dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        # Initializing p(kth_tree)
        randomWeights = np.random.rand(n_components, dataset.shape[0])
        for i in range(randomWeights.shape[0]):
            self.mixture_probs[i] = randomWeights[i]/np.sum(randomWeights[i])
        # Instantiating a variable for storing Chow-Liu probability
        tempDataWeights = np.zeros((n_components, dataset.shape[0]))
        dataWeights = np.zeros((n_components, dataset.shape[0]))

        # For storing the logliklihood from the previous iteration
        secondLL = 0

        # Initializing one CLT for each value of the latent variable
        self.clt_list = [CLT() for i in range(n_components)]
        # Learning Chow-Liu Trees for each value of the hidden variable
        for k in range(n_components):
            self.clt_list[k].learn(dataset)
            
        for itr in range(max_iter):
            # E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            # Your code for E-step here

            for k in range(n_components):
                # getProb gives one value for each example
                
                #for i, sample in enumerate(dataset):
                tempDataWeights[k] = np.apply_along_axis(self.clt_list[k].getProb, 1, dataset)

                #print(tempDataWeights[k])
                dataWeights[k] = tempDataWeights[k]/np.sum(tempDataWeights[k])
                #print(np.sum(dataWeights[k]))
                weights[k] = np.multiply(self.mixture_probs[k], dataWeights[k])/np.sum(np.multiply(self.mixture_probs[k], dataWeights[k]))
                self.clt_list[k].update(dataset, weights[k])
            # DataWeights is of (2, 12k) dimensions
            # for k in range(n_components):
            #     dataWeights[k] = tempDataWeights[k]/np.sum(tempDataWeights[k])
            # for k in range(n_components):
            #     self.mixture_probs[k] = weights[k]

            # # # weights becomes (n_component x dataset.shape[0]) 
            # for k in range(n_components):
            #     weights[k] = np.multiply(self.mixture_probs[k], dataWeights[k])/np.sum(np.multiply(self.mixture_probs[k], dataWeights[k]))
            #     print(np.sum(weights[k]))
            # # M-step: Update the Chow-Liu Trees and the mixture probabilities
            # # Your code for M-Step here
            # for k in range(n_components):
            #     self.clt_list[k].update(dataset, weights[k])
            
            # Compare two consecutive log liklihoods. And if the difference is less than Epsilon, break/converge.
            # Since logliklihood is only going to increase, we can take difference accordingly
            if(itr == 0):
                print("Debug print")
                secondLL = mix.computeLL(dataset)
                print("Completed the first iteration.")
                print(secondLL)
            else:
                print("Entered in the iteration ... ", itr)
                firstLL = mix.computeLL(dataset)
                print(firstLL)
                if(abs(firstLL - secondLL) < epsilon):
                    print("Exiting because the increase in log likelihood value is less than epsilon ... ")
                    print(secondLL)
                    break
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
        
        for k in range(self.n_components):
           # temp = np.sum(self.clt_list[k].computeLL(dataset))/self.clt_list[k].computeLL(dataset)
            #ll += np.sum((self.clt_list[k].computeLL(dataset) * self.mixture_probs[k])) 
            for sample in range(dataset.shape[0]):

                ll += np.prod(np.where(dataset[sample] == 1, self.mixture_probs[k][sample], 1-self.mixture_probs[k][sample]))
                # for variable in range(dataset.shape[1]):
                #     if(dataset[sample][variable] == 1):
                #         ll += self.mixture_probs[k][sample] * self.clt_list[k].computeLL(dataset)
                #     elif(dataset[sample][variable] == 0):
                #         ll += (1-self.mixture_probs[k][sample]) * self.clt_list[k].computeLL(dataset)
            #print(self.mixture_probs[k])
        #temp = 0
            ll * self.clt_list[k].computeLL(dataset)
        # for k in range(self.n_components):
        #     temp += self.clt_list[k].computeLL(dataset)
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

# VALIDATION - uncomment the next 10 lines for tuning the value of the hidden variable k
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

# TESTING - refer the code starting below
# Below are the values of a hidden variable (for each dataset) we have gotten based on validation sets
# kTest = [10, 20, 5, 10, 5, 10, 10, 2, 10, 20]
# hiddenVariableList = defaultdict(int)
# i = 0
# for key, values in index.items():
#     hiddenVariableList[key] = kTest[i]
#     i += 1

# start = timeit.default_timer()
# print("Learning models for appropriate value of the hidden variable before running it on the testset ...")
# print("Average runtime is 5-10 mins")
# print("NOTE - program is printing log likelihood after each iteration ...")
# mix.learn(dataset, n_components=hiddenVariableList[int(selection)], max_iter=10, epsilon=1e-5)
# print("Running on the testset when the hidden variable can take up to", hiddenVariableList[selection],"values")
# print("Log likelihood = ", mix.computeLL(testset))
# stop = timeit.default_timer()
# print("Runtime - ", stop-start)