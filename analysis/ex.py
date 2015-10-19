'''Examples'''

import os, time, sys

import numpy

from matplotlib import pyplot as plt

def softmax(x):
    '''Return the softmax function'''
    return numpy.exp(x)/numpy.sum(numpy.exp(x))
gap = numpy.zeros((50, 100))
for temp in numpy.arange(50):
    #print temp
    for i in numpy.arange(100):
        s = numpy.random.rand((25))
        t = softmax(s/(temp+1))
        H = (numpy.diag(t) - numpy.outer(t, t))/(temp+1)   
        U, s, _ = numpy.linalg.svd(H)
        #print (numpy.amax(t) - s[0])/s[0]
        gap[temp, i] = numpy.sum(numpy.log(s[:-1])) #(numpy.amax(t) - s[0])/s[0]
gap = numpy.mean(gap, axis=1)
fig = plt.figure()
plt.plot(gap)
plt.show()