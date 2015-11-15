'''Teacher predictive density'''

import os, sys, time

import numpy

# Load the targets and form the predictive mean
folder = './targets'
data = numpy.zeros((50000,10))
for i in range(100):
    data += numpy.load(folder + '/t' + str(i) + '.npy')
data = data/numpy.sum(data, axis=1)[:,numpy.newaxis]
savename = './targets/tpred.npy'
numpy.save(savename, data)
