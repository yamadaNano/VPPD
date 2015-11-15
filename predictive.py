'''Teacher predictive density'''

import os, sys, time

import numpy

folder = './targets'
data = numpy.zeros((50000,10))
for i in range(100
    print()
    data += numpy.load(folder + '/t' + str(i) + '.npy')

savename = './targets/tpred.npy'
numpy.save(savename, data)
