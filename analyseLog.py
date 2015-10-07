'''Analyse log'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy

from matplotlib import pyplot as plt


filename = '/home/daniel/Data/Experiments/N1MLnDAF/N1MLnDAFacc.npz'
data = numpy.load(filename)['running_error']
fig = plt.figure()
plt.plot(data[14800:15200])
plt.show()