'''Analyse the distribution of dirichlet samples'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy

from matplotlib import pyplot as plt
from matplotlib import cm

def softmax(x):
    '''Return the softmax output'''
    e_x = numpy.exp(x - numpy.amax(x))
    return e_x/numpy.sum(e_x)

filename = '/Users/danielernestworrall/Code/VPPD/n01443537_10007.npz'
logits = numpy.load(filename)['arr_0']
p = []
for x in logits:
    p.append(softmax(x))
P = numpy.vstack(p)
Ep = numpy.mean(P, axis=0)
Ep2 = numpy.mean(P**2, axis=0)
S = (Ep - Ep2)/(Ep2 - Ep**2)
alpha = Ep*S - 1
pnew = softmax(alpha)

print('Sum alpha: %f' % (numpy.sum(alpha),))
print('Sum distill: %f' % (numpy.sum(logits[3]/20.),))

'''
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.bar(numpy.arange(1000), pnew)
ax2.bar(numpy.arange(1000), softmax(logits[3]/20.))
plt.show()
'''
cov = numpy.dot(P.T,P) - numpy.outer(Ep,Ep)
fig =plt.figure()
plt.imshow(cov, cmap = cm.gray)
plt.show()

print numpy.amax(cov), numpy.amin(cov)













