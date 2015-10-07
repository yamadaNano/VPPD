'''Golden section line search'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy

from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar


def main():
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    logitfolder = '/home/daniel/Data/originalLogits/LogitsMean'
    dstfolder = '/home/daniel/Data/goldenLogits/LogitsMean'
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for i, line in enumerate(lines):
        base = os.path.basename(line.rstrip('.JPEG\n'))
        src_address = logitfolder + '/' + base + '.npz'
        dst_address = dstfolder + '/' + base + '.npz'
        logits = numpy.load(src_address)['arr_0']
        # Need to get the mean logit for range normalization
        mean_logit = getMeanLogit(logits)
        x = numpy.linspace(0.1, 25, 1000)
        y = []
        for i in x:
            t = softMax(i, mean_logit)
            y.append(numpy.sum(t - t**2))
        y = numpy.asarray(y)
        fig = plt.figure()
        plt.plot(x, y)
        plt.show()
        new_logits, T = lineSearch(mean_logit)
        numpy.savez(dst_address, logits=new_logits, T=T)
        #sys.stdout.flush()
        #sys.stdout.write('%i %s \r' % (i,dst_address))

def lineSearch(logits):
    '''Minimize trace of Hessian wrt temperature'''
    def fun(T):
        t = softMax(T, logits)
        return -numpy.sum(t - t**2)
    res = minimize_scalar(fun, bounds=(0.1, 25), method='golden')
    print res.x
    return logits/res.x, res.x

def softMax(T, logits):
    '''Return the softmax for the input logits'''
    return numpy.exp(logits/T)/numpy.sum(numpy.exp(logits/T), axis=1)[:,numpy.newaxis]

def getMeanLogit(logits):
    '''Return mean logit'''
    output = softMax(1, logits)
    mean_output = numpy.mean(output, axis=0)
    return numpy.log(mean_output)[numpy.newaxis,:]


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    