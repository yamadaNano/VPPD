'''Preprocess logits'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy

from matplotlib import pyplot as plt


def main():
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    logitfolder = '/home/daniel/Data/originalLogits/AugMeanLogits'
    dstfolder = '/home/daniel/Data/normedLogits/AugMeanLogits'
    D = 1000
    k = 1.85*numpy.log10(D + 4)
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for i, line in enumerate(lines):
        base = os.path.basename(line.rstrip('.JPEG\n'))
        src_address = logitfolder + '/' + base + '.npz'
        dst_address = dstfolder + '/' + base + '.npz'
        logits = numpy.load(src_address)['arr_0']
        # Need to get the mean logit for range normalization
        mean_logit = getMeanLogit(logits)
        new_logits, T = normalize(mean_logit, k)
        numpy.savez(dst_address, logits=new_logits, T=T)
        sys.stdout.flush()
        sys.stdout.write('%i %s \r' % (i,dst_address))

def normalize(logits, k):
    '''Return the logits and the equivalent distillation temperature'''
    R = numpy.amax(logits, axis=1) - numpy.amin(logits, axis=1)
    T = R/k
    normed_logits = logits/numpy.maximum(T,0.1)
    return (normed_logits, T)

def softMax(logits):
    '''Return the softmax for the input logits'''
    return numpy.exp(logits)/numpy.sum(numpy.exp(logits), axis=1)[:,numpy.newaxis]

def getMeanLogit(logits):
    '''Return mean logit'''
    output = softMax(logits)
    mean_output = numpy.mean(output, axis=0)
    return numpy.log(mean_output)[numpy.newaxis,:]


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    