'''Lasange and numpy layer-wise nonlinearities'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import lasagne
import numpy as np
import theano.tensor as T

class SoftermaxNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, k, **kwargs):
        super(SoftermaxNonlinearity, self).__init__(incoming, **kwargs)
        self.k = k

    def get_output_for(self, input, **kwargs):
        R = (T.max(input,axis=1)-T.min(input,axis=1)).dimshuffle(0,'x')
        input = self.k*input/T.maximum(R,0.1)
        return T.nnet.softmax(input)

class SoftmaxNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(SoftmaxNonlinearity, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return T.nnet.softmax(input)

def softerMax(logits, k):
    '''Return the softermax function'''
    R = np.max(logits, axis=1) - np.min(logits, axis=1)
    arg = k*logits/np.maximum(R,0.1)[:,np.newaxis]
    return np.exp(arg)/(1e-7+np.sum(np.exp(arg), axis=1)[:,np.newaxis])

def softMax(logits):
    '''Return the softermax function'''
    e_x = np.exp(logits - np.amax(logits, axis=1)[:,np.newaxis])
    return e_x/np.sum(e_x, axis=1)[:,np.newaxis]

class DistillationNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, temp, **kwargs):
        super(DistillationNonlinearity, self).__init__(incoming, **kwargs)
        self.temp = temp

    def get_output_for(self, input, **kwargs):
        input = input/self.temp
        return T.exp(input)/(1e-7+T.sum(T.exp(input), axis=1).dimshuffle(0,'x'))

def distill(logits, temp):
    '''Return the distilled softmax function'''
    return np.exp(logits/temp)/(1e-7+np.sum(np.exp(logits/temp), axis=1)[:,np.newaxis])