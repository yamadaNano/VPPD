'''Beta normalizers'''

import numpy

from scipy.special import gammaln

def softmax(x):
    '''Return the softmax function'''
    e_x = numpy.exp(x - numpy.amax(x, axis=1)[:,numpy.newaxis])
    return e_x/numpy.sum(e_x, axis=1)[:,numpy.newaxis]

def main():
    '''Return numbers'''
    # Generate a number
    x = numpy.random.rand(2, 1000)
    s = softmax(x/T)
    ln_num = numpy.sum(gammaln(T**2*s), axis=1)
    print ln_num
    

if __name__ == '__main__':
    main()