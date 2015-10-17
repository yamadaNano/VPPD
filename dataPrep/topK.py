'''Top k classes'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy


'''Direct logits'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy

from matplotlib import pyplot as plt
def main(T=10, K=10):
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    foldername = '/home/daniel/Data/targets/originalLogits/LogitsMean'
    saveFolder = '/home/daniel/Data/targets/clippedTargets/LogitsMean' + str(K)
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    lines = openFile(filename, foldername)
    for line in lines:
        base = os.path.basename(line)
        savename = saveFolder + '/' + base
        data = numpy.load(line)['arr_0']
        s = softmax(data/T)
        rank = numpy.argsort(s, axis=1)
        s[:,rank[:,:-K]] = numpy.mean(s[:,rank[:,:-K]])
        numpy.save(savename, s)
        sys.stdout.flush()
        sys.stdout.write('%s \r' % (savename,))
        

def openFile(filename, foldername):
    '''Return generator of logits'''
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = os.path.basename(line.rstrip('\n'))
        yield foldername + '/' + line.replace('npy','npz')

def softmax(x):
    '''Return the softmax output'''
    e_x = numpy.exp(x - numpy.amax(x, axis=1)[:,numpy.newaxis])
    return e_x/numpy.sum(e_x, axis=1)[:,numpy.newaxis]


if __name__ == '__main__':
    main(T=10, K=5)






















