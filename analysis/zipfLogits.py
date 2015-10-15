'''Zipf plots of logits'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy

from matplotlib import pyplot as plt


def main():
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    foldername = '/home/daniel/Data/originalLogits/LogitsMean'
    lines = openFile(filename, foldername)
    fig = plt.figure()
    for T in numpy.arange(1,21, step=2):
        print T
        lines = openFile(filename, foldername)
        t_x = numpy.zeros((1,1000))
        for i, line in enumerate(lines):
            data = numpy.load(line)['arr_0']
            sys.stdout.flush()
            sys.stdout.write('%s \r' % (line,))
            t_x += numpy.sort(softmax(data/T))[:,::-1]
            if i % 1000 == 999:
                plt.loglog(numpy.arange(1000)+1, t_x[0,:])
                break
    plt.show()

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
    main()