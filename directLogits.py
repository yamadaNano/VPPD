'''Direct logits'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy


def main():
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    foldername = '/home/daniel/Data/targets/originalLogits/LogitsMean'
    saveFolder = '/home/daniel/Data/targets/combinedTargets/normedLogitsMean'
    synsets = '/home/daniel/Data/ImageNetTxt/synsets.txt'
    lines = openFile(filename, foldername)
    pairs = getSynsets(synsets)
    T = 10.
    L = 0.1
    for line in lines:
        base = os.path.basename(line)
        savename = saveFolder + '/' + base
        syn = base.split('_')[0]
        data = numpy.load(line)['arr_0']
        t_x = softmax(data/T)
        y_x = getOneHot(t_x.shape, pairs[syn])
        c_x = L*y_x + T*t_x
        s_x = numpy.sum(c_x)
        c_x = c_x / s_x
        numpy.savez(savename, c_x=c_x, s_x=s_x)
        sys.stdout.flush()
        sys.stdout.write('%s \r' % (savename,))
        
def getOneHot(shape, arg):
    '''Return one hot vector with arg augmented'''
    z = numpy.zeros(shape)
    z[:,arg] = 1
    return z

def getSynsets(synsets):
    '''Return a dictionary with the synset mappings'''
    pairs = None
    if synsets is not None:
        pairs = {}
        with open(synsets, 'r') as sp:
            syns = sp.readlines()
        for i, syn in enumerate(syns):
            pairs[syn.rstrip('\n')] = i 
    return pairs

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

def getConcentration(data):
    '''Return concentration parameter'''
    return data.sum()

if __name__ == '__main__':
    main()