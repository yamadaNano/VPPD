'''Simplex analysis'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import numpy


def main():
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    foldername = '/home/daniel/Data/originalLogits/LogitsMean'
    lines = openFile(filename, foldername)
    for line in lines:
        data = numpy.load(line)['arr_0']
        print getConcentration(data)

def openFile(filename, foldername):
    '''Return generator of logits'''
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = os.path.basename(line.rstrip('\n'))
        yield foldername + '/' + line.replace('npy','npz')

def getConcentration(data):
    '''Return concentration parameter'''
    return data.sum()

if __name__ == '__main__':
    main()