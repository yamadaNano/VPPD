'''Partition MNIST into blocks'''

import os, sys, time

import cPickle as pickle
import gzip
import numpy

from urllib import urlretrieve


def partition(savename, N=10):
    '''Partition the data into N sets of increasing size'''
    def pickle_load(f, encoding):
        return pickle.load(f)
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    
    # Generate the partitions
    nest = partition_vector(numpy.arange(X_train.shape[0]), N)
    # Make and store partitions
    with open(savename, 'w') as fp:
        pickle.dump(nest, fp, pickle.HIGHEST_PROTOCOL)
    
    
def partition_vector(vec, N=10):
    '''Partition vector into N nested parts'''
    parts = numpy.array_split(vec, N)
    nest = []
    for i in numpy.arange(N):
        nest.append(numpy.hstack(parts[:(i+1)]))
    return nest


if __name__ == '__main__':
    savename = '/home/daniel/Data/MNIST/partitions.pkl'
    partition(savename, 10)