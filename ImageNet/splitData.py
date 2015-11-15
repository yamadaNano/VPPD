'''MNIST data splits'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import gzip
import lasagne
import numpy as np
import pickle
import theano
import theano.tensor as T

from lasagne import utils
from collections import OrderedDict
from matplotlib import pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams


# We'll now download the MNIST dataset if it is not yet available.
url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
filename = 'mnist.pkl.gz'
if not os.path.exists(filename):
    print("Downloading MNIST dataset...")
    urlretrieve(url, filename)
# We'll then load and unpickle the file.
def pickle_load(f, encoding):
    return pickle.load(f)
with gzip.open(filename, 'rb') as f:
    data = pickle_load(f, encoding='latin-1')
# Unpack data
X_train, y_train = data[0]
X_val, y_val = data[1]
X_test, y_test = data[2]
# Reshape to standard 4-array
X_train = X_train.reshape((-1, 1, 28, 28)).astype(theano.config.floatX)
X_val = X_val.reshape((-1, 1, 28, 28)).astype(theano.config.floatX)
X_test = X_test.reshape((-1, 1, 28, 28)).astype(theano.config.floatX)
# The targets are int64, we cast them to int8 for GPU compatibility.
y_train = y_train.astype(np.uint8)
y_val = y_val.astype(np.uint8)
y_test = y_test.astype(np.uint8)

# Split the training data into partitions
Xtrain = []; ytrain = []
for i in np.arange(10):
    idx = np.in1d(y_train, i)
    X = np.array_split(X_train[idx,...], 10, axis=0)
    Y = np.array_split(y_train[idx,...], 10, axis=0)
    Xtrain.append(X)
    ytrain.append(Y)

# Recombine into packets    
X_train = []; y_train = []
for i in np.arange(10):
    X_train.append(np.zeros((0,1,28,28)))
    y_train.append(np.zeros((0,)))
    x = []; y = []
    for j in np.arange(10):
        x = Xtrain[j][i]
        y = ytrain[j][i]
        X_train[i] = np.vstack((X_train[i], x))
        y_train[i] = np.hstack((y_train[i], y))

# Accumulate
for i in np.arange(10,0,-1):
    X_train[i-1] = np.vstack(X_train[:i])
    y_train[i-1] = np.hstack(y_train[:i])

data = X_train, y_train, X_val, y_val, X_test, y_test
# Save
fname = '/home/daniel/Code/VPPD/models/MNISTsplit.pkl'
with open(fname, 'w') as fp:
    cPickle.dump(data, fp, cPickle.HIGHEST_PROTOCOL)
    




































