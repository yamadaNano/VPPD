"""Get output"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    # We'll now download the MNIST dataset if it is not yet available.
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)

    # We'll then load and unpickle the file.
    import gzip
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))

    # The targets are int64, we cast them to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################

def rebuild(params, input_var=None):
    inc = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                    input_var=input_var)
    cv1 = recvLayer(inc, 32, (5,5), 'cv1', params, 0, 1)
    pl1 = plLayer(cv1, (3,3), 2, 'pl1')
    cv2 = recvLayer(pl1, 32, (5,5), 'cv2', params, 2, 3)
    pl2 = plLayer(cv2, (3,3), 2, 'pl2')
    pl2D = dropout(pl2, 0.5)
    fc1 = refcLayer(pl2D, 800, 'fc1', params, 4, 5)
    fc1D = dropout(fc1, 0.5)
    fc2 = refcLayer(fc1D, 800, 'fc2', params, 6, 7)
    fc2D = dropout(fc2, 0.5)
    l_out = lasagne.layers.DenseLayer(fc2D, num_units=10, W=params[8],
            b=params[9], nonlinearity=lasagne.nonlinearities.softmax)
    return l_out
    
def fcLayer(incoming, num_units, name):
    '''Build and return a fully-connected layer'''
    fc = lasagne.layers.DenseLayer(incoming, num_units=num_units,
                                   nonlinearity=lasagne.nonlinearities.rectify,
                                   W=lasagne.init.HeUniform(), name=name)
    return fc

def refcLayer(incoming, num_units, name, params, n1, n2):
    '''Build and return a fully-connected layer'''
    fc = lasagne.layers.DenseLayer(incoming, num_units=num_units,
                                   nonlinearity=lasagne.nonlinearities.rectify,
                                   W=params[n1], b=params[n2], name=name)
    return fc

def cvLayer(incoming, nFilters, filterSize, name):
    '''Build and return a conv layer'''
    conv = lasagne.layers.Conv2DLayer(
        incoming, num_filters=nFilters, filter_size=filterSize, name=name,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    return conv

def recvLayer(incoming, nFilters, filterSize, name, params, n1, n2):
    '''Build and return a conv layer'''
    nameW = name + '.W'
    nameb = name + '.b'
    conv = lasagne.layers.Conv2DLayer(
        incoming, num_filters=nFilters, filter_size=filterSize, name=name,
        W=params[n1], b=params[n2],
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    return conv

def plLayer(incoming, poolSize, stride, name):
    '''Build and return a max-pooling layer'''
    pool = lasagne.layers.MaxPool2DLayer(incoming, pool_size=poolSize,
                                         stride=stride, name=name)
    return pool

def dropout(incoming, dropProb):
    '''Build a dropout layer'''
    return lasagne.layers.DropoutLayer(incoming, p=dropProb)

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(filename, savename, nSamples):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    print("Building model and compiling functions...")
    params = np.load(filename)['arr_0']
    network = rebuild(params, input_var=input_var)
    # Output
    prediction = lasagne.layers.get_output(network, deterministic=False)
    # Flow graph compilations
    fn = theano.function([input_var,], prediction)
    # Finally, launch the training loop.
    print("Starting sampling")
    for i in np.arange(nSamples):
        sys.stdout.flush()
        sys.stdout.write('%i\r' % (i,))
        outputs = []
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
            inputs, __ = batch
            outputs.append(fn(inputs))
        outputs = np.vstack(outputs)
        np.save(savename + str(i), outputs)


if __name__ == '__main__':
    filename = './models/cnn.npz'
    savename = './targets/t'
    main(filename, savename, nSamples=100)
