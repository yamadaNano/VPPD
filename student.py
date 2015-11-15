"""Train a teacher"""

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

def loadTeacher(filename):
    '''Load targets from the teacher network'''
    return np.load(filename).astype(theano.config.floatX)


# ##################### Build the neural network model #######################

def build(input_var=None):
    inc = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                    input_var=input_var)
    fc1 = fcLayer(inc, 800, 'fc1')
    fc2 = fcLayer(fc1, 800, 'fc2')
    l_out = lasagne.layers.DenseLayer(fc2, num_units=10, name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def fcLayer(incoming, num_units, name):
    '''Build and return a fully-connected layer'''
    fc = lasagne.layers.DenseLayer(incoming, num_units=num_units,
                                   nonlinearity=lasagne.nonlinearities.rectify,
                                   W=lasagne.init.HeUniform(), name=name)
    return fc

def cvLayer(incoming, nFilters, filterSize, name):
    conv = lasagne.layers.Conv2DLayer(
        incoming, num_filters=nFilters, filter_size=filterSize, name=name,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    return conv

def plLayer(incoming, poolSize, stride, name):    
    pool = lasagne.layers.MaxPool2DLayer(incoming, pool_size=poolSize,
                                         stride=stride, name=name)
    return pool

def dropout(incoming, dropProb):
    '''Build a dropout layer'''
    return lasagne.layers.DropoutLayer(incoming, p=dropProb)

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, targets2, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], targets2[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(targetFile, nEpochs=500, lr=1e-2):
    # Load the dataset
    print("Loading data...")
    X_train, y_train2, X_val, y_val, X_test, y_test = load_dataset()
    y_train = loadTeacher(targetFile + 'pred.npy')
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')
    target_var2 = T.ivector('targets2')
    val_target_var = T.ivector('val_targets')
    print("Building model and compiling functions...")
    network = build(input_var)
    # Loss
    prediction = lasagne.layers.get_output(network, deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    loss = loss + 0.1*lasagne.objectives.categorical_crossentropy(prediction, target_var2).mean()
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            val_target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), val_target_var),
                      dtype=theano.config.floatX)
    # Updates
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=lr, momentum=0.9)
    # Flow graph compilations
    train_fn = theano.function([input_var, target_var, target_var2], loss, updates=updates)
    val_fn = theano.function([input_var, val_target_var], [test_loss, test_acc])
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(nEpochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #y_train = loadTeacher(targetFile + str(epoch%100) + '.npy')
        for batch in iterate_minibatches(X_train, y_train, y_train2, 500, shuffle=False):
            inputs, targets, targets2 = batch
            train_err += train_fn(inputs, targets, targets2)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, y_val, 500, shuffle=False):
            inputs, targets, __ = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, nEpochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=True):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('./models/mlp.npz', lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    targetFile = './targets/t'
    main(targetFile, lr=1e-2)
