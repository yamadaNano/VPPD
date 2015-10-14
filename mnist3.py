"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import cPickle
import numpy as np
import theano
import theano.tensor as T

import lasagne
from theano.compile.nanguardmode import NanGuardMode


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    from urllib import urlretrieve
    def pickle_load(f, encoding):
        return cPickle.load(f)

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
    X_train, y_train = stripData(X_train, y_train)
    X_val, y_val = stripData(X_val, y_val)
    X_test, y_test = stripData(X_test, y_test)

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
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.DropoutLayer(l_in, p=0.2), num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify, name='l_hid1',
            W=lasagne.init.GlorotUniform())
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.DropoutLayer(l_hid1, p=0.5), num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify, name='l_hid2',
            W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.DropoutLayer(l_hid2, p=0.5), num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax, name='l_out',
            W=lasagne.init.GlorotUniform())
    return l_out

def reload_mlp(filename, input_var=None, num_hid=800):
    params = np.load(filename)
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_in, 0.2), num_units=num_hid,
            W=params['l_hid1.W'], b=params['l_hid1.b'], name='l_hid1')
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid1, 0.5), num_units=num_hid,
            W=params['l_hid2.W'], b=params['l_hid2.b'], name='l_hid2')
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid2, 0.5), num_units=3,
            W=params['l_out.W'], b=params['l_out.b'], name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out
    
    
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

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

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    modelFile = './models/mnist3.pkl'
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    print('Building mlp')
    network = build_mlp(input_var)
    prediction = lasagne.layers.get_output(network, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=1e-2, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
    save_model(network, modelFile)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    save_model(network, modelFile)

def getTargets():
    '''Reload model and get the targets'''
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    input_var = T.tensor4('inputs')
    model = reload_mlp('./models/mnist3.pkl', input_var=input_var)
    prediction = lasagne.layers.get_output(model, deterministic=True)
    fn = theano.function([input_var], prediction)
    output = []
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, _ = batch
        t = fn(inputs)
        output.append(t)
    output = np.vstack(output)
    np.save('/home/daniel/Data/mnist/outputs/test.npy', output)
        
# ################################ Helpers ####################################

def save_model(model, file_name):
    '''Save the model parameters'''
    print('Saving model..')
    params = {}
    for param in lasagne.layers.get_all_params(model):
        params[str(param)] = param.get_value()
    
    file = open(file_name, 'w')
    cPickle.dump(params, file, cPickle.HIGHEST_PROTOCOL)
    file.close()

def stripData(X, y, n=3):
    '''Return the first n labelled data'''
    mask = (y < 3)
    y = np.compress(mask, y, axis=0)
    X = np.compress(mask, X, axis=0)
    return (X, y)

if __name__ == '__main__':
    getTargets()
    #main()















































