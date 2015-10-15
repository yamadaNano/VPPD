'''Cifar CNN'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import cPickle
import cv2
import handling as hd
import nonlinearities as nl
import numpy as np
import skimage
import skimage.io as skio
import theano
import theano.tensor as T

import lasagne



# ##################### Build the neural network model #######################

def build_cnn(im_shape, input_var=None):
    incoming = lasagne.layers.InputLayer(shape=(None, 3, im_shape[0], im_shape[1]),
                                        input_var=input_var)
    conv1 = lasagne.layers.Conv2DLayer(
            incoming, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool1 = lasagne.layers.MaxPool2DLayer(conv1, pool_size=(3, 3), stride=2)
    conv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride=2)
    conv3 = lasagne.layers.Conv2DLayer(
            pool2, num_filters=64, filter_size=(3, 3),
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool3 = lasagne.layers.MaxPool2DLayer(conv3, pool_size=(3, 3), stride=2)
    conv4 = lasagne.layers.Conv2DLayer(
            pool3, num_filters=64, filter_size=(3, 3),
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool4 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(3, 3), stride=2)
    full5 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool4, 0.5), num_units=2000,
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    full6 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(full5, 0.5), num_units=2000,
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    full7 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(full6, 0.5), num_units=1000,
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.softmax)
    return full7

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(train_file, logit_folder, val_file, savename, num_epochs=500,
         margin=25, base=0.01, mb_size=50, momentum=0.9, synsets=None,
         preproc=False, loss_type='crossentropy'):
    print("Loading data...")
    tr_addresses, tr_labels = hd.get_traindata(train_file, synsets)
    vl_addresses, vl_labels = hd.get_valdata(val_file)
    # Variables
    input_var = T.tensor4('inputs')
    soft_target = T.fmatrix('soft_target')
    hard_target = T.ivector('hard_target')
    learning_rate = T.fscalar('learning_rate')
    temp = T.fscalar('temp')
    im_shape = (227, 227)
    max_norm = 3.87
    t = 10.
    print("Building model and compiling functions...")
    network = build_cnn(im_shape, input_var=input_var)
    # Losses and updates
    prediction = lasagne.layers.get_output(network, deterministic=False)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    loss = losses(prediction, soft_target, loss_type)
    loss += regularization(prediction, t)
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1),
                        T.argmax(soft_target, axis=1)), dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(network)
    for param in params:
        if param.name == 'W':
            param = lasagne.updates.norm_constraint(param, max_norm, epsilon=1e-3)
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=learning_rate,
                                                momentum=momentum)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), hard_target),
                      dtype=theano.config.floatX)
    # Theano functions
    train_fn = theano.function([input_var, soft_target, learning_rate],
        [loss, train_acc], updates=updates)
    val_fn = theano.function([input_var, hard_target], test_acc)
    print("Starting training...")
    # We iterate over epochs:
    start_time = time.time()
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        learning_rate = get_learning_rate(epoch, margin, base)
        train_err = 0; train_batches = 0; running_error = []
        t_acc = 0; running_acc = []
        trdlg = hd.data_target_generator(tr_addresses, logit_folder,
                                              im_shape, mb_size, preproc=preproc,
                                              shuffle=True, synsets=synsets)
        for batch in hd.threaded_gen(trdlg, num_cached=500):
            inputs, soft = batch
            local_train_err, acc = train_fn(inputs, soft, learning_rate)
            train_err += local_train_err; t_acc += acc
            running_error.append(local_train_err); running_acc.append(acc)
            h, m, s = theTime(start_time)
            train_batches += 1
            if train_batches % 257 == 0:
                save_errors(savename, running_error, err_type='error')
                save_errors(savename, running_acc, err_type='acc')
                running_error = []; running_acc = []
            sys.stdout.write('Time: %d:%02d:%02d Minibatch: %i Training Error: %f\r' %
                             (h, m, s, train_batches, train_err/train_batches)),
            sys.stdout.flush()
        print
        val_acc = 0; val_batches = 0
        vldlg = hd.data_and_label_generator(vl_addresses, vl_labels, im_shape,
                                         mb_size)
        running_val_acc = []
        for batch in hd.threaded_gen(vldlg, num_cached=50):
            inputs, targets = batch
            val_acc += val_fn(inputs, targets)
            val_batches += 1
            sys.stdout.write('Minibatch: %i Validation Accuracy: %f\r' %
                             (val_batches, val_acc/val_batches * 100)),
            sys.stdout.flush()
        running_val_acc.append(val_acc/val_batches)
        save_errors(savename, running_val_acc, err_type='val_acc')
        print
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  train loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  valid acc:\t\t{:.6f}".format(val_acc / val_batches * 100.))

# ################################ Helpers ####################################

def get_learning_rate(epoch, margin, base):
    return base*margin/np.maximum(epoch,margin)

def theTime(start):
    '''Return the time in hh:mm:ss'''
    t = time.time() - start
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return (h, m, s)

def save_errors(filename, running_error, err_type='error'):
    running_error = np.asarray(running_error)
    savename = filename.split('.')
    savename = savename[0] + err_type + '.npz'
    if err_type == 'error':
        if os.path.isfile(savename):
            arr = np.load(savename)['running_error']
            running_error = np.hstack((arr, running_error))
    elif err_type == 'acc':
        if os.path.isfile(savename):
            arr = np.load(savename)['running_error']
            running_error = np.hstack((arr, running_error))
    elif err_type == 'val_acc':
        if os.path.isfile(savename):
            arr = np.load(savename)['running_error']
            running_error = np.hstack((arr, running_error))
    np.savez(savename, running_error=running_error)
    fig = plt.figure()
    plt.plot(running_error)
    plt.xlabel('Iterations')
    if err_type == 'error':
        plt.ylabel('Error')
    elif err_type == 'acc':
        plt.ylabel('Accuracy')
    elif err_type == 'val_acc':
        plt.ylabel('Validation Accuracy')
    plt.savefig(savename.replace('.npz','.png'))
    plt.close()

def losses(prediction, target, loss_type):
    if loss_type == 'crossentropy':
        loss = -T.sum(target*T.log(prediction), axis=1)
    elif loss_type == 'VPPD':
        loss = T.sum(prediction*(T.log(prediction) - T.log(target)), axis=1)
    else:
        print('Loss type not recognised')
        sys.exit()
    return loss.mean()

def regularization(prediction, t):
    return (t**2)*T.mean(T.log(T.sum(T.pow(prediction,1./t),axis=1)))

if __name__ == '__main__':
    data_root = '/home/daniel/Data/'
    loss_type = 'crossentropy'
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    if len(sys.argv) > 2:
        loss_type = sys.argv[2]
    main(train_file = data_root + 'ImageNetTxt/transfer.txt',
         logit_folder = data_root + 'combinedTargets/LogitsMean',
         val_file = data_root + 'ImageNetTxt/val50.txt',
         savename = data_root + 'Experiments/bridge/T10_0p005.npz',
         num_epochs=50, margin=25, base=5e-3, mb_size=50, momentum=0.9,
         preproc=True, synsets= data_root +'ImageNetTxt/synsets.txt')
        
# Savename codes
# N1-ML-(n)DA.npz
# Network 1,2,3...
# M = mean, L = Logit, A = Augmented logits
# (n)DA (no/yes to) data augmentation
# test for testing
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
