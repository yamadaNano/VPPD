'''Distillation experiments'''

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

def reload_cnn(im_shape, filename, input_var=None):
    params = np.load(filename)
    incoming = lasagne.layers.InputLayer(shape=(None, 3, im_shape[0], im_shape[1]),
                                        input_var=input_var)
    conv1 = lasagne.layers.Conv2DLayer(
            incoming, num_filters=32, filter_size=(3, 3), name='conv1',
            W=params['conv1.W'], b=params['conv1.b'],
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool1 = lasagne.layers.MaxPool2DLayer(conv1, pool_size=(3, 3), stride=2)
    conv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3), name='conv2',
            W=params['conv2.W'], b=params['conv2.b'],
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride=2)
    conv3 = lasagne.layers.Conv2DLayer(
            pool2, num_filters=64, filter_size=(3, 3), name='conv3',
            W=params['conv3.W'], b=params['conv3.b'],
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool3 = lasagne.layers.MaxPool2DLayer(conv3, pool_size=(3, 3), stride=2)
    conv4 = lasagne.layers.Conv2DLayer(
            pool3, num_filters=64, filter_size=(3, 3), name='conv4',
            W=params['conv4.W'], b=params['conv4.b'],
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool4 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(3, 3), stride=2)
    full5 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool4, 0.5), num_units=2000, name='full5',
            W=params['full5.W'], b=params['full5.b'],
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    full6 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(full5, 0.5), num_units=2000, name='full6',
            W=params['full6.W'], b=params['full6.b'],
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    full7 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(full6, 0.5), num_units=50, name='full7',
            W=params['full7.W'], b=params['full7.b'],
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return full7

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(train_file, logit_folder, val_file, savename, synmap_file, mb_size=50,
         preproc=False, synsets=None, deterministic=True,
         modelFile='./myModel.pkl'):
    print('Model file: %s' % (modelFile,))
    print("Loading data...")
    tr_addresses, tr_labels = hd.get_traindata(train_file, synsets)
    vl_addresses, vl_labels = hd.get_valdata(val_file)
    synmap = get_synmap(synmap_file)
    tr_labels = map_labels(tr_labels, synmap)
    vl_labels = map_labels(vl_labels, synmap)
    # Variables
    input_var = T.tensor4('inputs')
    im_shape = (227, 227)
    print("Building model and compiling functions...")
    network = reload_cnn(im_shape, modelFile, input_var=input_var)
    _, test_prediction = lasagne.layers.get_output(network, deterministic=deterministic)
    # Theano functions
    fn = theano.function([input_var], test_prediction)
    print("Starting training...")
    # We iterate over epochs:
    # In each epoch, we do a full pass over the training data:
    train_batches = 0
    t_acc = 0; running_acc = []
    trdlg = data_and_label_generator(addresses, labels, im_shape, mb_size,
                                     shuffle=False, preproc=preproc)
    for batch in hd.threaded_gen(trdlg, num_cached=500):
        inputs, _, _ = batch
        output = fn(inputs)
        train_batches += 1
        sys.stdout.write('Minibatch: %i\r' % (train_batches,)),
        sys.stdout.flush()
        sn = savename + str(train_batches) + '.npz'
        np.savez(sn, output)

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


if __name__ == '__main__':
    #data_root = '/home/dworrall/Data/'
    data_root = '/home/daniel/Data/'
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    main(train_file = data_root + 'ImageNetTxt/transfer.txt',
         logit_folder = data_root + 'originalLogits/LogitsMean',
         val_file = data_root + 'ImageNetTxt/val50.txt',
         savename = data_root + 'Experiments/trainRetrain/OP',
         synmap_file = data_root + 'ImageNetTxt/synmap.txt',
         mb_size=50, preproc=False, synsets=data_root +'ImageNetTxt/synsets.txt',
         modelFile= data_root + 'Experiments/alpha-3b/model0.pkl',
         deterministic = True)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
