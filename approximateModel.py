'''Cifar CNN'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import cv2
import numpy as np
import skimage.io as skio
import theano
import theano.tensor as T

import lasagne

from matplotlib import pyplot as plt
# ##################### Build the neural network model #######################

def build_cnn(im_shape, temp, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, im_shape[0], im_shape[1]),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, 0.5), num_units=500,
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, 0.5), num_units=500,
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network,0.5), num_units=10,
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.linear)
    network = SoftermaxNonlinearity(network, temp)
    return network

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

def main(num_epochs=500, margin=25, base=0.01):
    print("Loading data...")
    train_file = '/media/daniel/DATA/data_unencrypted/disc1_canada/train_file.txt'
    tr_addresses, tr_labels = get_metadata(train_file)
    valid_file = '/media/daniel/DATA/data_unencrypted/disc1_canada/valid_file.txt'
    vl_addresses, vl_labels = get_metadata(valid_file)
    # Variables
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learning_rate = T.fscalar('learning_rate')
    im_shape = (120, 160)
    mb_size = 20
    print("Building model and compiling functions...")
    network = build_cnn(im_shape, input_var)
    # Losses and updates
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
    #        loss, params, learning_rate=learning_rate, momentum=0.9)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=learning_rate)
    # Validation and testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Theano functions
    train_fn = theano.function([input_var, target_var, learning_rate],
        loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_acc)
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        learning_rate = get_learning_rate(epoch, margin, base)
        train_err = 0; train_batches = 0
        start_time = time.time()
        trdg = data_generator(tr_addresses, tr_labels, im_shape, mb_size,
                              preproc=True)
        for batch in threaded_gen(trdg, num_cached=50):
            inputs, targets = batch
            train_err += train_fn(inputs, targets, learning_rate)
            train_batches += 1
            sys.stdout.write('Minibatch: %i Training Error: %f\r' %
                             (train_batches, train_err/train_batches)),
            sys.stdout.flush()
        print
        val_acc = 0; val_batches = 0
        vldg = data_generator(vl_addresses, vl_labels, im_shape, mb_size)
        for batch in threaded_gen(vldg, num_cached=50):
            inputs, targets = batch
            val_acc += val_fn(inputs, targets)
            val_batches += 1
            sys.stdout.write('Minibatch: %i Validation Accuracy: %f\r' %
                             (val_batches, val_acc/val_batches * 100)),
            sys.stdout.flush()
        print
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  train loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  valid acc:\t\t{:.6f}".format(val_acc / val_batches * 100.))

# ################################ Helpers ####################################

def get_learning_rate(epoch, margin, base):
    return base*(margin*6)/np.maximum(epoch*6,margin)

# ################################ Layers #####################################

class SoftermaxNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, temp, **kwargs):
        super(SoftermaxNonlinearity, self).__init__(incoming, **kwargs)
        self.temp = temp

    def get_output_for(self, input, training=False, **kwargs):
        if training:
            R = (T.max(input,axis=1)-T.min(input,axis=1)).dimshuffle(0,'x')
            input = self.temp*input/T.maximum(R,0.1)
        return T.exp(input)/T.sum(T.exp(input), axis=1).dimshuffle(0,'x')

# ############################## Data handling ################################
def get_metadata(srcfile):
    '''Get all the addresses in the file'''
    with open(srcfile, 'r') as fp:
        lines = fp.readlines()
        num_lines = len(lines)
    return lines, num_lines

def data_generator(addresses, num_samples, preproc=False):
    '''Get images, resize and preprocess'''
    for line in addresses:
        line = line.rstrip('\n')
        base = os.path.basename(line).replace('.JPEG','')
        image = skio.imread(line)
        image = preprocess(image, num_samples, preproc=preproc)
        image = numpy.dstack(image)
        # Really need to add some kind of preprocessing
        yield (image, base)

def threaded_gen(generator, num_cached=50):
    '''Threaded generator to multithread the data loading pipeline'''
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
        
# ############################ Data preprocessing #############################
def preprocess(im, num_samples, preproc=True):
    '''Data normalizations and augmentations'''
    if im.ndim == 2:
        im = numpy.dstack((im,)*3)
    if preproc == True:
        img = []
        for i in numpy.arange(num_samples):
        # Random rotations
            angle = numpy.random.rand() * 360.
            M = cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2), angle, 1)
            img.append(cv2.warpAffine(im, M, (im.shape[1],im.shape[0])))
            # Random fliplr
            if numpy.random.rand() > 0.5:
                img[i] = img[i][:,::-1,...]
    else:
        img = (im,)*num_samples
    return img

if __name__ == '__main__':
    main(num_epochs=80, base=0.01)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
