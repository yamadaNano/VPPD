'''Outright train a CNN from scratch'''

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
            incoming, num_filters=32, filter_size=(3, 3), name='conv1',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool1 = lasagne.layers.MaxPool2DLayer(conv1, pool_size=(3, 3), stride=2)
    conv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=32, filter_size=(3, 3), name='conv2',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride=2)
    conv3 = lasagne.layers.Conv2DLayer(
            pool2, num_filters=64, filter_size=(3, 3), name='conv3',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool3 = lasagne.layers.MaxPool2DLayer(conv3, pool_size=(3, 3), stride=2)
    conv4 = lasagne.layers.Conv2DLayer(
            pool3, num_filters=64, filter_size=(3, 3), name='conv4',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    pool4 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(3, 3), stride=2)
    full5 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool4, 0.5), num_units=2000, name='full5',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    full6 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(full5, 0.5), num_units=2000, name='full6',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    full7 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(full6, 0.5), num_units=1000, name='full7',
            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(0.01),
            nonlinearity=lasagne.nonlinearities.softmax)
    return full7

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(train_file, val_file, savename, num_epochs=500, alpha=0.1,
         margin=25, base=0.01, mb_size=50, momentum=0.9, synsets=None):
    print("Loading data...")
    print('Alpha: %f' % (alpha,))
    tr_addresses, tr_labels = get_traindata(train_file, synsets)
    vl_addresses, vl_labels = get_valdata(val_file)
    N = len(tr_addresses)
    print('Num training examples: %i' % (N,))
    print('Alpha/N: %f' % (alpha/N,))
    # Variables
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learning_rate = T.fscalar('learning_rate')
    im_shape = (227, 227)
    print("Building model and compiling functions...")
    network = build_cnn(im_shape, input_var=input_var)
    # Losses and updates
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + regularization(prediction, alpha/N).mean()
    params = lasagne.layers.get_all_params(network, deterministic=False)
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                    learning_rate=learning_rate,
                                    momentum=momentum)
    # Validation and testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Theano functions
    train_fn = theano.function([input_var, target_var, learning_rate],
        [loss, train_acc], updates=updates)
    val_fn = theano.function([input_var, target_var], test_acc)
    print("Starting training...")
    # We iterate over epochs:
    start_time = time.time()
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        learning_rate = get_learning_rate(epoch, margin, base)
        train_err = 0; train_batches = 0; running_error = []; running_acc = []
        acc = 0.
        trdlg = data_and_label_generator(tr_addresses, tr_labels, im_shape,
                                         mb_size, shuffle=True)
        for batch in threaded_gen(trdlg, num_cached=500):
            inputs, targets = batch
            local_train_err, local_train_acc = train_fn(inputs, targets, learning_rate)
            train_err += local_train_err; acc += local_train_acc
            train_batches += 1
            if np.isnan(local_train_err):
                sys.exit()
            running_error.append(local_train_err)
            running_acc.append(local_train_acc)
            if train_batches % 257 == 0:
                save_errors(savename, running_error, err_type='error')
                save_errors(savename, running_acc, err_type='acc')
                running_error = []; running_acc = []
            h, m, s = theTime(start_time)
            sys.stdout.write('Time: %d:%02d:%02d Minibatch: %i Training Error: %f\r' %
                             (h, m, s, train_batches, train_err/train_batches)),
            sys.stdout.flush()
        print
        val_acc = 0; val_batches = 0; running_val_acc=[]
        vldlg = data_and_label_generator(vl_addresses, vl_labels, im_shape,
                                         mb_size)
        for batch in threaded_gen(vldlg, num_cached=50):
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
    savename = filename.replace('.npz','')
    savename = savename + err_type + '.npz'
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

def regularization(prediction, alpha):
    '''Return the bridge regularizer'''
    return -T.sum((alpha-1)*T.log(prediction), axis=1)

# ############################## Data handling ################################
def get_traindata(srcfile, synsets=None):
    '''Get the training data'''
    addresses = []; labels = []
    with open(srcfile, 'r') as fp:
        lines = fp.readlines()
    pairs = get_synsets(synsets)
    for line in lines:
        address = line.rstrip('\n')
        addresses.append(address)
        if pairs is not None:
            label = pairs[os.path.basename(address).split('_')[0]]
            labels.append(label)
    return (addresses, labels)

def get_valdata(srcfile):
    '''Get the validation data'''
    addresses = []; labels = []
    with open(srcfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        address, label = line.rstrip('\n').split(' ')
        label = np.int_(label)
        addresses.append(address)
        labels.append(label)
    return (addresses, labels)

def data_and_label_generator(addresses, labels, im_shape, mb_size, shuffle=False):
    '''Get images and pair up with logits'''
    order = ordering(len(addresses), shuffle=shuffle)
    batches = np.array_split(order, np.ceil(len(addresses)/(1.*mb_size)))
    for batch in batches:
        images = []; targets = []
        for idx in batch:
            # Load image
            line = addresses[idx].rstrip('\n')
            #image = cv2.resize(caffe_load_image(line), im_shape)
            #image = preprocess(image, 1, preproc=False)
            image = np.load(line)
            images.append(image)
            targets.append(labels[idx])
        im = np.dstack(images)
        im = np.transpose(im, (2,1,0)).reshape(-1,3,im_shape[0],im_shape[1])
        output = np.hstack(targets).astype(np.int32)
        # Really need to add some kind of preprocessing
        yield (im, output)

def load_image(address, im_shape, preproc=False):
    '''Return image in appropriate format'''
    image = cv2.resize(caffe_load_image(address), im_shape)
    return preprocess(image, 1, preproc=preproc)

def load_target(base, logit_folder, temp, pairs):
    '''Return the target in appropriate format''' 
    logit_address = logit_folder + '/' + base
    logits = np.load(logit_address)['arr_0']
    soft_target = np.mean(softerMax(logits, temp), axis=0)
    if pairs is not None:
        hard_target_idx = pairs[base.split('_')[0]]
        soft_target[hard_target_idx] += 1
        soft_target = soft_target/2.
    return soft_target[np.newaxis,:]

def ordering(num, shuffle=False):
    '''Return a possible random ordering'''
    order = np.arange(num)
    if shuffle:
        np.random.shuffle(order)
    return order

def get_synsets(synsets):
    '''Return a dictionary with the synset mappings'''
    pairs = None
    if synsets is not None:
        pairs = {}
        with open(synsets, 'r') as sp:
            syns = sp.readlines()
        for i, syn in enumerate(syns):
            pairs[syn.rstrip('\n')] = i 
    return pairs

def caffe_load_image(filename, color=True):
    '''Load an image converting from grayscale or alpha as needed.'''
    im = skimage.img_as_float(skio.imread(filename)).astype(theano.config.floatX)
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        if color:
            im = np.tile(im, (1, 1, 3))
    elif im.shape[2] == 4:
        im = im[:, :, :3]
    return im

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
    if preproc == True:
        img = []
        for i in np.arange(num_samples):
        # Random rotations
            angle = np.random.rand() * 360.
            M = cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2), angle, 1)
            img.append(cv2.warpAffine(im, M, (im.shape[1],im.shape[0])))
            # Random fliplr
            if np.random.rand() > 0.5:
                img[i] = img[i][:,::-1,...]
    else:
        img = (im,)*num_samples
    return np.dstack(img)


if __name__ == '__main__':
    #data_root = '/home/dworrall/Data/'
    data_root = '/home/daniel/Data/'
    alpha = 0.9
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    if len(sys.argv) > 2:
        alpha_txt = sys.argv[2]
        alpha = float(sys.argv[2])
    main(data_root + 'ImageNetTxt/transfer.txt',
         data_root + 'ImageNetTxt/val50.txt',
         data_root + 'Experiments/alpha/a'+alpha_txt+'.npz',
         num_epochs=50, margin=25, base=0.01, mb_size=50, momentum=0.9,
         alpha=alpha, synsets=data_root + 'ImageNetTxt/synsets.txt')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
