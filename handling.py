'''Data Handling'''

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
import nonlinearities as nl
import numpy as np
import skimage
import skimage.io as skio
import theano
import theano.tensor as T

import lasagne

def get_metadata(srcfile):
    '''Get all the addresses in the file'''
    with open(srcfile, 'r') as fp:
        lines = fp.readlines()
        num_lines = len(lines)
    return (lines, num_lines)

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

def data_and_label_generator(addresses, labels, im_shape, mb_size):
    '''Get images and pair up with logits'''
    order = ordering(len(addresses), shuffle=False)
    batches = np.array_split(order, np.ceil(len(addresses)/(1.*mb_size)))
    for batch in batches:
        images = []; targets = []
        for idx in batch:
            # Load image
            line = addresses[idx].rstrip('\n')
            #image = cv2.resize(caffe_load_image(line), im_shape)
            image = np.load(line).astype(theano.config.floatX)
            image = preprocess(image, 1, preproc=False)
            images.append(image)
            targets.append(labels[idx])
        im = np.dstack(images)
        im = np.transpose(im, (2,1,0)).reshape(-1,3,im_shape[0],im_shape[1])
        output = np.hstack(targets).astype(np.int32)
        yield (im, output)

def data_logit_label_generator(addresses, logit_folder, im_shape, mb_size,
                               k=1, preproc=False, shuffle=True, synsets=None):
    '''Get images and pair up with logits'''
    pairs = get_synsets(synsets)
    order = ordering(len(addresses), shuffle) 
    batches = np.array_split(order, np.ceil(len(addresses)/(1.*mb_size)))
    for batch in batches:
        images = []; soft = []; hard = []; temp = [] 
        for idx in batch:
            # Load image
            line = addresses[idx].rstrip('\n')
            images.append(load_image(line, im_shape, preproc))
            # Load logits
            base = os.path.basename(line).replace('.npy','.npz')
            target, t = load_target(base, logit_folder, k)
            soft.append(target)
            hard.append(pairs[base.split('_')[0]])
            temp.append(t)
        im = np.dstack(images)
        im = np.transpose(im, (2,1,0)).reshape(-1,3,im_shape[0],im_shape[1])
        soft_targets = np.vstack(soft).astype(np.float32)
        hard_targets = np.hstack(hard).astype(np.int32)
        temp = np.hstack(temp).astype(np.float32)
        yield (im, soft_targets, hard_targets, temp)

def load_image(address, im_shape, preproc=False):
    '''Return image in appropriate format'''
    #image = cv2.resize(caffe_load_image(address), im_shape)
    image = np.load(address).astype(theano.config.floatX)
    return preprocess(image, 1, preproc=preproc)

"""
def load_target(base, logit_folder, k):
    '''Return the target in appropriate format''' 
    logit_address = logit_folder + '/' + base
    data = np.load(logit_address)
    logits, t = data['logits'], data['T']
    soft_target = nl.softMax(logits)
    return (soft_target, t)
"""
def load_target(base, logit_folder, k):
    '''Return the target in appropriate format'''
    # Load logits
    logit_address = logit_folder + '/' + base
    data = np.load(logit_address)
    logits = data['arr_0']
    # Normalize the logits
    R = np.maximum(np.amax(logits, axis=1) - np.amin(logits, axis=1), 0.1)
    t = R[:,np.newaxis]/k
    soft_target = nl.softMax(logits/t)
    return (soft_target, t)

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

def preprocess(im, num_samples, preproc=True):
    '''Data normalizations and augmentations'''
    if preproc == True:
        img = []
        for i in np.arange(num_samples):
        # NEED TO IMPLEMENT RANDOM CROPS!!!
        # Random rotations
            angle = (np.random.rand()-0.5) * 40.
            M = cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2), angle, 1)
            img.append(cv2.warpAffine(im, M, (im.shape[1],im.shape[0])))
            # Random fliplr
            if np.random.rand() > 0.5:
                img[i] = img[i][:,::-1,...]
    else:
        img = (im,)*num_samples
    return np.dstack(img)