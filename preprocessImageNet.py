'''Preprocess ImageNet'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import caffe
import cv2
import numpy
import skimage.io as skio

from matplotlib import pyplot as plt


def main():
    im_shape = (227, 227)
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    meanfile = '/home/daniel/Data/ImageNetMoments/mean2.npy'
    varfile = '/home/daniel/Data/ImageNetMoments/var2.npy'
    addresses = getAddresses(filename)
    mean = getMean(addresses, im_shape)
    numpy.save(meanfile, mean)
    mean = numpy.load(meanfile)
    var = getVariance(addresses, im_shape, mean)
    numpy.save(varfile, var)

def meanImages():
    im_shape = (227, 227)
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    meanfile = '/home/daniel/Data/ImageNetMoments/mean.npy'
    savefolder = '/home/daniel/Data/ImageNet/pp_train'
    mean = numpy.load(meanfile)
    addresses = getAddresses(filename)
    for address in addresses:
        # Save name
        base = os.path.basename(address)
        folder = base.split('_')[0]
        directory = savefolder + '/' + folder
        if not os.path.exists(directory):
            os.makedirs(directory)
        savename = directory + '/' + base.replace('.JPEG','.npy')
        # Preprocess image
        im = caffe.io.load_image(address)
        im = cv2.resize(im, im_shape)
        im = im - mean
        numpy.save(savename, im)
        print savename

def getAddresses(filename):
    '''Return addresses'''
    print('Loading addresses')
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for i, line in enumerate(lines):
        lines[i] = line.rstrip('\n')
    return lines

def getMean(addresses, im_shape):
    '''Compute the mean of the images'''
    mean = 0
    print('Computing mean')
    for i, address in enumerate(addresses):
        im = caffe.io.load_image(address)
        im = cv2.resize(im, im_shape)
        mean = iMean(i, im, mean)
        sys.stdout.flush()
        sys.stdout.write('%i \r' % (i,))
    return mean

def getVariance(addresses, im_shape, mean):
    '''Compute the variance of the images'''
    var = 0
    print('Computing variance')
    for i, address in enumerate(addresses):
        im = caffe.io.load_image(address)
        im = cv2.resize(im, im_shape)
        var = iVar(i, im, mean, var)
        sys.stdout.flush()
        sys.stdout.write('%i \r' % (i,))
    return var
        
def iMean(i, X, mean):
    '''Incremental mean'''
    # i batch num, X data, mean previous mean
    return (i*mean + X)/(i+1.)

def iVar(i, X, mean, var):
    '''Incremental covariance'''
    V = (X - mean)**2
    return (i*var + V)/(i+1.)
        

if __name__ == '__main__':
    #main()
    meanImages()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    