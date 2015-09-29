'''Logit analyses'''

import os, sys, time

import matplotlib
import numpy
import seaborn as sns

from matplotlib import pyplot as plt


def main(srcfile, logitfolder):
    lines, num_lines = get_metadata(srcfile)
    logit_range = []
    mean = 0.
    for i, line in enumerate(lines):
        base = os.path.basename(line).rstrip('\n').split('.')[0]
        fname = logitfolder + '/' + base + '.npz'
        logits = numpy.load(fname)['arr_0']
        rn = numpy.amax(logits, axis=1) - numpy.amin(logits, axis=1)
        logit_range.extend(rn)
        if i % 100 == 0:
            sys.stdout.write('%i \r' % (i,))
            sys.stdout.flush()
    hist, bin_edges = numpy.histogram(logit_range, bins=100, density=True)
    plotHist(bin_edges, hist)

def ipca(srcfile, logitfolder, dstfolder):
    lines, num_lines = get_metadata(srcfile)
    mean = 0.
    for i, line in enumerate(lines):
        base = os.path.basename(line).rstrip('\n').split('.')[0]
        fname = logitfolder + '/' + base + '.npz'
        logits = numpy.load(fname)['arr_0']
        mean = iMean(i, zeroShift(logits), mean)
        sys.stdout.write('%i \r' % (i,))
        sys.stdout.flush()
    numpy.save(dstfolder + '/MeanLogitsZSMean.npy', mean)
    lines, num_lines = get_metadata(srcfile)
    cov = numpy.zeros((1000,1000))
    for i, line in enumerate(lines):
        base = os.path.basename(line).rstrip('\n').split('.')[0]
        fname = logitfolder + '/' + base + '.npz'
        logits = numpy.load(fname)['arr_0'] - mean
        cov = iCov(i, zeroShift(logits).T, cov)
        sys.stdout.write('%i \r' % (i,))
        sys.stdout.flush()
    numpy.save(dstfolder + '/MeanLogitsZSCov.npy', cov)
    _, s, _ = numpy.linalg.svd(cov)
    fig = plt.figure()
    plt.plot(s)
    plt.show()

def get_metadata(srcfile):
    '''Get all the addresses in the file'''
    with open(srcfile, 'r') as fp:
        lines = fp.readlines()
        num_lines = len(lines)
    return lines, num_lines

def plotHist(bin_edges, hist):
    '''Plot a histogram'''
    fig = plt.figure()
    plt.plot(bin_edges[:-1], hist)
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.xlabel('Logit range', fontsize=16)
    plt.ylabel('Probability density', fontsize=16)
    plt.title('Histogram of logit ranges on 50 random CaffeNet classes', fontsize=16)
    plt.show()

def iMean(i, X, mean):
    '''Incremental mean'''
    # i batch num, X data, mean previous mean
    return (i*mean + numpy.mean(X, axis=0))/(i+1.)

def iCov(i, X, cov):
    '''Incremental covariance'''
    return (i*cov + numpy.cov(X))/(i+1.)

def plotSvd(srcfile):
    '''Plot the singular value spectrum of the covariance matrix in srcfile'''
    C = numpy.load(srcfile)
    #_, s, _ = numpy.linalg.svd(C)
    #s = numpy.sort(numpy.diag(C))[::-1]
    fig = plt.figure()
    #plt.imshow(C)
    plt.imshow(orderedCov(C))
    #plt.plot(s)
    plt.show()

def plotMean(srcfile):
    '''Plot the mean outputs'''
    m = numpy.load(srcfile)
    fig = plt.figure()
    plt.plot(numpy.sort(m))
    plt.show()

def softmax(x):
    '''Plot the softmax of the input ROW-vector'''
    return numpy.exp(x)/numpy.sum(numpy.exp(x), axis=1)[:,numpy.newaxis]

def zeroShift(x):
    '''Align the minimum element of x to zero'''
    return x - numpy.amin(x, axis=1)[:,numpy.newaxis]

def orderedCov(C):
    '''Order the covariance matrix according to diagonal'''
    s = numpy.diag(C)
    idx = numpy.argsort(s)[::-1]
    P = numpy.zeros((C.shape[0],C.shape[0]))
    P[idx,numpy.arange(C.shape[0])] = 1
    return numpy.dot(numpy.dot(P.T,C),P)
    

if __name__ == '__main__':
    srcfile = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    logitfolder = '/home/daniel/Data/MeanLogits'
    logitsCov = '/home/daniel/Data/MeanLogitsZSCov.npy'
    logitsMean = '/home/daniel/Data/MeanLogitsZSMean.npy'
    dstfolder = '/home/daniel/Data'
    main(srcfile, logitfolder)
    #ipca(srcfile, logitfolder, dstfolder)
    #plotSvd(logitsCov)
    #plotMean(logitsMean)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
