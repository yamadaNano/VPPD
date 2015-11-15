'''Run sanity checks on logits'''

import os, sys, time

import numpy


def main():
    maxim = 0.
    minim = 10.
    filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
    logitFolder = '/home/daniel/Data/normedLogits/Logits'
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        base = os.path.basename(line.replace('.JPEG\n','.npz'))
        address = logitFolder + '/' + base
        logits = numpy.load(address)['logits']
        #temp = numpy.load(address)['T']
        #maxim = numpy.maximum(temp, maxim)
        #minim = numpy.minimum(temp, minim)
        R = numpy.amax(logits) - numpy.amin(logits)
        sys.stdout.flush()
        sys.stdout.write('%f \r' % (R,))


if __name__ == '__main__':
    main()