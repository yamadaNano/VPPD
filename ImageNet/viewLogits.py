'''View logits'''

import os, sys, time

import numpy
import seaborn as sns

from matplotlib import pyplot as plt

def main():
    color = ['green','blue','red']
    k = 3
    r = numpy.random.rand(k)
    y = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in numpy.arange(3):
        x = (3*i+1)*(r/(numpy.amax(r)-numpy.amin(r)))
        y = softmax(x)
        plt.bar(numpy.arange(k)/(k+1.)+i+0.1, y, width=0.25, color=color[i])
    ax.set_xticklabels(['','range = 1','','range = 4','','range = 7'])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.ylabel('Probability mass', fontsize=16)
    plt.title('The Effect of Range on Softmax Sharpness', fontsize=16)
    plt.show()

def softmax(x):
    '''Return the softmax function'''
    return numpy.exp(x)/numpy.sum(numpy.exp(x))


if __name__ == '__main__':
    main()