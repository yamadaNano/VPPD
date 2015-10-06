'''Run sanity checks on logits'''

import os, sys, time

import numpy


filename = '/home/daniel/Data/ImageNetTxt/transfer.txt'
logitFolder = '/home/daniel/Data/normedLogits/Logits'
with open(filename, 'r') as fp: