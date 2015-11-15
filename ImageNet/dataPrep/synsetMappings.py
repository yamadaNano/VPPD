'''Val synset mappings'''

import os, sys, time

import numpy


filename = '/home/daniel/Data/ImageNetTxt/val50.txt'
synmap = '/home/daniel/Data/ImageNetTxt/synmap.txt'
with open(filename, 'r') as fp:
    lines = fp.readlines()
nums = []
for line in lines:
    num = line.rstrip('\n').split(' ')[-1]
    if int(num) not in nums:
        nums.append(int(num))
nums = numpy.asarray(nums)
nums = numpy.sort(nums)
with open(synmap, 'w') as sp:
    for i in numpy.arange(nums.shape[0]):
        sp.write('%i %i\n' % (i,nums[i]))