'''Copy to SSD'''

import os, sys, time
from distutils.dir_util import copy_tree

filename = '/home/daniel/Data/ImageNetTxt/transfercategories.txt'
toDir = '/home/daniel/Data/ImageNet'
with open(filename, 'r') as fp:
    lines = fp.readlines()
for line in lines:
    fromDir = line.rstrip('\n')
    dstDir = toDir + '/' + os.path.basename(fromDir)
    print(dstDir)
    copy_tree(fromDir, dstDir)