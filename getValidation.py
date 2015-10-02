'''Get a relevant validation set'''

import os, sys, time


pairs = {}
toKeep = []
keepFiles = []
filename = '/home/daniel/Data/ImageNetTxt/transfercategories.txt'
synsets = '/home/daniel/Data/ImageNetTxt/synsets.txt'
valtxt = '/home/daniel/Data/ImageNetTxt/val.txt'
newvaltxt = '/home/daniel/Data/ImageNetTxt/val50.txt'
val_folder = '/home/daniel/Data/ImageNet/val'
with open(filename, 'r') as fp:
    lines = fp.readlines()
with open(synsets, 'r') as sp:
    syns = sp.readlines()
for i, syn in enumerate(syns):
    pairs[syn.rstrip('\n')] = i
for line in lines:
    syn = os.path.basename(line.rstrip('\n'))
    toKeep.append(pairs[syn])
with open(valtxt, 'r') as vp:
    vals = vp.readlines()
for val in vals:
    val = val.rstrip('\n').split(' ')
    if int(val[1]) in toKeep:
        keepFiles.append(val_folder + '/' + val[0] + ' ' + val[1] + '\n')
with open(newvaltxt, 'w') as np:
    np.writelines(keepFiles)