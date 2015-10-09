'''Execute distillation experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

k = [0.5, 1, 2, 5, 10]
for temp in k:
    command = "python approximateModel.py '/home/dworrall/Data/' " + str(temp)
    os.system(command)