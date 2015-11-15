'''Execute distillation experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

K = [5, 10, 25, 50, 100, 250, 500]
for k in K:
    command = "python distillation.py '/home/daniel/Data/' 10 crossentropy 0.01 " + str(k)
    os.system(command)