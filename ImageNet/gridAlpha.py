'''Execute alpha regularization experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

alphas = [-1e-1, -1e-2, -1e-3, -1e-4, 1e-1, 1e-2, 1e-3, 1e-4]
for alpha in alphas:
    command = "THEANO_FLAGS='device=gpu1' python scratchTrain50.py '/home/daniel/Data/' " + str(alpha) + " 1e-3 alpha-3"
    os.system(command)
