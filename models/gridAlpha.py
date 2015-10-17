'''Execute alpha regularization experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

alphas = [0.95, 0.9, 0.8, 0.75, 0.5, 0.25, 1.05, 1.1, 1.25, 1.5, 2]
for alpha in alphas:
    command = "THEANO_FLAGS='device=gpu1' python scratchTrain.py '/home/daniel/Data/' " + str(alpha)
    os.system(command)