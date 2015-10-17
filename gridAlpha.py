'''Execute alpha regularization experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

alphas = [0.9, 0.99, 0.999, 0.9999, 0.99999, 1.1, 1.01, 1.001, 1.0001, 1.00001]
for alpha in alphas:
    command = "THEANO_FLAGS='device=gpu1' python scratchTrain.py '/home/daniel/Data/' " + str(alpha)
    os.system(command)