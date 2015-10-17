'''Execute alpha regularization experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

alphas = [0.99999, 0.999995, 0.9999975, 1.0000025, 1.000005, 1.00001]
for alpha in alphas:
    command = "python scratchTrain.py '/home/dworrall/Data/' " + str(alpha)
    os.system(command)
