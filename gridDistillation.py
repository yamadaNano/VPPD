'''Execute distillation experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

temps = [1, 2, 5, 10, 20]
for temp in temps:
    command = "THEANO_FLAGS='device=gpu1' python distillation.py '/home/daniel/Data/' " + str(temp) + " VPPDtemp 1e-3"
    os.system(command)