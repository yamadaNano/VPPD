'''Execute distillation experiments'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

temps = [1, 2, 5, 10, 20]
for temp in temps:
    command = "python distillation.py '/home/dworrall/Data/' " + str(temps)
    os.system(command)