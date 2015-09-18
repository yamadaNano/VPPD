'''View the data for the compression tests'''

import os, sys, time
import numpy

from matplotlib import pyplot as plt

dataDark = numpy.load('./models/accumDark800.npz')
dataVPPD = numpy.load('./models/accumVPPD800.npz')

accumDark = dataDark['accum']
sizeDark = dataDark['size']
accumVPPD = dataVPPD['accum']
sizeVPPD = dataVPPD['size']

fig = plt.figure()
pltDark, = plt.plot(sizeDark, accumDark, color='blue')
pltVPPD, = plt.plot(sizeVPPD, accumVPPD, color='red')
plt.legend([pltDark, pltVPPD],['BDK', 'VPPD'])
plt.xlabel('Hidden layer size', size=16)
plt.ylabel('Test error')
plt.show()