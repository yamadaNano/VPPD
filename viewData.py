'''View the data for the compression tests'''

import os, sys, time
import numpy

from matplotlib import pyplot as plt

dataDark = numpy.load('./models/accumDarksub.npz')
dataVPPD = numpy.load('./models/accumVPPDsub.npz')
dataRaw = numpy.load('./models/accumRawsub.npz')

accumDark = dataDark['accum']
sizeDark = dataDark['size']
accumVPPD = dataVPPD['accum']
sizeVPPD = dataVPPD['size']
accumRaw = dataRaw['accum']
sizeRaw = dataRaw['size']

# Sort indicies
idxDark = numpy.argsort(sizeDark)
sizeDark = sizeDark[idxDark]
accumDark = accumDark[idxDark]
idxVPPD = numpy.argsort(sizeVPPD)
sizeVPPD = sizeVPPD[idxVPPD]
accumVPPD = accumVPPD[idxVPPD]
idxRaw = numpy.argsort(sizeRaw)
sizeRaw = sizeRaw[idxRaw]
accumRaw = accumRaw[idxRaw]

fig = plt.figure()
pltDark, = plt.plot(sizeDark, accumDark, color='blue')
pltVPPD, = plt.plot(sizeVPPD, accumVPPD, color='red')
pltRaw, = plt.plot(sizeRaw, accumRaw, color='green')
plt.legend([pltDark, pltVPPD, pltRaw],['BDK', 'VPPD', 'Raw'])
plt.xlabel('Hidden layer size', size=16)
plt.ylabel('Test error')
plt.show()