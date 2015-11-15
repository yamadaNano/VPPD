'''Get spread for TNS'''

import numpy

from matplotlib import pyplot as plt

D = 10000
k = numpy.linspace(0.1,10,100)
maxk = []

for d in numpy.arange(1,D):
    Z = (k**2)*((d/(numpy.exp(k) + d))**2 + d*(1/(d*numpy.exp(k) + 1))**2)
    maxk.append(k[numpy.argmax(Z)])

maxk = numpy.asarray(maxk)
print numpy.amax(maxk), numpy.amin(maxk)
d = numpy.arange(1,D)
fig = plt.figure()
plt.semilogx(d, maxk)
plt.semilogx(d, 1.85*numpy.log10(d+4))
plt.xlabel('$\log d$', size=18)
plt.ylabel('$k_{max}$', size=18)
plt.xlim(0,1e4)
plt.show()
