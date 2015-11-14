import numpy
from matplotlib import pyplot as plt

filename = '/home/daniel/Data/Experiments/trainRetrain/n01797886_32180.npy'

def soften(x, temp):
    e_x = x**(1./temp)
    return e_x/numpy.sum(e_x)

data = numpy.load(filename)
fig = plt.figure()
plt.bar(numpy.arange(1000), soften(data.T, 10.))
plt.show()