import numpy
import theano
filename = '/home/daniel/Data/Experiments/alpha-3b/model0.npz'
data = numpy.load(filename)
for d in data:
    print d, data[d].shape