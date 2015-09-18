'''VPPDGMM example'''

import numpy

from matplotlib import pyplot as plt

def rotate(theta):
    '''Return rotation matrix'''
    return numpy.asarray([[numpy.cos(theta),-numpy.sin(theta)],
                          [numpy.sin(theta), numpy.cos(theta)]])


# Num points
N = 20
# Set up original model
m1 = numpy.asarray([[-2],[-2]])
m2 = numpy.asarray([[2],[2]])
S1 = numpy.asarray([[1, 0],[0,1]])
S2 = numpy.asarray([[1, 0],[0,1]])

p = 0.5
# Generate data
e1 = numpy.random.randn(2,int(N*p))
e2 = numpy.random.randn(2,int(N*(1-p)))
x1 = m1 + numpy.dot(S1,e1)
x2 = m2 + numpy.dot(S2,e2)
# Plot
fig = plt.figure()
plt.scatter(x1[0,:], x1[1,:])
plt.scatter(x2[0,:], x2[1,:],color='red')
plt.show()

filename = './vppdtoy.npz'
X_train = numpy.hstack((x1,x2))
y_train = numpy.hstack((numpy.ones(x1[0.,:].shape),
                        numpy.zeros(x2[0.,:].shape)))
numpy.savez(filename, X_train=X_train, y_train=y_train)






































