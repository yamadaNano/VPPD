'''Functions for drawing contours of Dirichlet distributions.'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri

_corners = numpy.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 \
              for i in range(3)]

def xy2bc(xy, tol=1.e-7):
    '''Converts 2D Cartesian coordinates to barycentric'''
    s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 \
         for i in range(3)]
    return numpy.clip(s, tol, 1.0 - tol)

def softmax(x):
    '''Return the softmax output'''
    e_x = numpy.exp(x - numpy.max(x, axis=1)[:,numpy.newaxis])
    return e_x/numpy.sum(e_x, axis=1)[:,numpy.newaxis]

def RN(x, k=1):
    '''Return the range normalised points'''
    v = x/(numpy.max(x, axis=1) - numpy.min(x, axis=1))[:,numpy.newaxis]
    return softmax(k*v)

def plot_points(X, barycentric=True, border=True, color='k.', **kwargs):
    '''Plots a set of points in the simplex'''
    if barycentric is True:
        X = X.dot(_corners)
    plt.plot(X[:, 0], X[:, 1], color, ms=2, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.hold(1)
        plt.triplot(_triangle, linewidth=1)

def samplePoints(n=100):
    '''Generate points in the 3-simplex'''
    return numpy.random.randn(n,3)

def getRN():
    Z = samplePoints(n=1000)
    #T = numpy.logspace(numpy.log(0.5),numpy.log(100), 1000)
    Y = []
    for k in numpy.arange(10):
        Y.append(RN(Z, k/2.))
    return numpy.vstack(Y)

def getLocus(z=numpy.random.randn(20,3) , m=1000):
    #z = numpy.asarray([0, 1, 1.3])[numpy.newaxis,:]
    T = numpy.logspace(numpy.log(1),numpy.log(10), m)
    Z = []
    for t in T:
        Z.append(z/t)
    Z = numpy.vstack(Z)
    return softmax(Z)

def softTarget(l, m=1000):
    S = getLocus(numpy.random.randn(1,3), m=m)
    Y = numpy.zeros((3,))
    Y[numpy.argmax(S[0,:])] = 1
    T = numpy.logspace(numpy.log(1),numpy.log(10), m)
    c = []
    A = []
    for t in T:
        a = t/(t+l)
        b = l/(t+l)
        c.append(a*S[t,:] + b*Y)
        print a, S[t,:], Y, a*S[t,:] + b*Y
    return (numpy.vstack(c),S)

def convexCombinations():
    z = numpy.random.randn(2,3)
    s = softmax(z)
    l = numpy.linspace(0,1,100)
    L = numpy.vstack((l,1-l)).T
    return numpy.dot(L,s)

def randomErrors(n=10, sigma=0.1):
    '''Return Gaussian logits errors'''
    z = numpy.random.randn(1,3)
    z_e = z + sigma*numpy.random.randn(n,3)
    y = softmax(numpy.vstack((z,z_e)))
    return y

def circularErrors(n=10, r=0.1):
    '''Return circular logits errors'''
    z = numpy.random.randn(1,3)
    y = softmax(z)
    theta = numpy.linspace(0,2*numpy.pi, n)
    x_e = numpy.cos(theta)
    y_e = numpy.sin(theta)
    #e_simplex = numpy.random.randn(n,3)
    #bary_errors = r*softmax(e_simplex)
    errors = r*numpy.vstack((x_e, y_e))
    bary_errors = [xy2bc(errors[:,i]) for i in numpy.arange(errors.shape[1])]
    bary_errors = numpy.vstack(bary_errors)
    #errors = softmax(bary_errors + z)
    return y + bary_errors #errors

if __name__ == '__main__':
    '''
    Y = getRN()
    Y2 = getLocus()
    Y3 = circularErrors(n=100, r=0.1)
    #Y3 = randomErrors(sigma=0.1)
    Y3 = getLocus(Y3, m=20)
    Y4 = convexCombinations()
    Y5, Y6 = softTarget(0.1, m=1000)
    f = plt.figure(figsize=(12, 9))
    plot_points(Y, color='k.')
    plot_points(Y2, color='g.')
    plot_points(Y3, color='r.')
    plot_points(Y4, color='b.')
    plot_points(Y6, color='y.')
    plot_points(Y5, color='w.')
    plt.show()
    '''
    Y = numpy.load('/home/daniel/Data/mnist/outputs/train.npy')
    for t in numpy.arange(50, step=1):
        L = 10.
        a = t/(t+L)
        b = L/(t+L)
        h = numpy.zeros(Y.shape)
        maxes = numpy.argmax(Y, axis=1)
        maxes = numpy.ravel_multi_index((numpy.arange(maxes.shape[0]), maxes), (maxes.shape[0], 3))
        numpy.put(h, maxes, 1)
        Ys = a*softmax(numpy.log(Y)/(t+1.)) + b*h
        Yt = softmax(numpy.log(Y)/(t+1.))
        plot_points(Yt, color='r.')
        plot_points(Ys, color='b.')
        plt.ion()
        print t+1.
        raw_input('ENTER')
        plt.clf()
        #plot_points(Y2, color='g.')
        #plot_points(Y3, color='r.')
        plt.show()
        
















