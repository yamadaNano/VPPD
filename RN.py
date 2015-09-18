'''Range normalization example'''

import numpy
import seaborn as sns

from matplotlib import pyplot as plt

def softmax(x):
    return numpy.exp(x)/sum(numpy.exp(x))

def softmaxRN(x):
    x = x/(numpy.max(x)-numpy.min(x))
    k = 1.85*numpy.log10(x.shape[0] + 4)
    return softmax(x*k)

a = 5
x = a*numpy.random.randn(5)
ys = softmax(x)
yRN = softmaxRN(x)

fig = plt.figure()
plt.bar([1,2,3,4,5], ys, color='blue', width=0.4)
plt.bar([1.4,2.4,3.4,4.4,5.4], yRN, color='red', width=0.4)
plt.show()