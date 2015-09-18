'''Run toy VPPD'''

import numpy
import theano
import theano.tensor as T

import lasagne

from matplotlib import pyplot as plt

# ############################# Toy plotting ##################################

def reloadToy(filename, input_var=None):
    # Reload the model for the toy dataset
    params = numpy.load(filename)
   
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 2),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=100, W=params['l_hid1.W'],
            b=params['l_hid1.b'], name='l_hid1',
            nonlinearity=lasagne.nonlinearities.tanh)
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=2, W=params['l_out.W'],
            b=params['l_out.b'], name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def runModel(filename, bounds, num, data_file):
    print('Reloading model')
    input_var = T.tensor4('input_var')
    model = reloadToy(filename, input_var)
    prediction = lasagne.layers.get_output(model)
    print('Building expression graph')
    pred_fn = theano.function([input_var], prediction)
    print('Generating evaluation points')
    x_space = numpy.linspace(bounds[0], bounds[1], num)
    y_space = numpy.linspace(bounds[0], bounds[1], num)
    pred_space = numpy.meshgrid(x_space, y_space)
    pred = numpy.zeros((num**2,))
    
    for i in numpy.arange(num**2):
        input = numpy.asarray([x_space[i%num],y_space[i/num]])
        input = input.reshape((-1,1,1,2)).astype(theano.config.floatX)
        pred[i] = pred_fn(input)[0,0]
    pred = pred.reshape((num,num))
    plotResults(data_file, pred, x_space, y_space)
    

def plotResults(data_file, results, x_space, y_space):
    data = numpy.load(data_file)
    x_points = data['X_train']
    # Set up a regular grid of interpolation points
    x, y = numpy.meshgrid(x_space, y_space)
    #plt.contourf(x, y, results, vmin=results.min(), vmax=results.max(),
    #             origin='lower', interpolation='Gaussian')
    plt.imshow(results, origin='lower', interpolation='Gaussian',
               extent = [bounds[0],bounds[1],bounds[0],bounds[1]], aspect='auto')
    plt.scatter(x_points[0,:10], x_points[1,:10], color='white')
    plt.scatter(x_points[0,10:], x_points[1,10:], color='black')
    
    plt.show()


if __name__ == '__main__':
    bounds = [-10, 10]
    num = 200
    filename = './models/studenttoy.npz'
    data_file = './models/vppdtoy.npz'
    runModel(filename, bounds, num, data_file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    