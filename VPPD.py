'''BNN'''

__authors__   = "Daniel Worrall"
__copyright__ = "(c) 2015, University College London"
__license__   = "3-clause BSD License"
__contact__   = "d.worrall@cs.ucl.ac.uk"

import os, sys, time

import cPickle
import gzip
import lasagne
import numpy as np
import pickle
import theano
import theano.tensor as T

from lasagne import utils
from collections import OrderedDict
from matplotlib import pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams

# ################## Download and prepare appropriate dataset ##################
def load_dataset(dataset='MNIST'):
    if dataset == 'MNIST':
        # We'll now download the MNIST dataset if it is not yet available.
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            print("Downloading MNIST dataset...")
            urlretrieve(url, filename)
        # We'll then load and unpickle the file.
        def pickle_load(f, encoding):
            return pickle.load(f)
        with gzip.open(filename, 'rb') as f:
            data = pickle_load(f, encoding='latin-1')
        # Unpack data
        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]
        # Reshape to standard 4-array
        X_train = X_train.reshape((-1, 1, 28, 28))
        X_val = X_val.reshape((-1, 1, 28, 28))
        X_test = X_test.reshape((-1, 1, 28, 28))
        # The targets are int64, we cast them to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_val = y_val.astype(np.uint8)
        y_test = y_test.astype(np.uint8)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'CIFAR10':
        print('Loading CIFAR 10')
        file = '/media/daniel/DATA/Cifar/cifar-10-batches-py/data_batch_'
        data = []
        labels = []
        for i in ['1','2','3','4']:
            data_dict = unpickle(file+i)
            data.append(data_dict['data'])
            labels.append(np.asarray(data_dict['labels']))
        X_train = np.vstack(data[:3])
        y_train = np.hstack(labels[:3])
        X_val = data[-1]
        y_val = labels[-1]
        data_dict = unpickle('/media/daniel/DATA/Cifar/cifar-10-batches-py/test_batch')
        X_test = np.asarray(data_dict['data'])
        y_test = np.asarray(data_dict['labels'])
        # Need to rescale to [0,1]
        X_train = X_train.reshape((-1, 3, 32, 32))/255.
        X_val = X_val.reshape((-1, 3, 32, 32))/255.
        X_test = X_test.reshape((-1, 3, 32, 32))/255.
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        # The targets are int64, we cast them to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_val = y_val.astype(np.uint8)
        y_test = y_test.astype(np.uint8)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'TOY':
        print('Loading TOY')
        filename = '/home/daniel/Code/VPPD/models/vppdtoy.npz'
        data = np.load(filename)
        X_train = data['X_train'].T.reshape((-1, 1, 1, 2))
        X_train = X_train.astype(theano.config.floatX)
        y_train = data['y_train'].astype(np.uint8)
        return X_train, y_train, None, None, None, None

        
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# ##################### Build the neural network model #######################
# The various NN models for VPPD

def build_mlp(input_var=None, temp=-1, num_hid=800):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_in, 0.2), num_units=num_hid,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid1, 0.5), num_units=num_hid,
            W=lasagne.init.GlorotUniform(), name='l_hid2')
    l_out = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(l_hid2, 0.5), num_units=10,
            W=lasagne.init.GlorotUniform(), name='l_out',
            nonlinearity=lasagne.nonlinearities.linear)
    l_soft = SoftermaxNonlinearity(l_out, temp=temp)
    return l_soft

def build_cnn(input_var=None, temp=-1, num_hid=150):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_conv1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(3, 3), stride=2)
    l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1, num_filters=64, filter_size=(3, 3),
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.rectify)
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(3, 3), stride=2)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_pool2, 0.5), num_units=num_hid,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_out = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(l_hid1, 0.5), num_units=10,
            W=lasagne.init.GlorotUniform(), name='l_out',
            nonlinearity=lasagne.nonlinearities.linear)
    l_soft = SoftermaxNonlinearity(l_out, temp=temp)
    return l_soft

def build_toy(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 2),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=10,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=2,
            W=lasagne.init.GlorotUniform(), name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def build_pred(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 2),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=100,
            W=lasagne.init.GlorotUniform(), name='l_hid1')
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=2,
            W=lasagne.init.GlorotUniform(), name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def reloadToy(filename, input_var=None):
    # Reload the model for the toy dataset
    params = np.load(filename)
   
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 2),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=10, W=params['l_hid1.W'],
            b=params['l_hid1.b'], name='l_hid1',
            nonlinearity=lasagne.nonlinearities.tanh)
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=2, W=params['l_out.W'],
            b=params['l_out.b'], name='l_out',
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def reload_mlp(filename, input_var=None, temp=-1, num_hid=800):
    params = np.load(filename)
    
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_in, 0.2), num_units=num_hid,
            W=params['l_hid1.W'], b=params['l_hid1.b'], name='l_hid1')
    l_hid2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid1, 0.5), num_units=num_hid,
            W=params['l_hid2.W'], b=params['l_hid2.b'], name='l_hid2')
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hid2, 0.5), num_units=10,
            W=params['l_out.W'], b=params['l_out.b'], name='l_out',
            nonlinearity=lasagne.nonlinearities.linear)
    l_soft = SoftermaxNonlinearity(l_out, temp=temp)
    return l_soft

# ############################ Maximum likelihood #############################

def main(model='toy', num_epochs=100, file_name=None,
         save_name='./models/model.npz', dataset='MNIST', L2Radius=3.87,
         base_tlr=1e-4, base_slr=1e-2, mb_size=500, margin_lr=25, num_hid=800,
         **kwargs):
    # Load the dataset
    print("Loading data...")
    if dataset in ('MNIST','CIFAR10','TOY'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = dataset
    # Theano variables
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learning_rate = T.fscalar('learning_rate')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'toy':
        teacher = build_toy(input_var, num_hid)
    elif model == 'mlp':
        teacher = build_mlp(input_var, num_hid)
    elif model == 'cnn':
        teacher = build_cnn(input_var, num_hid)
    else:
        print('Model not recognised')
        sys.exit(1)
    # Networks
    t_pred = lasagne.layers.get_output(teacher, deterministic=False)
    v_pred = lasagne.layers.get_output(teacher, deterministic=True)
    # Loss functions
    t_loss = lasagne.objectives.categorical_crossentropy(t_pred, target_var)
    t_loss = t_loss.mean()
    t_acc = T.mean(T.eq(T.argmax(t_pred, axis=1), target_var),
                   dtype=theano.config.floatX)
    val_acc = T.mean(T.eq(T.argmax(v_pred, axis=1), target_var),
                     dtype=theano.config.floatX)
    # Learning updates
    t_params = lasagne.layers.get_all_params(teacher, trainable=True)
    t_updates = nesterov_momentum(t_loss, t_params, learning_rate)
    # Compile functions
    t_fn = theano.function([input_var, target_var, learning_rate],
        [t_loss, t_acc], updates=t_updates)
    v_fn = theano.function([input_var, target_var], val_acc)
    # Finally, launch the training loop.
    print("Burning in")
    for epoch in range(num_epochs):
        start_time = time.time()
        learning_rate = get_learning_rate(epoch, margin_lr, base_tlr)
        t_err = 0; t_accu = 0; t_batches = 0
        for batch in iterate_minibatches(X_train, y_train, mb_size, shuffle=True):
            inputs, targets = batch
            # SGD step
            err, acc = t_fn(inputs, targets, learning_rate=learning_rate)
            t_err += err; t_accu += acc; t_batches += 1
            
        if X_val is not None:
            val_acc = 0; val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, mb_size, shuffle=False):
                inputs, targets = batch
                acc = v_fn(inputs, targets)
                val_acc += acc; val_batches += 1
        
        print("Burn {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  train loss:\t\t{:.6f}".format(t_err / t_batches))
        print("  train acc:\t\t{:.6f}".format(t_accu / t_batches))
        if X_val is not None:
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
    
    # After training, we compute and print the test error:
    if X_test is not None:
        test_acc = 0; test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            acc = v_fn(inputs, targets)
            test_acc += acc; test_batches += 1
        print("Final results:")
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    print('Complete')
    save_model(teacher, save_name)
    if X_test is not None:
        return test_acc / test_batches * 100           
    print('Complete')
   
# ################ VARIATIONAL POSTERIOR PREDICTIVE INFERENCE #################
def main2(model='toy', num_epochs=100, file_name=None,
          save_name='./models/model.npz', dataset='MNIST', L2Radius=3.87,
          base_tlr=1e-4, base_slr=1e-2, update_W=True, mb_size=500,
          margin_lr=25, sampler='SGLD', thinning_interval=1, burn_in=20,
          method='VPPD', s_momentum=0.9, num_hid=800, **kwargs):
    # Load the dataset
    print("Loading data...")
    if dataset in ('MNIST','CIFAR10','TOY'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = dataset
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    t_learning_rate = T.fscalar('learning_rate')
    s_learning_rate = T.fscalar('learning_rate')
    temp = T.fscalar('temp')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'toy':
        teacher = reloadToy(file_name, input_var)
        student = build_pred(input_var)
    elif model == 'mlp':
        teacher = reload_mlp(file_name, input_var, temp=temp, num_hid=800)
        student = build_mlp(input_var, temp=temp, num_hid=num_hid)
    else:
        print('Model not recognised')
        sys.exit(1)
    # Hyperparameters
    copy_temp = 2.06
    # Networks
    t_pred = lasagne.layers.get_output(teacher, deterministic=False, training=True)
    t_test = lasagne.layers.get_output(teacher, deterministic=False, training=False)
    s_pred = lasagne.layers.get_output(student, deterministic=True, training=True)
    s_test = lasagne.layers.get_output(student, deterministic=True, training=False)
    t_loss = lasagne.objectives.categorical_crossentropy(t_pred, target_var)
    t_loss = t_loss.mean()
    if method == 'VPPD':
        s_loss = T.mean(s_pred*(T.log(s_pred)-T.log(t_pred)))
    elif method == 'Dark':
        s_loss = -T.mean(t_pred*T.log(s_pred))
    else:
        print('Method not recognised')
        sys.exit(1)
    t_acc = T.mean(T.eq(T.argmax(t_pred, axis=1), target_var),
                   dtype=theano.config.floatX)
    t_tar_acc = T.mean(T.eq(T.argmax(t_test, axis=1), target_var),
                       dtype=theano.config.floatX)
    s_tar_acc = T.mean(T.eq(T.argmax(s_test, axis=1), target_var),
                       dtype=theano.config.floatX)
    # Parameters
    t_params = lasagne.layers.get_all_params(teacher, trainable=True)
    s_params = lasagne.layers.get_all_params(student, trainable=True)
    # MCMC Sampler
    log_prior = log_prior_regularizer(t_params)
    t_updates = set_sampler(sampler,t_loss,t_params,t_learning_rate,log_prior)
    # Learning updates
    s_updates = nesterov_momentum(s_loss, s_params,
                                  learning_rate=s_learning_rate,
                                  momentum=s_momentum)
    # Theano functions
    if update_W == True:
        t_fn = theano.function([input_var, target_var, t_learning_rate, temp],
            [t_loss, t_tar_acc], updates=t_updates)
    else:
        t_fn = theano.function([input_var, target_var, temp], [t_loss, t_acc])
    s_fn = theano.function([input_var, s_learning_rate, temp], updates=s_updates)
    tval_fn = theano.function([input_var, target_var, temp], [t_loss, t_tar_acc])
    val_fn = theano.function([input_var, target_var], s_tar_acc)
    # Finally, launch the training loop.
    if update_W == True:
        print("Burning in")
        for epoch in range(burn_in):
            learning_rate = get_learning_rate(epoch, margin_lr, base_tlr)
            # In each epoch, we do a full pass over the training data:
            start_time = time.time()
            t_err = 0; t_accu = 0; t_batches = 0
            for batch in iterate_minibatches(X_train, y_train, mb_size, shuffle=True):
                inputs, targets = batch
                # Sample weights from teacher
                err, acc = t_fn(inputs, targets, learning_rate, temp=-1)
                t_err += err; t_accu += acc; t_batches += 1
            
            # And a full pass over the validation data:
            if X_val is not None:
                tv_acc = 0; val_batches = 0
                for batch in iterate_minibatches(X_val, y_val, mb_size, shuffle=False):
                    inputs, targets = batch
                    _, vacc = tval_fn(inputs, targets, temp=-1)
                    tv_acc += vacc; val_batches += 1
                
            print("Burn {} of {} took {:.3f}s".format(
                epoch + 1, burn_in, time.time() - start_time))
            print("  train loss:\t\t{:.6f}".format(t_err / t_batches))
            print("  train acc:\t\t{:.6f}".format(t_accu / t_batches * 100))
            if X_val is not None:
                print("  val acc:\t\t{:.6f}".format(tv_acc / val_batches * 100))
        
    # We iterate over epochs:
    print("Knowledge transfer")
    for epoch in range(num_epochs):
        tlearning_rate = get_learning_rate(epoch+burn_in, margin_lr, base_tlr)
        slearning_rate = get_learning_rate(epoch, margin_lr, base_slr)
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        t_err = 0; iterations = 0
        for i in np.arange(thinning_interval):
            t_batches = 0; s_batches = 0
            for batch in iterate_minibatches(X_train, y_train, mb_size, shuffle=True):
                inputs, targets = batch
                # Sample weights from teacher
                if update_W == True:
                    err, _ = t_fn(inputs, targets, learning_rate=tlearning_rate,
                                  temp=copy_temp)
                else:
                    err, _ = t_fn(inputs, targets, temp=copy_temp)
                # Train student
                if (iterations % thinning_interval) == 0:
                    #inputs = np.random.rand(mb_size, 1, 1, 2)*20 - 10
                    s_fn(inputs.astype(theano.config.floatX), slearning_rate,
                         temp=copy_temp)
                    s_batches += 1
                t_err += err/thinning_interval
                t_batches += 1; iterations += 1

        # And a full pass over the validation data:
        if X_val is not None:
            val_acc = 0; val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, mb_size, shuffle=False):
                inputs, targets = batch
                acc = val_fn(inputs, targets)
                val_acc += acc; val_batches += 1
        
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(t_err / t_batches))
        if X_val is not None:
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    if X_test is not None:
        test_acc = 0; test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            acc = val_fn(inputs, targets)
            test_acc += acc; test_batches += 1
        print("Final results:")
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    save_model(student, save_name)
    print('Complete')
    if X_test is not None:
        return test_acc / test_batches * 100
    else:
        return

# ######################### SPECIAL FUNCTIONS/METHODS #########################
    
def get_learning_rate(epoch, margin, base):
    return base*margin/np.maximum(epoch,margin)

def save_model(model, file_name):
    '''Save the model parameters'''
    print('Saving model..')
    params = {}
    for param in lasagne.layers.get_all_params(model):
        params[str(param)] = param.get_value()
    
    file = open(file_name, 'w')
    cPickle.dump(params, file, cPickle.HIGHEST_PROTOCOL)
    file.close()

class SoftermaxNonlinearity(lasagne.layers.Layer):
    def __init__(self, incoming, temp, **kwargs):
        super(SoftermaxNonlinearity, self).__init__(incoming, **kwargs)
        self.temp = temp

    def get_output_for(self, input, training=False, **kwargs):
        if training:
            R = (T.max(input,axis=1)-T.min(input,axis=1)).dimshuffle(0,'x')
            input = self.temp*input/T.maximum(R,0.1)
        return T.exp(input)/T.sum(T.exp(input), axis=1).dimshuffle(0,'x')

def L2BallConstraint(tensor_var, target_norm, norm_axes=None, epsilon=1e-7):
    ndim = tensor_var.ndim
    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )
    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    constrained_output = \
        (tensor_var * (target_norm / (dtype(epsilon) + norms)))

    return constrained_output

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def log_prior_regularizer(t_params):
    '''Return'''
    log_prior = 0.
    for param in t_params:
        if param.name[-1] == 'W':
            log_prior += -1.*T.sum(param**2)
        elif param.name[-1] == 'b':
            log_prior += -1.*T.sum(param**2)
    return log_prior

def set_sampler(sampler, t_loss, t_params, t_learning_rate, log_prior):
    if sampler == 'SGLD':
        t_updates = SGLD(t_loss, t_params, t_learning_rate, log_prior, N=20)
    elif sampler == 'SGHMC':
        t_updates = SGHMC(t_loss, t_params, t_learning_rate, log_prior, N=20,
                          friction=.01)
    elif sampler == None:
        t_updates = None
    else:
        print('Invalid sampler')
        sys.exit(1)
    return t_updates

# ######################### OPTIMIZATION ######################################

def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients"""
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form"""
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

def apply_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including momentum
    Generates update expressions of the form"""
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x
    return updates

def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum
    Generates update expressions of the form"""
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_momentum(updates, momentum=momentum)

def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including Nesterov momentum
    Generates update expressions of the form"""
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]
    return updates

def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum
    Generates update expressions of the form"""
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)

# ###################### Markov Chain Monte Carlo #############################

def SGLD(loss, params, learning_rate, log_prior, N):
    """Apply the SGLD MCMC sampler"""
    g_lik = get_or_compute_grads(-N*loss, params)
    g_prior = get_or_compute_grads(log_prior, params)
    smrg = MRG_RandomStreams()
    updates = OrderedDict()
    for param, gl, gp in zip(params, g_lik, g_prior):
        eta = T.sqrt(learning_rate)*smrg.normal(size=param.shape)
        delta = 0.5*learning_rate*(gl + gp) + eta
        updates[param] = param + delta
    return updates

def SGHMC(loss, params, lr, log_prior, N, friction):
    """Apply the SGHMC MCMC sampler"""
    g_lik = get_or_compute_grads(-N*loss, params)
    g_prior = get_or_compute_grads(log_prior, params)
    smrg = MRG_RandomStreams()
    updates = OrderedDict()
    C = friction
    for param, gl, gp in zip(params, g_lik, g_prior):
        value = param.get_value(borrow=True)
        mmtm = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        eta = T.sqrt(2*C*lr)*smrg.normal(size=param.shape)
        updates[mmtm] = mmtm*(1-lr*C) - lr*(gl + gp) + eta
        updates[param] = param + lr*mmtm
    return updates

# ########################## Experiments ######################################
def approximator_size(lower, upper, step):
    size = np.linspace(lower, upper, step)
    print('Experiment sizes % s' % (size,))
    accum = []
    for s in size:
        error = 101.
        for i in np.arange(10):
            print('s: %i \t trial: %i' %(s,i))
            accry = main2(model='mlp', save_name='./models/studentMNISTv.npz',
                          dataset='MNIST',
                          file_name = './models/teacherMNIST1200.npz',
                          num_epochs=100, L2Radius=3.87, base_tlr=1e-5,
                          base_slr=1e0, update_W=False, mb_size=100,
                          margin_lr=25, sampler='SGHMC', burn_in = 20,
                          thinning_interval = 1, s_momentum=0.99,
                          method='Dark', num_hid=s)
            error = np.minimum(100-accry, error)
        accum.append(error)
    accum = np.asarray(accum)
    print accum
    np.savez('./models/accumDark1200.npz', size=size, accum=accum)
    
    fig = plt.figure()
    plt.plot(size, accum)
    plt.show()

def cycle_mlps(lower, upper, step):
    size = np.linspace(lower, upper, step)
    print('Experiment sizes % s' % (size,))
    accum = []
    for s in size:
        error = 101.
        for i in np.arange(5):
            print('s: %i \t trial: %i' %(s,i))
            accry = main(model='mlp', save_name='./models/studentMNISTv.npz',
                         dataset='MNIST', num_epochs=500, L2Radius=3.87,
                         base_tlr=1e-1, base_slr=1e01, mb_size=500,
                         margin_lr=25, num_hid=s)
            error = np.minimum(100-accry, error)
        accum.append(error)
    accum = np.asarray(accum)
    print accum
    np.savez('./models/accumML.npz', size=size, accum=accum)
    
    fig = plt.figure()
    plt.plot(size, accum)
    plt.show()

def train_subset():
    fname = '/home/daniel/Code/VPPD/models/MNISTsplit.pkl'
    with open(fname, 'r') as fp:
        data = cPickle.load(fp)
    X_train, y_train, X_val, y_val, X_test, y_test = data
    for i in np.arange(10):
        X_train[i] = X_train[i].astype(theano.config.floatX)
    X_val = X_val.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    for i in np.arange(10):
        y_train[i] = y_train[i].astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    
    for i in np.arange(10):
        dataset = X_train[i], y_train[i], X_val, y_val, X_test, y_test
        save_name = './models/splits/teacherMNISTsplit' + str((i+1)*5000) + '.npz'
        main(model='mlp', save_name=save_name, dataset=dataset, num_epochs=500,
             L2Radius=3.87, base_tlr=1e-2, base_slr=1e-1, mb_size=500,
             margin_lr=25, num_hid=800)

def transfer_subset():
    '''Cycle through all subsets and log best of 5 accuracies'''
    # Load data subsets
    fname = '/home/daniel/Code/VPPD/models/MNISTsplit.pkl'
    with open(fname, 'r') as fp:
        data = cPickle.load(fp)
    X_train, y_train, X_val, y_val, X_test, y_test = data
    for i in np.arange(10):
        X_train[i] = X_train[i].astype(theano.config.floatX)
    X_val = X_val.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    for i in np.arange(10):
        y_train[i] = y_train[i].astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    
    # Load models
    dir_name = '/home/daniel/Code/VPPD/models/splits'
    accum = []; size = []
    for root, dirs, files in os.walk(dir_name):
        for f in files:
            # Specific model
            fname = root + '/' + f
            error = 101
            # Get model number
            i = f.replace('teacherMNISTsplit','')
            i = f.replace('.npz','')
            print i
            i = (int(i)/5000)-1
            size.append((i+1)*5000)
            print('Model number %i' % (i,))
            dataset = X_train[i], y_train[i], X_val, y_val, X_test, y_test
            for i in numpy.arange(5):
                accry = main2(model='mlp', save_name='./models/studentMNISTv.npz',
                              dataset=dataset, file_name = fname, num_epochs=100,
                              L2Radius=3.87, base_tlr=1e-5, base_slr=1e0, update_W=False,
                              mb_size=100, margin_lr=25, sampler='SGHMC',burn_in = 20,
                              thinning_interval = 1, s_momentum=0.99, method='VPPD',
                              num_hid=800)
                error = np.minimum(100-accry, error)
        accum.append(error)
    accum = np.asarray(accum)
    print accum
    np.savez('./models/accumVPPDsub.npz', size=size, accum=accum)
    
    fig = plt.figure()
    plt.plot(size, accum)
    plt.show()


if __name__ == '__main__':
    #main(model='cnn', save_name='./models/teacherMNISTCNN150.npz', dataset='MNIST',
    #     num_epochs=100, L2Radius=3.87, base_tlr=1e-1, base_slr=1e-1,
    #     mb_size=500, margin_lr=25, num_hid=150)
    #main2(model='mlp', save_name='./models/studentMNISTv.npz', dataset='MNIST',
    #      file_name = './models/teacherMNIST800.npz', num_epochs=100,
    #      L2Radius=3.87, base_tlr=1e-5, base_slr=1e0, update_W=False,
    #      mb_size=100, margin_lr=25, sampler='SGHMC', burn_in = 20,
    #      thinning_interval = 1, s_momentum=0.99, method='VPPD', num_hid=800)
    #approximator_size(100,1200,12)
    #cycle_mlps(100,1000,10)
    #train_subset()
    transfer_subset()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
