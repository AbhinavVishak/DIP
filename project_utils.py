import os
import sys
import numpy
import scipy.io
import theano
import theano.tensor as T
import cPickle

def shared_dataset(data_xy, borrow=True):    
    data_x, data_y = data_xy
    shared_x = theano.shared(data_x.astype(float),borrow=borrow)
    shared_y = theano.shared(data_y.astype(float),borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_mnist( ds_rate = None , theano_shared = True ):
    ''' Loads the MNIST dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    f = open('mnist.pkl', 'r')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set =( train_set[0][::ds_rate,:] , train_set[1][::ds_rate] )

    #return theano shared variables if corresponding flag true
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval