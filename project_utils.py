"""
Based on Source Code for Homework 3 of ECBM E6040, Spring 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
"""
import os
import sys
import numpy
import scipy.io
from sklearn import preprocessing
import theano
import theano.tensor as T

def translate_image(X,inaxis=0,amount=1):
    Y=[]
    print amount
    for ii in range(0,X.shape[0]):
        Xi = numpy.reshape(X[ii],(32,32,3))
        Yi = numpy.roll(Xi,amount,axis=inaxis)
        if (amount > 0) and (inaxis == 1):
            Yi[:,range(0,amount)] = 0
        if (amount < 0) and (inaxis == 1):
            #print [Yi.shape[1]-x-1 for x in range(1,numpy.abs(amount)+1)]
            Yi[:,[Yi.shape[1]-x for x in range(1,numpy.abs(amount)+1)]] = 0
        if (amount > 0) and (inaxis == 0):
            Yi[range(0,amount),:] = 0
        if (amount < 0) and (inaxis == 0):
            Yi[[Yi.shape[0]-x for x in range(1,numpy.abs(amount)+1)],:] = 0
        Yi=numpy.reshape(Yi,(32*32*3))
        Y.append(Yi)
    return numpy.asarray(Y)
	

	
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(ds_rate=None, theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://ufldl.stanford.edu/housenumbers/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    train_dataset = check_dataset('train_32x32.mat')
    test_dataset = check_dataset('test_32x32.mat')

    # Load the dataset
    train_set = scipy.io.loadmat(train_dataset)
    test_set = scipy.io.loadmat(test_dataset)

    # Convert data format
    def convert_data_format(data):
        X = numpy.reshape(data['X'],
                          (numpy.prod(data['X'].shape[:-1]), data['X'].shape[-1]),
                          order='C').T / 255.
        y = data['y'].flatten()
        y[y == 10] = 0
        return (X,y)
    train_set = convert_data_format(train_set)
    test_set = convert_data_format(test_set)

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval
    
def load_data_cifar10(ds_rate=None, theano_shared=True,translations=False):
    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    cifar_train=numpy.zeros((0,3072))
    cifar_train_labels=numpy.empty((0,0))
    for kind in range(1,6):
        k=unpickle('../data/data_batch_'+str(kind))
        cifar_train=numpy.vstack((cifar_train,k['data']))
        cifar_train_labels=numpy.append(cifar_train_labels,numpy.asarray(k['labels']))
        cifar_train

    k=unpickle('../data/test_batch') 
    cifar_test=k['data']
    cifar_test_labels=numpy.asarray(k['labels'])
    cifar_train = cifar_train / 255.
    cifar_test = cifar_test / 255.
    cifar_train = preprocessing.scale(cifar_train)
    cifar_test = preprocessing.scale(cifar_test)

    
    if translations==True:
        # Repeat the training set 5 times
        cifar_train_labels = numpy.tile(cifar_train_labels, 5)

        #  translate the dataset
        train_set_x_u = translate_image(cifar_train,inaxis=0,amount=-1)
        train_set_x_d = translate_image(cifar_train,inaxis=0,amount=1)
        train_set_x_r = translate_image(cifar_train,inaxis=1,amount=1)
        train_set_x_l = translate_image(cifar_train,inaxis=1,amount=-1)

        # Stack the original dataset and the synthesized datasets
        cifar_train = numpy.vstack((cifar_train,
                           train_set_x_u,
                           train_set_x_d,
                           train_set_x_r,
                           train_set_x_l))
    train_set=(cifar_train,cifar_train_labels)
    test_set=(cifar_test,cifar_test_labels)
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]
        
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval