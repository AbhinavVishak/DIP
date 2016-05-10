"""
Based on Source Code for Homework 3.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html

Adam: https://gist.github.com/skaae/ae7225263ca8806868cb
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from project_utils import shared_dataset, load_data, load_data_cifar10
from project_nn import LogisticRegression, GALayer, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn, Fig3Layer

def translate_image_theano(X,inaxis=0,amount=1):
    Y=[]
    print amount
    for ii in range(0,X.shape[0]):
        Xi = T.reshape(X[ii],(32,32,3))
        Yi = T.roll(Xi,amount,axis=inaxis)
        if (amount > 0) and (inaxis == 1):
            Yi[:,range(0,amount)] = 0
        if (amount < 0) and (inaxis == 1):
            #print [Yi.shape[1]-x-1 for x in range(1,numpy.abs(amount)+1)]
            Yi[:,[Yi.shape[1]-x for x in range(1,numpy.abs(amount)+1)]] = 0
        if (amount > 0) and (inaxis == 0):
            Yi[range(0,amount),:] = 0
        if (amount < 0) and (inaxis == 0):
            Yi[[Yi.shape[0]-x for x in range(1,numpy.abs(amount)+1)],:] = 0
        Yi=T.reshape(Yi,(32*32*3))
        Y.append(Yi)
    return T.asarray(Y)
    
def relu(x):
    return theano.tensor.switch(x<0, 0, x)


def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(numpy.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates
    
def test_lenet(learning_rate=0.001, n_epochs=1000, nkerns=[16, 16], ds_rate=1, dataset_type='cifar-10', power_reg=0.01, batch_size=200, dropout_rate=0.5, is_spectral=False, translations=False, verbose=False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)
    if dataset_type=='cifar-10':
        print 'Using cifar-10...'
        datasets = load_data_cifar10(ds_rate=ds_rate,translations=translations)
    else:
        print 'Using SVHN...'
        datasets = load_data(ds_rate=ds_rate)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    if is_spectral==False:
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, 3, 3),
            is_spectral=is_spectral,
            pool_ignore_border=True,
            poolsize=(2,2)
        )
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 15, 15),
            filter_shape=(nkerns[1], nkerns[0], 3, 3),
            is_spectral=is_spectral,
            pool_ignore_border=True,
            poolsize=(2,2)
        )
        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 6 * 6,
            n_out=4096,
            dropout_rate=dropout_rate,
            activation=relu
        )
        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in=4096,
            n_out=1024,
            dropout_rate=dropout_rate,
            activation=relu
        )
        layer4 = LogisticRegression(
             input=layer3.output,
             n_in=1024,
             n_out=10)
        cost = layer4.negative_log_likelihood(y)
    else:
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, 3, 3),
            is_spectral=is_spectral,
            filter_mask_perc=0.85,
            poolsize=(3,3)
        )
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 30, 30),
            filter_shape=(nkerns[1], nkerns[0], 3, 3),
            is_spectral=is_spectral,
            filter_mask_perc=0.85*0.85,
            poolsize=(3,3)
        )
        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 28 * 28,
            n_out=1024,
            dropout_rate=dropout_rate,
            activation=relu
        )   


    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    updates=adam(cost, params,learning_rate=learning_rate)
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    #theano.printing.pydotprint(cost, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)  
    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
        
    return layer0


def test_sec_51(learning_rate=0.001, n_epochs=1000, nkerns=[16, 16,100], ds_rate=1, dataset_type='cifar-10', power_reg=0.01, batch_size=200, dropout_rate=0.2, is_spectral=True, translations=False, remask=False, verbose=False):

    rng = numpy.random.RandomState(23455)
    if dataset_type=='cifar-10':
        print 'Using cifar-10...'
        datasets = load_data_cifar10(ds_rate=ds_rate,translations=translations)
    else:
        print 'Using SVHN...'
        datasets = load_data(ds_rate=ds_rate)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        is_spectral=is_spectral,
        filter_mask_perc=0.85,
        mc=1,
        poolsize=(3,3)
    )
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 26, 26),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        is_spectral=is_spectral,
        filter_mask_perc=0.15,
        mc=1,
        poolsize=(3,3)
    )
    layer3_input = layer1.output.flatten(2)
    layer3 = HiddenLayer(
            rng,
            input=layer3_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=4096,
            dropout_rate=dropout_rate,
            activation=relu
        )
    layer4 = HiddenLayer(
            rng,
            input=layer3.output,
            n_in=4096,
            n_out=1024,
            dropout_rate=dropout_rate,
            activation=relu
        )
    # layer2 = LeNetConvPoolLayer(
        # rng,
        # input=layer1.output,
        # image_shape=(batch_size, nkerns[1], 28, 28),
        # filter_shape=(nkerns[2], nkerns[1], 1, 1),
        # is_spectral=is_spectral,
        # filter_mask_perc=1,
        # mc=1,
        # poolsize=(3,3)
    # )
    # layer4 = GALayer(
        # rng,
        # input=layer1.output,
        # dropout_rate=dropout_rate,
        # activation=relu
    # )
    layer5 = LogisticRegression(
         input=layer4.output,
         n_in=1024,
         n_out=10)
         
    cost = layer5.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Create a list of all model parameters to be fit by gradient descent
    params = layer5.params +  layer4.params +  layer3.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    updates=adam(cost, params,learning_rate=learning_rate)
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    #theano.printing.pydotprint(cost, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)  
    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose,remask=remask)
        
    return layer0

def test_fig3(learning_rate=0.001, n_epochs=1000, nkerns=[128,32], ds_rate=1, dataset_type='cifar-10', power_reg=0.01, batch_size=200, dropout_rate=0.2, is_spectral=True, translations=False, remask=False, verbose=False):

    rng = numpy.random.RandomState(23455)
    if dataset_type=='cifar-10':
        print 'Using cifar-10...'
        datasets = load_data_cifar10(ds_rate=ds_rate,translations=translations)
    else:
        print 'Using SVHN...'
        datasets = load_data(ds_rate=ds_rate)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    layer0 = Fig3Layer(
       rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 7, 7),
        is_spectral=is_spectral,
        filter_mask_perc=0.85,
        mc=1,
        poolsize=(2,2)
    )
    
    layer1 = Fig3Layer(
       rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 13, 13),
        filter_shape=(nkerns[1], nkerns[0], 3, 3 ),
        is_spectral=is_spectral,
        filter_mask_perc=0.85,
        mc=1,
        poolsize=(1,1)
    )
    
    layer5 = GALayer(rng, input=layer1.output, dropout_rate=0,
                 activation=relu)
    
    layer6_input = layer5.output
    layer6 = LogisticRegression(
         input=layer6_input,
         n_in=nkerns[1],
         n_out=10
    )
         
    cost = layer6.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # Create a list of all model parameters to be fit by gradient descent
    params = layer6.params + layer0.params + layer1.params

    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    updates=adam(cost, params,learning_rate=learning_rate)
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    filter_wts = theano.function( inputs=[] ,  outputs = layer0.W )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    #theano.printing.pydotprint(cost, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)  
    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose,remask=remask,outputModel=True, filter_wts=filter_wts)
        
    return layer0