"""
Source Code for Homework 3 of ECBM E6040, Spring 2016, Columbia University

This code contains implementation of some basic components in neural network.

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html

Dropout: https://github.com/mdenil/dropout/blob/master/mlp.py
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import six.moves.cPickle as pickle
from cuda_fft_project import *

def relu(x):
    return theano.tensor.switch(x<0, 0, x)
    
def gaussian_filter(size, sigma = 1):
    x = numpy.arange(0, size, 1, float)
    y = x[:,numpy.newaxis]
    xc = (size-1) // 2
    yc = (size-1) // 2
    hh = numpy.exp(-1 * ((x-xc)**2 + (y-yc)**2) / ( 2 * sigma**2 ) )
    hh = hh / hh.sum()
    hh = hh - numpy.mean(hh)
    return hh
    
def gaussian_filters(size,sig): 
    a=gaussian_filter(size[3], sigma = sig)
    b = numpy.expand_dims(a, axis=0)
    b = numpy.repeat(b, size[1] , axis = 0)
    return b
    
def perform_dropout(rng, input, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(42))
    mask = srng.binomial(n=1, p=1-p, size=input.shape)
    output = input * T.cast(mask, theano.config.floatX)
    return output
    
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        #if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
        return T.mean(T.neq(self.y_pred, y))
        #else:
        #    raise NotImplementedError()

class GALayer(object):
    def __init__(self, rng, input, dropout_rate=0,
                 activation=relu):
        self.input = input
        
        self.output = perform_dropout(rng,T.mean(self.input,axis=(2,3)),dropout_rate)
        
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, dropout_rate=0.5,
                 activation=relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        output_candidate = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.output = perform_dropout(rng,output_candidate,dropout_rate)
        # parameters of the model
        self.params = [self.W, self.b]

class myMLP(object):
    """Multi-Layefr Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hiddenLayers):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int or list of ints
        :param n_hidden: number of hidden units. If a list, it specifies the
        number of units in each hidden layers, and its length should equal to
        n_hiddenLayers.

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type n_hiddenLayers: int
        :param n_hiddenLayers: number of hidden layers
        """

        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        if hasattr(n_hidden, '__iter__'):
            assert(len(n_hidden) == n_hiddenLayers)
        else:
            n_hidden = (n_hidden,)*n_hiddenLayers

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function.
        self.hiddenLayers = []
        for i in xrange(n_hiddenLayers):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = n_in if i == 0 else n_hidden[i-1]
            self.hiddenLayers.append(
                HiddenLayer(
                    rng=rng,
                    input=h_input,
                    n_in=h_in,
                    n_out=n_hidden[i],
                    activation=relu
            ))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = sum([x.params for x in self.hiddenLayers], []) + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),  dropout_rate=0.5, 
    is_spectral=False, pool_ignore_border=True,filter_mask_perc=0.85,mc=0.3):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:]) // 2
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        #W_ifft=(cuifft(self.W))[:,:,:,:,0]
        #self.powerleak=(cuifft(self.W))[:,:,:,:,1] / (cuifft(self.W))[:,:,:,:,0]
        #dtensor5 = T.TensorType('float32', (False,)*5)
        #x = T.dtensor5('x')
        #fft_caster = theano.function([x], (cufft(x))[:,:,:,:,0])
        #fft_realer=
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        #self.W_real=self.W_ifft[:,:,:,:,0]
        # convolve input feature maps with filters
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if is_spectral==False:
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape
            )
            self.pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=pool_ignore_border
            )   
        else:
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape
            )
            #self.W = theano.shared(
            #    numpy.asarray(
            #        rng.uniform(low=-W_bound, high=W_bound, size=numpy.append(filter_shape,2)),
            #        dtype=theano.config.floatX
            #    ),
            #    borrow=True
            #)
            #W_ifft=T.squeeze((cuifft(self.W))[:,:,:,:,0])
            # W_ifft=(cuifft(self.W))[:,:,:,:,0]
            # W_ifft=W_ifft.dimshuffle(0,1,2,3)
            # #print(W_ifft.eval().shape)
            # #print(input.eval().shape)
            # conv_out = conv2d(
                # input=input,
                # filters=W_ifft,
                # filter_shape=filter_shape,
                # image_shape=image_shape
            # )
            #pooling_mask=numpy.zeros(numpy.append(image_shape,2))
            self.image_shape=image_shape
            self.filter_shape=filter_shape
            self.filter_mask_perc=filter_mask_perc
            ones_length=numpy.round(self.filter_mask_perc*(self.image_shape[2]-self.filter_shape[2]+1))
            self.rng=rng
            if (ones_length%2)==1:
                ones_length=ones_length+1
            new_size_axis=(numpy.round(((self.image_shape[2]-self.filter_shape[2]+1)-ones_length)/2)).astype('int')
            conv_to_use = theano.shared(numpy.asarray(numpy.zeros((image_shape[0],filter_shape[0],(self.image_shape[2]-self.filter_shape[2]+1),(self.image_shape[2]-self.filter_shape[2]+1),2)),dtype=theano.config.floatX),borrow=True)
            conv_to_use=T.set_subtensor(conv_to_use[:,:,:,:,0],conv_out)
            conv_to_use=T.set_subtensor(conv_to_use[:,:,:,:,1],0)
            conv_to_use=cufft(conv_to_use)
            conv_to_use=T.roll(conv_to_use,(self.image_shape[2]-self.filter_shape[2]+1)/2,axis=2)
            conv_to_use=T.roll(conv_to_use,(self.image_shape[2]-self.filter_shape[2]+1)/2,axis=3)
            #conv_to_use = theano.shared(numpy.asarray(numpy.zeros((filter_shape[0],filter_shape[1],ones_length,ones_length,2)),dtype=theano.config.floatX),borrow=True)
            #conv_to_use[:,:,:,:,0]
            conv_to_use=conv_to_use[:,:,new_size_axis:(self.image_shape[2]-self.filter_shape[2]+1)-new_size_axis,new_size_axis:(self.image_shape[2]-self.filter_shape[2]+1)-new_size_axis,:] #T.set_subtensor(conv_to_use[:,:,:,:,0],conv_out[:,:,new_size_axis::-new_size_axis,new_size_axis::-new_size_axis])
            #conv_to_use=T.set_subtensor(conv_to_use[:,:,:,:,1],0)
            #conv_out=T.concatenate([conv_out,conv_out*0],axis=4)
            #conv_out=conv_out.dimshuffle(0,1,2,3,'x')
            #conv_out=T.tile(conv_out,(1,1,1,1,2))
            #conv_out=T.set_subtensor(conv_out[:,:,:,:,1],0)
            #self.pooled_out=(cuifft(cufft(conv_to_use) * self.fft_mask)/(image_shape[2]-filter_shape[2]+1)/(image_shape[2]-filter_shape[2]+1)/(image_shape[2]-filter_shape[2]+1)/filter_shape[0])[:,:,:,:,0]
            conv_to_use=T.roll(conv_to_use,-(self.image_shape[2]-self.filter_shape[2]+1)/2,axis=2)
            conv_to_use=T.roll(conv_to_use,-(self.image_shape[2]-self.filter_shape[2]+1)/2,axis=3)
            self.pooled_out=(cuifft(conv_to_use)/ones_length/ones_length/filter_shape[0])[:,:,:,:,0]
            #Multiply with the mask,back-project
            self.pooled_out=self.pooled_out.dimshuffle(0,1,2,3) #Decrease to 4 dimensions
        #conv_out = theano.function([input,self.W], T.reshape(input * T.tile(self.W,replicate_shape))) 
        #conv_out_part_b = theano.function([input,self.W],T.tile(self.W,replicate_shape))
        # downsample each feature map individually, using maxpooling
        #pool_printing = theano.printing.Print('this is a very important value')(cufft(conv_to_use))

        #theano.pp(self.pooled_out)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        output_candidate = relu(T.cast(self.pooled_out, 'float32') + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = perform_dropout(rng,output_candidate,dropout_rate)
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        
    def remask(self):
        ones_length=numpy.round(self.filter_mask_perc*(self.image_shape[2]-self.filter_shape[2]+1))
        self.rng=rng
        if (ones_length%2)==1:
            ones_length=ones_length+1
        #print(ones_length)
        fft_mask_shape=numpy.round(numpy.asarray(
                self.rng.uniform(low=mc*ones_length, high=ones_length, size=2),
                dtype=theano.config.floatX
            )/2)*2
        fft_mask=numpy.ones(shape=(fft_mask_shape[0],fft_mask_shape[1]))
        fft_mask=numpy.pad(fft_mask,(((self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[0])/2,(self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[0])/2),((self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[1])/2,(self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[1])/2)),mode='constant') # Prepare Centralized Mask
        fft_mask=numpy.fft.ifftshift(fft_mask) # Build the Full Mask by Shifting
        fft_mask=numpy.expand_dims(fft_mask, axis=0) #Expand to the Matching Dimensions
        fft_mask=numpy.expand_dims(fft_mask, axis=0) #Expand to the Matching Dimensions
        fft_mask=numpy.expand_dims(fft_mask, axis=4) #Expand to the Matching Dimensions
        fft_mask=numpy.tile(fft_mask,(self.image_shape[0],self.filter_shape[0],1,1,2)) #Tile along the right dimensions
        self.fft_mask=theano.shared(numpy.asarray(fft_mask,
                                       dtype=theano.config.floatX),borrow=True) # Compile the Mask

def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True, outputModel = False, classifier = None,remask=False,filter_wts=None):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 20000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))
                    best_model = classifier #this line seems useless.
                    #save model variables
                    if outputModel:    
                        with open('best_model-epoch'+str(epoch)+'.pkl', 'wb') as f:
                            pickle.dump(numpy.array(filter_wts()), f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return best_model

class Fig3Layer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),  dropout_rate=0.5, 
    is_spectral=False, pool_ignore_border=True,filter_mask_perc=0.85,mc=0.3):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:]) // 2
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        #W_ifft=(cuifft(self.W))[:,:,:,:,0]
        #self.powerleak=(cuifft(self.W))[:,:,:,:,1] / (cuifft(self.W))[:,:,:,:,0]
        #dtensor5 = T.TensorType('float32', (False,)*5)
        #x = T.dtensor5('x')
        #fft_caster = theano.function([x], (cufft(x))[:,:,:,:,0])
        #fft_realer=
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        #self.W_real=self.W_ifft[:,:,:,:,0]
        # convolve input feature maps with filters
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if is_spectral==False:
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape
            )
            self.pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=pool_ignore_border
            )   
        else:
            
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=numpy.append(filter_shape,2)),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            W_ifft=(cuifft(self.W)/filter_shape[2]/filter_shape[2]/filter_shape[1])[:,:,:,:,0]
            W_ifft=W_ifft.dimshuffle(0,1,2,3)
            #print(W_ifft.eval().shape)
             #print(input.eval().shape)
            conv_out = conv2d(
                input=input,
                filters=W_ifft,
                filter_shape=filter_shape,
                image_shape=image_shape
            )
            pooling_mask=numpy.zeros(numpy.append(image_shape,2))
           
        
            self.pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=pool_ignore_border
            )   
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        output_candidate = relu(T.cast(self.pooled_out, 'float32') + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = perform_dropout(rng,output_candidate,dropout_rate)
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        
    def remask(self):
        ones_length=numpy.round(self.filter_mask_perc*(self.image_shape[2]-self.filter_shape[2]+1))
        self.rng=rng
        if (ones_length%2)==1:
            ones_length=ones_length+1
        #print(ones_length)
        fft_mask_shape=numpy.round(numpy.asarray(
                self.rng.uniform(low=mc*ones_length, high=ones_length, size=2),
                dtype=theano.config.floatX
            )/2)*2
        fft_mask=numpy.ones(shape=(fft_mask_shape[0],fft_mask_shape[1]))
        fft_mask=numpy.pad(fft_mask,(((self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[0])/2,(self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[0])/2),((self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[1])/2,(self.image_shape[2]-self.filter_shape[2]+1-fft_mask.shape[1])/2)),mode='constant') # Prepare Centralized Mask
        fft_mask=numpy.fft.ifftshift(fft_mask) # Build the Full Mask by Shifting
        fft_mask=numpy.expand_dims(fft_mask, axis=0) #Expand to the Matching Dimensions
        fft_mask=numpy.expand_dims(fft_mask, axis=0) #Expand to the Matching Dimensions
        fft_mask=numpy.expand_dims(fft_mask, axis=4) #Expand to the Matching Dimensions
        fft_mask=numpy.tile(fft_mask,(self.image_shape[0],self.filter_shape[0],1,1,2)) #Tile along the right dimensions
        self.fft_mask=theano.shared(numpy.asarray(fft_mask,
                                       dtype=theano.config.floatX),borrow=True) # Compile the Mask