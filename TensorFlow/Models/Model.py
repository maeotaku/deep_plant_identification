import sys
sys.path.insert(0, '../')
from config import *


import pickle

import lasagne
import theano
import numpy as np

#from lasagne.layers import batch_norm
from lasagne.layers import ConcatLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers.normalization import batch_norm

#if RUN_MODE=='gpu':
#    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#    from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer
#else:
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
#from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax, linear, rectify

from lasagne.layers import BatchNormLayer
from lasagne.layers import ElemwiseSumLayer





def dev_var(initialiser, device):
    def init(shape):
        return theano.shared(initialiser(shape), target=device)
    return init

class Model():

    @classmethod
    def load_model_parameters(cls, filename, network):
        saved_parameters = pickle.load(open(filename))
        lasagne.layers.set_all_param_values(network, saved_parameters)
        return network

    @classmethod
    def load_model_parameters_npz(cls, filename, network):
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        #saved_parameters = pickle.load(open(filename))
        #lasagne.layers.set_all_param_values(network, saved_parameters)
        return network

    @classmethod
    def replace_layer(cls, net, deleteable_layer_name, new_layer):
        del net[deleteable_layer_name]
        net[new_layer.name] = new_layer
        return net

    @classmethod
    def get_last_layer(cls, network, layer_name):
        return network[layer_name]

    @classmethod
    def freeze_layers(cls, network, layer_list):
        for layer in lasagne.layers.get_all_layers(network):
            if not layer in layer_list:
                for param in layer.params:
                    layer.params[param].discard('trainable')
