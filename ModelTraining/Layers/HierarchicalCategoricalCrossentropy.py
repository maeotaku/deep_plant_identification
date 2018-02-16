import sys
sys.path.insert(0, '../')
import os
import pickle
import io

from theano import gof
from theano import scalar
from theano.tensor import extra_ops
#from theano.gof.opt import copy_stack_trace
from theano.tensor import basic as tensor, subtensor, opt, elemwise
from theano.tensor.type import (values_eq_approx_remove_inf,
                                values_eq_approx_remove_nan)
from theano.compile import optdb
from theano.gof import Apply

from theano.tensor.nnet.sigm import sigmoid, softplus
from theano.gradient import DisconnectedType
from theano.gradient import grad_not_implemented
from theano.tensor.nnet.blocksparse import sparse_block_dot

import numpy as np
import theano
import theano.tensor
from theano.tensor.nnet import *
import lasagne
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax

from HierarchyManagement import *


#coding_i has the softmax calculated values for all classes for the current item i. For the first hierarchy level send back the softmax value
def p_level_i(level_k, expected_prediction_idx, coding_i, num_classes, hierarchy, inv_hierarchy):
    if level_k==0:
        return coding_i[expected_prediction_idx]
    mask = build_level_mask(expected_prediction_idx, level_k, hierarchy, inv_hierarchy, num_classes)
    mask = np.array(mask)
    children = mask * coding_i
    return np.sum(children)

def hierarchical_categorical_crossentropy(coding_dist, true_dist, hierarchy, inv_hierarchy, level_list):
    """
    Return the cross-entropy between an approximating distribution and a true
    distribution taking into account a hierarchy of classes
    Mathematically it is defined as follows:
    .. math::
        H(p,q) = - \sum_x p(x) \log(q(x))
    Parameters
    ----------
    coding_dist : a dense matrix
        Each slice along axis represents one distribution.
    true_dist : a dense matrix or sparse matrix or integer vector
        In the case of a matrix argument, each slice along axis represents one
        distribution. In the case of an integer vector argument, each element
        represents the position of the '1' in a 1-of-N encoding.
    Returns
    -------
    tensor of rank one-less-than `coding_dist`
        The cross entropy between each coding and true distribution.
    Notes
    -----
    axis : int
        The dimension over which each distribution runs
        (1 for row distributions, 0 for column distributions).
    """
    if true_dist.ndim == coding_dist.ndim:
        return -tensor.sum(true_dist * tensor.log(coding_dist),
                           axis=coding_dist.ndim - 1)
    elif true_dist.ndim == coding_dist.ndim - 1:
        return hierarchical_categorical_crossentropy_1hot(coding_dist, true_dist, hierarchy, inv_hierarchy, level_list)
    else:
        raise TypeError('rank mismatch between coding and true distributions')

class HierarchicalCrossentropyCategorical1Hot(gof.Op):
    """
    Compute the cross entropy between a coding distribution and
    a true distribution of the form [0, 0, ... 0, 1, 0, ..., 0].
    .. math::
        y[i] = - \log(coding_dist[i, one_of_n[i])
    Notes
    -----
    In the case that the coding distribution is the output of a
    softmax, an application of this Op will probably be optimized
    away in favour of one with a C implementation.
    """
    __props__ = ()

    def make_node(self, coding_dist, true_one_of_n, hierarchy, inv_hierarchy, level_list):
        """
        Parameters
        ----------
        coding_dist : dense matrix
        true_one_of_n : lvector
        Returns
        -------
        dvector
        """
        _coding_dist = tensor.as_tensor_variable(coding_dist)
        _true_one_of_n = tensor.as_tensor_variable(true_one_of_n)
        self._hierarchy = hierarchy
        self._inv_hierarchy = inv_hierarchy
        self._level_list = level_list
        if _coding_dist.type.ndim != 2:
            raise TypeError('matrix required for argument: coding_dist')
        if _true_one_of_n.type not in (tensor.lvector, tensor.ivector):
            raise TypeError(
                'integer vector required for argument: true_one_of_n'
                '(got type: %s instead of: %s)' % (_true_one_of_n.type,
                                                   tensor.lvector))

        return Apply(self, [_coding_dist, _true_one_of_n],
                     [tensor.Tensor(dtype=_coding_dist.dtype,
                      broadcastable=[False])()])



    def perform(self, node, inp, out):
        coding, one_of_n = inp
        y_out, = out
        y = np.zeros_like(coding[:, 0])
        n = len(coding[0, :])
        #navigate over all items within the batch
        for i in xrange(len(y)):
            the_one = one_of_n[i]
            #calculate first the first level of teh hierarchy, equivalent to existing equations
            y[i] = 100000 #-np.log(coding[i, the_one])
            #calculate each hierarchy level k
            for k in range(0, len(self._level_list)):
                p_i_k = p_level_i(k, the_one, coding[i, :], n, self._hierarchy, self._inv_hierarchy)
                y[i] = min(-np.log(p_i_k), y[i])
                #temp[i] = -np.log(p_i_k)
            #y[i] = float(y[i]) / float(len(self._level_list))
        y_out[0] = y
        #print("F Coding dist", coding, coding.shape)
        #print("F True One of N", one_of_n, one_of_n.shape)
        #print("F y", y, y.shape)

    def infer_shape(self, node, in_shapes):
        return [(in_shapes[0][0],)]

    def grad(self, inp, grads):
        coding, one_of_n = inp
        g_y, = grads
        return [hierarchical_crossentropy_categorical_1hot_grad(g_y, coding, one_of_n, self._hierarchy, self._inv_hierarchy, self._level_list),
                grad_not_implemented(self, 1, one_of_n)]

hierarchical_categorical_crossentropy_1hot = HierarchicalCrossentropyCategorical1Hot()

class HierarchicalCrossentropyCategorical1HotGrad(gof.Op):

    __props__ = ()

    def make_node(self, g_y, coding_dist, true_one_of_n, hierarchy, inv_hierarchy, level_list):
        self._hierarchy = hierarchy
        self._inv_hierarchy = inv_hierarchy
        self._level_list = level_list
        return Apply(self, [g_y, coding_dist, true_one_of_n],
                     [coding_dist.type()])

    def perform(self, node, inp, out):
        g_y, coding_dist, true_one_of_n = inp
        g_coding_strg, = out
        g_coding = np.zeros_like(coding_dist)
        n = len(g_coding[0, :])
        #go through all items
        for i in xrange(len(g_y)):
            the_one = true_one_of_n[i]
            #g_coding[i, the_one] = (-g_y[i] / coding_dist[i, the_one]
            for k in range(0, len(self._level_list)):
                p_i_k = p_level_i(k, the_one, coding_dist[i, :], n, self._hierarchy, self._inv_hierarchy)
                p_i = coding_dist[i, the_one]
                g_coding[i, the_one] += (p_i * (1 - (1 / p_i_k)))
        g_coding_strg[0] = g_coding
        #print("B gy", g_y, g_y.shape)
        #print("B Coding dist", coding_dist, coding_dist.shape)
        #print("B True One of N", true_one_of_n, true_one_of_n.shape)
        #print("B g_coding", g_coding, g_coding.shape)

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

hierarchical_crossentropy_categorical_1hot_grad = HierarchicalCrossentropyCategorical1HotGrad()
