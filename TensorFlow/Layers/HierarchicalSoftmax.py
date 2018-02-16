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

class HSoftmax(gof.Op):
    """
    Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.
    """

    nin = 1
    nout = 1
    __props__ = ()

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s' %
                             x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        e_x = np.exp(x - x.max(axis=1)[:, None])
        sm = e_x / e_x.sum(axis=1)[:, None]
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = softmax_op(x)
        return [softmax_grad(g_sm, sm)]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)

    def infer_shape(self, node, shape):
        return shape

'''
    def c_headers(self):
        return ['<iostream>', '<cmath>']

    @staticmethod
    def c_code_template(dtype):
        # this implementation was lifted from
        # /u/bergstrj/cvs/bergstrj/src/feb07/nn.cxx

        # TODO: put this into a templated function, in the support code
        # TODO: declare the max of each row as an Op output

        # TODO: set error messages for failures in this code

        # TODO: use this to accept float32 and int32: node.inputs[0].type.dtype_specs()[1]
        init_decl = """
        npy_intp* Nx = PyArray_DIMS(%(x)s);
        npy_intp Sx1 = 0;
        npy_intp Ssm1 = 0;
        if (PyArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "not a 2d tensor");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(x)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(x)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "not a float");
            %(fail)s;
        }
        if ((NULL == %(sm)s)
            || (PyArray_DIMS(%(sm)s)[0] != PyArray_DIMS(%(x)s)[0])
            || (PyArray_DIMS(%(sm)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(sm)s);
            %(sm)s = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(%(x)s),
                                                       PyArray_TYPE(%(x)s));
            if(!%(sm)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc sm output");
                %(fail)s
            }
        }
        Sx1 = PyArray_STRIDES(%(x)s)[1]/sizeof(dtype_%(x)s);
        Ssm1 = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);
        """

        begin_row_loop = """
        for (size_t i = 0; i < Nx[0]; ++i)
        {
            size_t j;
            double sum = 0.0;
            const dtype_%(x)s* __restrict__ x_i = (dtype_%(x)s*)(PyArray_BYTES(%(x)s) + PyArray_STRIDES(%(x)s)[0] * i);
            dtype_%(sm) s* __restrict__ sm_i = (dtype_%(sm)s*)(PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);
            dtype_%(sm)s row_max = x_i[0];
            //std::cout << "0 " << row_max << "\\n";
            // Get the maximum value of the row
            for (j = 1; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx1] ;
                //std::cout << "1 " << row_ij << "\\n";
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }
        """

        inside_row_loop = """
            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx1] ;
                //std::cout << "2 " << j << " " << row_ij << " " << row_max << "\\n";
                dtype_%(sm)s sm_ij = exp(row_ij - row_max);
                //std::cout << "3 " << j << " " << sm_ij << "\\n";
                sum += sm_ij;
                sm_i[j * Ssm1] = sm_ij;
            }
            //cblas_dscal(x.N, 1.0 / sum, &mat_at(s,i,0), s.n);
            double sum_inv = 1.0 / sum;
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm1] *= sum_inv;
            }
        """
        # Get the vectorized version of exp if it exist
        try:
            vec_exp = theano.scalar.exp.c_code_contiguous_raw(dtype,
                                                              "Nx[1]", "sm_i", "sm_i")
            inside_row_loop_contig = """
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm1] = x_i[j * Sx1] - row_max;
            }
            %(vec_exp)s;
            for (j = 0; j < Nx[1]; ++j)
            {
                sum += sm_i[j * Ssm1];
            }
            //cblas_dscal(x.N, 1.0 / sum, &mat_at(s,i,0), s.n);
            double sum_inv = 1.0 / sum;
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm1] *= sum_inv;
            }
            """ % locals()

            inside_row_loop = """
            if(Ssm1 == 1){
                %(inside_row_loop_contig)s
            }else{
                %(inside_row_loop)s
            }
            """ % locals()
        except theano.gof.utils.MethodNotDefined:
            pass

        end_row_loop = """
        }
        """
        return (init_decl, begin_row_loop, inside_row_loop, end_row_loop)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        sm, = out
        code_template = ''.join(self.c_code_template(
            node.inputs[0].type.dtype_specs()[1]))
        return code_template % dict(locals(), **sub)

    @staticmethod
    def c_code_cache_version():
        return (3,)
'''

hsoftmax_op = HSoftmax()




class SoftmaxGrad(gof.Op):
    """
    Gradient wrt x of the Softmax Op.
    """

    nin = 2
    nout = 1
    __props__ = ()

    def make_node(self, dy, sm):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        if dy.type.ndim not in (1, 2) \
                or dy.type.dtype not in tensor.float_dtypes:
            raise ValueError('dy must be 1-d or 2-d tensor of floats. Got ',
                             dy.type)
        if dy.ndim == 1:
            dy = tensor.shape_padleft(dy, n_ones=1)
        if sm.ndim == 1:
            sm = tensor.shape_padleft(sm, n_ones=1)
        return Apply(self, [dy, sm], [sm.type()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = numpy.zeros_like(sm)
        # dx[i,j] = - (\sum_k dy[i,k] sm[i,k]) sm[i,j] + dy[i,j] sm[i,j]
        for i in xrange(sm.shape[0]):
            dy_times_sm_i = dy[i] * sm[i]
            dx[i] = dy_times_sm_i - sum(dy_times_sm_i) * sm[i]
        output_storage[0][0] = dx

    def grad(self, inp, grads):
        dy, sm = inp
        g, = grads

        tmp = g + tensor.neg(tensor.sum(g * sm, axis=1).dimshuffle((0, 'x')))
        g_dy = tmp * sm

        tmp2 = tensor.sum(dy * sm, axis=1).dimshuffle((0, 'x'))
        g_sm = tmp * dy - g * tmp2

        return g_dy, g_sm

    def infer_shape(self, node, shape):
        return [shape[1]]

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inp, out, sub):
        dy, sm = inp
        dx, = out
        return '''
        if ((PyArray_TYPE(%(dy)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(dy)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "types should be float or float64");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(sm)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(sm)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "types should be float or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(dy)s) != 2)
            || (PyArray_NDIM(%(sm)s) != 2))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(dy)s)[0] != PyArray_DIMS(%(sm)s)[0])
        {
            PyErr_SetString(PyExc_ValueError, "dy.shape[0] != sm.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyArray_DIMS(%(dx)s)[0] != PyArray_DIMS(%(sm)s)[0])
            || (PyArray_DIMS(%(dx)s)[1] != PyArray_DIMS(%(sm)s)[1]))
        {
            Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_SimpleNew(2,
                                                        PyArray_DIMS(%(sm)s),
                                                        PyArray_TYPE(%(sm)s));
            if (!%(dx)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc dx output");
                %(fail)s;
            }
        }
        for (size_t i = 0; i < PyArray_DIMS(%(dx)s)[0]; ++i)
        {
            const dtype_%(dy)s* __restrict__ dy_i = (dtype_%(dy)s*) (PyArray_BYTES(%(dy)s) + PyArray_STRIDES(%(dy)s)[0] * i);
            npy_intp Sdy = PyArray_STRIDES(%(dy)s)[1]/sizeof(dtype_%(dy)s);
            const dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*) (PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);
            npy_intp Ssm = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);
            dtype_%(dx) s* __restrict__ dx_i = (dtype_%(dx)s*) (PyArray_BYTES(%(dx)s) + PyArray_STRIDES(%(dx)s)[0] * i);
            npy_intp Sdx = PyArray_STRIDES(%(dx)s)[1]/sizeof(dtype_%(dx)s);
            double sum_dy_times_sm = 0.;
            for (size_t j = 0; j < PyArray_DIMS(%(dx)s)[1]; ++j)
            {
                dx_i[j * Sdx] = dy_i[j * Sdy] * sm_i[j * Ssm];
                sum_dy_times_sm += dx_i[j * Sdx];
            }
            for (size_t j = 0; j < PyArray_DIMS(%(dx)s)[1]; ++j)
            {
                dx_i[j * Sdx] -= sum_dy_times_sm * sm_i[j * Ssm];
            }
        }
        ''' % dict(locals(), **sub)
softmax_grad = SoftmaxGrad()
