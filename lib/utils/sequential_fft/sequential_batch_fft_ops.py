from __future__ import absolute_import, division, print_function

import os.path as osp

import tensorflow as tf
from tensorflow.python.framework import ops

# load module
module = tf.load_op_library(osp.join(osp.dirname(__file__),
                                     'build/sequential_batch_fft.so'))

sequential_batch_fft = module.sequential_batch_fft
sequential_batch_ifft = module.sequential_batch_ifft

# Shape registration is moved to C++ to be compatible with TensorFlow 1.0 API
# @tf.RegisterShape("SequentialBatchFFT")
# def _SequentialBatchFFTShape(op):
#     return [op.inputs[0].get_shape()]
# 
# @tf.RegisterShape("SequentialBatchIFFT")
# def _SequentialBatchIFFTShape(op):
#     return [op.inputs[0].get_shape()]

@ops.RegisterGradient("SequentialBatchFFT")
def _SequentialBatchFFTGrad(op, grad):
    if (grad.dtype == tf.complex64):
        size = tf.cast(tf.shape(grad)[1], tf.float32)
        return (sequential_batch_ifft(grad, op.get_attr("compute_size"))
            * tf.complex(size, 0.))
    else:
        size = tf.cast(tf.shape(grad)[1], tf.float64)
        return (sequential_batch_ifft(grad, op.get_attr("compute_size"))
            * tf.complex(size, tf.zeros([], tf.float64)))

@ops.RegisterGradient("SequentialBatchIFFT")
def _SequentialBatchIFFTGrad(op, grad):
    if (grad.dtype == tf.complex64):
        rsize = 1. / tf.cast(tf.shape(grad)[1], tf.float32)
        return (sequential_batch_fft(grad, op.get_attr("compute_size"))
            * tf.complex(rsize, 0.))
    else:
        rsize = 1. / tf.cast(tf.shape(grad)[1], tf.float64)
        return (sequential_batch_fft(grad, op.get_attr("compute_size"))
            * tf.complex(rsize, tf.zeros([], tf.float64)))
