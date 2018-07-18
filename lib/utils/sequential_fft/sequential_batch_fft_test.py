from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from sequential_batch_fft_ops import sequential_batch_fft, sequential_batch_ifft

compute_size = 128

x = tf.placeholder(tf.complex64, [None, None])
x_128 = tf.placeholder(tf.complex128, [None, None])
# FFT
x_fft = sequential_batch_fft(x, compute_size)
x_fft_128 = sequential_batch_fft(x_128, compute_size)
x_fft_tf = tf.fft(x)
# IFFT
x_ifft = sequential_batch_ifft(x, compute_size)
x_ifft_128 = sequential_batch_ifft(x_128, compute_size)
x_ifft_tf = tf.ifft(x)
# Grads
gx_fft = tf.gradients(x_fft, x)[0]
gx_fft_128 = tf.gradients(x_fft_128, x_128)[0]
gx_fft_tf = tf.gradients(x_fft_tf, x)[0]
gx_ifft = tf.gradients(x_ifft, x)[0]
gx_ifft_128 = tf.gradients(x_ifft_128, x_128)[0]
gx_ifft_tf = tf.gradients(x_ifft_tf, x)[0]

def test_shape():
    print("Testing shape...")

    # Test Shape inference. Output shape should be
    # the same as input shape
    input_pl = tf.placeholder(tf.complex64, [1000, 16000])
    output_fft = sequential_batch_fft(input_pl)
    output_ifft = sequential_batch_ifft(input_pl)
    g_fft = tf.gradients(output_fft, input_pl)[0]
    g_ifft = tf.gradients(output_ifft, input_pl)[0]
    assert(output_fft.get_shape() == input_pl.get_shape())
    assert(output_ifft.get_shape() == input_pl.get_shape())
    assert(g_fft.get_shape() == input_pl.get_shape())
    assert(g_ifft.get_shape() == input_pl.get_shape())

    print("Passed.")

def test_forward():
    # Test forward and compare with tf.batch_fft and tf.batch_ifft
    print("Testing forward...")

    sess = tf.Session()
    for dim in range(1000, 5000, 1000):
        for batch_size in range(1, 10):
            x_val = (np.random.randn(batch_size, dim) +
                     np.random.randn(batch_size, dim) * 1j).astype(np.complex64)

            # Forward complex64
            x_fft_val, x_ifft_tf_val = sess.run([x_fft, x_ifft], {x: x_val})
            # Forward complex128
            x_fft_128_val, x_ifft_128_val = sess.run([x_fft_128, x_ifft_128],
                                                     {x_128: x_val.astype(np.complex128)})
            # Forward with reference tf.batch_fft and tf.batch_ifft
            x_fft_tf_val, x_ifft_val = sess.run([x_fft_tf, x_ifft_tf], {x: x_val})

            ref_sum_fft  = np.sum(np.abs(x_fft_tf_val))
            ref_sum_ifft = np.sum(np.abs(x_ifft_tf_val))
            relative_diff_fft  = np.sum(np.abs(x_fft_val - x_fft_tf_val)) / ref_sum_fft
            relative_diff_ifft = np.sum(np.abs(x_ifft_val - x_ifft_tf_val)) / ref_sum_ifft
            relative_diff_fft128  = np.sum(np.abs(x_fft_128_val - x_fft_tf_val)) / ref_sum_fft
            relative_diff_ifft128 = np.sum(np.abs(x_ifft_128_val - x_ifft_tf_val)) / ref_sum_ifft

            assert(relative_diff_fft < 1e-5)
            assert(relative_diff_fft128 < 1e-5)
            assert(relative_diff_ifft < 1e-5)
            assert(relative_diff_ifft128 < 1e-5)

    sess.close()
    print("Passed.")

def test_gradient():
    # Test Backward and compare with tf.batch_fft and tf.batch_ifft
    print("Testing gradient...")

    sess = tf.Session()
    for dim in range(1000, 5000, 1000):
        for batch_size in range(1, 10):
            x_val = (np.random.randn(batch_size, dim) +
                     np.random.randn(batch_size, dim) * 1j).astype(np.complex64)

            # Backward complex64
            gx_fft_val, gx_ifft_tf_val = sess.run([gx_fft, gx_ifft], {x: x_val})
            # Backward complex128
            gx_fft_128_val, gx_ifft_128_val = sess.run([gx_fft_128, gx_ifft_128],
                                                       {x_128: x_val.astype(np.complex128)})
            # Backward with reference tf.batch_fft and tf.batch_ifft
            gx_fft_tf_val, gx_ifft_val = sess.run([gx_fft_tf, gx_ifft_tf], {x: x_val})

            ref_sum_fft  = np.sum(np.abs(gx_fft_tf_val))
            ref_sum_ifft = np.sum(np.abs(gx_ifft_tf_val))
            relative_diff_fft  = np.sum(np.abs(gx_fft_val - gx_fft_tf_val)) / ref_sum_fft
            relative_diff_ifft = np.sum(np.abs(gx_ifft_val - gx_ifft_tf_val)) / ref_sum_ifft
            relative_diff_fft128  = np.sum(np.abs(gx_fft_128_val - gx_fft_tf_val)) / ref_sum_fft
            relative_diff_ifft128 = np.sum(np.abs(gx_ifft_128_val - gx_ifft_tf_val)) / ref_sum_ifft

            assert(relative_diff_fft < 1e-5)
            assert(relative_diff_fft128 < 1e-5)
            assert(relative_diff_ifft < 1e-5)
            assert(relative_diff_ifft128 < 1e-5)

    sess.close()
    print("Passed.")

def test_large_input():
    # Very large input size, where tf.batch_fft and tf.batch_ifft
    # will run OOM
    print("Testing large input...")

    sess = tf.Session()
    batch_size, dim = 64*16*16, 16000
    print("Forwarding and Backwarding with input shape",
          [batch_size, dim], "This may take a while...")
    x_val = (np.random.randn(batch_size, dim) +
             np.random.randn(batch_size, dim) * 1j).astype(np.complex64)
    sess.run(tf.group(x_fft, x_ifft, gx_fft, gx_ifft), {x: x_val})

    sess.close()
    # Test passes iff no exception occurs.
    print("Passed.")

if __name__ == "__main__":
    test_shape()
    test_forward()
    test_gradient()
    test_large_input()
