# --------------------------------------------------------
# Two Stream Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Hangyan Jiang
# --------------------------------------------------------

# Testing part
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

def PlotImage(image):
    """
	PlotImage: Give a normalized image matrix which can be used with implot, etc.
	Maps to [0, 1]
	"""
    im = image.astype(float)
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def SRM(imgs):
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    q = [4.0, 12.0, 2.0]
    filter1 = np.asarray(filter1, dtype=float) / 4
    filter2 = np.asarray(filter2, dtype=float) / 12
    filter3 = np.asarray(filter3, dtype=float) / 2
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = np.einsum('klij->ijlk', filters)
    filters = tf.Variable(filters, dtype=tf.float32)
    imgs = np.array(imgs, dtype=float)
    input = tf.Variable(imgs, dtype=tf.float32)
    op = tf.nn.conv2d(input, filters, strides=[1, 1, 1, 1], padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        re = (sess.run(op))
        res = np.round(re[0] / q)
        res[res > 2] = 2
        res[res < -2] = -2
    return [res]
