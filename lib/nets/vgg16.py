# --------------------------------------------------------
# Tensorflow Two Stream Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Hangyan Jiang
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.utils.compact_bilinear_pooling import compact_bilinear_pooling_layer
import numpy as np

import lib.config.config as cfg
from lib.nets.network import Network


class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        with tf.variable_scope('vgg_16', 'vgg_16'):

            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            q = [4.0, 12.0, 2.0]
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
            filter1 = np.asarray(filter1, dtype=float) / q[0]
            filter2 = np.asarray(filter2, dtype=float) / q[1]
            filter3 = np.asarray(filter3, dtype=float) / q[2]
            filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
            filters = np.einsum('klij->ijlk', filters)
            filters = filters.flatten()
            initializer_srm = tf.constant_initializer(filters)

            # Build RGB stream head
            net = self.build_head(is_training)

            # Build Noise stream head
            net2 = self.build_head_forNoise(is_training, initializer, initializer_srm)

            # Build rpn
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)

            # Build proposals
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # Build predictions
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, net2, rois, is_training, initializer, initializer_bbox)

            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic, sess, pretrained_model):
        variables_to_restore = []
        noise_variable = {}
        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0' \
                    or v.name == 'vgg_16/cbp_fc6/weights:0' or v.name == 'vgg_16/cbp_fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == 'vgg_16/conv1/conv1_1/weights:0' or v.name == 'vgg_16/conv1n/conv1n_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
        # # From VGG pretrained weights file(RGB weights), load weights for noise stream
        #         name = v.name.split('/')
        #         if len(name) < 4:
        #             continue
        #         name[1] += 'n'
        #         name[2] = name[1] + name[2][len(name[1])-1:]
        #         noise_counterpart = name[0] + '/' + name[1] + '/' + name[2] + '/' + name[3]
        #         for u in variables:
        #             if u.name == noise_counterpart:
        #                 noise_variable[v.name.split(':')[0]] = u
        #                 print('Variables restored: %s' % u.name)
        #
        # with tf.variable_scope('Restore_Noise_Variables'):
        #     with tf.device("/cpu:0"):
        #         # fix the vgg16 noise stream variables
        #         restorer = tf.train.Saver(noise_variable)
        #         restorer.restore(sess, pretrained_model)

            # # From VGG pretrained weights file(RGB weights), load weights for noise stream except for conv layer 1, 2
            #     name = v.name.split('/')
            #     if len(name) < 4 or (name[1] != 'conv1' and name[1] != 'conv2'):
            #         continue
            #     name[1] += 'n'
            #     name[2] = name[1] + name[2][len(name[1]) - 1:]
            #     noise_counterpart = name[0] + '/' + name[1] + '/' + name[2] + '/' + name[3]
            #     for u in variables:
            #         if u.name == noise_counterpart:
            #             noise_variable[v.name.split(':')[0]] = u
            #             print('Variables restored: %s' % u.name)
            #
            # with tf.variable_scope('Restore_Noise_Variables'):
            #     with tf.device("/cpu:0"):
            #         # fix the vgg16 noise stream variables
            #         restorer = tf.train.Saver(noise_variable)
            #         restorer.restore(sess, pretrained_model)

        return variables_to_restore


    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                # cbp_fc6_conv = tf.get_variable("cbp_fc6_conv", [7, 7, 512, 4096], trainable=False)
                # cbp_fc7_conv = tf.get_variable("cbp_fc7_conv", [1, 1, 4096, 4096], trainable=False)
                # noise_conv1_rgb = tf.get_variable("noise_conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix[
                                                                                                  'vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix[
                                                                                                  'vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

                # restorer_cbp_noise = tf.train.Saver({"vgg_16/fc6/weights": cbp_fc6_conv,
                #                                      "vgg_16/fc7/weights": cbp_fc7_conv,
                #                                      "vgg_16/conv1/conv1_1/weights": noise_conv1_rgb})
                # restorer_cbp_noise.restore(sess, pretrained_model)
                #
                # sess.run(tf.assign(self._variables_to_fix['vgg_16/cbp_fc6/weights:0'], tf.reshape(cbp_fc6_conv,
                #                                                                               self._variables_to_fix[
                #                                                                                   'vgg_16/cbp_fc6/weights:0'].get_shape())))
                # sess.run(tf.assign(self._variables_to_fix['vgg_16/cbp_fc7/weights:0'], tf.reshape(cbp_fc7_conv,
                #                                                                               self._variables_to_fix[
                #                                                                                   'vgg_16/cbp_fc7/weights:0'].get_shape())))
                # sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1n/conv1n_1/weights:0'],
                #                    tf.reverse(noise_conv1_rgb, [2])))



    def build_head(self, is_training):

        # Main network
        # Layer  1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head'] = net

        return net

    def build_head_forNoise(self, is_training, initializer, initializer_srm):

        def truncate_2(x):
            neg = ((x + 2) + abs(x + 2)) / 2 - 2
            return -(2 - neg + abs(2 - neg)) / 2 + 2

        # Main network
        # Layer SRM
        net = slim.conv2d(self._image, 3, [5, 5], trainable=False, weights_initializer=initializer_srm,
                          activation_fn=None, padding='SAME', stride=1, scope='srm')
        net = truncate_2(net)

        # Layer  1
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], trainable=is_training, weights_initializer=initializer, scope='conv1n')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1n')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, weights_initializer=initializer, scope='conv2n')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2n')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, weights_initializer=initializer, scope='conv3n')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3n')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope='conv4n')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4n')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope='conv5n')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head2'] = net

        return net

    def build_rpn(self, net, is_training, initializer):

        # Build anchor component
        self._anchor_component()

        # Create RPN Layer
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")

        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # Change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):

        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, net2, rois, is_training, initializer, initializer_bbox):
        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        # pool5_flat = slim.flatten(pool5, scope='flatten')
        pool5_forNoise = self._crop_pool_layer(net2, rois, "pool5_forNoise")

        # Compact Bilinear Pooling
        cbp = compact_bilinear_pooling_layer(pool5, pool5_forNoise, 512)
        cbp_flat = slim.flatten(cbp, scope='cbp_flatten')

        # Fully connected layers
        # fc6 = slim.fully_connected(pool5_flat, 4096, scope='bbox_fc6')
        fc6_cbp = slim.fully_connected(cbp_flat, 4096, scope='fc6')
        if is_training:
            # fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
            fc6_cbp = slim.dropout(fc6_cbp, keep_prob=0.5, is_training=True, scope='cbp_dropout6')

        # fc7 = slim.fully_connected(fc6, 4096, scope='bbox_fc7')
        fc7_cbp = slim.fully_connected(fc6_cbp, 4096, scope='fc7')
        if is_training:
            # fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
            fc7_cbp = slim.dropout(fc7_cbp, keep_prob=0.5, is_training=True, scope='cbp_dropout7')

        # Scores and predictions
        cls_score = slim.fully_connected(fc7_cbp, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        bbox_prediction = slim.fully_connected(fc7_cbp, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction
