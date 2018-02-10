# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.ops import rnn
import numpy as np

from nets.network_v1 import Network1
from model.config import cfg

class mdlstmvgg16(Network1):
  def __init__(self):
    Network1.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

  def _image_to_head(self, is_training, reuse=None):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
     
      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
      #                  trainable=True, scope='conv22')
      h,w = 858, 600
      #self._im_info 
      #net  = separable_lstm(net, 128, h//2, w//2, 1, None,'l1')
      #net = slim.conv2d(net, 128, [5, 5], scope='conv2')
        
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                          trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net1 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      net = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

    self._act_summaries.append(net)
    self._layers['head'] = net
    
    return net,net1

  def _head_to_tail(self, pool5, is_training, reuse=None):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      if False:
          pool5  = separable_lstm(pool5, 128, cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1, [2,2],'l1')
    
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                            scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, 
                            scope='dropout7')

    return fc7

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.find('Momentum') != -1:
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)
        if False:
            sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                                self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))

def ndlstm_base_dynamic(inputs, noutput, sequence_length, scope=None, reverse=False):
  
  with tf.variable_scope(scope, "SeqLstm", [inputs]):
    # TODO(tmb) make batch size, sequence_length dynamic
    # example: sequence_length = tf.shape(inputs)[0]
    _, batch_size, _ = tf.unstack(tf.shape(inputs))#_shape(inputs)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(noutput, state_is_tuple=True)
    state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    #state = array_ops.zeros([batch_size, lstm_cell.state_size])
    #sequence_length = int(inputs.get_shape()[0])
    sequence_lengths = tf.to_int64(
        tf.fill([batch_size], sequence_length))
    if reverse:
      inputs = tf.reverse_v2(inputs, [0])
    outputs, _ = rnn.dynamic_rnn(
        lstm_cell, inputs, sequence_lengths, state, time_major=True)
    #print(outputs)
    if reverse:
      outputs = tf.reverse_v2(outputs, [0])
    return outputs

def _shape(tensor):
  """Get the shape of a tensor as an int list."""
  return tf.unstack(tf.shape(tensor))

def horizontal_lstm(images, num_filters_out, lengs, h, bb, scope=None):
  
  with tf.variable_scope(scope, "HorizontalLstm", [images]):
    batch_size, _, _, _ = tf.unstack(tf.shape(images))
    #sequence = images_to_sequence(images)
    num_image_batches, height, width, depth =  tf.unstack(tf.shape(images))
    transposed = tf.transpose(images, [2, 0, 1, 3])
    sequence = tf.reshape(transposed,
                           [lengs, num_image_batches * h, images.get_shape().as_list()[3]])
    
    with tf.variable_scope("lr"):
      hidden_sequence_lr = ndlstm_base_dynamic(sequence, num_filters_out // 2, lengs)
    with tf.variable_scope("rl"):
      hidden_sequence_rl = (ndlstm_base_dynamic(
          sequence, num_filters_out - num_filters_out // 2, lengs, reverse=1))
    output_sequence = tf.concat([hidden_sequence_lr, hidden_sequence_rl],
                                       2)
    #output = sequence_to_images(output_sequence, batch_size)
    width, num_batches, depth = _shape(output_sequence)
    height = num_batches // batch_size
    reshaped = tf.reshape(output_sequence,
                               [lengs, batch_size, h, output_sequence.get_shape().as_list()[2]])
    return tf.transpose(reshaped, [1, 2, 0, 3])
    

def separable_lstm(images, num_filters_out, lengs, h, bb,  kernel_size=None, scope=None , nhidden=None):
  
  with tf.variable_scope(scope, "SeparableLstm", [images]):
    if nhidden is None:
      nhidden = num_filters_out
    if kernel_size is not None:
      #images = get_blocks(images, kernel_size)
      images = get_blocks(images,lengs,h, bb, kernel_size)
      print(images)
      lengs = lengs//kernel_size[1]
      h = h//kernel_size[0]
    hidden = horizontal_lstm(images, nhidden,lengs, h, bb)
    with tf.variable_scope("vertical"):
      transposed = tf.transpose(hidden, [0, 2, 1, 3])
      output_transposed = horizontal_lstm(transposed, num_filters_out,h, lengs, bb)
    output = tf.transpose(output_transposed, [0, 2, 1, 3])
    return output

def get_blocks(images,  width, height,bb, kernel_size=[1,1]):
  """Split images in blocks

  Args:
    images: (num_images, height, width, depth) tensor
    kernel_size: A list of length 2 holding the [kernel_height, kernel_width] of
      of the pooling. Can be an int if both values are the same.

  Returns:
    (num_images, height/kernel_height, width/kernel_width,
    depth*kernel_height*kernel_width) tensor
  """
  with tf.variable_scope("image_blocks"):
    batch_size, _, _, chanels = _shape(images)
    h, w = height//kernel_size[0], width//kernel_size[1]
    features = kernel_size[1]*kernel_size[0]*images.get_shape().as_list()[3]    

    lines = tf.split(images, h, axis=1)
    line_blocks = []
    for line in lines:
      line = tf.transpose(line, [0, 2, 3, 1])
      line = tf.reshape(line, [batch_size, w, features])
      line_blocks.append(line)
    
    return tf.stack(line_blocks, axis=1)