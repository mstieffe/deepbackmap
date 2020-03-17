import tensorflow as tf
from tensorflow.contrib.layers.python.layers import variance_scaling_initializer
from configparser import SafeConfigParser

cfg = SafeConfigParser()
cfg.read('config.ini')


# the implements of leakyRelu
def lrelu(x , alpha= 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

"""
def conv2d(input_, nf, ks = (3,3), strides = (1,1), pad=False, name="conv2d"):
    (k_h, k_w) = ks
    (s_h, s_w) = strides
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], nf], initializer= variance_scaling_initializer())
        if pad:
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='VALID')
        biases = tf.get_variable('biases', [nf], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
    
def conv3d(input_, nf, ks = (3,3,3), strides = (1,1,1), pad=False, name="conv3d"):
    (k_h, k_w, k_d) = ks
    (s_h, s_w, s_d) = strides
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], nf], initializer= variance_scaling_initializer())
        if pad:
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        conv = tf.nn.conv3d(input_, w, strides=[1, s_h, s_w, s_d, 1], padding='VALID')
        biases = tf.get_variable('biases', [nf], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
"""  
def conv3d(input_, nf, ks = (3,3,3), strides = (1,1,1), pad=False, name="conv3d"):
    with tf.variable_scope(name):
        if pad:
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")
            padding = 'VALID'
        else:
            padding = 'SAME'
        conv = tf.layers.conv3d(input_, nf, ks, strides, padding=padding, activation = None)
        return conv

def conv3dd(input_, nf, ks = (3,3,3), strides = (1,1,1), pad=False, name="conv3d"):
    if pad:
        input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        padding = 'VALID'
    else:
        padding = 'SAME'
    conv = tf.layers.conv3d(input_, nf, ks, strides, padding=padding, activation = None)
    return conv

def conv3d_coords(input_, nf, ks = (3,3,3), strides = (1,1,1), pad=False, name="conv3d"):
    with tf.variable_scope(name):
        if pad:
            input_ = tf.pad(input_, [[0,0], [1, 1], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            padding = 'VALID'
        else:
            padding = 'SAME'
        conv = tf.layers.conv3d(input_, nf, ks, strides, padding=padding, activation = None)
        return conv
  
def Pixl_Norm_2d(input_, eps=1e-8):
    return input_ / tf.sqrt(tf.reduce_mean(input_**2, axis=3, keep_dims=True) + eps)

def Pixl_Norm_3d(input_, eps=1e-8):
    return input_ / tf.sqrt(tf.reduce_mean(input_**2, axis=4, keep_dims=True) + eps)




def minibatch_stddev_feat(input_):
    _, h, w, d, _ = input_.get_shape().as_list()
    new_feat_shape = [cfg.getint('model', 'batchsize'), h, w, d, 1]

    mean, var = tf.nn.moments(input_, axes=[0], keep_dims=True)
    stddev = tf.sqrt(tf.reduce_mean(var, keep_dims=True))
    new_feat = tf.tile(stddev, multiples=new_feat_shape)
    return tf.concat([input_, new_feat], axis=4)

def minibatch_stddev_feat2(input_):
    _, h, w, d, _ = input_.get_shape().as_list()
    new_feat_shape = [cfg.getint('model', 'batchsize'), h, w, d, 1]

    mean, var = tf.nn.moments(input_, axes=[0], keep_dims=True)
    stddev = tf.sqrt(tf.reduce_mean(var, keep_dims=True))
    new_feat = tf.tile(stddev, multiples=new_feat_shape)
    return tf.concat([input_, new_feat], axis=4), stddev


def energy_feat(input_, energies):
    _, h, w, d, _ = input_.get_shape().as_list()
    new_feat_shape = [1, h, w, d, 1]
    energies = tf.reshape(energies, [cfg.getint('model', 'batchsize'),1,1,1,1])

    #mean, var = tf.nn.moments(input_, axes=[0], keep_dims=True)
    #stddev = tf.sqrt(tf.reduce_mean(var, keep_dims=True))
    new_feat = tf.tile(energies, multiples=new_feat_shape)
    return tf.concat([input_, new_feat], axis=4)

def energy_feat2(input_, energies):
    _, h, w, d, _ = input_.get_shape().as_list()
    new_feat_shape = [cfg.getint('model', 'batchsize'), h, w, d, 1]
    energies = tf.reshape(energies, [1,1,1,1,cfg.getint('model', 'batchsize')])

    #mean, var = tf.nn.moments(input_, axes=[0], keep_dims=True)
    #stddev = tf.sqrt(tf.reduce_mean(var, keep_dims=True))
    new_feat = tf.tile(energies, multiples=new_feat_shape)
    return tf.concat([input_, new_feat], axis=4)