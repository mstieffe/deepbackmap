import tensorflow as tf
import numpy as np
#from ops import lrelu, conv3d, conv3dd, conv3d_coords, Pixl_Norm_3d# minibatch_stddev_feat,minibatch_stddev_feat2, energy_feat,energy_feat2 upscale, avgpool2d, WScaleLayer, 
from configparser import SafeConfigParser
#import tensorflow.layers as tfl


cfg = SafeConfigParser()
cfg.read('config.ini')

def lrelu(x , alpha= 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

#print(cfg.get('main', 'key1'))
def prelu(net):
    alpha = tf.Variable(0.0, dtype=net.dtype)
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)
    return tf.nn.leaky_relu(net, alpha)
    
    
def residual_3dconv_block(net, num_filters, kernel_size, stride, is_training, relu_type="lrelu"):
    
    # let us cache the input tensor and downsample it
    if stride != 1:
        #inp = tf.layers.max_pooling3d(net, 2, 2) 
        inp = tf.layers.conv3d(net, num_filters, 1, stride, padding='SAME', activation = None)
    else:
        inp = tf.identity(net)
    
    # now convolve with stride (potential downsampling)
    net = tf.layers.conv3d(net, num_filters, kernel_size, stride, padding='SAME', activation = None)
    net = tf.contrib.layers.layer_norm(net)    
    net = lrelu(net)

    
    # now convolve again but do not downsample
    net = tf.layers.conv3d(net, num_filters, kernel_size, 1, padding='SAME', activation = None)
    net = tf.contrib.layers.layer_norm(net)    
    
    #net = tf.concat((net, inp), axis=-1)
    net = tf.add(net, inp)


    net = lrelu(net)
        
    return net

def residual_3dconv_block_dis(net, num_filters, kernel_size, stride, is_training, relu_type="lrelu"):
    
    # let us cache the input tensor and downsample it
    if stride != 1:
        inp = tf.layers.max_pooling3d(net, 2, 2) 
    else:
        inp = tf.identity(net)
    
    # now convolve with stride (potential downsampling)
    #net = tfl.conv3d(net, num_filters, kernel_size, stride, activation_fn=tf.identity, padding="SAME")
    net = tf.layers.conv3d(net, num_filters, kernel_size, stride, padding='SAME', activation = None)
    net = tf.contrib.layers.layer_norm(net)    
    #net = tf.layers.batch_norm(net, is_training=True, activation_fn=tf.identity)    
    net = lrelu(net)
  
    # now convolve again but do not downsample
    net = tf.layers.conv3d(net, num_filters, kernel_size, 1, padding='SAME', activation = None)
    net = tf.contrib.layers.layer_norm(net)    
    
    net = tf.concat((net, inp), axis=-1)


    net = lrelu(net)
        
    return net



def residual_3d_deconv_block(net, num_filters, kernel_size, is_training):
    
    # let us cache the input tensor and downsample it
    net = tf.keras.backend.resize_volumes(net, 2, 2, 2, 'channels_last')
    net = tf.layers.conv3d(net, num_filters, 1, 1, padding='SAME', activation = None)
    inp = tf.identity(net)

    
    # now convolve with stride (potential downsampling)
    #net = tfl.conv3d(net, num_filters, kernel_size, stride, activation_fn=tf.identity, padding="SAME")
    net = tf.layers.conv3d(net, num_filters, kernel_size, 1, padding='SAME', activation = None)
    net = tf.contrib.layers.layer_norm(net)    
    net = lrelu(net)
    
    # now convolve again but do not downsample
    net = tf.layers.conv3d(net, num_filters, kernel_size, 1, padding='SAME', activation = None)
    net = tf.contrib.layers.layer_norm(net)    
    
    net = tf.add(net, inp)


    net = lrelu(net)
        
    return net
        

def dis(inp, is_training):
    with tf.variable_scope('discriminator', reuse= tf.AUTO_REUSE):
        print(inp)
        conv = tf.layers.conv3d(inp, 128, 4, 1, padding='SAME', activation = None)
        conv = lrelu(conv)

        conv = residual_3dconv_block_dis(conv, 128, 3, 1, is_training)
        conv = residual_3dconv_block_dis(conv, 256, 3, 2, is_training)
        #conv = residual_3dconv_block_dis(conv, 256, 3, 1, is_training)
        conv = residual_3dconv_block_dis(conv, 256, 3, 1, is_training)
        
        #conv = minibatch_stddev_feat(conv)
        conv = tf.layers.conv3d(conv, 256, 3, 1, padding='SAME', activation = None)
        conv = tf.contrib.layers.layer_norm(conv)
        conv = lrelu(conv)    
        
        conv = tf.layers.conv3d(conv, 128, (4,4,4), (1,1,1), padding='VALID', activation = None)
        conv = tf.contrib.layers.layer_norm(conv)
        conv = lrelu(conv)        
        
        conv = tf.reshape(conv, [cfg.getint('model', 'batchsize'), -1])
    
        output = tf.layers.dense(conv, 1, activation = None)
        
        return output



def gen_atom(z, c, l, is_training):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        #conditional_info = tf.concat([c, h], 4)    
        
        cond = tf.layers.conv3d(c, 128, 4, 1, padding='SAME', activation = None)
        cond = tf.contrib.layers.layer_norm(cond)    
        cond = lrelu(cond)
    
        cond = residual_3dconv_block(cond, 128, 3, 1, is_training)
        #cond = residual_3dconv_block(cond, 128, 3, 1, is_training)    
        cond = residual_3dconv_block(cond, 128, 3, 1, is_training)
        cond = residual_3dconv_block(cond, 128, 3, 1, is_training)
             
        print(cond)
        cond_downsampled = residual_3dconv_block(cond, 256, 3, 2, is_training)            
             
        lb = tf.ones([cfg.getint('model', 'batchsize'), 4, 4, 4, 4])*l
        
        noise = tf.reshape(z, [cfg.getint('model', 'batchsize'), 1, 1, 1, 256])
        
        noise = tf.concat([noise, l], axis = 4)        
        
        noise = tf.pad(noise, [[0,0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        noise = tf.layers.conv3d(noise, 256, 4, 1, padding='VALID', activation = None)
        noise = lrelu(noise)

        noise = tf.concat([noise, lb, cond_downsampled], axis = 4)        
        noise = tf.layers.conv3d(noise, 256, 3, 1, padding='SAME', activation = None)
        noise = tf.contrib.layers.layer_norm(noise)    
        noise = lrelu(noise)

        noise = residual_3dconv_block(noise, 256, 3, 1, is_training)
        noise = residual_3dconv_block(noise, 256, 3, 1, is_training)

        noise = residual_3d_deconv_block(noise,128,3, is_training)
       
    
        im = tf.concat([noise, cond], 4)
        print(im)
    
        im = residual_3dconv_block(im, 256, 3, 1, is_training)
        im = residual_3dconv_block(im, 256, 3, 1, is_training)
        im = residual_3dconv_block(im, 256, 3, 1, is_training)
    
        print(im)
    
        #im = tf.layers.conv3d(im, 256, (3,3,3), (1,1,1), padding='VALID', activation = None)
        #im = Pixl_Norm_3d(lrelu(im))
        im = tf.layers.conv3d(im, 1, 1, 1, padding='SAME', activation = None)
    
        im = tf.sigmoid(im)
    
        return im
        
def gen_atom_withtype(z, c, l, is_training):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        #conditional_info = tf.concat([c, h], 4)    
        
        cond = tf.layers.conv3d(c, 128, 4, 1, padding='SAME', activation = None)
        cond = tf.contrib.layers.layer_norm(cond)    
        cond = lrelu(cond)
    
        cond = residual_3dconv_block(cond, 128, 3, 1, is_training)
        #cond = residual_3dconv_block(cond, 128, 3, 1, is_training)    
        cond = residual_3dconv_block(cond, 128, 3, 1, is_training)
        cond = residual_3dconv_block(cond, 128, 3, 1, is_training)
             
        print(cond)
        cond_downsampled = residual_3dconv_block(cond, 256, 3, 2, is_training)            
             
        lb = tf.ones([cfg.getint('model', 'batchsize'), 4, 4, 4, 4])*l
        
        noise = tf.reshape(z, [cfg.getint('model', 'batchsize'), 1, 1, 1, 256])
        
        noise = tf.concat([noise, l], axis = 4)        
        
        noise = tf.pad(noise, [[0,0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        noise = tf.layers.conv3d(noise, 256, 4, 1, padding='VALID', activation = None)
        noise = lrelu(noise)

        noise = tf.concat([noise, lb, cond_downsampled], axis = 4)        
        noise = tf.layers.conv3d(noise, 256, 3, 1, padding='SAME', activation = None)
        noise = tf.contrib.layers.layer_norm(noise)    
        noise = lrelu(noise)

        noise = residual_3dconv_block(noise, 256, 3, 1, is_training)
        noise = residual_3dconv_block(noise, 256, 3, 1, is_training)

        noise = residual_3d_deconv_block(noise,128,3, is_training)
       
    
        im = tf.concat([noise, cond], 4)
        print(im)
    
        im = residual_3dconv_block(im, 256, 3, 1, is_training)
        im = residual_3dconv_block(im, 256, 3, 1, is_training)
        im = residual_3dconv_block(im, 256, 3, 1, is_training)
    
        print(im)
    
        #im = tf.layers.conv3d(im, 256, (3,3,3), (1,1,1), padding='VALID', activation = None)
        #im = Pixl_Norm_3d(lrelu(im))
        im = tf.layers.conv3d(im, 4, 1, 1, padding='SAME', activation = None)
    
        im = tf.sigmoid(im)
    
        return im

