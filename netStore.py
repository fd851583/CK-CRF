# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:36:36 2020

@author: Xiuxiu
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import cv2

def net3(image, name="net3"):
    reuse = len([t for t in tf.global_variables() if t.name.startswith(name)]) > 0
    with tf.variable_scope(name, reuse=reuse):
        print(name + ":")
        print(image.shape)
        x = slim.conv2d(image, 8, kernel_size=[3,3], stride=1, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = slim.max_pool2d(x, [3,3], 2, 'SAME')
        
        x1 = x
        print(x.shape)
        
        x = slim.conv2d(x, 16, kernel_size=[3,3], stride=1, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = slim.max_pool2d(x, [3,3], 2, 'SAME')
        
        x2 = x
        print(x.shape)
        
        x = slim.conv2d(x, 32, kernel_size=[3,3], stride=1, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = slim.max_pool2d(x, [3,3], 2, 'SAME')
        
        x3 = x
        print(x.shape)
        
        x = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = slim.max_pool2d(x, [3,3], 2, 'SAME')
        
        x4 = x
        print(x.shape)
        
        x = slim.conv2d(x, 128, kernel_size=[3,3], stride=1, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = slim.max_pool2d(x, [3,3], 2, 'SAME')
        
        x5 = x
        print(x.shape)
        
        x = slim.conv2d(x, 256, kernel_size=[3,3], stride=1, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = slim.max_pool2d(x, [3,3], 2, 'SAME')
        print(x.shape)
             
#        x = slim.conv2d_transpose(x, 128, kernel_size=[3,3], stride = 1, activation_fn=None)
        x = slim.conv2d_transpose(x, 128, kernel_size=[3,3], stride = 2, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = tf.concat([x, x5], axis=-1)
        print(x.shape)
                    
#        x = slim.conv2d_transpose(x, 64, kernel_size=[3,3], stride = 1, activation_fn=None)
        x = slim.conv2d_transpose(x, 64, kernel_size=[3,3], stride = 2, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = tf.concat([x, x4], axis=-1)
        print(x.shape)
          
#        x = slim.conv2d_transpose(x, 32, kernel_size=[3,3], stride = 1, activation_fn=None)
        x = slim.conv2d_transpose(x, 32, kernel_size=[3,3], stride = 2, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = tf.concat([x, x3], axis=-1)
        print(x.shape)
       
#        x = slim.conv2d_transpose(x, 16, kernel_size=[3,3], stride = 1, activation_fn=None)
        x = slim.conv2d_transpose(x, 16, kernel_size=[3,3], stride = 2, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = tf.concat([x, x2], axis=-1)
        print(x.shape)
           
#        x = slim.conv2d_transpose(x, 8, kernel_size=[3,3], stride = 1, activation_fn=None)
        x = slim.conv2d_transpose(x, 8, kernel_size=[3,3], stride = 2, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        x = tf.concat([x, x1], axis=-1)
        print(x.shape)
                  
#        x = slim.conv2d_transpose(x, 1, kernel_size=[3,3], stride = 1, activation_fn=None)
        x = slim.conv2d_transpose(x, 1, kernel_size=[3,3], stride = 2, activation_fn=None)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        print(x.shape)
        
    return x

def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 0):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    kernel = np.multiply(kx,np.transpose(ky))
    kernel[int(kernel_size/2), int(kernel_size/2)] = (1-kernel[int(kernel_size/2), int(kernel_size/2)]) * -1
    
    kernel = np.reshape(kernel, [kernel_size, kernel_size, 1, 1])
    return kernel

def crf_net(x, kernel_r=3, name="net0"):
    reuse = len([t for t in tf.global_variables() if t.name.startswith(name)]) > 0
    with tf.variable_scope(name, reuse=reuse):
        print(name+":")
        print(x.shape)
        
        filter1 = gaussian_kernel_2d_opencv(kernel_r)
        
        x = tf.nn.conv2d(x, filter1, [1, 1, 1, 1], 'SAME', name=None)
        x = tf.nn.relu(x)
        
        return x
    
def crf_net1(x, kernel_r=3, name="net0"):
    reuse = len([t for t in tf.global_variables() if t.name.startswith(name)]) > 0
    with tf.variable_scope(name, reuse=reuse):
        print(name+":")
        print(x.shape)
        
        filter1 = gaussian_kernel_2d_opencv(kernel_r)
        
        x = tf.nn.conv2d(x, filter1, [1, 1, 1, 1], 'SAME', name=None)
        
        return x