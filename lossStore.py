# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:53:11 2020

@author: Xiuxiu
"""

import tensorflow as tf

def cross_entropy_loss(outputs, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=labels))

def pix_loss(outputs, labels):
    return tf.reduce_mean(tf.square(outputs-labels))

def crf_loss1(output):  
    temp1 = tf.where(output>0, tf.ones_like(output), tf.zeros_like(output))
    temp2 = tf.reduce_sum(temp1, axis=[1,2,3])
    temp3 = tf.reduce_sum(output, axis=[1,2,3])
    temp2 = tf.where(tf.equal(temp2, 0), tf.ones_like(temp2), temp2)
    
    return tf.reduce_mean(tf.divide(temp3, temp2))

def crf_loss2(output):  
    return tf.reduce_mean(tf.reduce_sum(output, axis=[1,2,3]))

def crf_loss3(output):  
    return tf.reduce_mean(output)

def crf_loss4(output):
    return tf.reduce_sum(output)

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice