# Based on code from https://github.com/tensorflow/cleverhans
#
# This is the code for the paper
#
# Certifying Some Distributional Robustness with Principled Adversarial Training
# Link: https://openreview.net/forum?id=Hk6kPgZA-
#
# Authors: Aman Sinha, Hongseok Namkoong, John Duchi


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf

import utils_tf

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def wrm(x, preds, y=None, eps=0.3, ord=2, model=None, steps=15):
  
    """
        TensorFlow implementation of the Wasserstein distributionally
        adversarial training method. 
        :param x: the input placeholder
        :param preds: the model's output tensor
        :param y: (optional) A placeholder for the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param eps: .5 / gamma (Lagrange dual parameter) 
        in the ICLR paper (see link above)
        :param ord: (optional) Order of the norm (mimics Numpy).
        Possible values: 2.
        :param model: TF graph model
        :param steps: hwo many gradient ascent steps to take
        when finding adversarial example 
        :return: a tensor for the adversarial example
        """
    
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)

    grad, = tf.gradients(eps*loss, x)
    x_adv = tf.stop_gradient(x+grad)
    x = tf.stop_gradient(x)
    
    for t in xrange(steps):
        loss = utils_tf.model_loss(y, model(x_adv), mean=False)
        grad, = tf.gradients(eps*loss, x_adv)
        grad2, = tf.gradients(tf.nn.l2_loss(x_adv-x), x_adv)
        grad = grad - grad2
        x_adv = tf.stop_gradient(x_adv+1./np.sqrt(t+2)*grad)
    return x_adv
