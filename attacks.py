# Based on code from https://github.com/tensorflow/cleverhans
#
# This is the code for the paper
#
# Certifying Some Distributional Robustness with Principled Adversarial Training
# Link: https://openreview.net/forum?id=Hk6kPgZA-
#
# Authors: Aman Sinha, Hongseok Namkoong, John Duchi

from abc import ABCMeta
import numpy as np
import warnings

from attacks_tf import wrm

class Attack:
    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None):
        """
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        :param back: The backend to use. Either 'tf' (default) or 'th'.
        :param sess: The tf session to run graphs in (use None for Theano)
        """
        if not(back == 'tf' or back == 'th'):
            raise ValueError("Backend argument must either be 'tf' or 'th'.")
        if back == 'tf' and sess is None:
            raise Exception("A tf session was not provided in sess argument.")
        if back == 'th' and sess is not None:
            raise Exception("A session should not be provided when using th.")
        if not hasattr(model, '__call__'):
            raise ValueError("model argument must be a function that returns "
                             "the symbolic output when given an input tensor.")

        # Prepare attributes
        self.model = model
        self.back = back
        self.sess = sess
        self.inf_loop = False

    def generate(self, x, **kwargs):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically. Otherwise, it will wrap the
        numerical implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        if not self.inf_loop:
            self.inf_loop = True
            assert self.parse_params(**kwargs)
            import tensorflow as tf
            graph = tf.py_func(self.generate_np, [x], tf.float32)
            self.inf_loop = False
            return graph
        else:
            error = "No symbolic or numeric implementation of attack."
            raise NotImplementedError(error)

    def parse_params(self, params=None):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        :param params: a dictionary of attack-specific parameters
        :return: True when parsing was successful
        """
        return True

class WassersteinRobustMethod(Attack):
    def __init__(self, model, back='tf', sess=None):
        super(WassersteinRobustMethod, self).__init__(model, back, sess)
    
    def generate(self, x, **kwargs):
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)
        return wrm(x, self.model(x), y=self.y, eps=self.eps, ord=self.ord, \
                   model=self.model, steps=self.steps)

    def parse_params(self, eps=0.3, ord=2, y=None, steps=15,**kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        
        Attack-specific parameters:
        :param eps: (optional float) .5/gamma (Lagrange dual parameter) 
        in the ICLR paper (see link above), 
        :param ord: (optional) Order of the norm (mimics Numpy).
        Possible values: 2.
        :param y: (optional) A placeholder for the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param steps: how many gradient ascent steps to take in finding
        the adversarial example
        """
        # Save attack-specific parameters
        self.eps = eps
        self.ord = ord
        self.y = y
        self.steps = steps
                   
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [int(2)]:
            raise ValueError("Norm order must be 2.")
        return True
