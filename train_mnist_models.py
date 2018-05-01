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
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_mnist import data_mnist
from utils_tf import model_train, model_eval
from utils import cnn_model

from keras.models import load_model
from keras.backend import manual_variable_initialization
from attacks import WassersteinRobustMethod

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 25, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_string('train_dir', '.', 'Training directory')
flags.DEFINE_string('filename_erm', 'erm.h5', 'Training directory')
flags.DEFINE_string('filename_wrm', 'wrm.h5', 'Training directory')

train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
}
eval_params = {'batch_size': FLAGS.batch_size}

seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

def main(argv=None):

    keras.layers.core.K.set_learning_phase(1)
    manual_variable_initialization(True)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model(activation='elu')
    predictions = model(x)
    wrm = WassersteinRobustMethod(model, sess=sess)
    wrm_params = {'eps': 1.3, 'ord': 2, 'y':y, 'steps': 15}
    predictions_adv_wrm = model(wrm.generate(x, **wrm_params))

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

        # Accuracy of the model on Wasserstein adversarial examples
        accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_wrm, X_test, \
                                       Y_test, args=eval_params)
        print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)

    # Train the model
    model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate, \
                args=train_params, save=False)
    model.model.save(FLAGS.train_dir + '/' + FLAGS.filename_erm)


    print('')
    print("Repeating the process, using Wasserstein adversarial training")
    # Redefine TF model graph
    model_adv = cnn_model(activation='elu')
    predictions_adv = model_adv(x)
    wrm2 = WassersteinRobustMethod(model_adv, sess=sess)
    predictions_adv_adv_wrm = model_adv(wrm2.generate(x, **wrm_params))
    
    def evaluate_adv():
        # Accuracy of adversarially trained model on legitimate test inputs
        accuracy = model_eval(sess, x, y, predictions_adv, X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)
        
        # Accuracy of the adversarially trained model on Wasserstein adversarial examples
        accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_adv_wrm, \
                                       X_test, Y_test, args=eval_params)
        print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)

    model_train(sess, x, y, predictions_adv_adv_wrm, X_train, Y_train, \
                predictions_adv=predictions_adv_adv_wrm, evaluate=evaluate_adv, \
                args=train_params, save=False)
    model_adv.model.save(FLAGS.train_dir + '/' + FLAGS.filename_wrm)

if __name__ == '__main__':
    app.run()
