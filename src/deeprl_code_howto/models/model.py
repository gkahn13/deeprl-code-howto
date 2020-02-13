"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""

from dotmap import DotMap
import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self, params, env_spec):
        super(Model, self).__init__()

        self._env_spec = env_spec
        raise NotImplementedError

    def call(self, inputs, training=False):
        """
        Args:
            inputs (DotMap):
            training (bool):

        Returns:
            outputs (DotMap):
        """
        raise NotImplementedError

    def restore(self, fname):
        raise NotImplementedError
