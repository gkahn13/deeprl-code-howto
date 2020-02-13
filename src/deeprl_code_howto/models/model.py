from dotmap import DotMap
import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self, params):
        super(Model, self).__init__()

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
