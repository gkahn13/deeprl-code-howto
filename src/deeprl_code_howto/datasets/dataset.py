"""
This is where you'll store data for training.

This is one of the classes you are most likely to subclass.
If you are training online in simulation you'll probably use a replay buffer in RAM.
If you are training offline from images you may use something like a tfrecord.
If you need to rebalance your data, you may override get_batch.
If you need to pass in previous observations/actions, you may override get_batch.
The examples go on.
"""

from dotmap import DotMap


class Dataset(object):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        raise NotImplementedError

    def add(self, obs, goal, action, done):
        # NOTE: if you are training from a fixed dataset, you won't use this
        raise NotImplementedError

    def get_batch(self):
        """
        Returns:
            inputs (DotMap)
            outputs (DotMap)
        """
        raise NotImplementedError
        # NOTE: What goes inputs and outputs will vary significantly depending on algorithm.
        #       If you are doing Q-learning, you need s,a,r,s'.
        #       If you are doing MBRL, you'll need s_0, ..., s_H, a_0, ..., a_H
        inputs = DotMap()
        outputs = DotMap()
        return inputs, outputs
