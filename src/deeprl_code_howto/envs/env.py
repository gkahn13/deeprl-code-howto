"""
This is where actions actually get executed on the robot, and observations are received.
"""

class Env(object):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        # NOTE: Why is there no reward? Are we even doing reinforcement learning???
        #       The reward is just another observation! Viewing it this way is much more flexible,
        #       especially with model-based RL
        return obs, goal, done

    def reset(self):
        raise NotImplementedError
        return obs, goal
