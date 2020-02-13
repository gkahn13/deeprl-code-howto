
class Env(object):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        return obs, goal, done

    def reset(self):
        raise NotImplementedError
        return obs, goal
