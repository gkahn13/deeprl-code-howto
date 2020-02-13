

class EnvSpec(object):

    def __init__(self, params):
        raise NotImplementedError

    @property
    def observation_names(self):
        raise NotImplementedError

    @property
    def action_names(self):
        raise NotImplementedError

    @property
    def names(self):
        return self.observation_names + self.action_names

    @property
    def names_to_shapes(self):
        raise NotImplementedError

    @property
    def names_to_limits(self):
        raise NotImplementedError

    @property
    def names_to_dtypes(self):
        raise NotImplementedError
