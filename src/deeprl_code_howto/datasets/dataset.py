

class Dataset(object):

    def __init__(self, params):
        self._env_spec = params.env_spec
        raise NotImplementedError

    def get_batch(self):
        raise NotImplementedError
