

class Policy(object):

    def __init__(self, params):
        raise NotImplementedError

    def get_action(self, model, observation, goal):
        raise NotImplementedError
