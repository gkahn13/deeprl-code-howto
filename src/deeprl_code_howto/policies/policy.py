"""
The policy uses the model to select actions using the current observation and goal.

The policy will vary significantly depending on the algorithm.
If the model is a policy that maps observation/goal directly to actions, then all you have to do is call the model.
If the model is a dynamics model, the policy will need to be a planner (e.g., random shooting, CEM).

Having the policy be separate from the model is advantageous since it allows you to easily swap in
different policies for the same model.
"""

class Policy(object):

    def __init__(self, params):
        raise NotImplementedError

    def get_action(self, model, observation, goal):
        """
        Args:
            model (Model):
            observation (DotMap):
            goal (DotMap):

        Returns:
            DotMap
        """
        raise NotImplementedError
