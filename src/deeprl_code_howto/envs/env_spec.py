"""
The EnvSpec defines the "hyperparameters" of the environment: the shapes/limits/dtypes of the
observations, goals, and actions.

Why is the EnvSpec separate from the Env? One way to think about it is that EnvSpec should probably be named
RobotSpec, since it defines what the robot's observations, goals, and actions. So with this separation, you can
have different robots (i.e., different EnvSpec) for the same Env.

The EnvSpec is needed for the dataset---so it knows what to store---and the model---so it knows what inputs/outputs
to expect.

Another advantage of separating the EnvSpec from the Env is if you do offline training (i.e., no on-policy data
gathering), you don't need the Env (which may have ugly robot-specific code like ROS)!
"""

class EnvSpec(object):

    def __init__(self, params):
        raise NotImplementedError

    @property
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def action_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def names(self):
        """
        Returns:
            list(str)
        """
        return self.observation_names + self.goal_names + self.action_names

    @property
    def names_to_shapes(self):
        """
        Knowing the dimensions is useful for building neural networks

        Returns:
            DotMap
        """
        raise NotImplementedError

    @property
    def names_to_limits(self):
        """
        Knowing the limits is useful for normalizing data

        Returns:
            DotMap
        """
        raise NotImplementedError

    @property
    def names_to_dtypes(self):
        """
        Knowing the data type is useful for building neural networks and datasets

        Returns:
            DotMap
        """
        raise NotImplementedError
