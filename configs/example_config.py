"""
The config file should be the minimal description of a specific experiment you wish to run.

It should mostly contain aspects of the experiment that are unique.
If you find yourself replicating code in config files, you may consider moving that code into the main code base.

Having this file be python code (as opposed to something like a YAML) is very useful because some things you'll want
to vary will be code. For example, the cost function for training the neural network. While it may be tempting to still
use a YAML and have the string specify "cost_function_v0" and "cost_function_v1", you can get hidden bugs
where you change the code but not the config, and now your experiment is less reproducible.

I recommend using attribute dictionaries (e.g., DotMap) everywhere. It makes the code much cleaner compared to strings.
"""

from dotmap import DotMap as d

from deeprl_code_howto.datasets.dataset import Dataset
from deeprl_code_howto.envs.env import Env
from deeprl_code_howto.envs.env_spec import EnvSpec
from deeprl_code_howto.models.model import Model
from deeprl_code_howto.policies.policy import Policy
from deeprl_code_howto.trainers.trainer import Trainer


def get_env_spec_params():
    # NOTE: each "sub" param has a class and a params that's passed into it
    #       this separation is because the config file should not instantiate any classes, only define
    return d(
        cls=EnvSpec,
        params=d(
            # TODO
        )
    )

def get_env_params():
    return d(
        cls=Env,
        params=d(
            # TODO
        )
    )

def get_dataset_params(folders):
    # NOTE: you can pass in arguments so that you can reuse,
    #       such as using this function to create the training and holdout params
    return d(
        cls=Dataset,
        params=d(
            # TODO
        )
    )

def get_model_params():
    return d(
        cls=Model,
        params=d(
            # TODO
        )
    )

def get_trainer_params():
    return d(
        cls=Trainer,
        params=d(
            max_steps=int(1e5),
            step_train_env_every_n_steps=1,
            step_holdout_env_every_n_steps=50,
            holdout_every_n_steps=50,
            log_every_n_steps=int(1e3),
            save_every_n_steps=int(1e4),

            cost_fn = lambda model_outputs, outputs: d(),

            # TODO
        )
    )

def get_policy_params():
    return d(
        cls=Policy,
        params=d(
            # TODO
        )
    )

def get_params():
    return d(
        exp_name='bumpy',

        # NOTE: this is where all the params get created
        env_spec=get_env_spec_params(),
        env=get_env_params(),
        dataset_train=get_dataset_params(['/some/training/folders']),
        dataset_holdout=get_dataset_params(['/some/holdout/folders']),
        model=get_model_params(),
        trainer=get_trainer_params(),
        policy=get_policy_params(),
    )

# NOTE: this params is what will be imported
params = get_params()