import argparse
import os

from deeprl_code_howto.experiments.file_manager import FileManager
from deeprl_code_howto.utils.file_utils import import_config

# NOTE: You can add command line arguments here (e.g., which GPU to run things on)
#       I prefer against specifying experiment arguments here, and solely using the config file
#       But if you need to run large hyperparameter sweeps, you may need to reconsider.
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--continue', action='store_true')
args = parser.parse_args()

# import the config params
config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)

file_manager = FileManager(params.exp_name,
                           is_continue=getattr(args, 'continue'),
                           log_fname='log_train.txt',
                           config_fname=config_fname)

# instantiate classes from the params
env_spec = params.env_spec.cls(params.env_spec.params)
env_train = params.env.cls(params.env.params, env_spec)
env_holdout = params.env.cls(params.env.params, env_spec)
dataset_train = params.dataset_train.cls(params.dataset_train.params, env_spec)
dataset_holdout = params.dataset_holdout.cls(params.dataset_holdout.params, env_spec)
model = params.model.cls(params.model.params, env_spec)
policy = params.policy.cls(params.policy.params)
trainer = params.trainer.cls(params.trainer.params,
                             file_manager=file_manager,
                             model=model,
                             policy=policy,
                             dataset_train=dataset_train,
                             dataset_holdout=dataset_holdout,
                             env_train=env_train,
                             env_holdout=env_holdout)

# run training
trainer.run()