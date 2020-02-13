"""
This file is very similar to train; it's basically a subset of train
"""

import argparse
import os

from deeprl_code_howto.experiments.file_manager import FileManager
from deeprl_code_howto.utils.file_utils import import_config
from deeprl_code_howto.utils.python_utils import exit_on_ctrl_c

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()

config_fname = os.path.abspath(args.config)
model_fname = os.path.abspath(args.model)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
assert os.path.exists(model_fname), '{0} does not exist'.format(model_fname)

params = import_config(config_fname)

file_manager = FileManager(params.exp_name, is_continue=True)
env_spec = params.env_spec.cls(params.env_spec.params)
env = params.env.cls(params.env.params, env_spec)
model = params.model.cls(params.model.params, env_spec)
policy = params.policy.cls(params.policy.params)

### restore model
model.restore(args.model_fname)

### eval loop
exit_on_ctrl_c()
done = True
while True:
    if done:
        obs, goal = env.reset()

    action = policy.get_action(model, obs, goal)
    obs, goal, done = env.step(action)

