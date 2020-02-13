import argparse
import os

from deeprl_code_howto.experiments.file_manager import FileManager
from deeprl_code_howto.trainers.trainer import Trainer
from deeprl_code_howto.utils.file_utils import import_config

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--continue', action='store_true')
args = parser.parse_args()

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)

file_manager = FileManager(params.exp_name,
                           is_continue=getattr(args, 'continue'),
                           log_fname='log_train.txt',
                           config_fname=config_fname)
dataset_train = params.dataset.cls(params.dataset.kwargs_train)
dataset_holdout = params.dataset.cls(params.dataset.kwargs_holdout)
model = params.model.cls(params.model.kwargs_train)

trainer = Trainer(params.trainer, file_manager, model, dataset_train, dataset_holdout)
trainer.run()