import os
import shutil
import subprocess

from deeprl_code_howto.experiments import logger


class FileManager(object):

    base_dir = os.path.abspath(__file__)[:os.path.abspath(__file__).find('src/deeprl_code_howto')]
    experiments_dir = os.path.join(base_dir, 'experiments')
    configs_dir = os.path.join(base_dir, 'configs')

    def __init__(self, exp_name, is_continue=False, config_fname=None, log_fname=None):
        self._exp_name = exp_name
        self._exp_dir = os.path.join(FileManager.base_dir, self._exp_name)

        if is_continue:
            assert os.path.exists(self._exp_dir),\
                'Experiment folder "{0}" does not exists, but continue = True'.format(self._exp_name)
        else:
            assert not os.path.exists(self._exp_dir),\
                'Experiment folder "{0}" exists, but continue = False'.format(self._exp_name)

        self._save_git()

        if config_fname is not None:
            shutil.copy(config_fname, os.path.join(self.exp_dir, 'config.py'))

        if log_fname is not None:
            logger.setup(log_fname)

    def _save_git(self):
        git_dir = os.path.join(self._exp_dir, 'git')
        os.makedirs(git_dir, exist_ok=True)

        git_commit_fname = os.path.join(git_dir, 'commit.txt')
        git_diff_fname = os.path.join(git_dir, 'diff.txt')

        if not os.path.exists(git_commit_fname):
            subprocess.call('cd {0}; git log -1 > {1}'.format(FileManager.base_dir, git_commit_fname), shell=True)
        if not os.path.exists(git_diff_fname):
            subprocess.call('cd {0}; git diff > {1}'.format(FileManager.base_dir, git_diff_fname), shell=True)

    ###################
    ### Experiments ###
    ###################

    @property
    def exp_dir(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        return self._exp_dir

    ##############
    ### Models ###
    ##############

    @property
    def models_dir(self):
        models_dir = os.path.join(self.exp_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        return models_dir