"""
The trainer is where everything is combined.
"""

from deeprl_code_howto.experiments import logger
from deeprl_code_howto.utils.python_utils import timeit


class Trainer(object):

    def __init__(self, params, file_manager, model, policy, dataset_train, dataset_holdout, env_train, env_holdout):
        self._file_manager = file_manager
        self._model = model
        self._policy = policy
        self._dataset_train = dataset_train
        self._dataset_holdout= dataset_holdout
        self._env_train = env_train
        self._env_holdout = env_holdout

        # steps
        # NOTE: Define everything according to training steps. Quantities such as "rollouts" and "epochs"
        #       are bad because they are environment/data dependent.
        self._max_steps = int(params.max_steps)
        self._step_train_env_every_n_steps = int(params.step_train_env_every_n_steps)
        self._step_holdout_env_every_n_steps = int(params.step_holdout_env_every_n_steps)
        self._holdout_every_n_steps = int(params.holdout_every_n_steps)
        self._log_every_n_steps = int(params.log_every_n_steps)
        self._save_every_n_steps = int(params.save_every_n_steps)

        # cost
        self._cost_fn = params.cost_fn

        raise NotImplementedError # TODO

    def run(self):
        """
        This is the main loop:
            - gather data
            - train the model
            - save the model
            - log progress
        """
        # NOTE: make sure you if you're experiment is killed that you can restart it where you left off
        self._restore_latest_checkpoint()

        obs_train, goal_train = self._env_train.reset()
        obs_holdout, goal_holdout = self._env_holdout.reset()

        for step in range(self._get_current_step(), self._max_steps + 1):
            # NOTE: always have some form of timing so that you can find bugs
            with timeit('total'):
                with timeit('train'):
                    self._train_step()

                if step > 0 and step % self._step_train_env_every_n_steps == 0:
                    with timeit('train env'):
                        obs_train, goal_train = self._env_step(self._env_train, self._dataset_train,
                                                               obs_train, goal_train)
                if step > 0 and step % self._step_holdout_env_every_n_steps == 0:
                    with timeit('holdout env'):
                        obs_holdout, goal_holdout = self._env_step(self._env_holdout, self._dataset_holdout,
                                                                   obs_holdout, goal_holdout)

                if step > 0 and step % self._holdout_every_n_steps == 0:
                    with timeit('holdout'):
                        self._holdout_step()

                if step > 0 and step % self._env_every_n_steps == 0:
                    with timeit('env'):
                        self._env_step()

                if step > 0 and step % self._save_every_n_steps == 0:
                    with timeit('save'):
                        self._save()

            if step > 0 and step % self._log_every_n_steps == 0:
                self._log()

    def _restore_latest_checkpoint(self):
        raise NotImplementedError

    def _get_current_step(self):
        raise NotImplementedError

    def _train_step(self):
        inputs, outputs = self._dataset_train.get_batch()
        model_outputs = self._model.call(inputs, training=True)
        cost = self._cost_fn(model_outputs, outputs)
        raise NotImplementedError

        # TODO: add to logger

    def _holdout_step(self):
        inputs, outputs = self._dataset_holdout.get_batch(self._batch_size, self._model.horizon)
        model_outputs = self._model.call(inputs, training=True)
        cost_dict = self._cost_fn(model_outputs, outputs)

        # TODO: add to logger

    def _env_step(self, env, dataset, obs, goal):
        action = self._policy.get_action(self._model, obs, goal)
        next_obs, next_goal, done = env.step(action)
        dataset.add(obs, goal, action, done)
        if done:
            next_obs, next_goal = env.reset()
        return next_obs, next_goal

    def _save(self):
        raise NotImplementedError

    def _log(self):
        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()

        # NOTE: log to something useful, like tensorboard

        raise NotImplementedError
