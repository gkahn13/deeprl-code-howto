from deeprl_code_howto.experiments import logger
from deeprl_code_howto.utils.python_utils import timeit


class Trainer(object):

    def __init__(self, params, file_manager, model, dataset_train, dataset_holdout):
        self._file_manager = file_manager
        self._model = model
        self._dataset_train = dataset_train
        self._dataset_holdout= dataset_holdout

        # steps
        self._max_steps = int(params.max_steps)
        self._holdout_every_n_steps = int(params.holdout_every_n_steps)
        self._log_every_n_steps = int(params.log_every_n_steps)
        self._save_every_n_steps = int(params.save_every_n_steps)

        raise NotImplementedError # TODO

    def run(self):
        self._restore_latest_checkpoint()

        for step in range(self._get_current_step(), self._max_steps + 1):
            with timeit('total'):
                self._train_step()

                if step > 0 and step % self._holdout_every_n_steps == 0:
                    self._holdout_step()

                # save
                if step > 0 and step % self._save_every_n_steps == 0:
                    with timeit('save'):
                        self._save()

            # log
            if step > 0 and step % self._log_every_n_steps == 0:
                self._log()

    def _restore_latest_checkpoint(self):
        raise NotImplementedError

    def _get_current_step(self):
        raise NotImplementedError

    def _train_step(self):
        with timeit('batch'):
            inputs, outputs = self._dataset_train.get_batch()

        with timeit('model'):
            model_outputs = self._model.call(inputs, training=True)

        with timeit('cost_fn'):
            cost = self._cost_fn(model_outputs, outputs)

        with timeit('optimizer'):
            raise NotImplementedError

        # TODO: add to logger

    def _holdout_step(self):
        with timeit('holdout'):
            inputs, outputs = self._dataset_holdout.get_batch(self._batch_size, self._model.horizon)
            model_outputs = self._model.call(inputs, training=True)
            cost_dict = self._cost_fn(model_outputs, outputs)

            # TODO: add to logger

    def _save(self):
        raise NotImplementedError

    def _log(self):
        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()

        raise NotImplementedError
