import abc

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings


# TODO: `BaseRLAlgorithm` requires a replaybuffer...
#        This only fit the off-policy situation, which are all of the algorithms
#        before. We need to maybe rewrite BaseRLAlgorithm so that it can also be
#        used for on-policy RL.
class OnPolicyRLAlgorithm(metaclass=abc.ABCMeta):

    # TODO: Add an evaluation environment
    def __init__(
        self,
        trainer,
        exploration_env,
        exploration_data_collector,):
        
        self.trainer = trainer
        self.expl_env = exploration_env
        self.expl_data_collector = exploration_data_collector
        # Originally training mode can be True/False. Here, training mode
        # is actually a string considering in the future, for meta-learning
        # algorithms, training mode might include 'meta-training', 'adaptation'
        # and etc.. However, in a simple reinforcement learning setting, this
        # should be 'training'/'testing'.
        self._training_mode = 'training'
        self._start_epoch = 0
        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    @abc.abstractmethod
    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, mode):
        self._training_mode = mode

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        return snapshot
    
    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
