import abc
from collections import OrderedDict

import gtimer as gt
from torch import nn as nn
from typing import Iterable

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.onpolicy_rl_algorithm import OnPolicyRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass


class TorchOnpolicyRLAlgorithm(OnPolicyRLAlgorithm):
    
    def __init__(
            self,
            trainer,
            exploration_env,
            exploration_data_collector,
            batch_size,
            max_path_length,
            num_epochs,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,):
        super().__init__(trainer, exploration_env, exploration_data_collector)

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
    
    def process_batch(self, batch):
        """Process paths to be compatible."""
        batch = self.trainer.process_batch(batch) 
        return batch

    def _train(self):
        self.training_mode = 'training'
        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True
        ):
            gt.stamp('exploration sampling', unique=False)
            batch = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.batch_size,  # num steps
                discard_incomplete_paths=False,
            )
            gt.stamp('samples processing', unique=False)
            processed_batch = self.process_batch(batch)
            gt.stamp('networks optimizing', unique=False)
            self.trainer.train(processed_batch)
            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
