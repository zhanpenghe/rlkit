import numpy as np

import torch.optim as optim

from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class PPO(TorchRLAlgorithm):

    def __init__(
            self,
            policy,
            baseline,
            lr_clip_range=1e-2,
            max_kl_step=1e-2,
            optimizer_class=optim.Adam,
            ):
        pass
