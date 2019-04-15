"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import functional as F

from rlkit.policies.base import Policy, StochasticPolicy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class GaussianMlpPolicy(MlpPolicy, StochasticPolicy):

    def __init__(
            self,
            observation_dim,
            action_dim,
            *args,
            **kwargs,):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        super().__init__(
            input_size=observation_dim,
            output_size=2*action_dim,  # mean and std share network for now.
            *args,
            **kwargs,)
        self._distribution = Normal

    def forward(self, input, **kwargs):
        mlp_outputs = super().forward(input, **kwargs)
        
        # Split the mean and log std
        means = mlp_outputs.narrow(1, 0, self._action_dim)
        log_stds = mlp_outputs.narrow(1, self._action_dim, self._action_dim)
        stds = torch.exp(log_stds)
        distribution = self._distribution(means, stds)
        samples = distribution.sample()
        info = dict(
            mean=means,
            log_std=log_stds,
            dist=distribution,
        )
        return samples, info


class GaussianMlpBaseline(Mlp):
    def __init__(
        self,
        hidden_sizes,
        input_size,
        init_std=1.,
        *args,
        **kwargs):
        
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=1,
            input_size=input_size,
            *args,
            **kwargs)
        self._distribution = Normal

        # Create std variables
        init_std_log = torch.tensor(np.log(init_std))
        self._log_std = Variable(init_std_log, requires_grad=True,)

    def forward(self, input, **kwargs):
        means = super().forward(input, **kwargs)
        dist = self._distribution(means, torch.exp(self._log_std))
        samples = dist.sample()
        info = dict(mean=means, log_std=self._log_std, dist=dist)
        return samples, info
    
    @property
    def distribution(self):
        return self._distribution
