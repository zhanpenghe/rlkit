"""
Some implemented loss functions utilities.

These function are also pytorch modules but they
don't contain variables that need updates.
"""

import torch
from torch.distributions import Normal
import torch.nn as nn


class NegLogLikeli(nn.Module):

    def __init__(self, distribution=Normal, normalize_inputs=True):
        self._distribution = distribution
        # TODO Add normalization of inputs
        self._normalize_inputs = normalize_inputs
        super().__init__()
    
    def forward(self, input, **kwargs):
        """
        Compute the negative log likelihood.

        This forward pass assume that the input is a list
        and should be in form [samples, mean, log_std]
        """
        samples, means, log_stds = input
        dist = self._distribution(means, torch.exp(log_stds))
        loss =  - dist.log_prob(samples).mean(dim=0)
        return loss
