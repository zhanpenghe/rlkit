"""
Some tests for policies, baselines and loss functions.

TODO add assertion for all tests.
"""

import numpy as np
import torch

from rlkit.torch.pytorch_util import from_numpy
from rlkit.torch.networks import GaussianMlpPolicy, GaussianMlpBaseline
from rlkit.torch.loss_func import NegLogLikeli


def test_gaussian_mlp_policy():
    policy = GaussianMlpPolicy(
        observation_dim=2,
        action_dim=3,
        hidden_sizes=[10, 10],)

    actions, agent_info = policy.get_actions(
        np.random.uniform(size=[2, 2]))
    print(actions, agent_info)


def test_gaussian_mlp_baseline():
    baseline = GaussianMlpBaseline(input_size=2, hidden_sizes=[2, 2])
    val, infos = baseline(from_numpy(np.random.uniform(size=[2, 2])))
    print(val, infos)


def test_neg_log_likeli_loss():
    # INTEGRATION_TEST
    baseline = GaussianMlpBaseline(input_size=2, hidden_sizes=[2, 2])
    loss = NegLogLikeli()

    val, infos = baseline(from_numpy(np.random.uniform(size=[2, 2])))
    loss_inputs = [val, infos['mean'], infos['log_std']]
    loss_vals = loss(loss_inputs)
    print(loss_vals) 

    # UNIT_TEST
    means = torch.zeros([2, 2], dtype=torch.float32)
    log_stds = torch.zeros([2, 2], dtype=torch.float32)
    samples = torch.zeros([2, 2], dtype=torch.float32)
    loss_vals = loss([samples, means, log_stds])
    print(loss_vals)


if __name__ == "__main__":
    test_gaussian_mlp_policy()
    test_gaussian_mlp_baseline()
    test_neg_log_likeli_loss()
