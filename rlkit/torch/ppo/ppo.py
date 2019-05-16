import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim

from rlkit.torch.torch_rl_algorithm import TorchTrainer


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPO(TorchTrainer):

    def __init__(
        self,
        policy,
        baseline,
        lr_clip_range=1e-2,
        max_kl_step=1e-2,
        discount=0.99,
        num_train_steps=1,
        policy_optimizer_cls=optim.Adam,
        baseline_optimizer_cls=optim.Adam):

        self.policy = policy
        self.baseline = baseline
        self.lr_clip_range = lr_clip_range
        self.max_kl_step = max_kl_step
        self.discount = discount

        self.policy_optimizer = policy_optimizer_cls(self.policy.parameters())
        self.baseline_optimizer = baseline_optimizer_cls(self.baseline.parameters())

        super().__init__(num_train_steps)

    def _train_baseline(self, batch):
        # observations and returns
        observations = np.concatenate([o for o in batch['observations']])
        returns = np.concatenate([ret for ret in batch['returns']])

        # optimize baseline
        self.baseline_optimizer.zero_grad()
        returns_pred = self.baseline(observations)
        critic = nn.MSELoss()
        loss = critic(returns_pred, returns)
        loss.backward()
        self.baseline_optimizer.step()

    def _train_policy(self, batch):
        # forward pass
        # o --> policy --> action_dist

        # obs.shape = [N_TRAJS, H, OBS_DIM]        
        obs = batch['observations']
        obs = obs.reshape((-1, self.policy.observation_dim))
        _, infos = self.policy(obs)
        action_dist = infos['dist']

        # calculate discounted rewards
        baselines = self.baseline(obs)
        rewards = batch['rewards']
        discounted_rew = rewards - baselines

        # loglikelihood ratio of actions
        actions = batch['actions']
        log_prob = action_dist.log_likelihood(actions)
        loss = -torch.mean(log_prob, axis=1)

        loss.backward()

    def train_from_torch(self, batch):
        self._train_policy(batch)
        self._train_baseline(batch)

    def networks(self):
        return [self.policy, self.baseline]
