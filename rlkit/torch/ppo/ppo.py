import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim

# This is currently the only distribution supporting..
# because rlkit did not implemented the distribution field.
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.torch_rl_algorithm import TorchTrainer


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOTrainer(TorchTrainer):

    def __init__(
        self,
        policy,
        baseline,
        lr_clip_range=1e-2,
        max_kl_step=1e-2,
        discount=0.99,
        gae_lambda=1,
        num_train_steps=1,
        policy_optimizer_cls=optim.Adam,
        baseline_optimizer_cls=optim.Adam):

        self.policy = policy
        self.baseline = baseline
        self.lr_clip_range = lr_clip_range
        self.max_kl_step = max_kl_step
        self.discount = discount
        self.gae_lambda = gae_lambda
        self._num_train_steps = num_train_steps

        self.policy_optimizer = policy_optimizer_cls(self.policy.parameters())
        self.baseline_optimizer = baseline_optimizer_cls(self.baseline.parameters())

        super().__init__()

    def _train_baseline(self, batch):
        # observations and returns
        observations = batch['observations']
        returns = batch['returns']

        # optimize baseline
        self.baseline_optimizer.zero_grad()
        returns_pred = self.baseline(observations)[0]
        critic = nn.MSELoss()
        loss = critic(returns_pred, returns)
        loss.backward()
        self.baseline_optimizer.step()

    def _train_policy(self, batch):
        # forward pass
        # o --> policy --> action_dist

        # obs.shape = [N_TRAJS, H, OBS_DIM]        
        obs = batch['observations']
        obs = obs.reshape((-1, self.policy.input_size))
        policy_outputs = self.policy(obs)
        mean, std = policy_outputs[1], policy_outputs[5]
        action_dist = TanhNormal(mean, std)

        # calculate discounted rewards
        advantages  = batch['advantages']

        self.policy_optimizer.zero_grad()
        # loglikelihood ratio of actions
        actions = batch['actions']
        # TODO: currently the distribution is not correct because
        # the policy class is not using multivariate distribution. To
        # fix these issues, I think we need to upgrade sac's policy
        # but this needs more effort to verify if sac is still working
        # with the multivariate version.
        log_prob = action_dist.log_prob(actions)
        loss = -torch.mean(
            torch.sum(log_prob, dim=1) * advantages)
        loss.backward()
        self.policy_optimizer.step()

    def train_from_torch(self, batch):
        self._train_policy(batch)
        self._train_baseline(batch)

    @property
    def networks(self):
        return [self.policy, self.baseline]

    def process_batch(self, batch):
        # calculate baslines, deltas, advantages for each path
        baselines = []
        returns = []
        all_path_baselines = [
            self.baseline(torch.Tensor(path['observations']))[0].detach()
            for path in batch
        ]

        # calculate advantages as it is done in https://arxiv.org/abs/1506.02438
        # for a more illustrative understanding, this is a blog post that describe
        # some insights of it: http://www.breloff.com/DeepRL-OnlineGAE/
        # TODO (Yong): write advantage computation as a PyTorch module
        for idx, path in enumerate(batch):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path['rewards'].squeeze() \
                + self.discount * path_baselines[1:] \
                - path_baselines[:-1]
            path['advantages'] = discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path['deltas'] = deltas

        # Trajectories --> Batch
        for idx, path in enumerate(batch):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])
            # returns
            path['returns'] = discount_cumsum(path['rewards'],
                                                      self.discount)
            returns.append(path['returns'])

        obs = [path['observations'] for path in batch]       
        actions = [path['actions'] for path in batch]
        rewards = [path['rewards'] for path in batch]
        returns = [path['returns'] for path in batch]
        advantages = [path['advantages'] for path in batch]
        # baseline is a list already, this is just a placeholder to consider
        # padding it?? (TODO: remove)

        batch_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            baselines=baselines,
        )

        for key, val in batch_data.items():
            batch_data[key] = np.concatenate(val, axis=0)

        # Check dimension 0 are matching
        assert batch_data['observations'].shape[0] == batch_data['actions'].shape[0]\
            == batch_data['rewards'].shape[0] == batch_data['returns'].shape[0]\
            == batch_data['advantages'].shape[0], 'Shape 0 of the fields in batch_data ' +\
                'are not matching!'

        return batch_data
