import numpy as np

from rlkit.torch.pytorch_util import from_numpy
from rlkit.torch.networks import GaussianMlpPolicy, GaussianMlpBaseline

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

if __name__ == "__main__":
    test_gaussian_mlp_policy()
    test_gaussian_mlp_baseline()
