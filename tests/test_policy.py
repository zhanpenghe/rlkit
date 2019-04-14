import numpy as np

from rlkit.torch.networks import GaussianMlpPolicy

def test_gaussian_mlp_policy():
    policy = GaussianMlpPolicy(
        observation_dim=2,
        action_dim=3,
        hidden_sizes=[10, 10],)

    actions, agent_info = policy.get_actions(
        np.random.uniform(size=[2, 2]))
    print(actions, agent_info)

if __name__ == "__main__":
    test_gaussian_mlp_policy()