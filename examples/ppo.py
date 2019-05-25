from gym.envs.mujoco import HalfCheetahEnv

from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import GaussianMlpBaseline
from rlkit.torch.ppo import PPOTrainer
import rlkit.torch.pytorch_util as ptu
# TODO `MakeDeterministic` is not used
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchOnpolicyRLAlgorithm

from point_env import PointEnv


def experiment(variant):

    expl_env = PointEnv(goal=(3, 3), random_start=False, action_scale=0.1)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    baseline = GaussianMlpBaseline(
        input_size=obs_dim,
        hidden_sizes=(32, 32),)

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(32, 32),)

    trainer = PPOTrainer(policy=policy, baseline=baseline)

    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )

    algo = TorchOnpolicyRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        exploration_data_collector=expl_path_collector,
        batch_size=512,
        num_epochs=500,
        num_expl_steps_per_train_loop=512,
        num_train_loops_per_epoch=1,
        num_trains_per_train_loop=32,
        max_path_length=100,
    )

    algo.to(ptu.device)
    algo.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(algorithm="PPO", version="normal",)
    setup_logger('ppo-halfcheetah', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
