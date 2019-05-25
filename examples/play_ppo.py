import time

import joblib
from point_env import PointEnv

def play():
    path = '/Users/adamhzp/Desktop/codes_local/rlkit/data/ppo-halfcheetah/ppo-halfcheetah_2019_05_24_17_54_01_0000--s-0/params.pkl'
    pkl = joblib.load(path)
    env = pkl['exploration/env']
    policy = pkl['exploration/policy']
    for _ in range(10):
        o = env.reset()
        for _ in range(100):
            env.render()
            a, _ = policy.get_action(o)
            o, _, done, _ = env.step(a)
            if done:
                break

play()
