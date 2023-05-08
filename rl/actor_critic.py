import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# Create the environment
env = gym.make("CartPole-v1")

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

if __name__ == '__main__':
    # Create the environment
    gym.envs.register(
        id='PolyWorld-v0',
        entry_point='env.pursuer_control:PolygonEnv',
        max_episode_steps=300,
    )
    env = gym.make('PolyWorld-v0', render_mode='human', size=10)
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    env.reset()
    env.render()
    env.step(0)
    env.step(1)
    env.step(0)
    env.step(1)
    # Set seed for experiment reproducibility
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()
