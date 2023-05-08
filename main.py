# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import tensorflow as tf
import numpy as np
import jurisdiction
from renderer import qt_display


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
