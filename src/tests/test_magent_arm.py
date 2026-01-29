import time
from functools import partial
import numpy as np
import envs  # noqa
import gymnasium as gym

import unittest
from utils.videos import convert_mp4_and_save


class TestMAgentArm(unittest.TestCase):
    def testA(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/combined-arms-v6",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            map_size = 20
        )

        n_range = 0
        n_melee = 0
        print(env._env.action_space)
        for a in env._env.action_space:
            if a.n > 10:
                n_range += 1
            else: 
                n_melee += 1

        env.reset()
        env.reset()
        import matplotlib.pyplot as plt
        plt.imshow(env.render())
        plt.show()

        for _ in range(5):
            action = [8] * (n_melee // 2) + [6] * (n_range // 2)
            action = action + action
            _, _, truncated, done, info = env.step(action)
        plt.imshow(env.render())
        plt.show()
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
