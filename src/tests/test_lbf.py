import time
from functools import partial
import numpy as np
import envs  # noqa
import gymnasium as gym

import unittest


def setup(lbf_env):
    lbf_env = lbf_env.unwrapped
    # lbf_env.field_size=(4, 4)
    lbf_env.field = np.zeros(lbf_env.field_size, np.int32)

    row = [0, 1, 3]
    col = [0, 0, 3]
    i = 0

    for player in lbf_env.players:
        player.reward = 0

        player.setup(
            (row[i], col[i]),
            i+1,
            lbf_env.field_size,
        )
        i += 1

    row = [1, 3, 0]
    col = [1, 2, 3]

    for r, c in zip(row, col):
        lbf_env.field[row, col] = 3


class TestLBF(unittest.TestCase):
    def test(self):
        import envs.custom_lbf as custom_lbf
        from envs.custom_lbf.foraging.environment import ForagingEnv
        kwargs={
                "min_player_level": 1,
                "max_player_level": 2,
                "max_food_level": 3,
                "min_food_level": 1,
                "sight": 2,
                "max_episode_steps": 50,
                "force_coop": False,
                "grid_observation": False,
                "penalty":0.0,
            }

        players=3
        env = ForagingEnv(field_size=(4, 4), players=players, max_num_food=3, **kwargs)
        obs, _ = env.reset()
        for o in obs:
            print(o)
        
        setup(env)
        # env.render(np.ones((players, )*2), np.random.rand(players, players))
        # import matplotlib.pyplot as plt

        # plt.plot(a)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
