from functools import partial
import numpy as np
import envs  # noqa
import gymnasium as gym
import numpy as np

import unittest


class TestMPESpread(unittest.TestCase):
    
    def testMPE(self):
        from envs.star_spread_mpe import StarSpreadMPEv2Env

        env = StarSpreadMPEv2Env(
            n_agents=100
        )

        env.reset()
        for _ in range(10000):
            action = np.random.randint(0, 5, size=(100,))
            obs, _, truncated, terminated, infor = env.step(action)
            vobs = env.get_obs_vectorized()
            obs = env.get_obs()
            self.assertTrue(np.all(obs == vobs))
            if truncated or terminated:
                env.reset()
        env.close()


if __name__ == "__main__":
    unittest.main()
