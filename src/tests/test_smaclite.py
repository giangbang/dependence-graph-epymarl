import time
import unittest
import envs  # noqa
import gymnasium as gym
import numpy as np
from tests.test_mpe import eq_rw


def sample_legal_action(env):
    avail_action = env.unwrapped.get_avail_actions()
    actions = []
    for ac_space, avail_ac in zip(env.action_space, avail_action):
        while True:
            ac = ac_space.sample()
            if avail_ac[ac] == 1:
                actions.append(ac)
                break
    actions = [int(a) for a in actions]
    return actions


class TestSMAClite(unittest.TestCase):
    def test_custom_smaclite_rw(self):
        from smaclite.env.maps.map import MapPreset

        for preset in MapPreset:
            map_info_name = preset.value.name

            with self.subTest(map_info_name):
                env_a = gym.make(f"smaclite/{map_info_name}-v0")
                env_b = gym.make(f"custom-smaclite/{map_info_name}-v0")
                for seed in range(10):
                    eq_rw(
                        env_a,
                        env_b,
                        seed=seed,
                        sample_ac_fn=sample_legal_action,
                        check_fn=self.assertLess,
                    )

    def test_APISMAClite(self):
        from envs.smaclite_wrapper import SMACliteWrapper

        env = SMACliteWrapper("custom-smaclite/10m_vs_11m", 1, 150)
        env.reset()
        action = sample_legal_action(env.env)

        obs, reward, truncated, terminated, info = env.step(action)

        env.get_state()
        env.get_obs_size()
        env.get_state_size()

        print(env.get_env_info())
        print("obs[0].shape:", obs[0].shape)
        print("reward.shape:", len(reward), reward, type(reward))

        env.close()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
