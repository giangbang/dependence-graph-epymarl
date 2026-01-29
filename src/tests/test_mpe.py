from functools import partial
import numpy as np
import envs  # noqa
import gymnasium as gym

import unittest


def sample_action(env):
    return [ac.sample() for ac in env.action_space]


def eq_rw(env_a, env_b, seed, check_fn, sample_ac_fn=sample_action):
    """Test if two environments' rewards are similar under same seeds"""
    env = env_a
    env.reset(seed=seed)
    tot_rw = []
    actions = []
    while True:
        action = sample_ac_fn(env)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        tot_rw.append(np.sum(reward))
        if terminated or truncated:
            break

    env = env_b
    env.reset(seed=seed)
    t = 0

    while True:
        action = actions[t]

        obs, reward, terminated, truncated, info = env.step(action)
        check_fn(abs(tot_rw[t] - np.sum(reward)), 1e-6)
        if terminated or truncated:
            break
        t += 1

    return True


class TestMPE(unittest.TestCase):
    def test_custom_pz_rw(self):
        env_a = gym.make("pz-mpe-simple-spread-v3")
        env_b = gym.make("custom-pz-mpe-simple-spread-v3")
        for seed in range(100):
            eq_rw(
                env_a,
                env_b,
                seed=seed,
                check_fn=self.assertLess,
            )

    def testMPEAPI(self):
        from envs import MPEWrapper

        env = MPEWrapper(
            key="custom-pz-mpe-simple-spread-v3",
            time_limit=25,
            pretrained_wrapper=None,
            seed=1,
            common_reward=False,
            reward_scalarisation="sum",
            render_mode="rgb_array",
        )

        env.reset()
        action = env._env.action_space.sample()

        obs, reward, truncated, terminated, info = env.step(action)

        env.get_state()
        env.get_obs_size()
        env.get_state_size()
        img = env.render()
        print(env.get_env_info())
        assert isinstance(img, np.ndarray)

        print("obs[0].shape:", obs[0].shape)
        print("reward.shape:", len(reward), reward, type(reward))
        env.close()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
