import time
from functools import partial
import numpy as np
import envs  # noqa
import gymnasium as gym

import unittest
from utils.videos import convert_mp4_and_save


class TestMAgentBattle(unittest.TestCase):

    def testMAGentBattleAPI(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
        )

        env.reset()
        action = env._env.action_space.sample()

        env.step(action)

        env.get_state()
        env.get_obs_size()
        env.get_state_size()
        img = env.render()
        print(env.get_env_info())
        assert isinstance(img, np.ndarray)
        env.close()

        self.assertTrue(True)

    def testMagentBattleBotEnv(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            True,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )
        env.reset()

        renders = [env.render()]

        for _ in range(200):
            action = env._env.action_space.sample()
            obs, reward, done, truncate, info = env.step(action)
            renders.append(env.render())

            if done or truncate:
                break
        convert_mp4_and_save(
            renders, "../videos", "action_space_sample_vs_vecbot_API.mp4"
        )
        self.assertTrue(True)

    def testVectorizedVSBot(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )

        from envs.pretrained.magent.battle import (
            SimpleVectorizedChaseAttackAgent,
            SimpleChaseAttackAgent,
        )

        agent = SimpleVectorizedChaseAttackAgent()
        other_agent = SimpleChaseAttackAgent()

        step = 0
        obs, _ = env.reset()
        renders = [env.render()]
        for _ in range(200):
            action = agent.act(obs[: len(obs) // 2])
            other_action = []
            for o in obs[len(obs) // 2 :]:
                other_action.append(other_agent.act(o))
            cat_actions = np.concatenate([action, other_action], axis=0)
            obs, rew, done, truncated, info = env.step(cat_actions)
            step += 1
            renders.append(env.render())
            if done or truncated:
                break
        convert_mp4_and_save(renders, "../videos", "vectorized_vs_bot")
        self.assertTrue(True)

    def testFPS(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )

        from envs.pretrained.magent.battle import (
            SimpleVectorizedChaseAttackAgent,
            SimpleChaseAttackAgent,
        )

        agent = SimpleVectorizedChaseAttackAgent()
        other_agent = SimpleChaseAttackAgent()

        agent_time = 0
        vectorized_agent_time = 0
        step = 0
        for _ in range(10):
            obs, _ = env.reset()
            for _ in range(200):
                start_time = time.time()
                action = agent.act(obs[: len(obs) // 2])
                vectorized_agent_time += time.time() - start_time
                other_action = []
                start_time = time.time()
                for o in obs[len(obs) // 2 :]:
                    other_action.append(other_agent.act(o))
                agent_time += time.time() - start_time
                cat_actions = np.concatenate([action, other_action], axis=0)
                obs, rew, done, truncated, info = env.step(cat_actions)
                step += 1
                if done or truncated:
                    break
        print(f"Bot FPS: {(step / agent_time):.2f}")
        print(f"Vectorized Bot FPS: {(step / vectorized_agent_time):.2f}")
        self.assertTrue(True)

    def testMagentBattleDifferentBots(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )

        obs, _ = env.reset()
        from envs.pretrained.magent.battle import (
            SimpleVectorizedChaseAttackAgent,
        )

        agent = SimpleVectorizedChaseAttackAgent()
        renders = []

        for _ in range(200):
            action = agent.act(obs)
            obs, rew, done, truncated, info = env.step(action)
            renders.append(env.render())
            if done or truncated:
                break

        # save videos

        convert_mp4_and_save(renders, "../videos", "vectorized_vs_vectorized")
        self.assertTrue(True)

    def testMAGentBattleCombatBot(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )

        obs, _ = env.reset()
        from envs.pretrained.magent.battle import (
            SimpleChaseAttackAgent,
            RandomAgent,
            StillAgent,
        )

        agent = SimpleChaseAttackAgent()
        random_agent = SimpleChaseAttackAgent()
        renders = []

        for _ in range(200):
            a = []
            for i, o in enumerate(obs):
                if i >= len(obs) // 2:
                    a.append(agent.act(o))
                else:
                    a.append(random_agent.act(o))
            obs, rew, done, truncated, info = env.step(a)
            renders.append(env.render())
            if done or truncated:
                break

        # save videos

        convert_mp4_and_save(renders, "../videos", "bot_vs_bot")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
