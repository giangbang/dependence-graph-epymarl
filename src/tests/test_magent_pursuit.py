import envs  # noqa
import unittest
import numpy as np
from utils.videos import convert_mp4_and_save


class TestMAgentPursuit(unittest.TestCase):
    def testRender(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            map_size=18,
        )

        env.reset()
        import matplotlib.pyplot as plt

        plt.imshow(env.render())
        plt.show()

        self.assertTrue(True)
        

    def testAction(self):
        return
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            map_size=18,
        )

        env.reset()
        import matplotlib.pyplot as plt

        plt.imshow(env.render())
        plt.show()
        n_agent = len(env._env.action_space)
        n_predator = 0
        for ac in env._env.action_space:
            n_predator += 1 if ac.n == 13 else 0
        n_prey = n_agent - n_predator
        print("predator", n_predator)
        print("prey", n_prey)

        for _ in range(10):
            action = [13] * n_predator + [1] * n_prey
            # action = env._env.action_space.sample()
            print(action)
            env.step(action)
            plt.imshow(env.render())
            plt.show()
        # self.assertTrue(True)

    def testMAGentPursuitAPI(self):
        return
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
        )

        env.reset()
        action = env._env.action_space.sample()

        print(env._env.action_space)

        env.step(action)

        env.get_state()
        env.get_obs_size()
        env.get_state_size()
        img = env.render()
        print(env.get_env_info())
        assert isinstance(img, np.ndarray)
        env.close()

        # self.assertTrue(True)

    def testFPS(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )
        
        env.flatten_obs = False

        from envs.pretrained.magent.pursuit import (
            SimplePrey,
            SimpleVectorizedPrey,
            SimplePredator,
        )

        agent = SimplePredator()
        other_agent = SimplePrey()

        other_vectorized_agent = SimpleVectorizedPrey()

        agent_time = 0
        vectorized_agent_time = 0
        step = 0
        import time

        n_predator = 25
        n_prey = env.n_agents - n_predator
        for _ in range(10):
            obs, _ = env.reset()
            print(obs[0].shape)
            for _ in range(200):
                action = []
                for o in obs[:n_predator]:
                    action.append(agent.act(o))

                other_action = []
                start_time = time.time()
                for o in obs[n_predator:]:
                    other_action.append(other_agent.act(o))
                agent_time += time.time() - start_time
                cat_actions = np.concatenate([action, other_action], axis=0)
                obs, rew, done, truncated, info = env.step(cat_actions)
                step += 1
                if done or truncated:
                    break

        vectorized_step = 0
        for _ in range(10):
            obs, _ = env.reset()
            for _ in range(200):
                action = []
                for o in obs[:n_predator]:
                    action.append(agent.act(o))

                other_action = []
                start_time = time.time()
                other_action = other_vectorized_agent.act(obs[n_predator:])
                vectorized_agent_time += time.time() - start_time
                cat_actions = np.concatenate([action, other_action], axis=0)
                obs, rew, done, truncated, info = env.step(cat_actions)
                vectorized_step += 1
                if done or truncated:
                    break

        print(f"Bot FPS: {(step / agent_time):.2f}")
        print(f"Vectorized Bot FPS: {(vectorized_step / vectorized_agent_time):.2f}")
        self.assertTrue(True)

    def testPursuitEnvAPI(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            True,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=9,
        )
        obs, _ = env.reset()
        renders = [env.render()]
        from envs.pretrained.magent.pursuit import SimplePredator

        agent = SimplePredator()

        while True:
            action = []
            for o in obs:
                action.append(agent.act(o))
            obs, _, terminated, truncated, info = env.step(action)
            renders.append(env.render())
            if truncated or terminated:
                break
        convert_mp4_and_save(renders, "../videos", "vectorized_prey_vs_predator")
        self.assertTrue(True)

    def testpursuitBots(self):
        return
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=9,
        )

        obs, _ = env.reset()
        print(obs[-1][..., 3])
        self.assertTrue(True)

        n_agent = len(env._env.action_space)
        n_predator = 0
        for ac in env._env.action_space:
            n_predator += 1 if ac.n == 13 else 0
        n_prey = n_agent - n_predator
        print("predator", n_predator)
        print("prey", n_prey)

        import matplotlib.pyplot as plt

        # plt.imshow(env.render())
        # plt.show()
        # return

        from envs.pretrained.magent.pursuit import SimplePredator, SimplePrey

        predator = SimplePredator()
        prey = SimplePrey()

        renders = [env.render()]

        for _ in range(300):
            actions = []
            for i, o in enumerate(obs):
                if i < n_predator:
                    actions.append(predator.act(o))
                else:
                    actions.append(prey.act(o))
            obs, reward, terminated, truncated, info = env.step(actions)
            renders.append(env.render())
            if terminated or truncated:
                break
        convert_mp4_and_save(renders, "../videos", "predator_prey")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
