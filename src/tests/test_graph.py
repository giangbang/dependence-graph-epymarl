import time
import numpy as np
import envs  # noqa
import gymnasium as gym

import unittest


class TestGraphfn(unittest.TestCase):

    def testAPIPursuitMAgent(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            True,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )
        from envs.graph_env import get_knn_graph

        # env.reset()

        graph, dist = get_knn_graph(env, 3)
        # graph, dist = get_knn_graph(env, 3.5)

        print("==== MAgent2/adversarial-pursuit ====")
        print(graph)
        print(dist.shape)
        print(dist)
        print("==== MAgent2/adversarial-pursuit ====")

        env = envs.MAgentWrapper(
            1,
            "magent2/adversarial-pursuit-v4",
            False,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )
        from envs.graph_env import get_knn_graph

        # env.reset()

        graph, dist = get_knn_graph(env, 3)
        print("==== MAgent2/adversarial-pursuit ====")
        print(graph)
        print(graph.shape)
        print(dist)
        print("==== MAgent2/adversarial-pursuit ====")
        graph, dist = get_knn_graph(env, 3.5)

        assert isinstance(graph, np.ndarray) and len(graph.shape) == 2
        self.assertTrue(True)

    def testAPIBattleMAgent(self):
        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            True,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )
        from envs.graph_env import get_knn_graph

        # env.reset()

        graph, dist = get_knn_graph(env, 3)
        graph, dist = get_knn_graph(env, 3.5)

        print("==== MAgent2/battle ====")
        print(graph)
        print(graph.shape)
        print(dist)
        print("==== MAgent2/battle ====")

        env = envs.MAgentWrapper(
            1,
            "magent2/battle-v4",
            False,
            False,
            "sum",
            render_mode="rgb_array",
            # map_size=45,
        )
        from envs.graph_env import get_knn_graph

        # env.reset()

        graph, dist = get_knn_graph(env, 3)
        print("==== MAgent2 ====")
        print(graph)
        print(graph.shape)
        print(dist)
        print("==== MAgent2 ====")
        graph, dist = get_knn_graph(env, 3.5)

        assert isinstance(graph, np.ndarray) and len(graph.shape) == 2
        self.assertTrue(True)

    def testAPIMPE(self):
        from envs import MPEWrapper

        env = MPEWrapper(
            key="custom-pz-mpe-simple-spread-v3",
            time_limit=25,
            pretrained_wrapper=None,
            seed=1,
            N=10,
            common_reward=False,
            reward_scalarisation="sum",
        )

        from envs.graph_env import get_knn_graph

        graph, dist = get_knn_graph(env, 3)
        graph, dist = get_knn_graph(env, 3.5)

        assert isinstance(graph, np.ndarray) and len(graph.shape) == 2
        self.assertTrue(True)

    def testAPISMAClite(self):
        from envs import SMACliteWrapper

        env = SMACliteWrapper("custom-smaclite/10m_vs_11m", 1, 150)
        env.reset()

        from envs.graph_env import get_knn_graph

        graph, dist = get_knn_graph(env, 3)
        graph, dist = get_knn_graph(env, 3.5)

        print("==== SMAClite ====")
        print(dist)
        print(graph.shape)
        print(graph)
        print("==== SMAClite ====")

        assert isinstance(graph, np.ndarray) and len(graph.shape) == 2
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
