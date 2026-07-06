import time
from functools import partial
import numpy as np
import gymnasium as gym

import unittest

class TestCognac(unittest.TestCase):
    def test(self):
        from envs.cognac_wrapper import COGNACWrapper
        from envs.cognac.utils.graph_utils import generate_adjacency_matrix

        # Utility for generating random adjacency matrix
        adjacency_matrix = generate_adjacency_matrix(10)

        config={
                "adjacency_matrix": adjacency_matrix
            }

        env_list = [
            "binary_consensus",
            # "grid_firefighting_graph",
            # "row_firefighting_graph",
            # "multi_commodity_flow",
            "sysadmin_network",
        ]
        for env_name in env_list:
            print(f"Testing {env_name}...")
            env = COGNACWrapper(
                seed=1,
                key=env_name,
                common_reward=False,
                reward_scalarisation="sum",
                **config if env_name not in ["grid_firefighting_graph", "row_firefighting_graph"] else {},
            )
            obs, _ = env.reset()
            for o in obs:
                print("o:", o)
            env.step([0] * env.n_agents)

if __name__ == "__main__":
    unittest.main()
