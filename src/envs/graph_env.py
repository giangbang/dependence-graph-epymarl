import gymnasium as gym
import numpy as np
from .multiagentenv import MultiAgentEnv
from envs.stag_hunt import StagHunt


def get_erdos_renyi_graph(batch, n_agents, p):
    """Random graph, every two vertices has a probability of p to form an edge"""
    graph = np.random.rand(batch, n_agents, n_agents) < p
    # graph.diagonal(dim1=-2, dim2=-1).fill_(1)
    return graph


def get_knn_graph(env, k):
    original_env = env
    if isinstance(env, MultiAgentEnv) and not isinstance(env, StagHunt):
        try:
            env = env.env
        except:
            env = env._env
    if isinstance(env, gym.Env):
        env = env.unwrapped
    env_name = env.__class__.__name__
    threshold = None
    dead_agents = []

    if env_name == "Warehouse": 
        from rware.warehouse import Warehouse  # type: ignore

        env: Warehouse

        agent_pos = []

        for agent in env.agents:
            agent_pos.append([agent.x, agent.y])

        agent_pos = np.array(agent_pos)
        assert len(agent_pos.shape) == 2, agent_pos.shape

        dist = np.linalg.norm(
            np.expand_dims(agent_pos, axis=0) - np.expand_dims(agent_pos, axis=1),
            axis=-1,
        )  # n_agents x n_agents
    elif env_name == "PettingZooWrapper":
        from envs.pz_wrapper import PettingZooWrapper
        env: PettingZooWrapper

        _env = env._env
        if env.lib_name == "mpe":
            from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
            _env: SimpleEnv

            agent_pos = []

            for agent in _env.unwrapped.world.agents:
                agent_pos.append(agent.state.p_pos)

            agent_pos = np.array(agent_pos)
            assert len(agent_pos.shape) == 2, agent_pos.shape
            n_agents = len(agent_pos)

            dist = np.linalg.norm(
                np.expand_dims(agent_pos, axis=0) - np.expand_dims(agent_pos, axis=1),
                axis=-1,
            )  # n_agents x n_agents
            assert dist.shape == (n_agents, n_agents)
        else:
            raise NotImplementedError(env.lib_name + " not support agent graph")    

    elif env_name == "ForagingEnv":
        from lbforaging.foraging.environment import ForagingEnv  # type: ignore

        env: ForagingEnv

        agent_pos = []

        for agent in env.players:
            agent_pos.append(agent.position)

        agent_pos = np.array(agent_pos)
        assert len(agent_pos.shape) == 2, agent_pos.shape

        dist = np.abs(
            np.expand_dims(agent_pos, axis=0) - np.expand_dims(agent_pos, axis=1)
        ).sum(axis=-1)  # n_agents x n_agents
        threshold = 3


    elif env_name == "CustomSMACliteEnv":
        from envs.custom_smaclite.env import CustomSMACliteEnv

        env: CustomSMACliteEnv
        if not env.use_targeter_graph:
            n_agents = env.n_agents
            agent_pos = []
            dead_agents = []
            for agent in range(n_agents):
                if agent in env.agents:  # alive
                    agent_pos.append(env.agents[agent].pos)
                else:  # dead
                    agent_pos.append(np.zeros((2,), dtype=np.float32))
                    dead_agents.append(agent)

            agent_pos = np.array(agent_pos)

            dist = np.linalg.norm(
                np.expand_dims(agent_pos, axis=0) - np.expand_dims(agent_pos, axis=1),
                axis=-1,
            )  # n_agents x n_agents
            assert dist.shape == (n_agents, n_agents)
        else:
            return env.graph, None
    elif env_name == "MAgentEnv":
        from magent2.environments.magent_env import magent_parallel_env
        from envs.magent_wrapper import MAgentEnv

        magent_env: MAgentEnv = env
        env: magent_parallel_env = magent_env._env

        agent_pos = []
        if not original_env.pretrained_wrapper:
            for handle in env._all_handles:
                pos = env.env.get_pos(handle)
                agent_pos.append(pos)
            agent_pos = np.concatenate(agent_pos, axis=0)
        else:  # else only return the controlled agents coordinates
            env_name = env.metadata["name"]
            if "battle" in env_name:
                agent_pos = env.env.get_pos(env._all_handles[0])
                agent_pos = np.array(agent_pos)
            elif "adversarial_pursuit" in env.metadata["name"]:
                agent_pos = env.env.get_pos(env._all_handles[0])
                agent_pos = np.array(agent_pos)
            else:
                raise NotImplementedError(env_name + " not support graph env")

        assert agent_pos.shape[-1] == 2 and len(agent_pos.shape) == 2

        if "battle" in env_name or "arm" in env_name:
            # dead agents are not included in agent_pos
            live_agents = set(env.agents)
            n_agents = original_env.n_agents
            all_agent_pos = []
            dead_agents = []
            assert len(agent_pos) <= n_agents, f"{n_agents} {len(agent_pos)}"

            j = 0
            for i, agent in enumerate(env.possible_agents[:n_agents]):
                if agent in live_agents:
                    all_agent_pos.append(agent_pos[j])
                    j += 1
                else:
                    all_agent_pos.append(np.zeros((2,), dtype=np.float32)) 
                    dead_agents.append(i)

            # now it includes
            agent_pos = np.array(all_agent_pos)

        dist = np.abs(
            np.expand_dims(agent_pos, axis=0) - np.expand_dims(agent_pos, axis=1)
        ).sum(
            axis=-1
        )  # n_agents x n_agents
        assert dist.shape == (n_agents, n_agents), f"{dist.shape} and {n_agent} does not match"
        threshold = 3
    else:
        raise NotImplementedError(env_name + " not support agent graph")

    # masking out deads in the distance matrix
    if len(dead_agents) > 0:
        # dead_agents contains the indices of the deads
        dead_agents = np.array(dead_agents, dtype=int)
        n_dead = len(dead_agents)
        rows = np.repeat(np.arange(n_agents), n_dead)
        cols = np.repeat(dead_agents, n_agents)
        assert rows.shape == cols.shape
        dist[rows, cols] = 1e6  # not compare distance with dead agents

    n = len(dist)
    k = min(k, n)
    if isinstance(k, int):
        topk_indx = np.argpartition(dist, k - 1, axis=-1)[:, :k]

        rows = np.repeat(np.arange(n), k)
        cols = topk_indx.reshape(-1)

        graph = np.diag(np.zeros(len(dist)))
        if threshold is None:
            graph[rows, cols] = 1
        else:
            graph[rows, cols] = dist[rows, cols] <= (threshold + 1e-8)

        return graph, dist
    elif isinstance(k, float):
        k_rup = min(int(k + 1), n)
        k_rdown = min(int(k), n)
        k_rem = max(k - k_rdown, 0)
        # print(k_rup, k_rdown, k_rem)
        assert k_rem <= 1 and k_rem >= 0, k_rem
        topk_indx = np.argpartition(dist, k_rup - 1)[:, :k_rup]

        removed_sampple = np.random.uniform(size=n) < k_rem
        # print("removed", removed_sampple)
        rows = np.repeat(np.arange(n), k_rup)
        # print("rows", rows)
        cols = topk_indx.reshape(-1)

        graph = np.diag(np.zeros(len(dist)))
        if threshold is None:
            graph[rows, cols] = 1
        else:
            graph[rows, cols] = dist[rows, cols] <= (threshold + 1e-8)

        graph[np.arange(n), topk_indx[:, -1]] = removed_sampple

        return graph, dist
    else:
        raise ValueError(f"`k` must be either `float` or `int, instead got `{type(k)}`")
