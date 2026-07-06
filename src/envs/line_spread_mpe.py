import numpy as np
from envs.multiagentenv import MultiAgentEnv


class LineSpreadMPEEnv(MultiAgentEnv):
    """
    Line/Cycle-structured Spread environment.

    Each agent has one co-located landmark (same index). Agents move on a
    directed line/cycle graph:

        graph[i, j] == 1  <=>  agent j can observe agent i

    i.e. graph[:, j] (a COLUMN) is the boolean mask of everything agent j
    observes, and graph[i, :] (a ROW) is the mask of everything that
    observes agent i. These are NOT the same thing, and it's easy to mix
    them up -- see `_observed_mask` below, which is the single place that
    resolves "what does agent i see".

    Topology: every agent always observes itself (self-loop). In addition,
    for a "line" graph, agent i (i > 0) observes agent i-1 one-directionally
    (agent 0 has no predecessor). For a "cycle" graph, the same holds with
    wraparound, so agent 0 also observes agent n-1.

    Reward: for each agent i, take the set of agents it observes (itself
    + its predecessor, if any) and the landmarks belonging to that same
    set. Reward is the negative sum, over those landmarks, of the distance
    to the closest agent within that same observed set.
    """

    N_ACTIONS = 5
    STEP = 0.10

    _DIRS = np.array([
        [0., 0.],
        [0., 1.],
        [0., -1.],
        [-1., 0.],
        [1., 0.],
    ], dtype=np.float32)

    def __init__(
        self,
        n_agents=5,
        graph_type="line",      # {"line", "cycle"}
        time_limit=50,
        seed=None,
        common_reward=False,
        noisy_rewards=True,
        **kwargs,
    ):
        assert graph_type in ("line", "cycle")
        assert n_agents >= 2

        self.n_agents = n_agents
        self.graph_type = graph_type
        self.episode_limit = time_limit
        self.common_reward = common_reward
        self.noisy_rewards = noisy_rewards

        self._rng = np.random.RandomState(seed)
        if n_agents > 30:
            self.STEP /= 2

        angles = np.linspace(
            0, 2 * np.pi,
            n_agents,
            endpoint=False
        )

        self._landmarks = (
            0.7 *
            np.stack(
                [np.cos(angles), np.sin(angles)],
                axis=1
            )
        ).astype(np.float32)
        np.random.shuffle(self._landmarks)

        self._pos = np.zeros((n_agents, 2), dtype=np.float32)
        self._t = 0

        self._graph = self._build_graph()

        self._obs_size = 2 + 2 * n_agents + 2 * n_agents

    def _build_graph(self):
        """
        graph[i, j] == 1  <=>  agent j observes agent i.

        Self-loops: every agent observes itself -> diagonal is 1.
        Chain edges: agent (i+1) observes agent i, i.e. graph[i, i+1] = 1.
        For "cycle", this wraps so agent 0 observes agent n-1.
        """
        n = self.n_agents
        g = np.eye(n, dtype=np.uint8)

        for i in range(n):

            right = i + 1

            if self.graph_type == "cycle":
                right %= n

            if right < n:
                g[i, right] = 1

        return g

    def _observed_mask(self, agent_id):
        """
        Boolean mask over all agents (length n) indicating which agents
        `agent_id` observes (always includes itself, plus its predecessor
        in the line/cycle if one exists).

        This is graph[:, agent_id] -- a COLUMN of self._graph -- since
        self._graph[i, j] == 1 means "j observes i".
        """
        return self._graph[:, agent_id].astype(bool)

    # ----------------------------------------------------

    def reset(self, seed=None, options=None):

        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._t = 0

        self._pos = self._rng.uniform(
            -1,
            1,
            (self.n_agents, 2)
        ).astype(np.float32)

        return self.get_obs_vectorized(), {}

    def step(self, actions):

        self._t += 1

        actions = np.asarray(actions)

        delta = self._DIRS[actions] * self.STEP

        self._pos = np.clip(
            self._pos + delta,
            -1,
            1
        )

        rewards = self._compute_rewards()

        truncated = (
            self._t >= self.episode_limit
        )

        info = {}

        if truncated:
            info["episode_limit"] = True

        if self.common_reward:
            return (
                self.get_obs_vectorized(),
                float(rewards.sum()),
                False,
                truncated,
                info,
            )

        return (
            self.get_obs_vectorized(),
            rewards.tolist(),
            False,
            truncated,
            info,
        )

    def _compute_rewards(self):

        n = self.n_agents

        rewards = np.zeros(
            n,
            dtype=np.float32
        )

        for i in range(n):

            visible = self._observed_mask(i)

            pos = self._pos[visible]
            lm = self._landmarks[visible]

            d = np.linalg.norm(
                pos[:, None] - lm[None],
                axis=-1,
            )

            rewards[i] = -d.min(axis=0).sum()

        if self.noisy_rewards:
            rewards += self._rng.normal(
                0,
                1,
                size=n,
            ).astype(np.float32)

        return rewards

    # ----------------------------------------------------

    def get_obs_vectorized(self):

        n = self.n_agents

        obs = np.zeros(
            (n, self._obs_size),
            dtype=np.float32,
        )

        obs[:, :2] = self._pos

        pos_start = 2
        lm_start = pos_start + 2 * n

        for i in range(n):

            visible = self._observed_mask(i)

            tmp_pos = np.zeros(
                (n, 2),
                dtype=np.float32
            )

            # tmp_lm = np.zeros(
            #     (n, 2),
            #     dtype=np.float32
            # )

            tmp_pos[visible] = self._pos[visible]
            # tmp_lm[visible] = self._landmarks[visible]

            obs[
                i,
                pos_start:lm_start
            ] = tmp_pos.flatten()

            obs[
                i,
                lm_start:
            ] = self._landmarks.flatten()

        return obs

    def get_obs(self):

        return [
            self.get_obs_agent(i)
            for i in range(self.n_agents)
        ]

    def get_obs_agent(self, agent_id):

        return self.get_obs_vectorized()[agent_id]

    def get_obs_size(self):

        return self._obs_size

    def get_state(self):

        return np.concatenate([
            self._pos.flatten(),
            self._landmarks.flatten(),
        ]).astype(np.float32)

    def get_state_size(self):

        return 4 * self.n_agents

    def get_total_actions(self):

        return self.N_ACTIONS

    def get_avail_actions(self):

        return [
            [1] * self.N_ACTIONS
        ] * self.n_agents

    def get_avail_agent_actions(
        self,
        agent_id,
    ):
        return [1] * self.N_ACTIONS

    def get_graph(self):

        return self._graph.copy()

    def get_stats(self):

        d = np.linalg.norm(
            self._pos[:, None]
            - self._landmarks[None],
            axis=-1,
        )

        return {
            "coverage_loss":
            float(
                d.min(axis=0).sum()
            )
        }

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):

        self._rng = np.random.RandomState(seed)

    def save_replay(self):
        pass