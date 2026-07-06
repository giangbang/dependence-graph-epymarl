import numpy as np
from envs.multiagentenv import MultiAgentEnv


class StarSpreadMPEv2Env(MultiAgentEnv):
    N_ACTIONS = 5      # no-op, up, down, left, right
    HUB_STEP  = 0.10   
    LEAF_STEP = 0.10

    _DIRS = np.array([
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0, -1.0],
        [-1.0,  0.0],
        [ 1.0,  0.0],
    ], dtype=np.float32)

    def __init__(
        self,
        n_agents: int = 5,
        time_limit: int = 50,
        seed: int = None,
        common_reward: bool = False,
        reward_scalarisation: str = "sum",
        noisy_rewards: bool = True,
        **kwargs,
    ):
        assert n_agents >= 2
        if n_agents > 30:
            self.HUB_STEP /= 2
            self.LEAF_STEP /= 2

        self.n_agents      = n_agents
        self.episode_limit = time_limit
        self.common_reward = common_reward
        self._rng          = np.random.RandomState(seed)
        self.noisy_rewards = noisy_rewards

        # N landmarks on unit circle, evenly spaced, radius 0.7
        angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
        self._landmarks = (0.7 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                           ).astype(np.float32)   # [N, 2]

        self._pos = np.zeros((n_agents, 2), dtype=np.float32)
        self._t   = 0

        # Oracle star graph: hub (col 0) influences every agent + self-loops
        g = np.eye(n_agents, dtype=np.uint8)
        g[:, 0] = 1.

        self._star_graph = g

        # Per-agent step sizes
        self._steps = np.array(
            [self.HUB_STEP] + [self.LEAF_STEP] * (n_agents - 1),
            dtype=np.float32,
        )

        # Observation sizes
        # [pos(2), all_landmark_pos(2N)]       = 2N+2
        self._obs_size_hub  = 4 * n_agents + 2
        self._obs_size_leaf = 4 * n_agents + 2
        self._obs_size = max(self._obs_size_hub, self._obs_size_leaf)

        self.lm_flat = self._landmarks.ravel()

    # ------------------------------------------------------------------
    # MultiAgentEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._t = 0
        self._pos = self._rng.uniform(-1.0, 1.0, (self.n_agents, 2)).astype(np.float32)
        return self.get_obs_vectorized(), {}

    def step(self, actions):
        self._t += 1
        # for i, a in enumerate(actions):
        #     delta = self._DIRS[int(a)] * self._steps[i]
        #     self._pos[i] = np.clip(self._pos[i] + delta, -1.0, 1.0)
        actions = np.asarray(actions, dtype=np.int64)

        deltas = self._DIRS[actions] * self._steps[:, None]
        self._pos = np.clip(self._pos + deltas, -1.0, 1.0)

        rewards    = self._compute_rewards()
        truncated = (self._t >= self.episode_limit)

        info = {}
        if truncated:
            info["episode_limit"] = True

        if self.common_reward:
            team_r = float(rewards.sum())
            return self.get_obs_vectorized(), team_r, False, truncated, info

        return self.get_obs_vectorized(), rewards.tolist(), False, truncated, info 

    def _compute_rewards(self) -> np.ndarray:
        # Pairwise distances: dists[i,j] = ||pos_i - landmark_j||
        diffs = self._pos[:, None, :] - self._landmarks[None, :, :]  # [N, N, 2]
        dists = np.linalg.norm(diffs, axis=-1)                        # [N, N]

        # Individual reward for leaves (and hub in common-reward mode)
        leaf_rewards = -dists.min(axis=-1).astype(np.float32)         # [N]

        # Hub reward: team coverage = -Σⱼ min_i ||pos_i - lm_j||
        # = negative sum of each landmark's distance to nearest agent
        hub_reward = float(-dists.min(axis=0).sum())                  # scalar

        rewards = leaf_rewards.copy()
        rewards[0] = hub_reward   # override hub reward

        if self.noisy_rewards:
            rewards += self._rng.normal(0, 1, size=self.n_agents).astype(np.float32)

        return rewards
    
    def get_obs_vectorized(self):
        n = self.n_agents

        pos = self._pos.astype(np.float32)          # (N, 2)
        lm = self._landmarks.astype(np.float32)     # (N, 2)

        pos_flat = pos.reshape(-1)                  # (2N,)
        lm_flat = lm.reshape(-1)                    # (2N,)

        obs = np.zeros((n, self._obs_size), dtype=np.float32)

        # First block: own position (all agents)
        obs[:, :2] = pos

        # Second block: visible agent positions
        pos_start = 2
        pos_end = pos_start + 2 * n

        # Hub (agent 0): sees all positions
        obs[0, pos_start:pos_end] = pos_flat

        # Leaves: only keep their own position
        leaf_pos = np.zeros((n - 1, n, 2), dtype=np.float32)
        leaf_idx = np.arange(1, n)
        leaf_pos[np.arange(n - 1), leaf_idx] = pos[1:]
        obs[1:, pos_start:pos_end] = leaf_pos.reshape(n - 1, -1)

        # Landmark block (shared for all)
        lm_start = pos_end
        lm_end = lm_start + 2 * lm.shape[0]
        obs[:, lm_start:lm_end] = lm_flat

        return obs

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        lm_flat = self._landmarks.flatten()   # 2N
        if agent_id == 0:
            # Hub: full state — own pos, all agent positions, all landmarks
            obs = np.concatenate([
                self._pos[0],           # 2
                self._pos.flatten(),    # 2N
                lm_flat,                # 2N
            ])
        else:
            # Leaf: own pos, all landmarks (no other leaves)
            tmp = np.zeros((self.n_agents, 2), dtype=np.float32)
            tmp[agent_id] = self._pos[agent_id]

            obs = np.concatenate([
                self._pos[agent_id],    # 2
                # self._pos[0],           # 2 (hub pos)
                tmp.flatten(),
                lm_flat,                # 2N
            ])
        out = np.zeros(self._obs_size, dtype=np.float32)
        out[:len(obs)] = obs
        return out

    def get_obs_size(self) -> int:
        return self._obs_size

    def get_state(self) -> np.ndarray:
        return np.concatenate([self._pos.flatten(),
                               self._landmarks.flatten()]).astype(np.float32)

    def get_state_size(self) -> int:
        return 2 * self.n_agents + 2 * self.n_agents

    def get_avail_actions(self):
        return [[1] * self.N_ACTIONS] * self.n_agents

    def get_avail_agent_actions(self, agent_id: int):
        return [1] * self.N_ACTIONS

    def get_total_actions(self) -> int:
        return self.N_ACTIONS

    def get_graph(self) -> np.ndarray:
        # reward graph and state graph
        return self._star_graph.copy()

    def get_stats(self):
        # Team coverage: fraction of landmarks with at least one agent within 0.15
        diffs = self._pos[:, None, :] - self._landmarks[None, :, :]  # [N, N, 2]
        dists = np.linalg.norm(diffs, axis=-1)                        # [N, N]
        covered = int((dists.min(axis=0) < 0.15).sum())
        return {
            "coverage": covered / self.n_agents,
            "team_coverage_loss": float(dists.min(axis=0).sum()),
        }

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def save_replay(self):
        pass
