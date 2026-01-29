from gymnasium.spaces import Tuple
import gymnasium as gym
import numpy as np
from magent2.environments.magent_env import magent_parallel_env
import magent2


class Battle(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from magent2.environments.magent_env import magent_parallel_env
        env: magent_parallel_env = self.env.unwrapped._env.unwrapped

        self.tot_agents = len(env.possible_agents)
        self.n_agents = self.tot_agents // 2
        self.n_pt_agents = self.tot_agents - self.n_agents

        self.pt_action_space = self.action_space[-self.n_pt_agents:]
        self.pt_observation_space = self.observation_space[-self.n_pt_agents:]

        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])

        self.bot_agents = SimpleVectorizedChaseAttackAgent()
        self.prev_obs = None
        self.prev_mask = None

    def reset(self, seed=None, options=None):
        obss, info = super().reset(seed=seed, options=options)
        self.prev_obs = obss[self.n_agents :]
        self.prev_mask = np.ones(self.n_agents, dtype=bool)
        return obss[:self.n_agents], info

    def step(self, action):
        self.prev_obs = np.array(self.prev_obs)
        bot_action = tuple(self.bot_agents.act(self.prev_obs))
        action = tuple(action) + bot_action
        obs, rew, done, truncated, info = super().step(action)

        self.prev_obs = obs[self.n_agents :]
        self.prev_mask = done or truncated
        obs = obs[:self.n_agents]
        rew = rew[:self.n_agents]

        return obs, rew, done, truncated, info


# action mapping:
MOVE_UP_FAR = 0
MOVE_UP_LEFT = 1
MOVE_UP = 2
MOVE_UP_RIGHT = 3
MOVE_LEFT_FAR = 4
MOVE_LEFT = 5
DO_NOTHING = 6
MOVE_RIGHT = 7
MOVE_RIGHT_FAR = 8
MOVE_DOWN_LEFT = 9
MOVE_DOWN = 10
MOVE_DOWN_RIGHT = 11
MOVE_DOWN_FAR = 12

ATTACK_UP_LEFT = 13
ATTACK_UP = 14
ATTACK_UP_RIGHT = 15
ATTACK_LEFT = 16
ATTACK_RIGHT = 17
ATTACK_DOWN_LEFT = 18
ATTACK_DOWN = 19
ATTACK_DOWN_RIGHT = 20

_MOVE_ACTIONS = (
    MOVE_UP_FAR,
    MOVE_UP_LEFT,
    MOVE_UP,
    MOVE_UP_RIGHT,
    MOVE_LEFT_FAR,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_RIGHT_FAR,
    MOVE_DOWN_LEFT,
    MOVE_DOWN,
    MOVE_DOWN_RIGHT,
    MOVE_DOWN_FAR,
)


class RandomAgent:
    def act(self, observation: np.ndarray) -> int:
        return np.random.choice(_MOVE_ACTIONS)


class StillAgent:
    def act(self, observation: np.ndarray) -> int:
        return DO_NOTHING


class SimpleChaseAttackAgent:
    """
    Given an observation (expected 13x13xC, enemy presence channel default=3),
    this agent returns a single discrete action index:
      - Move toward nearest seen enemy until within attack range, then attack.
      - If no enemy seen, move randomly.
    """

    def __init__(self, enemy_channel: int = 3, rng: np.random.Generator = None):
        self.enemy_channel = enemy_channel
        self.rng = rng or np.random.default_rng()

    def _nearest_enemy_offset(self, enemy_map: np.ndarray):
        """Return (dx, dy) from center to nearest enemy (dx: cols, dy: rows), or None."""
        if enemy_map is None:
            return None
        h, w = enemy_map.shape
        cy, cx = h // 2, w // 2
        ys, xs = np.nonzero(enemy_map)
        if len(xs) == 0:
            return None
        dists = (xs - cx) ** 2 + (ys - cy) ** 2
        idx = int(np.argmin(dists))
        dx = int(xs[idx] - cx)
        dy = int(ys[idx] - cy)
        return dx, dy

    @staticmethod
    def _in_attack_range(dx: int, dy: int) -> bool:
        """Chebyshev distance ≤ 1 means adjacent/diagonal -> attackable."""
        return max(abs(dx), abs(dy)) <= 1

    @staticmethod
    def _direction_to_attack(dx: int, dy: int) -> int:
        """Map dx,dy to the correct attack action (uses your mapping)."""
        if dx == 0 and dy < 0:
            return ATTACK_UP
        if dx == 0 and dy > 0:
            return ATTACK_DOWN
        if dy == 0 and dx < 0:
            return ATTACK_LEFT
        if dy == 0 and dx > 0:
            return ATTACK_RIGHT
        if dx < 0 and dy < 0:
            return ATTACK_UP_LEFT
        if dx > 0 and dy < 0:
            return ATTACK_UP_RIGHT
        if dx < 0 and dy > 0:
            return ATTACK_DOWN_LEFT
        if dx > 0 and dy > 0:
            return ATTACK_DOWN_RIGHT
        return DO_NOTHING

    @staticmethod
    def _direction_to_move(dx: int, dy: int) -> int:
        """
        Choose a move action that advances toward (dx,dy).
        If the target is far (max abs >= 2) choose a *_FAR straight move along the primary axis.
        Otherwise prefer diagonal/straight short moves.
        """
        if dx == 0 and dy < 0:
            return MOVE_UP_FAR if abs(dy) > 2 else MOVE_UP
        if dx == 0 and dy > 0:
            return MOVE_DOWN_FAR if abs(dy) > 2 else MOVE_DOWN
        if dy == 0 and dx < 0:
            return MOVE_LEFT_FAR if abs(dx) > 2 else MOVE_LEFT
        if dy == 0 and dx > 0:
            return MOVE_RIGHT_FAR if abs(dx) > 2 else MOVE_RIGHT

        # diagonal when both non-zero and close
        if abs(dx) <= 1 and abs(dy) <= 1:
            if dx < 0 and dy < 0:
                return MOVE_UP_LEFT
            if dx > 0 and dy < 0:
                return MOVE_UP_RIGHT
            if dx < 0 and dy > 0:
                return MOVE_DOWN_LEFT
            if dx > 0 and dy > 0:
                return MOVE_DOWN_RIGHT

        # If target is farther, move along primary axis using *_FAR (or normal if far not available)
        if abs(dx) >= abs(dy):
            return MOVE_LEFT_FAR if dx < 0 else MOVE_RIGHT_FAR
        else:
            return MOVE_UP_FAR if dy < 0 else MOVE_DOWN_FAR

    def _random_move(self) -> int:
        """Return a random movement action."""
        return np.random.choice(_MOVE_ACTIONS)

    def act(self, observation: np.ndarray) -> int:
        """
        Primary method: given observation, return action index.
        - observation: expected shape (H,W,C) and enemy presence in channel self.enemy_channel.
        """
        if observation is None:
            return self._random_move()

        # Safely extract enemy presence channel if exists
        if observation.ndim >= 3 and observation.shape[2] > self.enemy_channel:
            enemy_map = observation[:, :, self.enemy_channel]
        else:
            # fallback: if map not in expected format, act randomly
            return self._random_move()

        d = self._nearest_enemy_offset(enemy_map)
        if d is None:
            return self._random_move()

        dx, dy = d
        if self._in_attack_range(dx, dy):
            return self._direction_to_attack(dx, dy)
        else:
            return self._direction_to_move(dx, dy)


class SimpleVectorizedChaseAttackAgent:
    """
    Vectorized version of SimpleChaseAttackAgent.
    Processes a batch of observations (N,H,W,C) at once and returns (N,) actions.
    """

    def __init__(self, enemy_channel: int = 3, rng: np.random.Generator = None):
        self.enemy_channel = enemy_channel
        self.rng = rng or np.random.default_rng()

        self._dx_flat = None

        self._dy_flat = None
        self._d2_flat = None
        self._shape = None
        
        self.epsilon = 0.5

    def act(self, observations: np.ndarray) -> np.ndarray:
        observations = np.array(observations)
        if observations is None or observations.ndim < 4:
            return self.rng.choice(
                _MOVE_ACTIONS,
                size=(observations.shape[0] if observations is not None else 1,),
            )

        N, H, W, C = observations.shape
        if C <= self.enemy_channel:
            return self.rng.choice(_MOVE_ACTIONS, size=N)

        # --- Lazy cache of dx, dy, d2 ---
        if self._dx_flat is None or self._shape != (H, W):
            cy, cx = H // 2, W // 2
            ys, xs = np.meshgrid(np.arange(H) - cy, np.arange(W) - cx, indexing="ij")
            self._dx_flat = xs.ravel().astype(np.int32)
            self._dy_flat = ys.ravel().astype(np.int32)
            self._d2_flat = (xs**2 + ys**2).ravel()
            self._shape = (H, W)

        enemy_maps = observations[..., self.enemy_channel]  # (N,H,W)

        # Flatten to (N, H*W)
        enemy_present = enemy_maps.reshape(N, -1) > 0

        # Masked squared distance
        d2 = np.where(enemy_present, self._d2_flat, np.inf)
        flat_idx = np.argmin(d2, axis=1)
        has_enemy = np.any(enemy_present, axis=1)

        # dx, dy per agent
        dx = self._dx_flat[flat_idx].copy()
        dy = self._dy_flat[flat_idx].copy()
        dx[~has_enemy] = 0
        dy[~has_enemy] = 0

        # Start with random moves, overwrite later
        actions = self.rng.choice(_MOVE_ACTIONS, size=N)

        # Attack range mask
        in_attack_range = (np.maximum(np.abs(dx), np.abs(dy)) <= 1) & has_enemy
        move_mask = has_enemy & ~in_attack_range

        # Fill in attack and move actions
        actions[in_attack_range] = self._direction_to_attack_batch(
            dx[in_attack_range], dy[in_attack_range]
        )
        actions[move_mask] = self._direction_to_move_batch(dx[move_mask], dy[move_mask])
        assert len(actions.shape) == 1, actions.shape

        # epsilon-greedy, bots act suboptimal for easier RL learning
        n = len(actions)
        k = int(np.round(self.epsilon * n))  # number of entries to replace

        if k > 0:
            idx = self.rng.choice(n, size=k, replace=False)  # random positions
            actions[idx] = self.rng.choice(_MOVE_ACTIONS, size=k)  # random replacements

        return actions

    @staticmethod
    def _direction_to_attack_batch(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Vectorized attack mapping."""
        out = np.full(dx.shape, DO_NOTHING, dtype=np.int32)

        # Orthogonal
        out[(dx == 0) & (dy < 0)] = ATTACK_UP
        out[(dx == 0) & (dy > 0)] = ATTACK_DOWN
        out[(dy == 0) & (dx < 0)] = ATTACK_LEFT
        out[(dy == 0) & (dx > 0)] = ATTACK_RIGHT

        # Diagonal
        out[(dx < 0) & (dy < 0)] = ATTACK_UP_LEFT
        out[(dx > 0) & (dy < 0)] = ATTACK_UP_RIGHT
        out[(dx < 0) & (dy > 0)] = ATTACK_DOWN_LEFT
        out[(dx > 0) & (dy > 0)] = ATTACK_DOWN_RIGHT
        return out

    @staticmethod
    def _direction_to_move_batch(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Vectorized movement mapping."""
        out = np.empty(dx.shape, dtype=np.int32)

        # Vertical
        up = (dx == 0) & (dy < 0)
        down = (dx == 0) & (dy > 0)
        out[up] = np.where(np.abs(dy[up]) > 2, MOVE_UP_FAR, MOVE_UP)
        out[down] = np.where(np.abs(dy[down]) > 2, MOVE_DOWN_FAR, MOVE_DOWN)

        # Horizontal
        left = (dy == 0) & (dx < 0)
        right = (dy == 0) & (dx > 0)
        out[left] = np.where(np.abs(dx[left]) > 2, MOVE_LEFT_FAR, MOVE_LEFT)
        out[right] = np.where(np.abs(dx[right]) > 2, MOVE_RIGHT_FAR, MOVE_RIGHT)

        # Diagonal when both small
        diag_close = (np.abs(dx) <= 1) & (np.abs(dy) <= 1) & (dx != 0) & (dy != 0)
        out[diag_close & (dx < 0) & (dy < 0)] = MOVE_UP_LEFT
        out[diag_close & (dx > 0) & (dy < 0)] = MOVE_UP_RIGHT
        out[diag_close & (dx < 0) & (dy > 0)] = MOVE_DOWN_LEFT
        out[diag_close & (dx > 0) & (dy > 0)] = MOVE_DOWN_RIGHT

        # Farther away: move along primary axis
        remaining = ~(up | down | left | right | diag_close)
        x_big = np.abs(dx[remaining]) >= np.abs(dy[remaining])
        y_big = ~x_big

        out[remaining] = np.where(
            x_big,
            np.where(dx[remaining] < 0, MOVE_LEFT_FAR, MOVE_RIGHT_FAR),
            np.where(dy[remaining] < 0, MOVE_UP_FAR, MOVE_DOWN_FAR),
        )
        return out
