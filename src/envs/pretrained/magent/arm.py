from enum import IntEnum
from gymnasium.spaces import Tuple
import gymnasium as gym
import numpy as np
from magent2.environments.magent_env import magent_parallel_env
import magent2


class Arm(gym.Wrapper):
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

class RangeAction(IntEnum):
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

    ATTACK_UP_FAR = 13
    ATTACK_UP_LEFT = 14
    ATTACK_UP = 15
    ATTACK_UP_RIGHT = 16
    ATTACK_LEFT_FAR = 17
    ATTACK_LEFT = 18
    ATTACK_RIGHT = 19
    ATTACK_RIGTH_FAR = 20
    ATTACK_LEFT_DOWN = 21
    ATTACK_DOWN = 22
    ATTACK_RIGHT_DOWN = 23
    ATTACK_DOWN_FAR = 24



class MeleeAction(IntEnum):
    MOVE_UP = 0
    MOVE_LEFT = 1
    DO_NOTHING = 2
    MOVE_RIGHT = 3
    MOVE_DOWN = 4

    ATTACK_UP = 5
    ATTACK_LEFT = 6
    ATTACK_RIGHT = 7
    ATTACK_DOWN = 8
